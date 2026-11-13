"""The code graders that come out of ``evals/base.py``'s ``score()`` branches, one constructor per rule.

Each constructor returns a ``Grader`` (``kind="code"``); the branch it replaces is named in the catalogue. They read
``sample.output_parsed`` / ``sample.output_text`` and the item's ``expected`` and nothing else.

A grader that reads the reply as a JSON structure (``label_share``, ``bool_field``, ``ranks_first``,
``verdict_equals``) takes ``sample.output_parsed`` when the run parsed something, else parses ``output_text`` with
``parse_json_text``; ``json_valid`` and ``schema_valid`` always parse the text, because the reply's JSON is what they
judge. An ``expected`` that is not the shape a grader reads raises ``CannotGrade`` naming the shape.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence

from ..models import CoreModel
from ..sample import Sample
from ..score import GraderKind, Score
from ..values import JsonValue, ParseStatus
from .protocol import CannotGrade, Grader

Rule = Callable[[Sample, JsonValue], Score]


class ParsedJson(CoreModel):
    """What ``parse_json_text`` made of a model's text."""

    value: JsonValue
    status: ParseStatus


def parse_json_text(text: str) -> ParsedJson:
    """Parse the first JSON value in ``text`` (a fenced block or bare); ``status`` is ``"ok"`` or ``"not_json"``."""
    body = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, flags=re.DOTALL)
    if fenced:
        body = fenced.group(1)
    try:
        value: JsonValue = json.loads(body)
    except (json.JSONDecodeError, RecursionError):
        return ParsedJson(value=None, status="not_json")
    return ParsedJson(value=value, status="ok")


class _CodeGrader:
    """A ``kind="code"`` grader over one rule function; every constructor below builds one."""

    kind: GraderKind = "code"

    def __init__(self, id: str, rule: Rule, *, version: str = "1") -> None:
        self._id = id
        self._version = version
        self._rule = rule

    @property
    def id(self) -> str:
        return self._id

    @property
    def version(self) -> str:
        return self._version

    def grade(self, sample: Sample, expected: JsonValue) -> Score:
        return self._rule(sample, expected)


class _WrongLabels:
    """``with_wrong_labels``: the wrapped grader, graded against ``wrong(sample, expected)``."""

    def __init__(self, grader: Grader, wrong: Callable[[Sample, JsonValue], JsonValue]) -> None:
        self._grader = grader
        self._wrong = wrong

    @property
    def id(self) -> str:
        return f"{self._grader.id}+wrong_label"

    @property
    def version(self) -> str:
        return self._grader.version

    @property
    def kind(self) -> GraderKind:
        return self._grader.kind

    def grade(self, sample: Sample, expected: JsonValue) -> Score:
        return self._grader.grade(sample, self._wrong(sample, expected))


def _number(text: str) -> float | None:
    try:
        return float(text.replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def _json_of(sample: Sample) -> JsonValue:
    if sample.output_parsed is not None:
        return sample.output_parsed
    return parse_json_text(sample.output_text).value


def _mapping(id: str, value: JsonValue, what: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict):
        raise CannotGrade(f"{id}: {what} must be a JSON object, not {type(value).__name__}")
    return value


def _field(id: str, expected: JsonValue, key: str) -> JsonValue:
    mapping = _mapping(id, expected, "expected")
    if key not in mapping:
        raise CannotGrade(f"{id}: expected has no {key!r} key")
    return mapping[key]


def _verdict(passed: bool, *, answer: str | None = None, explanation: str | None = None) -> Score:
    return Score(value=float(passed), answer=answer, explanation=explanation)


def last_number(*, tolerance: float = 0.0) -> Grader:
    """1.0 when the last number in the reply equals ``expected`` within ``tolerance``.

    [tinygsm8k]
    """

    def rule(sample: Sample, expected: JsonValue) -> Score:
        numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", sample.output_text)
        got = _number(numbers[-1]) if numbers else None
        want = _number(str(expected))
        passed = got is not None and want is not None and abs(got - want) <= tolerance
        return _verdict(passed, answer=numbers[-1] if numbers else None)

    return _CodeGrader("last_number", rule)


def answer_letter(*, choices: str = "ABCD") -> Grader:
    """1.0 when the last ``ANSWER: X`` letter (X in ``choices``) equals ``expected``.

    [quality, tinymmlu]
    """
    marker = re.compile(rf"ANSWER:\s*\(?([{re.escape(choices)}])\b")

    def rule(sample: Sample, expected: JsonValue) -> Score:
        found = marker.findall(sample.output_text)
        passed = bool(found) and found[-1] == str(expected).strip().upper()
        return _verdict(passed, answer=found[-1] if found else None)

    return _CodeGrader("answer_letter", rule)


def json_valid() -> Grader:
    """1.0 when the reply parses as JSON, 0.0 when it does not."""

    def rule(sample: Sample, expected: JsonValue) -> Score:
        status = parse_json_text(sample.output_text).status
        return _verdict(status == "ok", explanation=status)

    return _CodeGrader("json_valid", rule)


def schema_valid(schema: Mapping[str, JsonValue] | None = None) -> Grader:
    """1.0 when the reply parses and validates against ``schema`` (default: the item's ``expected``).

    Needs the ``schema`` extra; the import is lazy. [jsonschemabench]
    """

    def rule(sample: Sample, expected: JsonValue) -> Score:
        parsed = parse_json_text(sample.output_text)
        if parsed.status != "ok":
            return _verdict(False, explanation=parsed.status)
        against = schema if schema is not None else _mapping("schema_valid", expected, "expected")
        return _verdict(_schema_valid(dict(against), parsed.value))

    return _CodeGrader("schema_valid", rule)

def _schema_valid(schema: dict[str, JsonValue], value: JsonValue) -> bool:
    try:
        from jsonschema.exceptions import SchemaError
        from jsonschema.validators import validator_for
    except ModuleNotFoundError as missing:
        raise ModuleNotFoundError("schema_valid needs the schema extra: pip install 'gigai-core[schema]'") from missing
    try:
        return bool(validator_for(schema)(schema).is_valid(value))
    except (SchemaError, TypeError, ValueError, RecursionError, re.error):
        return False


def label_share(*, field: str = "labels") -> Grader:
    """The share of keys in ``expected[field]`` the reply's ``field`` object got exactly right.

    [contractnli]
    """
    id = "label_share"

    def rule(sample: Sample, expected: JsonValue) -> Score:
        want = _mapping(id, _field(id, expected, field), f"expected[{field!r}]")
        parsed = _json_of(sample)
        got = parsed.get(field) if isinstance(parsed, dict) else None
        if not isinstance(got, dict) or not want:
            return Score(value=0.0)
        right = sum(got.get(key) == value for key, value in want.items())
        return Score(value=right / len(want), explanation=f"{right} of {len(want)} labels right")

    return _CodeGrader(id, rule)


def bool_field(field: str, *, invert: bool = False, expected_key: str | None = None) -> Grader:
    """1.0 when the reply's boolean ``field`` equals ``expected[expected_key]`` (negated when ``invert``).

    ``expected_key`` defaults to ``field``. [ragtruth: ``supported`` equals not ``hallucinated``:
    ``bool_field("supported", invert=True, expected_key="hallucinated")``]
    """
    id = "bool_field"
    key = field if expected_key is None else expected_key

    def rule(sample: Sample, expected: JsonValue) -> Score:
        want = _field(id, expected, key)
        if not isinstance(want, bool):
            raise CannotGrade(f"{id}: expected[{key!r}] must be a boolean, not {type(want).__name__}")
        parsed = _json_of(sample)
        got = parsed.get(field) if isinstance(parsed, dict) else None
        return _verdict(isinstance(got, bool) and got == (want != invert))

    return _CodeGrader(id, rule)


_IFEVAL_UNVERIFIED = "ifeval: no check for instruction type"


def ifeval() -> Grader:
    """Pass when every instruction in ``expected`` is followed.

    15 of IFEval's 25 instruction types are implemented; any other type raises ``CannotGrade`` (the item is
    ungraded, not failed). [ifeval]
    """
    id = "ifeval"

    def rule(sample: Sample, expected: JsonValue) -> Score:
        instructions = _field(id, expected, "instruction_id_list")
        kwargs_list = _field(id, expected, "kwargs")
        if not isinstance(instructions, list) or not isinstance(kwargs_list, list) or len(instructions) != len(kwargs_list):
            raise CannotGrade(f"{id}: expected needs instruction_id_list and kwargs lists of the same length")
        verdicts: list[bool] = []
        for instruction, kwargs in zip(instructions, kwargs_list, strict=True):
            if not isinstance(instruction, str):
                raise CannotGrade(f"{id}: instruction ids must be strings, not {type(instruction).__name__}")
            given = {key: value for key, value in kwargs.items() if value is not None} if isinstance(kwargs, dict) else {}
            try:
                verdict = _ifeval_check(instruction, sample.output_text, given)
            except (KeyError, TypeError, ValueError) as bad:
                raise CannotGrade(f"{id}: instruction {instruction} has unusable kwargs ({bad})") from bad
            if verdict is None:
                raise CannotGrade(f"{_IFEVAL_UNVERIFIED} {instruction}")
            verdicts.append(verdict)
        return _verdict(all(verdicts), explanation=f"{sum(verdicts)} of {len(verdicts)} instructions followed")

    return _CodeGrader(id, rule)


def _relation(count: int, relation: JsonValue, n: int) -> bool:
    if relation == "less than":
        return count < n
    return count >= n


def _int(value: JsonValue) -> int:
    if isinstance(value, int | float | str):
        return int(value)
    raise TypeError(f"not a number: {value!r}")


def _words(value: JsonValue) -> Sequence[str]:
    if isinstance(value, list) and all(isinstance(word, str) for word in value):
        return [str(word) for word in value]
    raise TypeError(f"not a list of words: {value!r}")


def _ifeval_check(instruction: str, text: str, kwargs: Mapping[str, JsonValue]) -> bool | None:
    """One IFEval instruction checked with the standard library; ``None`` for a type this does not verify.

    Approximate: 15 of the 25 upstream instruction types. Known gaps against upstream: english_capital and
    english_lowercase skip its langdetect == "en" check, and number_sentences splits on a regex where it uses
    nltk punkt.
    """
    if instruction == "keywords:existence":
        return all(re.search(re.escape(word), text, flags=re.IGNORECASE) for word in _words(kwargs["keywords"]))
    if instruction == "keywords:forbidden_words":
        forbidden = _words(kwargs["forbidden_words"])
        return not any(re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE) for word in forbidden)
    if instruction == "keywords:frequency":
        keyword = kwargs["keyword"]
        if not isinstance(keyword, str):
            raise TypeError("keyword must be a string")
        found = len(re.findall(re.escape(keyword), text, flags=re.IGNORECASE))
        return _relation(found, kwargs.get("relation"), _int(kwargs["frequency"]))
    if instruction == "punctuation:no_comma":
        return "," not in text
    if instruction == "detectable_format:title":
        return re.search(r"<<[^\n]*\S[^\n]*>>", text) is not None
    if instruction == "length_constraints:number_sentences":
        sentences = [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
        return _relation(len(sentences), kwargs.get("relation"), _int(kwargs["num_sentences"]))
    if instruction == "length_constraints:number_paragraphs":
        paragraphs = re.split(r"\s?\*\*\*\s?", text)
        if any(not part.strip() for index, part in enumerate(paragraphs) if 0 < index < len(paragraphs) - 1):
            return False
        return len([part for part in paragraphs if part.strip()]) == _int(kwargs["num_paragraphs"])
    if instruction == "detectable_content:number_placeholders":
        return len(re.findall(r"\[.*?\]", text)) >= _int(kwargs["num_placeholders"])
    if instruction == "startend:quotation":
        stripped = text.strip()
        return len(stripped) > 1 and stripped[0] == '"' and stripped[-1] == '"'
    if instruction == "detectable_format:number_highlighted_sections":
        single = [part for part in re.findall(r"\*[^\n\*]*\*", text) if part.strip("*").strip()]
        double = [part for part in re.findall(r"\*\*[^\n\*]*\*\*", text) if part.strip("*").strip()]
        return len(single) + len(double) >= _int(kwargs["num_highlights"])
    if instruction == "change_case:english_capital":
        return text.isupper()
    if instruction == "change_case:english_lowercase":
        return text.islower()
    if instruction == "combination:repeat_prompt":
        return text.strip().lower().startswith(str(kwargs["prompt_to_repeat"]).strip().lower())
    if instruction == "startend:end_checker":
        return text.strip().strip('"').lower().endswith(str(kwargs["end_phrase"]).strip().lower())
    if instruction == "detectable_format:json_format":
        return parse_json_text(text).status == "ok"
    return None


def ranks_first(*, id_key: str = "id", score_key: str = "score") -> Grader:
    """The reply is a list of ``{id, score}`` rows; 1.0 when ``expected["chosen"]``'s score is strictly the highest.

    Every score must be numeric. [rewardbench2]
    """
    id = "ranks_first"

    def rule(sample: Sample, expected: JsonValue) -> Score:
        chosen_id = _field(id, expected, "chosen")
        parsed = _json_of(sample)
        if not isinstance(parsed, list):
            return Score(value=0.0)
        scores = {row.get(id_key): row.get(score_key) for row in parsed if isinstance(row, dict)}
        chosen = scores.get(chosen_id)
        others = [value for key, value in scores.items() if key != chosen_id]
        numbers = [value for value in (chosen, *others) if isinstance(value, int | float) and not isinstance(value, bool)]
        if len(numbers) != len(others) + 1 or not others:
            return Score(value=0.0)
        best, rest = numbers[0], numbers[1:]
        return _verdict(all(best > value for value in rest), answer=str(chosen))

    return _CodeGrader(id, rule)


def verdict_equals(*, field: str = "verdict", expected_key: str = "label") -> Grader:
    """1.0 when the reply's ``field`` equals ``expected[expected_key]``.

    [judgebench: ``verdict`` vs ``label``; mtbench_human: ``field="winner"``, ``expected_key="winner"``]
    """
    id = "verdict_equals"

    def rule(sample: Sample, expected: JsonValue) -> Score:
        want = _field(id, expected, expected_key)
        parsed = _json_of(sample)
        got = parsed.get(field) if isinstance(parsed, dict) else None
        return _verdict(isinstance(parsed, dict) and got == want, answer=None if got is None else str(got))

    return _CodeGrader(id, rule)


def constant(value: float, *, id: str = "constant") -> Grader:
    """Always ``value``: the harness's own check that a run and a comparison work end to end (book lesson one)."""

    def rule(sample: Sample, expected: JsonValue) -> Score:
        return Score(value=value)

    return _CodeGrader(id, rule)


def with_wrong_labels(grader: Grader, wrong: Callable[[Sample, JsonValue], JsonValue]) -> Grader:
    """The same grader, scoring against ``wrong(sample, expected)`` instead of ``expected``: the N7 canary.

    A grader that still passes the canary is not measuring anything. ``id`` is ``<id>+wrong_label``.
    """
    return _WrongLabels(grader, wrong)
