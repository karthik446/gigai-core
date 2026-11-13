"""Graders lifted from inspect_ai: the typed boundary over ``_vendor/inspect_ai/scorers``.

``includes``, ``match`` and ``pattern`` wrap the vendored bodies verbatim; the bodies are ``async`` upstream and never
await anything, so they are driven to completion here without an event loop. ``model_graded_qa`` keeps inspect_ai's
judge template and parser; ``autoevals_judge`` and ``openevals_judge`` put the other two vendored prompt families
behind the same ``Judge`` shape (``ModelJudge`` from ``.judge``).
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Mapping, Sequence
from typing import Literal

from .._vendor.inspect_ai._metric import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    PARTIAL,
    is_finite_number,
)
from .._vendor.inspect_ai._metric import Score as VendorScore
from .._vendor.inspect_ai.scorers import includes as _includes
from .._vendor.inspect_ai.scorers import match as _match
from .._vendor.inspect_ai.scorers import pattern as _pattern
from .._vendor.inspect_ai.scorers._shim import ModelOutput, Scorer, TaskState
from .._vendor.inspect_ai.scorers._target import Target
from .._vendor.inspect_ai.scorers.judge_template import (  # the parser's pieces are private upstream; pinned data
    _GRADE_WORD_VALUES,  # pyright: ignore[reportPrivateUsage]
    _PERMISSIVE_GRADE_PATTERN,  # pyright: ignore[reportPrivateUsage]
    DEFAULT_MODEL_GRADED_QA_TEMPLATE,
    _trailing_verdict_disagrees,  # pyright: ignore[reportPrivateUsage]
    default_instructions,
)
from .._vendor.openevals.prompts.correctness import CORRECTNESS_PROMPT
from .._vendor.openevals.prompts.hallucination import HALLUCINATION_PROMPT
from ..canonical import sha256_hex
from ..providers import CallSettings, ModelProvider
from ..sample import Sample
from ..score import GraderKind, Score
from ..values import JsonValue
from .judge import (
    ModelJudge,
    conversation,
    expected_text,
    load_template,
    render_template,
    template_choice,
    verdict_score,
)
from .protocol import CannotGrade, Grader, Judge

LIFTED_VERSION = "inspect_ai@9567b431"

MatchLocation = Literal["end", "begin", "any", "exact"]

_LETTERS: dict[str, float] = {CORRECT: 1.0, PARTIAL: 0.5, INCORRECT: 0.0, NOANSWER: 0.0}


class _Lifted:
    """A vendored scorer body as a core ``Grader``."""

    kind: GraderKind = "code"
    version = LIFTED_VERSION

    def __init__(self, id: str, body: Scorer) -> None:
        self._id = id
        self._body = body

    @property
    def id(self) -> str:
        return self._id

    def grade(self, sample: Sample, expected: JsonValue) -> Score:
        return _run(self._body, sample.output_text, _targets(expected))


def _targets(expected: JsonValue) -> list[str]:
    """The reference strings a vendored body compares against: every item of a sequence, else the value itself."""
    if isinstance(expected, Sequence) and not isinstance(expected, str | bytes):
        return [str(item) for item in expected]
    return [str(expected)]


def _complete(awaitable: Awaitable[VendorScore]) -> VendorScore:
    """Drive a body that never awaits to its result; one that does is refused rather than silently run."""
    steps = awaitable.__await__()
    try:
        next(steps)
    except StopIteration as done:
        if isinstance(done.value, VendorScore):
            return done.value
        raise RuntimeError(f"lifted body returned {type(done.value).__name__}, not a Score") from None
    steps.close()
    raise RuntimeError("lifted body awaited something; inspect_ai text scorers must not need an event loop")


def _run(body: Scorer, text: str, targets: list[str]) -> Score:
    verdict = _complete(body(TaskState(output=ModelOutput(completion=text)), Target(targets)))
    if isinstance(verdict.value, str) and verdict.value in _LETTERS:
        value = _LETTERS[verdict.value]
    elif isinstance(verdict.value, int | float) and not isinstance(verdict.value, bool):
        value = float(verdict.value)
    else:
        raise RuntimeError(f"lifted body returned an unmapped value {verdict.value!r}")
    return Score(value=value, explanation=verdict.reason, answer=verdict.answer)


def includes(*, ignore_case: bool = True) -> Grader:
    """1.0 when the reply contains one of the target strings (inspect_ai ``includes``)."""
    return _Lifted("includes", _includes(ignore_case=ignore_case))


def match(*, location: MatchLocation = "end", ignore_case: bool = True, numeric: bool | None = None) -> Grader:
    """1.0 when the reply matches a target at ``location`` (inspect_ai ``match``).

    ``numeric=None`` compares as numbers when every target is a finite number, the way 0.3.2 did; ``True``/``False``
    forces it.
    """

    def body(state: TaskState, target: Target) -> Awaitable[VendorScore]:
        compare_numbers = all(is_finite_number(item) for item in target.target) if numeric is None else numeric
        return _match(location=location, ignore_case=ignore_case, numeric=compare_numbers)(state, target)

    return _Lifted("match", body)


def pattern(regex: str, *, ignore_case: bool = True, match_all: bool = False) -> Grader:
    """1.0 when the regex's capture group(s) in the reply equal a target (inspect_ai ``pattern``)."""
    return _Lifted("pattern", _pattern(regex, ignore_case=ignore_case, match_all=match_all))


AutoevalsTemplate = Literal[
    "battle", "closed_q_a", "factuality", "humor", "possible", "security", "sql", "summary", "translation"
]
OpenevalsPrompt = Literal["correctness", "hallucination"]

_OPENEVALS_PROMPTS: Mapping[OpenevalsPrompt, str] = {
    "correctness": CORRECTNESS_PROMPT,
    "hallucination": HALLUCINATION_PROMPT,
}
_OPENEVALS_ASK = (
    "\n\nEnd your reply with one JSON object on its own line: "
    '{"verdict": "pass" or "fail", "explanation": "<one sentence>"}.\n'
)
#: The autoevals template fields and what they take from a sample: the input, the output, and the expected value.
_AUTOEVALS_FIELDS = ("input", "output", "expected")


def model_graded_qa(provider: ModelProvider, *, settings: CallSettings, partial_credit: bool = False) -> Judge:
    """inspect_ai's model-graded QA: its judge prompt and its last-``GRADE:`` parser (C is 1.0, P 0.5, I 0.0).

    The field markers' nonce is the first 8 hex digits of the SHA-256 of the question and the answer (upstream
    draws it at random), so the same sample always gets the same prompt. The final ``GRADE: <word>`` decides: the
    word must be a grade the instructions offered (``C``/``I``, ``P`` with ``partial_credit``, or spelled out) and a
    later verdict line in a form the pattern does not read must agree; anything else is ``CannotGrade`` (an
    off-menu final verdict never falls back to an earlier mention).

    Args:
        provider: The model that judges.
        settings: The call settings, including the judge model.
        partial_credit: Offer ``P`` as upstream does.

    Returns:
        The judge; its ``prompt_sha256`` digests the template and the instructions.
    """
    instructions = default_instructions(partial_credit)
    offered = (CORRECT, PARTIAL, INCORRECT) if partial_credit else (CORRECT, INCORRECT)

    def render(sample: Sample, expected: JsonValue, swapped: bool) -> str:
        question = conversation(sample)
        nonce = sha256_hex(f"{question}\0{sample.output_text}".encode())[:8]
        return DEFAULT_MODEL_GRADED_QA_TEMPLATE.format(
            question=question,
            answer=sample.output_text,
            criterion="\n".join(_targets(expected)),
            instructions=instructions,
            grader_nonce=nonce,
        )

    def read(reply: str) -> Score:
        found = re.search(_PERMISSIVE_GRADE_PATTERN, reply)
        if found is None:
            raise CannotGrade("the judge's reply has no GRADE")
        word = found.group(1).strip().strip("_")
        grade = _GRADE_WORD_VALUES.get(word.lower(), word.upper() if len(word) == 1 else None)
        if grade is None or grade not in offered or _trailing_verdict_disagrees(reply, found.end(), grade):
            raise CannotGrade("the judge's final GRADE is not one of the offered grades")
        return Score(value=_LETTERS[grade], explanation=reply, answer=grade)

    return ModelJudge(
        provider,
        settings,
        id="model_graded_qa",
        version=LIFTED_VERSION,
        prompt_sha256=sha256_hex(f"{DEFAULT_MODEL_GRADED_QA_TEMPLATE}\0{instructions}".encode()),
        render=render,
        read=read,
    )


def autoevals_judge(template: AutoevalsTemplate, provider: ModelProvider, *, settings: CallSettings) -> Judge:
    """One of the nine vendored autoevals templates, read by the choice its reply names (``template_choice``).

    The template's ``{{input}}``, ``{{output}}`` and ``{{expected}}`` placeholders take the sample's input, its
    output and the expected value (as text); a template with other placeholders (``battle``, ``humor``, ...) cannot
    be rendered from a sample and raises ``CannotGrade`` naming them. The score is the choice's ``choice_scores``
    entry; a reply naming no choice is ``CannotGrade``.

    Args:
        template: The template name.
        provider: The model that judges.
        settings: The call settings, including the judge model.

    Returns:
        The judge, with id ``autoevals_<template>``; its ``prompt_sha256`` digests the template prompt.
    """
    loaded = load_template(template)

    def render(sample: Sample, expected: JsonValue, swapped: bool) -> str:
        fields = dict(
            zip(_AUTOEVALS_FIELDS, (conversation(sample), sample.output_text, "\n".join(_targets(expected))))
        )
        try:
            return render_template(loaded, **fields)
        except ValueError as exc:
            raise CannotGrade(str(exc)) from None

    def read(reply: str) -> Score:
        choice = template_choice(loaded, reply)
        if choice is None:
            raise CannotGrade(f"the judge's reply names none of {', '.join(loaded.choice_scores)}: {reply[:200]!r}")
        return Score(value=loaded.choice_scores[choice], explanation=reply, answer=choice)

    return ModelJudge(
        provider,
        settings,
        id=f"autoevals_{template}",
        version=LIFTED_VERSION,
        prompt_sha256=sha256_hex(loaded.prompt.encode()),
        render=render,
        read=read,
    )


def openevals_judge(name: OpenevalsPrompt, provider: ModelProvider, *, settings: CallSettings) -> Judge:
    """One of the two vendored openevals prompts, read as a pass/fail ``JudgeVerdict``.

    ``{inputs}`` takes the sample's input, ``{outputs}`` its output and ``{reference_outputs}`` the expected value
    as JSON; ``hallucination`` also takes ``{context}`` from ``expected["context"]`` (the reference is then
    ``expected["reference"]`` when present, else ``null``). The prompt ends with the request for the verdict object:
    exactly ``verdict`` and an optional ``explanation`` (``JudgeVerdict``), last in the reply; a last object with any
    other key is ``CannotGrade``.

    Args:
        name: The prompt name.
        provider: The model that judges.
        settings: The call settings, including the judge model.

    Returns:
        The judge, with id ``openevals_<name>``; its ``prompt_sha256`` digests the prompt text and the request.
    """
    prompt = _OPENEVALS_PROMPTS[name]

    def render(sample: Sample, expected: JsonValue, swapped: bool) -> str:
        reference = expected
        fields: dict[str, str] = {"inputs": conversation(sample), "outputs": sample.output_text}
        if name == "hallucination":
            if not isinstance(expected, Mapping) or not isinstance(expected.get("context"), str):
                raise CannotGrade("hallucination needs expected['context'] as text")
            fields["context"] = str(expected["context"])
            reference = expected.get("reference")
        fields["reference_outputs"] = expected_text(reference)
        return prompt.format_map(fields) + _OPENEVALS_ASK

    return ModelJudge(
        provider,
        settings,
        id=f"openevals_{name}",
        version=LIFTED_VERSION,
        prompt_sha256=sha256_hex(f"{prompt}{_OPENEVALS_ASK}".encode()),
        render=render,
        read=verdict_score,
    )
