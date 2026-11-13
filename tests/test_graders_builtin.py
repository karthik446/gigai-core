"""The code graders of ``graders/builtin.py``: each 0.3.2 ``score()`` branch as a constructor, right and wrong.

The IFEval and wrong-labels assertions are the ones ``tests/behaviors/evals/test_base.py`` made in 0.3.2
(``test_approximate_ifeval_checks_a_non_blank_title_and_a_quoted_ending`` and the wrong-labels half of
``test_canary_transforms_are_seeded_and_wrong_labels_differ_from_gold``), over hand fixtures instead of the ci slices.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from gigai_core.graders import builtin
from gigai_core.graders.protocol import CannotGrade, Grader
from gigai_core.providers import Turn
from gigai_core.sample import Sample
from gigai_core.values import JsonValue, ParseStatus

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


def _sample(output: str, parsed: JsonValue = None, status: ParseStatus = "not_parsed") -> Sample:
    return Sample(
        trace_id="trace_1",
        run_id="run_1",
        item_id="item_1",
        item_version="1",
        cluster_id="item_1",
        repeat_index=0,
        input=(Turn(role="user", content="What is it?"),),
        output_text=output,
        output_parsed=parsed,
        parse_status=status,
        outcome="succeeded",
        error=None,
        recorded_at=NOW,
    )


def _value(grader: Grader, output: str, expected: JsonValue) -> float:
    return grader.grade(_sample(output), expected).value


def test_parse_json_text_reads_bare_and_fenced_json_and_names_the_rest() -> None:
    assert builtin.parse_json_text('{"a": 1}') == builtin.ParsedJson(value={"a": 1}, status="ok")
    assert builtin.parse_json_text('```json\n{"a": [1, 2]}\n```') == builtin.ParsedJson(value={"a": [1, 2]}, status="ok")
    assert builtin.parse_json_text("```\n  [1]  \n```") == builtin.ParsedJson(value=[1], status="ok")
    assert builtin.parse_json_text("  42 ") == builtin.ParsedJson(value=42, status="ok")
    assert builtin.parse_json_text("not json") == builtin.ParsedJson(value=None, status="not_json")
    assert builtin.parse_json_text("") == builtin.ParsedJson(value=None, status="not_json")
    assert builtin.parse_json_text("[" * 100_000).status == "not_json"  # RecursionError is "not_json" too


def test_every_builtin_grader_has_its_id_version_and_kind() -> None:
    graders: list[Grader] = [
        builtin.last_number(),
        builtin.answer_letter(),
        builtin.json_valid(),
        builtin.schema_valid(),
        builtin.label_share(),
        builtin.bool_field("supported"),
        builtin.ifeval(),
        builtin.ranks_first(),
        builtin.verdict_equals(),
        builtin.constant(0.5),
        builtin.constant(1.0, id="always_right"),
    ]
    assert [grader.id for grader in graders] == [
        "last_number",
        "answer_letter",
        "json_valid",
        "schema_valid",
        "label_share",
        "bool_field",
        "ifeval",
        "ranks_first",
        "verdict_equals",
        "constant",
        "always_right",
    ]
    assert {grader.version for grader in graders} == {"1"}
    assert {grader.kind for grader in graders} == {"code"}


def test_last_number_reads_the_last_number_with_commas_and_dollars() -> None:
    grader = builtin.last_number()
    assert _value(grader, "First 3 apples, then 5. ANSWER: 1,234", "1234") == 1.0
    assert _value(grader, "ANSWER: $1,234.00", 1234) == 1.0
    assert _value(grader, "ANSWER: 1234", "1,235") == 0.0
    assert _value(grader, "-7 degrees", "-7") == 1.0
    assert _value(grader, "no digits here", "4") == 0.0
    assert _value(grader, "4", "four") == 0.0
    score = grader.grade(_sample("so 12 then 13.5"), "13.5")
    assert (score.value, score.answer) == (1.0, "13.5")
    assert _value(builtin.last_number(tolerance=0.01), "about 3.16", "3.14") == 0.0
    assert _value(builtin.last_number(tolerance=0.01), "about 3.141", "3.14") == 1.0


def test_answer_letter_takes_the_last_marked_letter_case_insensitively_on_expected() -> None:
    grader = builtin.answer_letter()
    assert _value(grader, "ANSWER: A\nActually ANSWER: (B)", "b") == 1.0
    assert _value(grader, "ANSWER: A\nActually ANSWER: (B)", "A") == 0.0
    assert _value(grader, "The answer is B", "B") == 0.0
    assert _value(grader, "ANSWER: E", "E") == 0.0  # not a choice
    assert _value(grader, "ANSWER: BC", "B") == 0.0  # the letter must stand alone
    assert _value(builtin.answer_letter(choices="ABCDE"), "ANSWER: E", "E") == 1.0
    assert grader.grade(_sample("ANSWER: C"), "C").answer == "C"


def test_json_valid_and_schema_valid_judge_the_reply_text() -> None:
    assert _value(builtin.json_valid(), '{"a": 1}', None) == 1.0
    assert _value(builtin.json_valid(), "```json\n[1]\n```", None) == 1.0
    assert _value(builtin.json_valid(), "{a: 1}", None) == 0.0
    schema: dict[str, JsonValue] = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    assert _value(builtin.schema_valid(), '{"a": 1}', schema) == 1.0
    assert _value(builtin.schema_valid(), '{"a": "one"}', schema) == 0.0
    assert _value(builtin.schema_valid(), "{}", schema) == 0.0
    assert _value(builtin.schema_valid(), "not json", schema) == 0.0
    assert _value(builtin.schema_valid(), '"x"', {"type": "string", "pattern": "("}) == 0.0  # a regex that does not compile
    assert _value(builtin.schema_valid(schema), '{"a": 1}', {"schema": schema}) == 1.0  # an explicit schema wins
    with pytest.raises(CannotGrade, match="expected must be a JSON object"):
        builtin.schema_valid().grade(_sample('{"a": 1}'), "not a schema")


def test_label_share_is_the_share_of_expected_keys_labelled_right() -> None:
    grader = builtin.label_share()
    want: JsonValue = {"labels": {"nda-1": "Entailment", "nda-2": "Contradiction", "nda-3": "NotMentioned"}}
    assert _value(grader, '{"labels": {"nda-1": "Entailment", "nda-2": "Contradiction", "nda-3": "NotMentioned"}}', want) == 1.0
    assert _value(grader, '{"labels": {"nda-1": "Entailment", "nda-2": "Entailment"}}', want) == pytest.approx(1 / 3)
    assert _value(grader, '{"labels": {"nda-1": "Entailment", "nda-2": "Contradiction", "nda-3": "NotMentioned", "x": 1}}', want) == 1.0
    assert _value(grader, '{"labels": []}', want) == 0.0
    assert _value(grader, "[1, 2]", want) == 0.0
    assert _value(grader, "not json", want) == 0.0
    assert _value(grader, '{"labels": {}}', {"labels": {}}) == 0.0  # nothing to get right
    assert _value(builtin.label_share(field="tags"), '{"tags": {"a": 1}}', {"tags": {"a": 1}}) == 1.0
    with pytest.raises(CannotGrade, match="expected has no 'labels' key"):
        grader.grade(_sample('{"labels": {}}'), {"tags": {}})


def test_bool_field_compares_or_negates_the_expected_boolean() -> None:
    same = builtin.bool_field("ok")
    assert _value(same, '{"ok": true}', {"ok": True}) == 1.0
    assert _value(same, '{"ok": false}', {"ok": True}) == 0.0
    assert _value(same, '{"ok": "true"}', {"ok": True}) == 0.0  # a string is not a boolean
    assert _value(same, '{"ok": 1}', {"ok": True}) == 0.0  # neither is 1
    assert _value(same, "[true]", {"ok": True}) == 0.0
    ragtruth = builtin.bool_field("supported", invert=True, expected_key="hallucinated")
    assert _value(ragtruth, '{"supported": false}', {"hallucinated": True}) == 1.0
    assert _value(ragtruth, '{"supported": true}', {"hallucinated": True}) == 0.0
    assert _value(ragtruth, '{"supported": true}', {"hallucinated": False, "answer": "x"}) == 1.0
    with pytest.raises(CannotGrade, match="must be a boolean"):
        ragtruth.grade(_sample('{"supported": true}'), {"hallucinated": "yes"})


def test_structure_graders_prefer_the_parsed_output_and_fall_back_to_the_text() -> None:
    parsed = _sample("garbage", {"verdict": "A>B"}, "ok")
    assert builtin.verdict_equals().grade(parsed, {"label": "A>B"}).value == 1.0
    assert builtin.verdict_equals().grade(_sample('{"verdict": "A>B"}'), {"label": "A>B"}).value == 1.0
    assert builtin.verdict_equals().grade(_sample('{"verdict": "A>B"}', "the text", "ok"), {"label": "A>B"}).value == 0.0


def test_ranks_first_needs_the_chosen_row_strictly_above_every_other_numeric_score() -> None:
    grader = builtin.ranks_first()
    expected: JsonValue = {"chosen": "r1", "rejected": ["r2", "r3"]}
    assert _value(grader, '[{"id": "r1", "score": 9}, {"id": "r2", "score": 3.5}, {"id": "r3", "score": 1}]', expected) == 1.0
    assert _value(grader, '[{"id": "r1", "score": 3}, {"id": "r2", "score": 3}]', expected) == 0.0  # a tie is not first
    assert _value(grader, '[{"id": "r1", "score": 3}, {"id": "r2", "score": 5}]', expected) == 0.0
    assert _value(grader, '[{"id": "r1", "score": 3}]', expected) == 0.0  # nobody to beat
    assert _value(grader, '[{"id": "r1", "score": "9"}, {"id": "r2", "score": 3}]', expected) == 0.0  # text score
    assert _value(grader, '[{"id": "r1", "score": true}, {"id": "r2", "score": 0}]', expected) == 0.0  # bool is not a number
    assert _value(grader, '[{"id": "r2", "score": 3}]', expected) == 0.0  # the chosen row is missing
    assert _value(grader, '{"r1": 9}', expected) == 0.0
    assert _value(builtin.ranks_first(id_key="name", score_key="rank"), '[{"name": "r1", "rank": 2}, {"name": "r2", "rank": 1}]', expected) == 1.0
    with pytest.raises(CannotGrade, match="expected has no 'chosen' key"):
        grader.grade(_sample("[]"), {"winner": "r1"})


def test_verdict_equals_compares_one_field_with_one_expected_key() -> None:
    judgebench = builtin.verdict_equals()
    assert _value(judgebench, '{"verdict": "A>B"}', {"label": "A>B"}) == 1.0
    assert _value(judgebench, '{"verdict": "B>A"}', {"label": "A>B"}) == 0.0
    assert _value(judgebench, '{"winner": "A>B"}', {"label": "A>B"}) == 0.0
    assert _value(judgebench, '["A>B"]', {"label": "A>B"}) == 0.0
    mtbench = builtin.verdict_equals(field="winner", expected_key="winner")
    assert _value(mtbench, '{"winner": "A"}', {"winner": "A"}) == 1.0
    assert _value(mtbench, '{"winner": "B"}', {"winner": "A"}) == 0.0
    assert judgebench.grade(_sample('{"verdict": "B>A"}'), {"label": "A>B"}).answer == "B>A"


def test_constant_scores_every_sample_the_same() -> None:
    assert _value(builtin.constant(0.0), "anything", {"label": "x"}) == 0.0
    assert _value(builtin.constant(1.0), "", None) == 1.0
    assert _value(builtin.constant(0.5, id="coin"), "x", 3) == 0.5


def _ifeval(output: str, instruction: str, **kwargs: JsonValue) -> float:
    expected: JsonValue = {"instruction_id_list": [instruction], "kwargs": [dict(kwargs)]}
    return builtin.ifeval().grade(_sample(output), expected).value


def test_approximate_ifeval_checks_a_non_blank_title_and_a_quoted_ending() -> None:
    assert _ifeval("<<Rain>>\nIt rained.", "detectable_format:title") == 1.0
    assert _ifeval("<< >>\nIt rained.", "detectable_format:title") == 0.0
    assert _ifeval("<<>> no title", "detectable_format:title") == 0.0
    assert _ifeval("<<\nRain>>", "detectable_format:title") == 0.0
    end = "Any other questions?"
    assert _ifeval(f'"It rained. {end}"', "startend:end_checker", end_phrase=end) == 1.0
    assert _ifeval(f"It rained. {end}  ", "startend:end_checker", end_phrase=end) == 1.0
    assert _ifeval(f"{end} It rained.", "startend:end_checker", end_phrase=end) == 0.0
    with pytest.raises(CannotGrade, match="no check for instruction type length_constraints:number_words"):
        _ifeval("two words", "length_constraints:number_words", num_words=2, relation="at least")


def test_ifeval_checks_the_other_implemented_instruction_types() -> None:
    assert _ifeval("Bring an umbrella and a coat.", "keywords:existence", keywords=["umbrella", "Coat"]) == 1.0
    assert _ifeval("Bring an umbrella.", "keywords:existence", keywords=["umbrella", "coat"]) == 0.0
    assert _ifeval("Bring an umbrella.", "keywords:forbidden_words", forbidden_words=["rain"]) == 1.0
    assert _ifeval("Bring an umbrella.", "keywords:forbidden_words", forbidden_words=["Umbrella"]) == 0.0
    assert _ifeval("A rainbow appeared.", "keywords:forbidden_words", forbidden_words=["rain"]) == 1.0
    with pytest.raises(CannotGrade, match="unusable kwargs"):
        _ifeval("rain", "keywords:frequency", keyword=1, frequency=1)
    assert _ifeval("rain rain rain", "keywords:frequency", keyword="rain", relation="at least", frequency=3) == 1.0
    assert _ifeval("rain rain", "keywords:frequency", keyword="rain", relation="at least", frequency=3) == 0.0
    assert _ifeval("rain rain", "keywords:frequency", keyword="rain", relation="less than", frequency=3) == 1.0
    assert _ifeval("rain", "keywords:frequency", keyword="rain", frequency="1") == 1.0  # no relation means at least
    assert _ifeval("no commas here", "punctuation:no_comma") == 1.0
    assert _ifeval("one, two", "punctuation:no_comma") == 0.0
    assert _ifeval("One. Two! Three?", "length_constraints:number_sentences", relation="at least", num_sentences=3) == 1.0
    assert _ifeval("One. Two!", "length_constraints:number_sentences", relation="less than", num_sentences=3) == 1.0
    assert _ifeval("One. Two! Three?", "length_constraints:number_sentences", relation="less than", num_sentences=3) == 0.0
    assert _ifeval("a\n***\nb\n***\nc", "length_constraints:number_paragraphs", num_paragraphs=3) == 1.0
    assert _ifeval("a\n***\n\n***\nc", "length_constraints:number_paragraphs", num_paragraphs=2) == 0.0  # blank middle
    assert _ifeval("a\n***\nb", "length_constraints:number_paragraphs", num_paragraphs=3) == 0.0
    assert _ifeval("Dear [name], see [date].", "detectable_content:number_placeholders", num_placeholders=2) == 1.0
    assert _ifeval("Dear [name].", "detectable_content:number_placeholders", num_placeholders=2) == 0.0
    assert _ifeval(' "quoted" ', "startend:quotation") == 1.0
    assert _ifeval('"', "startend:quotation") == 0.0
    assert _ifeval('"open', "startend:quotation") == 0.0
    assert _ifeval("*one* and **two**", "detectable_format:number_highlighted_sections", num_highlights=2) == 1.0
    assert _ifeval("*one* and ** **", "detectable_format:number_highlighted_sections", num_highlights=2) == 0.0
    assert _ifeval("ALL CAPS", "change_case:english_capital") == 1.0
    assert _ifeval("Not all caps", "change_case:english_capital") == 0.0
    assert _ifeval("all lower", "change_case:english_lowercase") == 1.0
    assert _ifeval("Not all lower", "change_case:english_lowercase") == 0.0
    assert _ifeval("Write a poem.\nRoses are red.", "combination:repeat_prompt", prompt_to_repeat="write a poem.") == 1.0
    assert _ifeval("Roses are red.", "combination:repeat_prompt", prompt_to_repeat="Write a poem.") == 0.0
    assert _ifeval('```json\n{"a": 1}\n```', "detectable_format:json_format") == 1.0
    assert _ifeval("{a: 1}", "detectable_format:json_format") == 0.0


def test_ifeval_needs_every_instruction_followed_and_usable_kwargs() -> None:
    both: JsonValue = {
        "instruction_id_list": ["punctuation:no_comma", "change_case:english_lowercase"],
        "kwargs": [None, {"ignored": None}],
    }
    assert builtin.ifeval().grade(_sample("all lower no commas"), both).value == 1.0
    score = builtin.ifeval().grade(_sample("Lower, no"), both)
    assert (score.value, score.explanation) == (0.0, "0 of 2 instructions followed")
    one_of_two = builtin.ifeval().grade(_sample("lower, with comma"), both)
    assert (one_of_two.value, one_of_two.explanation) == (0.0, "1 of 2 instructions followed")
    with pytest.raises(CannotGrade, match="unusable kwargs"):
        _ifeval("rain", "keywords:frequency", keyword="rain", frequency="three")
    with pytest.raises(CannotGrade, match="unusable kwargs"):
        _ifeval("rain", "keywords:existence", keywords="rain")
    with pytest.raises(CannotGrade, match="unusable kwargs"):
        _ifeval("rain", "keywords:frequency", keyword="rain")
    with pytest.raises(CannotGrade, match="same length"):
        builtin.ifeval().grade(_sample("x"), {"instruction_id_list": ["punctuation:no_comma"], "kwargs": []})
    with pytest.raises(CannotGrade, match="expected has no 'kwargs' key"):
        builtin.ifeval().grade(_sample("x"), {"instruction_id_list": []})


# The 0.3.2 ``base.wrong_label`` transforms, one per set shape, as ``with_wrong_labels`` callables.
_NEXT_LETTER = {"A": "B", "B": "C", "C": "D", "D": "A"}
_NEXT_NLI = {"Entailment": "Contradiction", "Contradiction": "NotMentioned", "NotMentioned": "Entailment"}
_NEXT_VERDICT = {"A>B": "B>A", "B>A": "A>B", "A": "B", "B": "A"}


def _next_letter(sample: Sample, expected: JsonValue) -> JsonValue:
    return _NEXT_LETTER.get(str(expected).strip().upper(), "A")


def _next_number(sample: Sample, expected: JsonValue) -> JsonValue:
    try:
        return str(float(str(expected).replace(",", "")) + 1)
    except ValueError:
        return "0"


def _negate_hallucinated(sample: Sample, expected: JsonValue) -> JsonValue:
    assert isinstance(expected, dict)
    return {**expected, "hallucinated": not expected["hallucinated"]}


def _rotate_labels(sample: Sample, expected: JsonValue) -> JsonValue:
    assert isinstance(expected, dict)
    labels = expected["labels"]
    assert isinstance(labels, dict)
    return {"labels": {key: _NEXT_NLI.get(str(value), "Entailment") for key, value in labels.items()}}


def _swap_chosen(sample: Sample, expected: JsonValue) -> JsonValue:
    assert isinstance(expected, dict)
    rejected = expected.get("rejected")
    rejected = list(rejected) if isinstance(rejected, list) else []
    return {**expected, "chosen": rejected[0] if rejected else "none", "rejected": [expected["chosen"], *rejected[1:]]}


def _next_label(sample: Sample, expected: JsonValue) -> JsonValue:
    assert isinstance(expected, dict)
    return {**expected, "label": _NEXT_VERDICT.get(str(expected["label"]), "A>B")}


def _next_winner(sample: Sample, expected: JsonValue) -> JsonValue:
    assert isinstance(expected, dict)
    return {**expected, "winner": _NEXT_VERDICT.get(str(expected["winner"]), "A")}


CANARIES: list[tuple[str, Grader, Callable[[Sample, JsonValue], JsonValue], str, JsonValue]] = [
    ("quality", builtin.answer_letter(), _next_letter, "ANSWER: D", "D"),
    ("tinymmlu", builtin.answer_letter(), _next_letter, "ANSWER: (A)", "a"),
    ("tinygsm8k", builtin.last_number(), _next_number, "ANSWER: 1,234", "1234"),
    ("ragtruth", builtin.bool_field("supported", invert=True, expected_key="hallucinated"), _negate_hallucinated, '{"supported": false}', {"hallucinated": True}),
    ("contractnli", builtin.label_share(), _rotate_labels, '{"labels": {"nda-1": "Entailment", "nda-2": "NotMentioned"}}', {"labels": {"nda-1": "Entailment", "nda-2": "NotMentioned"}}),
    ("rewardbench2", builtin.ranks_first(), _swap_chosen, '[{"id": "r1", "score": 9}, {"id": "r2", "score": 3}, {"id": "r3", "score": 1}]', {"chosen": "r1", "rejected": ["r2", "r3"]}),
    ("judgebench", builtin.verdict_equals(), _next_label, '{"verdict": "A>B"}', {"label": "A>B"}),
    ("mtbench_human", builtin.verdict_equals(field="winner", expected_key="winner"), _next_winner, '{"winner": "B"}', {"winner": "B"}),
]


@pytest.mark.parametrize(("name", "grader", "wrong", "output", "expected"), CANARIES, ids=[row[0] for row in CANARIES])
def test_wrong_labels_differ_from_gold_and_fail_a_gold_reply(
    name: str, grader: Grader, wrong: Callable[[Sample, JsonValue], JsonValue], output: str, expected: JsonValue
) -> None:
    sample = _sample(output)
    assert wrong(sample, expected) != expected, name
    canary = builtin.with_wrong_labels(grader, wrong)
    assert (canary.id, canary.version, canary.kind) == (f"{grader.id}+wrong_label", grader.version, grader.kind)
    assert grader.grade(sample, expected).value == 1.0, name
    assert canary.grade(sample, expected).value == 0.0, name


def test_wrong_labels_fall_back_the_way_0_3_2_did() -> None:
    sample = _sample("")
    assert _next_letter(sample, "Z") == "A"
    assert _next_number(sample, "four") == "0"
    assert _swap_chosen(sample, {"chosen": "r1"}) == {"chosen": "none", "rejected": ["r1"]}
    assert _next_label(sample, {"label": "tie"}) == {"label": "A>B"}
    assert _next_winner(sample, {"winner": "tie"}) == {"winner": "A"}
    assert _rotate_labels(sample, {"labels": {"k": "other"}}) == {"labels": {"k": "Entailment"}}
