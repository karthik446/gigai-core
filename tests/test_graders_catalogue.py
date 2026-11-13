"""Every code grader row of DESIGN.md's catalogue that 0.1.0 ships constructs and grades a fixture.

A constructor added to ``graders/builtin.py`` or ``graders/lifted.py`` without a fixture here fails
``test_every_code_grader_constructor_has_a_fixture``; the judge, label and agentic rows arrive with their packets.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from datetime import UTC, datetime
from types import ModuleType

import pytest

from gigai_core.graders import builtin, lifted
from gigai_core.graders.protocol import Grader
from gigai_core.providers import Turn
from gigai_core.sample import Sample
from gigai_core.values import JsonValue

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


def _sample(output: str) -> Sample:
    return Sample(
        trace_id="trace_1",
        run_id="run_1",
        item_id="item_1",
        item_version="1",
        cluster_id="item_1",
        repeat_index=0,
        input=(Turn(role="user", content="What is it?"),),
        output_text=output,
        output_parsed=None,
        parse_status="not_parsed",
        outcome="succeeded",
        error=None,
        recorded_at=NOW,
    )


def _next_letter(sample: Sample, expected: JsonValue) -> JsonValue:
    return "B" if expected == "A" else "A"


IFEVAL: JsonValue = {"instruction_id_list": ["punctuation:no_comma"], "kwargs": [None]}

# row -> (grader, a right reply, a wrong reply, expected)
FIXTURES: dict[str, tuple[Grader, str, str, JsonValue]] = {
    "constant": (builtin.constant(1.0), "anything", "", None),
    "includes": (lifted.includes(), "I think it is Paris.", "I think it is Rome.", "paris"),
    "match": (lifted.match(), "So the total is $1,234.00", "So the total is 1235", "1234"),
    "pattern": (lifted.pattern(r"ANSWER:\s*(\w+)"), "blah\nANSWER: B", "blah\nANSWER: C", "B"),
    "answer_letter": (builtin.answer_letter(), "ANSWER: (B)", "ANSWER: A", "B"),
    "last_number": (builtin.last_number(), "ANSWER: 1,234", "ANSWER: 1235", "1234"),
    "json_valid": (builtin.json_valid(), '{"a": 1}', "{a: 1}", None),
    "schema_valid": (builtin.schema_valid(), '{"a": 1}', '{"a": "one"}', {"type": "object", "properties": {"a": {"type": "integer"}}}),
    "label_share": (builtin.label_share(), '{"labels": {"k": "Entailment"}}', '{"labels": {"k": "Contradiction"}}', {"labels": {"k": "Entailment"}}),
    "bool_field": (builtin.bool_field("supported", invert=True, expected_key="hallucinated"), '{"supported": false}', '{"supported": true}', {"hallucinated": True}),
    "ifeval": (builtin.ifeval(), "no commas", "one, two", IFEVAL),
    "ranks_first": (builtin.ranks_first(), '[{"id": "r1", "score": 9}, {"id": "r2", "score": 1}]', '[{"id": "r1", "score": 1}, {"id": "r2", "score": 9}]', {"chosen": "r1"}),
    "verdict_equals": (builtin.verdict_equals(), '{"verdict": "A>B"}', '{"verdict": "B>A"}', {"label": "A>B"}),
    "with_wrong_labels": (builtin.with_wrong_labels(builtin.answer_letter(), _next_letter), "ANSWER: B", "ANSWER: A", "A"),
}


def _constructors(module: ModuleType) -> set[str]:
    def returns_a_grader(member: Callable[..., object]) -> bool:
        return inspect.signature(member).return_annotation in ("Grader", Grader)

    return {name for name, member in inspect.getmembers(module, inspect.isfunction) if not name.startswith("_") and member.__module__ == module.__name__ and returns_a_grader(member)}


def test_every_code_grader_constructor_has_a_fixture() -> None:
    assert _constructors(builtin) | _constructors(lifted) == set(FIXTURES)


@pytest.mark.parametrize("row", sorted(FIXTURES))
def test_the_catalogue_row_constructs_and_grades_its_fixture(row: str) -> None:
    grader, right, wrong, expected = FIXTURES[row]
    assert grader.kind == "code"
    assert grader.id and grader.version
    assert grader.grade(_sample(right), expected).value == 1.0, row
    if row != "constant":
        assert grader.grade(_sample(wrong), expected).value == 0.0, row
