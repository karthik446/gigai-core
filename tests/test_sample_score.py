"""`Sample`, `Score`/`ScoreRecord` and the `Grader` Protocol: round trips, strictness, a satisfiable contract."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from gigai_core.graders.protocol import CannotGrade, Grader
from gigai_core.providers import (
    CallSettings,
    ProviderIdentity,
    TokenUsage,
    ToolCall,
    Turn,
)
from gigai_core.sample import Sample
from gigai_core.score import GraderKind, JudgeNote, Score, ScoreRecord
from gigai_core.trace import (
    CallError,
    ModelCall,
    ModelExchange,
    ModelSpan,
    Span,
    ToolSpan,
    TrialSpan,
)
from gigai_core.values import JsonValue

NOW = datetime(2026, 11, 13, 10, 0, tzinfo=UTC)
IDENTITY = ProviderIdentity(provider="scripted", endpoint="memory", adapter="tests:1")
SETTINGS = CallSettings(model="some/model")


def _call(latency_ms: int = 5) -> ModelCall:
    return ModelCall(
        identity=IDENTITY,
        target=None,
        settings=SETTINGS,
        resolved_model="some/model",
        usage=TokenUsage(input_tokens=3, output_tokens=1, total_tokens=4),
        cost_usd=0.0,
        cost_status="provider_reported",
        latency_ms=latency_ms,
    )


def _model_span(span_id: str, parent_id: str, text: str, tool_calls: tuple[ToolCall, ...] = ()) -> ModelSpan:
    return ModelSpan(
        span_id=span_id,
        parent_id=parent_id,
        turn=0,
        started_at=NOW,
        completed_at=NOW,
        call=_call(),
        request_sha256="0" * 64,
        last_turn=Turn(role="user", content="2+2"),
        text=text,
        finish_reason="tool_calls" if tool_calls else "stop",
        tool_calls=tool_calls,
    )


def _sample(spans: tuple[Span, ...] = (), *, trace_id: str = "trace_1") -> Sample:
    return Sample(
        trace_id=trace_id,
        run_id="run_1",
        item_id="q1",
        item_version="1",
        cluster_id="sums",
        repeat_index=0,
        input=(Turn(role="system", content="Answer tersely."), Turn(role="user", content="2+2")),
        output_text='{"answer": "4"}',
        output_parsed={"answer": "4"},
        parse_status="ok",
        outcome="succeeded",
        error=None,
        spans=spans,
        recorded_at=NOW,
    )


def test_a_sample_round_trips_and_carries_what_the_0_3_2_trace_row_carried() -> None:
    # 0.3.2 test_records.py: the trace row had repeat_index, cluster_id, parse_status, outcome and a latency.
    sample = _sample((TrialSpan(span_id="t1", turn=0, started_at=NOW, completed_at=NOW, item_id="q1", repeat_index=0),))
    assert (sample.repeat_index, sample.cluster_id, sample.parse_status) == (0, "sums", "ok")
    assert sample.outcome == "succeeded" and sample.error is None and sample.final_state is None
    assert sample.output_parsed == {"answer": "4"}
    assert Sample.model_validate_json(sample.model_dump_json()) == sample
    failed = sample.model_copy(update={"outcome": "failed", "error": CallError(code="model_call_timed_out", message="t", retryable=True)})
    assert Sample.model_validate_json(failed.model_dump_json()) == failed
    with pytest.raises(ValidationError, match="parse_status"):
        Sample.model_validate_json(sample.model_dump_json().replace('"ok"', '"parsed"'))


def test_the_span_views_keep_order_and_last_call_is_the_last_model_call() -> None:
    call = ToolCall(id="c1", name="add", arguments='{"a": 2, "b": 2}')
    trial = TrialSpan(span_id="t1", turn=0, started_at=NOW, completed_at=NOW, item_id="q1", repeat_index=0)
    first = _model_span("m1", "t1", "", (call,))
    tool = ToolSpan(span_id="k1", parent_id="t1", turn=0, started_at=NOW, completed_at=NOW, call=call, result=4)
    second = ModelSpan.place(
        ModelExchange(call=_call(9), request_sha256="1" * 64, last_turn=Turn(role="tool", content="4", tool_call_id="c1"), text="4", finish_reason="stop"),
        span_id="m2",
        parent_id="t1",
        turn=0,
        started_at=NOW,
        completed_at=NOW,
    )
    sample = _sample((trial, first, tool, second))
    assert tuple(sample.model_spans) == (first, second)
    assert tuple(sample.tool_spans) == (tool,)
    assert tuple(sample.tool_calls) == (call,)
    assert sample.last_call is not None and sample.last_call.latency_ms == 9
    assert _sample().last_call is None and tuple(_sample().tool_calls) == ()


def test_a_sample_promotes_to_an_item_that_names_its_run_and_trace() -> None:
    item = _sample().promoted(item_id="q1-promoted", cluster_id="sums", expected="4")
    assert (item.id, item.cluster_id, item.expected) == ("q1-promoted", "sums", "4")
    assert item.turns == (Turn(role="user", content="2+2"),) and item.prompt == ""
    assert item.system == "Answer tersely."
    assert item.source_ids == {"run_id": "run_1", "trace_id": "trace_1"}
    assert item.user_turns() == item.turns


def test_score_and_judge_note_round_trip() -> None:
    plain = Score(value=1.0, explanation="matched", answer="4")
    assert Score.model_validate_json(plain.model_dump_json()) == plain and plain.judge is None
    note = JudgeNote(
        model="judge/model",
        prompt_sha256="a" * 64,
        position_swapped=True,
        calls=(ModelExchange(call=_call(), request_sha256="2" * 64, last_turn=Turn(role="user", content="judge"), text="C", finish_reason="stop"),),
    )
    judged = Score(value=0.0, judge=note)
    assert Score.model_validate_json(judged.model_dump_json()) == judged
    with pytest.raises(ValidationError):
        Score(value=float("nan"))


def _record(value: float, grader_id: str = "exact_match", kind: GraderKind = "code") -> ScoreRecord:
    return ScoreRecord(
        score_id="score_1", trace_id="trace_1", grader_id=grader_id, grader_version="1", kind=kind, value=value, recorded_at=NOW
    )


def test_a_score_record_round_trips_and_a_judge_record_links_its_spans() -> None:
    assert ScoreRecord.model_validate_json(_record(1.0).model_dump_json()) == _record(1.0)
    judged = ScoreRecord(
        score_id="score_2",
        trace_id="trace_1",
        grader_id="pairwise",
        grader_version="3",
        kind="model",
        value=0.0,
        explanation="B was better",
        judge_model="judge/model",
        judge_prompt_sha256="a" * 64,
        position_swapped=False,
        judge_span_ids=("j1", "j2"),
        recorded_at=NOW,
    )
    assert ScoreRecord.model_validate_json(judged.model_dump_json()) == judged


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("value", "1", "float"),
        ("value", float("inf"), "finite"),
        ("kind", "judge", "code"),
        ("grader_version", 1, "string"),
        ("recorded_at", datetime(2026, 11, 13, 10, 0), "timezone"),
        ("scorer_id", "exact_match", "extra_forbidden"),
    ],
)
def test_a_score_record_is_strict(field: str, value: JsonValue | datetime, reason: str) -> None:
    fields: dict[str, JsonValue | datetime] = {
        "score_id": "score_1",
        "trace_id": "trace_1",
        "grader_id": "exact_match",
        "grader_version": "1",
        "kind": "code",
        "value": 1.0,
        "recorded_at": NOW,
    }
    fields[field] = value
    with pytest.raises(ValidationError, match=reason):
        ScoreRecord.model_validate(fields)


def test_a_score_record_needs_its_timestamp() -> None:
    with pytest.raises(ValidationError, match="recorded_at"):
        ScoreRecord.model_validate_json('{"score_id": "s", "trace_id": "t", "grader_id": "g", "grader_version": "1", "kind": "code", "value": 1.0}')


class Exact:
    """A five-line grader: the Protocol needs no base class."""

    id = "exact"
    version = "1"
    kind: GraderKind = "code"

    def grade(self, sample: Sample, expected: JsonValue) -> Score:
        return Score(value=1.0 if sample.output_parsed == expected else 0.0)


class FromFunction:
    """A grader over a plain function; `None` from the function is `CannotGrade`."""

    def __init__(self, id: str, rule: Callable[[str, JsonValue], float | None]) -> None:
        self._id = id
        self._rule = rule

    @property
    def id(self) -> str:
        return self._id

    @property
    def version(self) -> str:
        return "2"

    @property
    def kind(self) -> GraderKind:
        return "code"

    def grade(self, sample: Sample, expected: JsonValue) -> Score:
        value = self._rule(sample.output_text, expected)
        if value is None:
            raise CannotGrade(f"{self._id}: no rule for {expected!r}")
        return Score(value=value, explanation=f"{self._id} on {sample.output_text!r}")


def _length_rule(text: str, expected: JsonValue) -> float | None:
    if not isinstance(expected, int):
        return None
    return 1.0 if len(text) <= expected else 0.0


def test_the_grader_protocol_is_satisfied_by_a_small_class_and_a_function_backed_one() -> None:
    graders: list[Grader] = [Exact(), FromFunction("short", _length_rule)]
    sample = _sample()
    assert [grader.id for grader in graders] == ["exact", "short"]
    assert [(grader.version, grader.kind) for grader in graders] == [("1", "code"), ("2", "code")]
    assert graders[0].grade(sample, {"answer": "4"}).value == 1.0
    assert graders[0].grade(sample, {"answer": "5"}).value == 0.0
    assert graders[1].grade(sample, 40).value == 1.0 and graders[1].grade(sample, 3).value == 0.0
    with pytest.raises(CannotGrade, match="no rule for 'four'"):
        graders[1].grade(sample, "four")
    assert isinstance(CannotGrade("x"), Exception)
