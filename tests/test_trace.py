"""The trace: spans round-trip through a list and a JSONL file, bad lines are named, a cut file still opens."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gigai_core.providers import (
    CallSettings,
    ModelRequest,
    ModelResult,
    ToolCall,
    Turn,
    request_sha256,
)
from gigai_core.trace import (
    SPAN,
    CallError,
    JsonlTrace,
    ListTrace,
    ModelCall,
    ModelExchange,
    ModelSpan,
    Span,
    ToolSpan,
    TraceSchemaError,
    TrialSpan,
    TurnSpan,
    read_spans,
)
from tests.fakes import SCRIPTED_IDENTITY, ScriptedProvider

NOW = datetime(2026, 1, 1, tzinfo=UTC)
REQUEST = ModelRequest(settings=CallSettings(model="m"), turns=(Turn(role="user", content="2 + 2?"),))


def trial(span_id: str = "t1") -> TrialSpan:
    return TrialSpan(span_id=span_id, turn=0, started_at=NOW, completed_at=NOW, item_id="q1", repeat_index=0)


def exchange(result: ModelResult) -> ModelExchange:
    return ModelExchange(
        call=ModelCall.of(REQUEST, result, SCRIPTED_IDENTITY),
        request_sha256=request_sha256(REQUEST),
        last_turn=REQUEST.turns[-1],
        text=result.text,
        finish_reason=result.finish_reason,
    )


def model_span(span_id: str = "m1", *, request_turns: tuple[Turn, ...] | None = None) -> ModelSpan:
    result = ScriptedProvider(["4"]).invoke(REQUEST)
    placed = ModelSpan.place(exchange(result), span_id=span_id, parent_id="t1", turn=0, started_at=NOW, completed_at=NOW)
    return placed if request_turns is None else placed.model_copy(update={"request_turns": request_turns})


def four_spans() -> list[Span]:
    tool = ToolSpan(
        span_id="x1",
        parent_id="u1",
        turn=1,
        started_at=NOW,
        completed_at=None,
        call=ToolCall(id="c1", name="add", arguments='{"a": 2'),
        result={"sum": 4},
        error=CallError(code="tool_failed", message="bad json", retryable=False),
    )
    return [trial(), TurnSpan(span_id="u1", parent_id="t1", turn=1, started_at=NOW, completed_at=NOW), model_span(), tool]


def test_model_call_of_and_model_span_place_keep_the_facts() -> None:
    result = ScriptedProvider(["4"]).invoke(REQUEST)
    call = ModelCall.of(REQUEST, result, SCRIPTED_IDENTITY, target="default")
    assert (call.identity, call.target, call.settings, call.usage) == (SCRIPTED_IDENTITY, "default", REQUEST.settings, result.usage)
    span = model_span()
    assert (span.kind, span.parent_id, span.text, span.request_turns) == ("model_call", "t1", "4", None)
    assert span.last_turn == REQUEST.turns[-1]


def test_list_trace_and_jsonl_trace_round_trip_the_same_spans(tmp_path: Path) -> None:
    spans = four_spans()
    memory = ListTrace()
    path = tmp_path / "run.jsonl"
    with JsonlTrace(path) as file:
        for span in spans:
            memory.write(span)
            file.write(span)
    assert memory.spans == spans
    assert list(read_spans(path)) == spans
    assert [SPAN.validate_json(line).kind for line in path.read_bytes().splitlines()] == [
        "trial",
        "turn",
        "model_call",
        "tool_call",
    ]


def test_jsonl_trace_drops_request_turns_unless_store_inputs(tmp_path: Path) -> None:
    span = model_span(request_turns=REQUEST.turns)
    with JsonlTrace(tmp_path / "drop.jsonl") as drop, JsonlTrace(tmp_path / "keep.jsonl", store_inputs=True) as keep:
        drop.write(span)
        keep.write(span)
    dropped, kept = next(read_spans(tmp_path / "drop.jsonl")), next(read_spans(tmp_path / "keep.jsonl"))
    assert isinstance(dropped, ModelSpan) and dropped.request_turns is None
    assert isinstance(kept, ModelSpan) and kept.request_turns == REQUEST.turns


def test_a_bad_line_names_the_file_and_the_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    good = SPAN.dump_json(trial())
    path.write_bytes(good + b"\n" + good + b"\n" + b'{"kind": "trial", "span_id": 1}\n' + good + b"\n")
    with pytest.raises(TraceSchemaError) as error:
        list(read_spans(path))
    assert str(path) in str(error.value) and "line 3" in str(error.value)
    assert error.value.line == 3

    newer = good.replace(b"gigai-core-trace:1", b"gigai-core-trace:2")
    path.write_bytes(good + b"\n" + newer + b"\n")
    with pytest.raises(TraceSchemaError, match="line 2: unknown schema_id 'gigai-core-trace:2'"):
        list(read_spans(path))


def test_a_file_cut_by_a_crash_gives_back_the_finished_spans_first(tmp_path: Path) -> None:
    path = tmp_path / "cut.jsonl"
    with JsonlTrace(path) as file:
        for span in four_spans():
            file.write(span)
    whole = path.read_bytes()
    path.write_bytes(whole[: len(whole) - 20])  # the fourth line loses its tail
    spans = read_spans(path)
    assert [next(spans).span_id for _ in range(3)] == ["t1", "u1", "m1"]
    with pytest.raises(TraceSchemaError, match="line 4"):
        next(spans)


def test_eight_threads_writing_at_once_keep_every_line_intact(tmp_path: Path) -> None:
    path = tmp_path / "threads.jsonl"
    with JsonlTrace(path, store_inputs=True) as file:

        def write_batch(worker: int) -> None:
            for index in range(25):
                file.write(model_span(f"w{worker}-{index}", request_turns=REQUEST.turns * 20))

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write_batch, range(8)))
    ids = [span.span_id for span in read_spans(path)]
    assert len(ids) == 200 and set(ids) == {f"w{w}-{i}" for w in range(8) for i in range(25)}
