"""The `Run` value and its file: `save_run`/`open_run` round trips, error cases, and the four views."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from gigai_core.dataset import Dataset, DatasetItem
from gigai_core.providers import CallSettings, ProviderIdentity, TokenUsage, Turn
from gigai_core.records import RunRecord
from gigai_core.run import (
    RUN_LINE,
    RUN_SCHEMA,
    Run,
    RunFileError,
    RunHeaderLine,
    open_run,
    save_run,
)
from gigai_core.sample import Sample
from gigai_core.score import GraderKind, ScoreRecord
from gigai_core.trace import ModelCall, ModelSpan, Span, TrialSpan

NOW = datetime(2026, 11, 13, 10, 0, tzinfo=UTC)
IDENTITY = ProviderIdentity(provider="scripted", endpoint="memory", adapter="tests:1")
SETTINGS = CallSettings(model="some/model")
ITEMS = ("a", "b", "c")
REPEATS = 2
# grader -> item -> value per repeat; None is CannotGrade (no score row).
GRADES: dict[str, dict[str, tuple[float | None, ...]]] = {
    "g1": {"a": (1.0, 1.0), "b": (1.0, 0.0), "c": (1.0, 1.0)},
    "g2": {"a": (1.0, 1.0), "b": (0.5, 0.5), "c": (1.0, None)},
    "g3": {"a": (0.0, 0.0), "b": (0.5, 0.5), "c": (1.0, 1.0)},
}
KINDS: dict[str, GraderKind] = {"g1": "code", "g2": "code", "g3": "model"}


def _dataset() -> Dataset:
    return Dataset.of("letters", [DatasetItem(id=item, cluster_id=item, prompt=f"say {item}", expected=item) for item in ITEMS])


def _record(dataset: Dataset) -> RunRecord:
    return RunRecord(
        run_id="run_1", dataset=dataset.ref, settings=SETTINGS, provider=IDENTITY, created_at=NOW, completed_at=NOW, repeats=REPEATS
    )


def _spans(item: str, repeat: int) -> tuple[Span, ...]:
    trial = TrialSpan(span_id=f"t-{item}-{repeat}", turn=0, started_at=NOW, completed_at=NOW, item_id=item, repeat_index=repeat, outcome="succeeded")
    call = ModelCall(
        identity=IDENTITY,
        target=None,
        settings=SETTINGS,
        resolved_model="some/model",
        usage=TokenUsage(input_tokens=2, output_tokens=1, total_tokens=3),
        cost_usd=None,
        cost_status="unavailable",
        latency_ms=1,
    )
    model = ModelSpan(
        span_id=f"m-{item}-{repeat}",
        parent_id=trial.span_id,
        turn=0,
        started_at=NOW,
        completed_at=NOW,
        call=call,
        request_sha256="0" * 64,
        last_turn=Turn(role="user", content=f"say {item}"),
        text=item,
        finish_reason="stop",
    )
    return (trial, model)


def _sample(item: str, repeat: int) -> Sample:
    return Sample(
        trace_id=f"trace-{item}-{repeat}",
        run_id="run_1",
        item_id=item,
        item_version="1",
        cluster_id=item,
        repeat_index=repeat,
        input=(Turn(role="user", content=f"say {item}"),),
        output_text=item,
        output_parsed=item,
        parse_status="not_parsed",
        outcome="succeeded",
        error=None,
        spans=_spans(item, repeat),
        recorded_at=NOW,
    )


def _run() -> Run:
    dataset = _dataset()
    samples = tuple(_sample(item, repeat) for item in ITEMS for repeat in range(REPEATS))
    scores: list[ScoreRecord] = []
    for sample in samples:
        for grader_id, by_item in GRADES.items():
            value = by_item[sample.item_id][sample.repeat_index]
            if value is None:
                continue
            scores.append(
                ScoreRecord(
                    score_id=f"score-{grader_id}-{sample.trace_id}",
                    trace_id=sample.trace_id,
                    grader_id=grader_id,
                    grader_version="1",
                    kind=KINDS[grader_id],
                    value=value,
                    recorded_at=NOW,
                )
            )
    return Run(record=_record(dataset), items={item.id: item for item in dataset.items}, samples=samples, scores=tuple(scores))


def test_values_are_per_item_means_over_the_repeats_with_ungraded_items_absent() -> None:
    run = _run()
    assert run.values("g1") == {"a": 1.0, "b": 0.5, "c": 1.0}
    assert run.values("g2") == {"a": 1.0, "b": 0.5}
    assert run.values("g3") == {"a": 0.0, "b": 0.5, "c": 1.0}
    assert run.values("never") == {}


def test_ungraded_lists_the_items_a_grader_had_no_verdict_for() -> None:
    run = _run()
    assert run.ungraded("g1") == () and run.ungraded("g3") == ()
    assert run.ungraded("g2") == ("c",)  # one of the two repeats has no score: the item is not measured
    assert run.ungraded("never") == ("a", "b", "c")
    assert set(run.values("g2")) | set(run.ungraded("g2")) == set(ITEMS)


def test_contested_names_the_items_where_one_grader_says_met_and_another_says_less() -> None:
    run = _run()
    assert run.contested() == ("a",)  # g1 and g2 at 1.0, g3 at 0.0
    # b: nobody at 1.0; c: the graders with a verdict all say 1.0 (g2's missing verdict is not a disagreement).
    unanimous = run.model_copy(update={"scores": tuple(score for score in run.scores if score.grader_id != "g3")})
    assert unanimous.contested() == ()
    assert Run(record=run.record, items=run.items, samples=run.samples, scores=()).contested() == ()


def test_contested_needs_a_grader_at_exactly_1_against_one_below_it() -> None:
    run = _run()

    def with_value(grader_id: str, item_id: str, value: float) -> Run:
        traces = {sample.trace_id for sample in run.samples if sample.item_id == item_id}
        scores = tuple(
            score.model_copy(update={"value": value})
            if score.grader_id == grader_id and score.trace_id in traces
            else score
            for score in run.scores
        )
        return run.model_copy(update={"scores": scores})

    no_one_at_1 = with_value("g3", "b", 0.0)  # b: 0.5, 0.5, 0.0
    assert "b" not in no_one_at_1.contested()
    near_1 = with_value("g1", "b", 1.0)  # b: 1.0, 0.5, 0.5
    assert "b" in near_1.contested()
    assert "b" not in with_value("g2", "b", 0.75).contested()  # b: 0.5, 0.75, 0.5: nobody at 1.0


def test_trace_yields_an_items_spans_across_its_trials_in_recorded_order() -> None:
    run = _run()
    assert [span.span_id for span in run.trace("b")] == ["t-b-0", "m-b-0", "t-b-1", "m-b-1"]
    assert list(run.trace("missing")) == []
    reversed_run = run.model_copy(update={"samples": tuple(reversed(run.samples))})
    assert [span.span_id for span in reversed_run.trace("b")] == ["t-b-1", "m-b-1", "t-b-0", "m-b-0"]


def test_a_run_saved_by_value_reopens_equal_with_no_dataset(tmp_path: Path) -> None:
    run = _run()
    path = tmp_path / "run.jsonl"
    save_run(run, path, inputs_by_reference=False)
    lines = path.read_bytes().splitlines()
    assert [RUN_LINE.validate_json(line).kind for line in lines] == ["run"] + ["item"] * 3 + ["sample"] * 6 + ["score"] * 17
    assert RUN_LINE.validate_json(lines[0]) == RunHeaderLine(record=run.record)
    assert open_run(path) == run
    assert sorted(child.name for child in tmp_path.iterdir()) == ["run.jsonl"]  # no temp file left behind


def test_a_run_saved_by_reference_needs_its_dataset_to_reopen(tmp_path: Path) -> None:
    run = _run()
    path = tmp_path / "run.jsonl"
    save_run(run, path)
    assert "item" not in [RUN_LINE.validate_json(line).kind for line in path.read_bytes().splitlines()]
    with pytest.raises(RunFileError, match="saved by reference") as info:
        open_run(path)
    assert info.value.line == 0 and info.value.path == path and str(path) in str(info.value)
    assert open_run(path, dataset=_dataset()) == run
    other = Dataset.of("letters", [DatasetItem(id="a", cluster_id="a", prompt="say A", expected="a")])
    with pytest.raises(RunFileError, match=f"has sha256 {other.ref.sha256}, the run measured {run.record.dataset.sha256}"):
        open_run(path, dataset=other)


def test_save_run_refuses_to_overwrite_and_never_leaves_a_half_file(tmp_path: Path) -> None:
    run = _run()
    path = tmp_path / "run.jsonl"
    path.write_text("kept\n")
    with pytest.raises(FileExistsError, match="never overwrites"):
        save_run(run, path)
    assert path.read_text() == "kept\n"
    assert sorted(child.name for child in tmp_path.iterdir()) == ["run.jsonl"]


def test_open_run_names_the_bad_line_by_its_1_based_number(tmp_path: Path) -> None:
    run = _run()
    path = tmp_path / "run.jsonl"
    save_run(run, path, inputs_by_reference=False)
    lines = path.read_bytes().splitlines()
    bad = [*lines[:2], b'{"kind": "item", "item": {"id": "x"}}', *lines[3:]]
    (tmp_path / "bad.jsonl").write_bytes(b"\n".join(bad) + b"\n")
    with pytest.raises(RunFileError, match="bad.jsonl line 3: ") as info:
        open_run(tmp_path / "bad.jsonl")
    assert info.value.line == 3 and "cluster_id" in str(info.value)
    (tmp_path / "cut.jsonl").write_bytes(b"\n".join(lines[:5]) + b"\n" + lines[5][:-3] + b"\n")
    with pytest.raises(RunFileError, match="cut.jsonl line 6: "):
        open_run(tmp_path / "cut.jsonl")
    (tmp_path / "unknown-kind.jsonl").write_bytes(lines[0] + b'\n{"kind": "label"}\n')
    with pytest.raises(RunFileError, match="(?s)line 2: .*Input tag 'label'"):
        open_run(tmp_path / "unknown-kind.jsonl")


def test_open_run_refuses_an_unknown_schema_id_naming_it_and_the_file(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    save_run(_run(), path, inputs_by_reference=False)
    newer = path.read_bytes().replace(RUN_SCHEMA.encode(), b"gigai-core-run:2", 1)
    (tmp_path / "newer.jsonl").write_bytes(newer)
    with pytest.raises(RunFileError, match=f"newer.jsonl line 1: unknown schema_id 'gigai-core-run:2' \\(this version reads '{RUN_SCHEMA}'\\)"):
        open_run(tmp_path / "newer.jsonl")


def test_open_run_wants_the_header_first_and_once_and_every_sample_item_present(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    save_run(_run(), path, inputs_by_reference=False)
    lines = path.read_bytes().splitlines()
    (tmp_path / "no-header.jsonl").write_bytes(b"\n".join(lines[1:]) + b"\n")
    with pytest.raises(RunFileError, match="line 1: expected the run header first, found a 'item' line"):
        open_run(tmp_path / "no-header.jsonl")
    (tmp_path / "two-headers.jsonl").write_bytes(b"\n".join([lines[0], *lines]) + b"\n")
    with pytest.raises(RunFileError, match="line 2: a second run header"):
        open_run(tmp_path / "two-headers.jsonl")
    (tmp_path / "empty.jsonl").write_bytes(b"")
    with pytest.raises(RunFileError, match="empty.jsonl line 0: empty file"):
        open_run(tmp_path / "empty.jsonl")
    (tmp_path / "missing-item.jsonl").write_bytes(b"\n".join(lines[:3] + lines[4:]) + b"\n")  # item c dropped
    with pytest.raises(RunFileError, match="missing-item.jsonl line 8: sample 'trace-c-0' names item 'c', which is not there"):
        open_run(tmp_path / "missing-item.jsonl")
