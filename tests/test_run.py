"""``run()``: dataset x provider x graders -> a ``Run``, with every call recorded and no key or network.

The calling-and-recording assertions of 0.3.2 ``tests/behaviors/evals/test_run.py`` live here, re-expressed over a
value-in/value-out ``run()`` with ``tests.fakes.ScriptedProvider``; everything about gig, steps, suites, the CLI,
sqlite and redaction stayed in gigai.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gigai_core.budget import Budget, BudgetExceeded, CallBudget, Reservation, budgeted
from gigai_core.dataset import Dataset, DatasetItem
from gigai_core.graders.judge import llm_judge, pairwise_judge
from gigai_core.graders.lifted import includes, match
from gigai_core.graders.protocol import CannotGrade
from gigai_core.providers import (
    CallSettings,
    ModelRequest,
    ModelResult,
    ProviderError,
    TokenUsage,
    Turn,
)
from gigai_core.records import Prereg
from gigai_core.run import Run, open_run, run, save_run
from gigai_core.sample import Sample
from gigai_core.score import GraderKind, Score
from gigai_core.trace import JsonlTrace, ModelSpan, TrialSpan, read_spans
from gigai_core.values import JsonValue
from tests.fakes import ScriptedProvider

SETTINGS = CallSettings(model="scripted/model")
NOW = datetime(2026, 1, 1, tzinfo=UTC)
PASS = 'Looks right.\n{"verdict": "pass", "explanation": "it is the capital"}'
FAIL = 'Not it.\n{"verdict": "fail", "explanation": "wrong city"}'
PREREG = Prereg(primary_metric="includes", pass_rule="min_lower_bound", margin=0.05, n_planned=2, registered_at=NOW)


def _dataset(count: int = 10, *, expected: JsonValue = "yes", system: str | None = None) -> Dataset:
    items = [
        DatasetItem(id=f"item-{n}", cluster_id=f"cluster-{n}", prompt=f"question {n}", expected=expected, system=system)
        for n in range(count)
    ]
    return Dataset.of("ten", items)


def _budget(max_calls: int | None = None) -> CallBudget:
    return CallBudget(max_calls=max_calls)


class _ReserveCap:
    """A ``Budget`` whose ``plan`` accepts everything and whose ``reserve`` refuses from call ``cap + 1`` on.

    ``CallBudget.plan`` refuses a run that cannot fit before the first call, so a cap that bites in the middle of a
    run needs a budget that only counts at reservation time (the way 0.3.2's ``max_calls`` did).
    """

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.reserved = 0

    def plan(self, requests: Sequence[ModelRequest]) -> float:
        return 0.0

    def reserve(self, request: ModelRequest, *, cap_usd: float | None = None) -> Reservation:
        if self.reserved >= self.cap:
            raise BudgetExceeded(f"1 more call would pass the cap of {self.cap}")
        self.reserved += 1
        return Reservation(estimate_usd=0.0)

    def settle(self, reservation: Reservation, result: ModelResult | None) -> None:
        return None


class _Refusing:
    """A grader that raises ``CannotGrade`` for the items in ``refused`` and grades 1.0 otherwise."""

    id = "refusing"
    version = "1"
    kind: GraderKind = "code"

    def __init__(self, refused: Sequence[str]) -> None:
        self.refused = tuple(refused)
        self.graded: list[str] = []

    def grade(self, sample: Sample, expected: JsonValue) -> Score:
        self.graded.append(sample.item_id)
        if sample.item_id in self.refused:
            raise CannotGrade(f"no verdict on {sample.item_id}")
        return Score(value=1.0)


class _Broken:
    """A grader with a bug."""

    id = "broken"
    version = "1"
    kind: GraderKind = "code"

    def grade(self, sample: Sample, expected: JsonValue) -> Score:
        raise RuntimeError("grader bug")


class _Timeout(ProviderError):
    """A provider failure with its own code, the way an adapter's timeout error has one."""

    code = "timeout"
    retryable = True


def _trace_path(result: Run) -> Path:
    assert result.record.trace_path is not None
    return result.record.trace_path


def _record_closes(monkeypatch: pytest.MonkeyPatch) -> list[JsonlTrace]:
    """Every ``JsonlTrace`` closed from now on, in order; the real close still runs."""
    closed: list[JsonlTrace] = []
    real_close = JsonlTrace.close

    def recording_close(self: JsonlTrace) -> None:
        closed.append(self)
        real_close(self)

    monkeypatch.setattr(JsonlTrace, "close", recording_close)
    return closed


def _result(text: str, cost_usd: float | None) -> ModelResult:
    return ModelResult(
        text=text,
        finish_reason="stop",
        resolved_model="scripted/model-2026",
        usage=TokenUsage(input_tokens=3, output_tokens=1, total_tokens=4),
        cost_status="provider_reported" if cost_usd is not None else "unavailable",
        cost_usd=cost_usd,
        latency_ms=7,
        started_at=NOW,
        completed_at=NOW,
    )


# --- relocated from 0.3.2 tests/behaviors/evals/test_run.py ---


def test_solve_on_the_ci_slice_traces_ten_calls_and_scores_each_with_every_scorer(tmp_path: Path) -> None:
    provider = ScriptedProvider(["yes"] * 10)
    result = run(_dataset(10), provider, [includes(), match()], settings=SETTINGS, budget=_budget(), trace_dir=tmp_path)
    assert len(provider.requests) == 10
    assert result.record.calls_made == 10
    assert len(result.samples) == 10
    assert all(sample.outcome == "succeeded" for sample in result.samples)
    assert len(result.scores) == 20
    assert {score.grader_id for score in result.scores} == {"includes", "match"}
    assert result.values("includes") == {f"item-{n}": 1.0 for n in range(10)}
    assert result.values("match") == {f"item-{n}": 1.0 for n in range(10)}
    assert result.record.trace_path == tmp_path / f"{result.record.run_id}.jsonl"
    spans = list(read_spans(_trace_path(result)))
    assert [span.kind for span in spans] == ["model_call", "trial"] * 10


def test_k_repeats_every_item_with_repeat_indexes_0_to_k_minus_1() -> None:
    provider = ScriptedProvider(["yes", "no", "yes"] * 2)
    result = run(_dataset(2), provider, [includes()], settings=SETTINGS, budget=_budget(), repeats=3)
    assert result.record.repeats == 3
    assert [(sample.item_id, sample.repeat_index) for sample in result.samples] == [
        ("item-0", 0), ("item-0", 1), ("item-0", 2), ("item-1", 0), ("item-1", 1), ("item-1", 2),
    ]
    assert [request.metadata["repeat_index"] for request in provider.requests] == [0, 1, 2, 0, 1, 2]
    assert result.values("includes") == {"item-0": pytest.approx(2 / 3), "item-1": pytest.approx(2 / 3)}


def test_max_calls_stops_before_the_call_that_would_pass_it() -> None:
    provider = ScriptedProvider(["yes"] * 10)
    with pytest.raises(BudgetExceeded, match="1 more call would pass the cap of 3"):
        run(_dataset(10), provider, [includes()], settings=SETTINGS, budget=_ReserveCap(3))
    assert len(provider.requests) == 3


def test_max_calls_counts_judge_calls_too() -> None:
    provider = ScriptedProvider(["yes", PASS] * 6)
    budget = _budget(max_calls=8)
    judge = llm_judge(budgeted(provider, budget), rubric="Is it yes?", settings=SETTINGS)
    with pytest.raises(BudgetExceeded, match=r"1 more call\(s\) would pass max_calls=8 \(already 8\)"):
        run(_dataset(6), provider, [judge], settings=SETTINGS, budget=budget)
    assert budget.calls == 8
    assert len(provider.requests) == 8
    # core grades after each trial, so the cap of 8 is reached as 4 subject calls and 4 judge calls (0.3.2 graded at
    # the end: 6 + 2); the judge's calls are the ones that count, a subject-only count would have reached 6
    prompts = [request.turns[-1].content for request in provider.requests]
    subject = [prompt for prompt in prompts if prompt.startswith("question ")]
    judged = [prompt for prompt in prompts if prompt.startswith("Is it yes?")]
    assert (len(subject), len(judged)) == (4, 4)
    assert prompts == [prompt for pair in zip(subject, judged, strict=True) for prompt in pair]


def test_a_validated_judge_scores_through_its_own_calls(tmp_path: Path) -> None:
    provider = ScriptedProvider(["yes", PASS, "yes", FAIL])
    budget = _budget()
    judge = llm_judge(budgeted(provider, budget), rubric="Is it yes?", settings=SETTINGS, grader_id="rubric-judge")
    result = run(_dataset(2), provider, [judge], settings=SETTINGS, budget=budget, trace_dir=tmp_path)
    assert result.values("rubric-judge") == {"item-0": 1.0, "item-1": 0.0}
    assert result.record.calls_made == 4
    for sample, score in zip(result.samples, result.scores, strict=True):
        trial, subject, judged = sample.spans
        assert isinstance(trial, TrialSpan)
        assert isinstance(subject, ModelSpan) and isinstance(judged, ModelSpan)
        assert subject.parent_id == trial.span_id and judged.parent_id == trial.span_id
        assert judged.text in (PASS, FAIL) and judged.started_at >= subject.started_at
        assert score.kind == "model"
        assert score.judge_model == "scripted/model"
        assert score.judge_prompt_sha256 == judge.prompt_sha256
        assert score.judge_span_ids == (judged.span_id,)
    written = [(span.kind, span.span_id) for span in read_spans(_trace_path(result))]
    expected = [(kind, span.span_id) for sample in result.samples for kind, span in zip(("model_call", "model_call", "trial"), (sample.spans[1], sample.spans[2], sample.spans[0]), strict=True)]
    assert written == expected


def test_a_pairwise_judge_keeps_its_two_calls_in_order_under_the_trial(tmp_path: Path) -> None:
    pair = "Which is right?\n\nQUESTION:\n2 + 2?\n\nRESPONSE A:\n4\n\nRESPONSE B:\n5"
    item = DatasetItem(id="pair", cluster_id="pair", prompt=pair, expected={"label": "A"})
    provider = ScriptedProvider(["noted", '{"verdict": "A"}', '{"verdict": "B"}'])
    budget = _budget()
    judge = pairwise_judge(budgeted(provider, budget), rubric="Which is correct?", settings=SETTINGS, grader_id="pair-judge")
    result = run(Dataset.of("pairs", [item]), provider, [judge], settings=SETTINGS, budget=budget, trace_dir=tmp_path)
    assert result.values("pair-judge") == {"pair": 1.0} and result.record.calls_made == 3
    (sample,) = result.samples
    (score,) = result.scores
    trial, subject, first, second = sample.spans
    assert isinstance(trial, TrialSpan) and isinstance(subject, ModelSpan)
    assert isinstance(first, ModelSpan) and isinstance(second, ModelSpan)
    assert (first.text, second.text) == ('{"verdict": "A"}', '{"verdict": "B"}')
    assert "RESPONSE A:\n4\n\nRESPONSE B:\n5" in first.last_turn.content
    assert "RESPONSE A:\n5\n\nRESPONSE B:\n4" in second.last_turn.content
    assert score.judge_span_ids == (first.span_id, second.span_id)
    assert first.span_id != second.span_id
    assert {subject.parent_id, first.parent_id, second.parent_id} == {trial.span_id}
    assert subject.started_at <= first.started_at <= second.started_at
    written = [(span.kind, span.span_id) for span in read_spans(_trace_path(result))]
    assert written == [("model_call", subject.span_id), ("model_call", first.span_id), ("model_call", second.span_id), ("trial", trial.span_id)]


# --- new ---


def test_the_plan_runs_before_the_first_call_and_a_cap_that_cannot_fit_leaves_no_file(tmp_path: Path) -> None:
    provider = ScriptedProvider(["yes"] * 10)
    trace_dir = tmp_path / "traces"
    with pytest.raises(BudgetExceeded, match="max_calls=3"):
        run(_dataset(10), provider, [includes()], settings=SETTINGS, budget=_budget(max_calls=3), trace_dir=trace_dir)
    assert provider.requests == []
    assert not trace_dir.exists()


def test_a_failing_call_in_the_middle_is_a_failed_sample_and_the_run_goes_on() -> None:
    provider = ScriptedProvider(["yes", _Timeout("the model took too long"), "yes"])
    result = run(_dataset(3), provider, [includes()], settings=SETTINGS, budget=_budget())
    assert [sample.outcome for sample in result.samples] == ["succeeded", "failed", "succeeded"]
    failed = result.samples[1]
    assert failed.error is not None
    assert (failed.error.code, failed.error.message, failed.error.retryable) == ("timeout", "the model took too long", True)
    assert failed.output_text == "" and failed.output_parsed is None and failed.parse_status == "not_parsed"
    trial, model = failed.spans
    assert isinstance(trial, TrialSpan) and trial.outcome == "failed" and trial.error == failed.error
    assert isinstance(model, ModelSpan) and model.error == failed.error and model.text == ""
    assert model.finish_reason == "error" and model.call.cost_status == "unavailable"
    assert result.record.calls_made == 3
    assert result.values("includes") == {"item-0": 1.0, "item-2": 1.0}
    assert result.ungraded("includes") == ("item-1",)


def test_a_failed_sample_is_not_graded() -> None:
    provider = ScriptedProvider([_Timeout("gone"), "yes"])
    grader = _Refusing(())
    result = run(_dataset(2), provider, [grader], settings=SETTINGS, budget=_budget())
    assert grader.graded == ["item-1"]
    assert [score.trace_id for score in result.scores] == [result.samples[1].trace_id]


def test_cannot_grade_leaves_no_score_and_ungraded_names_the_item_while_values_omits_it() -> None:
    provider = ScriptedProvider(["yes"] * 3)
    result = run(_dataset(3), provider, [_Refusing(["item-1"])], settings=SETTINGS, budget=_budget())
    assert len(result.scores) == 2
    assert result.values("refusing") == {"item-0": 1.0, "item-2": 1.0}
    assert result.ungraded("refusing") == ("item-1",)


def test_a_grader_bug_propagates_and_the_trace_is_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    closed = _record_closes(monkeypatch)
    provider = ScriptedProvider(["yes"] * 3)
    with pytest.raises(RuntimeError, match="grader bug"):
        run(_dataset(3), provider, [_Broken()], settings=SETTINGS, budget=_budget(), trace_dir=tmp_path)
    (path,) = tmp_path.iterdir()
    assert [span.kind for span in read_spans(path)] == ["model_call"]
    assert [trace.path for trace in closed] == [path]


def test_two_graders_that_disagree_make_the_item_contested() -> None:
    provider = ScriptedProvider(["yes", "no"])
    result = run(_dataset(2), provider, [includes(), _Refusing(())], settings=SETTINGS, budget=_budget())
    assert result.values("includes") == {"item-0": 1.0, "item-1": 0.0}
    assert result.values("refusing") == {"item-0": 1.0, "item-1": 1.0}
    assert result.contested() == ("item-1",)


def test_repeats_3_makes_three_trials_per_item_and_values_the_mean() -> None:
    provider = ScriptedProvider(["yes", "yes", "no"])
    result = run(_dataset(1), provider, [includes()], settings=SETTINGS, budget=_budget(), repeats=3)
    assert [sample.repeat_index for sample in result.samples] == [0, 1, 2]
    assert result.values("includes") == {"item-0": pytest.approx(2 / 3)}


def test_the_trace_file_holds_the_spans_run_trace_yields(tmp_path: Path) -> None:
    provider = ScriptedProvider(["yes", "no"])
    result = run(_dataset(2), provider, [includes()], settings=SETTINGS, budget=_budget(), trace_dir=tmp_path, repeats=1)
    written = list(read_spans(_trace_path(result)))
    assert len(written) == 4
    for item in ("item-0", "item-1"):
        from_run = sorted(result.trace(item), key=lambda span: span.span_id)
        from_file = sorted((span for span in written if span.span_id in {s.span_id for s in from_run}), key=lambda span: span.span_id)
        assert from_file == from_run
        assert [span.kind for span in result.trace(item)] == ["trial", "model_call"]


def test_store_inputs_decides_whether_request_turns_are_written(tmp_path: Path) -> None:
    for store_inputs in (False, True):
        provider = ScriptedProvider(["yes"])
        result = run(
            _dataset(1), provider, [includes()], settings=SETTINGS, budget=_budget(),
            trace_dir=tmp_path / str(store_inputs), store_inputs=store_inputs,
        )
        (model,) = [span for span in read_spans(_trace_path(result)) if isinstance(span, ModelSpan)]
        (in_memory,) = result.samples[0].model_spans
        expected = provider.requests[0].turns if store_inputs else None
        assert model.request_turns == expected
        assert in_memory.request_turns == expected
        assert model.last_turn == Turn(role="user", content="question 0")
        assert len(model.request_sha256) == 64


def test_the_item_system_wins_over_the_system_argument() -> None:
    items = [
        DatasetItem(id="own", cluster_id="own", prompt="q", expected="yes", system="Be terse."),
        DatasetItem(id="shared", cluster_id="shared", prompt="q", expected="yes"),
    ]
    provider = ScriptedProvider(["yes", "yes"])
    result = run(Dataset.of("sys", items), provider, [includes()], settings=SETTINGS, budget=_budget(), system="Be kind.")
    own, shared = provider.requests
    assert own.turns == (Turn(role="system", content="Be terse."), Turn(role="user", content="q"))
    assert shared.turns == (Turn(role="system", content="Be kind."), Turn(role="user", content="q"))
    assert [sample.input for sample in result.samples] == [own.turns, shared.turns]
    assert own.metadata == {"item_id": "own", "repeat_index": 0}


def test_no_system_anywhere_sends_the_user_turn_alone() -> None:
    provider = ScriptedProvider(["yes"])
    run(_dataset(1), provider, [includes()], settings=SETTINGS, budget=_budget())
    assert provider.requests[0].turns == (Turn(role="user", content="question 0"),)


def test_schema_mode_takes_the_items_own_schema_and_parses_the_reply() -> None:
    schema: Mapping[str, JsonValue] = {"type": "object", "properties": {"city": {"type": "string"}}}
    own_expected: JsonValue = {"schema": dict(schema), "answer": "Paris"}
    items = [
        DatasetItem(id="own", cluster_id="own", prompt="q", expected=own_expected),
        DatasetItem(id="shared", cluster_id="shared", prompt="q", expected={"answer": "Rome"}),
    ]
    shared_schema: Mapping[str, JsonValue] = {"type": "object"}
    settings = CallSettings(model="scripted/model", output_mode="schema", json_schema=shared_schema)
    provider = ScriptedProvider(['{"city": "Paris"}', "not json"])
    result = run(Dataset.of("schema", items), provider, [], settings=settings, budget=_budget())
    own, shared = provider.requests
    assert own.settings.json_schema == schema
    assert shared.settings.json_schema == shared_schema
    assert result.samples[0].output_parsed == {"city": "Paris"} and result.samples[0].parse_status == "ok"
    assert result.samples[1].output_parsed is None and result.samples[1].parse_status == "not_json"
    assert result.record.settings == settings


def test_schema_mode_without_any_schema_is_refused_before_any_call() -> None:
    provider = ScriptedProvider(["{}"])
    settings = CallSettings(model="scripted/model", output_mode="schema")
    with pytest.raises(ValueError, match="item-0"):
        run(_dataset(1), provider, [includes()], settings=settings, budget=_budget())
    assert provider.requests == []


def test_text_mode_parses_nothing_and_json_mode_parses_the_reply() -> None:
    provider = ScriptedProvider(['{"a": 1}', '{"a": 1}'])
    text = run(_dataset(1), provider, [], settings=SETTINGS, budget=_budget())
    json_mode = run(_dataset(1), provider, [], settings=CallSettings(model="scripted/model", output_mode="json"), budget=_budget())
    assert (text.samples[0].output_parsed, text.samples[0].parse_status) == (None, "not_parsed")
    assert (json_mode.samples[0].output_parsed, json_mode.samples[0].parse_status) == ({"a": 1}, "ok")


def test_a_duplicate_grader_id_repeats_0_and_a_conversation_item_are_refused_before_any_call(tmp_path: Path) -> None:
    provider = ScriptedProvider(["yes"] * 4)
    with pytest.raises(ValueError, match="duplicate grader id 'includes'"):
        run(_dataset(1), provider, [includes(), includes()], settings=SETTINGS, budget=_budget(), trace_dir=tmp_path)
    twice = [DatasetItem(id="same", cluster_id=f"c{n}", prompt=f"q{n}", expected="yes") for n in range(2)]
    doubled = Dataset.of("doubled", [*twice, DatasetItem(id="other", cluster_id="other", prompt="q", expected="yes"), *twice])
    with pytest.raises(ValueError, match=r"duplicate item ids \['same'\]: one trial per item needs every id once"):
        run(doubled, provider, [includes()], settings=SETTINGS, budget=_budget(), trace_dir=tmp_path)
    with pytest.raises(ValueError, match="repeats must be at least 1, not 0"):
        run(_dataset(1), provider, [includes()], settings=SETTINGS, budget=_budget(), repeats=0, trace_dir=tmp_path)
    conversation = DatasetItem(
        id="chat", cluster_id="chat", prompt="hi", expected="yes", turns=(Turn(role="user", content="hi"), Turn(role="user", content="more")),
    )
    with pytest.raises(ValueError, match=r"items \['chat'\] have scripted turns: scripted turns need core 0.2.0"):
        run(Dataset.of("chat", [conversation]), provider, [includes()], settings=SETTINGS, budget=_budget(), trace_dir=tmp_path)
    assert provider.requests == []
    assert list(tmp_path.iterdir()) == []


def test_a_mid_run_budget_refusal_propagates_and_leaves_a_readable_partial_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    closed = _record_closes(monkeypatch)
    provider = ScriptedProvider(["yes"] * 5)
    with pytest.raises(BudgetExceeded):
        run(_dataset(5), provider, [includes()], settings=SETTINGS, budget=_ReserveCap(2), trace_dir=tmp_path)
    assert len(closed) == 1
    (path,) = tmp_path.iterdir()
    spans = list(read_spans(path))
    assert [span.kind for span in spans] == ["model_call", "trial", "model_call", "trial"]
    assert [span.item_id for span in spans if isinstance(span, TrialSpan)] == ["item-0", "item-1"]


def test_save_run_then_open_run_round_trips_a_run_by_reference_and_with_inputs(tmp_path: Path) -> None:
    provider = ScriptedProvider(["yes", "no"])
    dataset = _dataset(2)
    result = run(dataset, provider, [includes()], settings=SETTINGS, budget=_budget(), trace_dir=tmp_path / "t", prereg=PREREG, tag="smoke")
    save_run(result, tmp_path / "by-reference.jsonl")
    save_run(result, tmp_path / "with-inputs.jsonl", inputs_by_reference=False)
    assert open_run(tmp_path / "by-reference.jsonl", dataset=dataset) == result
    assert open_run(tmp_path / "with-inputs.jsonl") == result


def test_spent_usd_and_calls_made_are_the_sums_over_the_spans() -> None:
    provider = ScriptedProvider([_result("yes", 0.25), _result("no", None), _result("yes", 0.5)])
    result = run(_dataset(3), provider, [includes()], settings=SETTINGS, budget=_budget())
    assert result.record.spent_usd == pytest.approx(0.75)
    assert result.record.calls_made == 3
    assert [span.call.cost_usd for sample in result.samples for span in sample.model_spans] == [0.25, None, 0.5]
    assert result.record.prices == ()


def test_the_record_carries_what_the_run_was_made_of() -> None:
    provider = ScriptedProvider(["yes"])
    dataset = _dataset(1)
    result = run(dataset, provider, [includes()], settings=SETTINGS, budget=_budget(), tag="t")
    record = result.record
    assert record.run_id.startswith("run_") and len(record.run_id) == 20
    assert record.dataset == dataset.ref
    assert record.provider == provider.identity
    assert record.settings == SETTINGS
    assert record.trace_path is None
    assert record.tag == "t" and record.prereg is None and record.replay_of is None
    assert record.completed_at is not None and record.completed_at >= record.created_at
    assert result.items == {"item-0": dataset.items[0]}
    assert not provider.closed
    sample = result.samples[0]
    assert sample.run_id == record.run_id and sample.trace_id.startswith("trace_")
    assert sample.item_version == "1" and sample.cluster_id == "cluster-0"
    assert result.scores[0].score_id.startswith("score_") and result.scores[0].trace_id == sample.trace_id


def test_a_budget_is_a_protocol_a_test_may_implement() -> None:
    budget: Budget = _ReserveCap(10)
    provider = ScriptedProvider(["yes"])
    result = run(_dataset(1), provider, [includes()], settings=SETTINGS, budget=budget)
    assert result.record.calls_made == 1
