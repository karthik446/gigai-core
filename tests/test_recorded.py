"""``RecordedProvider``: a run replayed from its own trace, with no key and no network."""

from __future__ import annotations

import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gigai_core.budget import CallBudget, budgeted
from gigai_core.dataset import Dataset, DatasetItem
from gigai_core.graders.judge import llm_judge
from gigai_core.graders.lifted import includes
from gigai_core.providers import (
    CallCancelled,
    CallSettings,
    ModelRequest,
    ProviderError,
    ProviderIdentity,
    TokenUsage,
    Turn,
    request_sha256,
)
from gigai_core.recorded import EVERY_CAPABILITY, RecordedProvider
from gigai_core.run import Run, run
from gigai_core.trace import CallError, ModelCall, ModelSpan, Span, read_spans
from tests.fakes import SCRIPTED_IDENTITY, ScriptedProvider

SETTINGS = CallSettings(model="scripted/model")
NOW = datetime(2026, 1, 1, tzinfo=UTC)
REPLAY_IDENTITY = ProviderIdentity(provider="recorded", endpoint="trace", adapter="gigai_core.recorded.RecordedProvider:1")


class _Timeout(ProviderError):
    code = "timeout"
    retryable = True


def _dataset(prompts: list[str]) -> Dataset:
    items = [DatasetItem(id=f"item-{n}", cluster_id=f"cluster-{n}", prompt=prompt, expected="yes") for n, prompt in enumerate(prompts)]
    return Dataset.of("replay", items)


def _record(tmp_path: Path, prompts: list[str], replies: list[str | ProviderError]) -> tuple[Run, list[Span]]:
    result = run(_dataset(prompts), ScriptedProvider(replies), [includes()], settings=SETTINGS, budget=CallBudget(), trace_dir=tmp_path)
    assert result.record.trace_path is not None
    return result, list(read_spans(result.record.trace_path))


def _request(prompt: str, settings: CallSettings = SETTINGS) -> ModelRequest:
    return ModelRequest(settings=settings, turns=(Turn(role="user", content=prompt),))


def _span(prompt: str, text: str, *, error: CallError | None = None) -> ModelSpan:
    request = _request(prompt)
    call = ModelCall(
        identity=SCRIPTED_IDENTITY,
        target=None,
        settings=SETTINGS,
        resolved_model="scripted/model-2026",
        usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        cost_usd=0.01,
        cost_status="derived",
        latency_ms=5,
    )
    return ModelSpan(
        span_id=f"span-{prompt}-{text}",
        parent_id="trial",
        turn=0,
        started_at=NOW,
        completed_at=None,
        call=call,
        request_sha256=request_sha256(request),
        last_turn=request.turns[-1],
        text=text,
        finish_reason="stop" if error is None else "error",
        error=error,
    )


def test_a_run_replayed_from_its_trace_gives_the_same_texts_values_and_run(tmp_path: Path) -> None:
    first, spans = _record(tmp_path / "first", ["q0", "q1", "q2"], ["yes", "no", "yes"])
    replayed = run(_dataset(["q0", "q1", "q2"]), RecordedProvider(spans, SCRIPTED_IDENTITY), [includes()], settings=SETTINGS, budget=CallBudget())
    assert [sample.output_text for sample in replayed.samples] == ["yes", "no", "yes"]
    assert replayed.values("includes") == first.values("includes") == {"item-0": 1.0, "item-1": 0.0, "item-2": 1.0}
    volatile = {"run_id", "created_at", "completed_at", "trace_path"}
    assert replayed.record.model_dump(exclude=volatile) == first.record.model_dump(exclude=volatile)
    assert replayed.items == first.items
    sample_volatile = {"trace_id", "run_id", "recorded_at", "spans"}
    for mine, theirs in zip(replayed.samples, first.samples, strict=True):
        assert mine.model_dump(exclude=sample_volatile) == theirs.model_dump(exclude=sample_volatile)
        assert [span.kind for span in mine.spans] == [span.kind for span in theirs.spans]
        (mine_call,) = mine.model_spans
        (theirs_call,) = theirs.model_spans
        assert (mine_call.text, mine_call.request_sha256, mine_call.call) == (theirs_call.text, theirs_call.request_sha256, theirs_call.call)
    score_volatile = {"score_id", "trace_id", "recorded_at"}
    assert [s.model_dump(exclude=score_volatile) for s in replayed.scores] == [s.model_dump(exclude=score_volatile) for s in first.scores]


def test_the_same_prompt_asked_twice_gets_the_first_then_the_second_recorded_answer() -> None:
    provider = RecordedProvider([_span("same", "first"), _span("same", "second"), _span("other", "elsewhere")], REPLAY_IDENTITY)
    assert provider.invoke(_request("same")).text == "first"
    assert provider.invoke(_request("other")).text == "elsewhere"
    assert provider.invoke(_request("same")).text == "second"


def test_a_missing_answer_is_a_provider_error_naming_the_model_and_the_digest() -> None:
    provider = RecordedProvider([_span("once", "only")], REPLAY_IDENTITY)
    provider.invoke(_request("once"))
    digest = request_sha256(_request("once"))
    with pytest.raises(ProviderError, match=f"'scripted/model'.*{digest}") as caught:
        provider.invoke(_request("once"))
    assert caught.value.code == "no_recorded_answer"
    with pytest.raises(ProviderError) as never:
        provider.invoke(_request("never asked"))
    assert never.value.code == "no_recorded_answer"
    assert "recorded 0" in str(never.value)


def test_a_recorded_error_replays_with_its_code_and_retryable(tmp_path: Path) -> None:
    first, spans = _record(tmp_path, ["q0", "q1"], ["yes", _Timeout("too slow")])
    provider = RecordedProvider(spans, SCRIPTED_IDENTITY)
    with pytest.raises(ProviderError, match="too slow") as caught:
        provider.invoke(_request("q1"))
    assert caught.value.code == "timeout" and caught.value.retryable is True
    replayed = run(_dataset(["q0", "q1"]), RecordedProvider(spans, SCRIPTED_IDENTITY), [includes()], settings=SETTINGS, budget=CallBudget())
    assert [sample.outcome for sample in replayed.samples] == ["succeeded", "failed"]
    assert replayed.samples[1].error == first.samples[1].error


def test_settings_other_than_the_model_do_not_change_the_match() -> None:
    provider = RecordedProvider([_span("q", "answer")], REPLAY_IDENTITY)
    warmer = CallSettings(model="scripted/model", temperature=0.9, max_output_tokens=12, seed=7)
    result = provider.invoke(_request("q", warmer))
    assert result.text == "answer"
    assert result.resolved_model == "scripted/model-2026" and result.cost_usd == 0.01 and result.latency_ms == 5
    assert result.started_at == NOW and result.completed_at == NOW
    with pytest.raises(ProviderError) as caught:
        provider.invoke(_request("q", CallSettings(model="other/model")))
    assert caught.value.code == "no_recorded_answer"


def test_a_replayed_run_with_another_budget_and_temperature_still_matches(tmp_path: Path) -> None:
    _first, spans = _record(tmp_path, ["q0", "q1"], ["yes", "no"])
    warmer = CallSettings(model="scripted/model", temperature=1.0)
    replayed = run(_dataset(["q0", "q1"]), RecordedProvider(spans, SCRIPTED_IDENTITY), [includes()], settings=warmer, budget=CallBudget(max_calls=2))
    assert [sample.output_text for sample in replayed.samples] == ["yes", "no"]
    assert replayed.record.settings == warmer


def test_a_set_cancel_raises_before_an_answer_is_consumed() -> None:
    provider = RecordedProvider([_span("q", "answer")], REPLAY_IDENTITY)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CallCancelled):
        provider.invoke(_request("q"), cancel=cancel)
    assert provider.invoke(_request("q"), cancel=threading.Event()).text == "answer"


def test_eight_threads_each_take_one_distinct_recorded_answer() -> None:
    provider = RecordedProvider([_span("q", f"answer {n}") for n in range(8)], REPLAY_IDENTITY)
    texts: list[str] = []
    lock = threading.Lock()

    def take() -> None:
        text = provider.invoke(_request("q")).text
        with lock:
            texts.append(text)

    threads = [threading.Thread(target=take) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(texts) == [f"answer {n}" for n in range(8)]
    with pytest.raises(ProviderError):
        provider.invoke(_request("q"))


def test_identity_capabilities_and_close() -> None:
    provider = RecordedProvider([], REPLAY_IDENTITY)
    assert provider.identity == REPLAY_IDENTITY
    assert provider.capabilities == EVERY_CAPABILITY
    assert {"text", "json", "schema", "system", "messages", "tools", "logprobs", "seed", "temperature", "top_p"} <= provider.capabilities
    provider.close()
    provider.close()


def test_recorded_imports_no_cli_db_or_numeric_library() -> None:
    code = "import sys, gigai_core.recorded; print(sorted({'click', 'numpy', 'sqlite3'} & set(sys.modules)))"
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_a_replayed_judge_reads_its_recorded_replies_and_calls_no_model(tmp_path: Path) -> None:
    """0.3.2 ``test_replay``: the judge's own calls are in the trace, so a replay answers the judge from it too."""
    verdicts = ['Looks right.\n{"verdict": "pass", "explanation": "yes"}', 'Not it.\n{"verdict": "fail", "explanation": "no"}']
    live = ScriptedProvider(["yes", verdicts[0], "no", verdicts[1]])
    budget = CallBudget()
    judge = llm_judge(budgeted(live, budget), rubric="Is it yes?", settings=SETTINGS, grader_id="rubric-judge")
    first = run(_dataset(["q0", "q1"]), live, [judge], settings=SETTINGS, budget=budget, trace_dir=tmp_path)
    assert first.record.trace_path is not None and first.record.calls_made == 4
    spans = list(read_spans(first.record.trace_path))

    recorded = RecordedProvider(spans, SCRIPTED_IDENTITY)
    replay_budget = CallBudget()
    replayed_judge = llm_judge(budgeted(recorded, replay_budget), rubric="Is it yes?", settings=SETTINGS, grader_id="rubric-judge")
    replayed = run(_dataset(["q0", "q1"]), recorded, [replayed_judge], settings=SETTINGS, budget=replay_budget)
    assert replayed.values("rubric-judge") == first.values("rubric-judge") == {"item-0": 1.0, "item-1": 0.0}
    assert replayed.record.calls_made == 4 and len(live.requests) == 4
    for mine, theirs in zip(replayed.samples, first.samples, strict=True):
        assert [span.text for span in mine.model_spans] == [span.text for span in theirs.model_spans]
    for mine_score, theirs_score in zip(replayed.scores, first.scores, strict=True):
        assert (mine_score.kind, mine_score.judge_prompt_sha256, mine_score.explanation) == ("model", theirs_score.judge_prompt_sha256, theirs_score.explanation)

    subject_only = [span for span in spans if not isinstance(span, ModelSpan) or span.last_turn.content.startswith("q")]
    assert len(subject_only) == len(spans) - 2
    without_judge = RecordedProvider(subject_only, SCRIPTED_IDENTITY)
    starved_judge = llm_judge(without_judge, rubric="Is it yes?", settings=SETTINGS, grader_id="rubric-judge")
    with pytest.raises(ProviderError) as caught:
        run(_dataset(["q0", "q1"]), without_judge, [starved_judge], settings=SETTINGS, budget=CallBudget())
    assert caught.value.code == "no_recorded_answer"
