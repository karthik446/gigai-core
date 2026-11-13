"""DESIGN.md section 4.9: the lesson snippets 1, 2, 4, 5 and 6, run keyless with ``tests.fakes.ScriptedProvider``.

The snippets import from ``gigai_core``; that façade is merged in A8, so until then the names come from their
defining modules (``gigai_core.run``, ``gigai_core.budget`` and so on). Nothing else differs from the snippet text,
except where a comment says what did.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from gigai_core.agreement import judge_agreement
from gigai_core.budget import CallBudget, Price
from gigai_core.comparison import compare
from gigai_core.dataset import Dataset, DatasetItem
from gigai_core.gate import MinLowerBound, NoWorseThan, gate
from gigai_core.graders.judge import llm_judge
from gigai_core.graders.labels import read_labels, write_labels
from gigai_core.graders.lifted import includes
from gigai_core.providers import CallSettings
from gigai_core.records import LabelRecord
from gigai_core.run import Run, RunFileError, open_run, run, save_run
from tests.fakes import ScriptedProvider

NOW = datetime(2026, 1, 1, tzinfo=UTC)
PRICE = Price(model="some/model", prompt_usd_per_token=1e-6, completion_usd_per_token=1e-6, fetched_at=NOW, source="lesson")
PASS = 'Reasoning.\n{"verdict": "pass", "explanation": "meets the rubric"}'
FAIL = 'Reasoning.\n{"verdict": "fail", "explanation": "misses the rubric"}'


def _capitals() -> Dataset:
    return Dataset.of("capitals", (DatasetItem(id="fr", cluster_id="fr", prompt="Capital of France?", expected="Paris"),))


def test_snippet_1_run_returns_a_typed_run_record(capsys: pytest.CaptureFixture[str]) -> None:
    provider = ScriptedProvider(["Paris, of course."])
    dataset = Dataset.of("capitals", (DatasetItem(id="fr", cluster_id="fr", prompt="Capital of France?", expected="Paris"),))
    # ``CallBudget(max_usd=0.05, max_calls=20)`` as written refuses an unpriced model under a money cap (nothing is
    # guessed); the lesson's budget needs a price for "some/model", so one is supplied here.
    budget = CallBudget(max_usd=0.05, max_calls=20, prices={"some/model": PRICE})
    result = run(dataset, provider, [includes()], settings=CallSettings(model="some/model"), budget=budget)
    print(result.record.spent_usd, result.values("includes"))
    assert capsys.readouterr().out == "0.0 {'fr': 1.0}\n"
    assert result.record.calls_made == 1
    assert isinstance(result, Run)


def test_snippet_2_the_local_trace_is_one_jsonl_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    provider = ScriptedProvider(["Paris"])
    settings = CallSettings(model="some/model")
    budget = CallBudget(max_calls=20)
    result = run(_capitals(), provider, [includes()], settings=settings, budget=budget, trace_dir=tmp_path / "traces")
    for span in result.trace("fr"):
        print(span.turn, span.kind, span.started_at)
    lines = capsys.readouterr().out.splitlines()
    assert [line.split(" ", 2)[:2] for line in lines] == [["0", "trial"], ["0", "model_call"]]
    trace_path = tmp_path / "traces" / f"{result.record.run_id}.jsonl"
    assert result.record.trace_path == trace_path
    assert len(trace_path.read_text(encoding="utf-8").splitlines()) == 2


def _twenty() -> Dataset:
    items = [DatasetItem(id=f"c{n}", cluster_id=f"c{n}", prompt=f"Capital of country {n}?", expected="Paris") for n in range(20)]
    return Dataset.of("twenty-capitals", items)


def _save(tmp_path: Path, name: str, replies: list[str]) -> Path:
    result = run(_twenty(), ScriptedProvider(replies), [includes()], settings=CallSettings(model="some/model"), budget=CallBudget())
    path = tmp_path / name
    save_run(result, path, inputs_by_reference=False)
    return path


def test_snippet_4_paired_comparison_with_a_gate_is_keyless(tmp_path: Path) -> None:
    gem = _save(tmp_path, "gem.jsonl", ["Paris"] * 20)
    minimal = _save(tmp_path, "min.jsonl", ["Paris"] * 18 + ["Lyon"] * 2)
    a, b = open_run(gem), open_run(minimal)
    comparison = compare(a, b, ["includes"])
    verdict = gate(candidate=b, baseline=a, rules=[MinLowerBound(metric="includes", floor=0.6), NoWorseThan(metric="includes", margin=0.05)])
    assert not verdict.passed and verdict.failed[0].rule == NoWorseThan(metric="includes", margin=0.05)
    (measured,) = comparison.metrics
    assert measured.metric == "includes" and measured.difference == pytest.approx(-0.1)
    assert verdict.verdicts[0].passed and verdict.status == "fail"


def test_snippet_5_judge_agreement_against_human_labels(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    items = [DatasetItem(id=f"c{n}", cluster_id=f"c{n}", prompt=f"Capital of country {n}?", expected="Paris") for n in range(4)]
    dataset = Dataset.of("four", items)
    provider = ScriptedProvider(["Paris", PASS, "Lyon", FAIL, "Paris", PASS, "Nice", FAIL])
    judge = llm_judge(provider, rubric="The answer names the capital.", settings=CallSettings(model="some/model"), grader_id="rubric-judge")
    judged_run = run(dataset, provider, [judge], settings=CallSettings(model="some/model"), budget=CallBudget())
    labels = [
        LabelRecord(label_id=f"label-{n}", dataset_id="four", item_id=f"c{n}", item_version="1", label=label, source="human", recorded_at=NOW)
        for n, label in enumerate(("met", "not met", "met", "not met"))
    ]
    write_labels(tmp_path / "labels.jsonl", labels)
    agreement = judge_agreement(judged_run, "rubric-judge", read_labels(tmp_path / "labels.jsonl"))
    print(agreement.kappa.kappa if agreement.kappa else "one category only")
    assert capsys.readouterr().out == "1.0\n"
    assert agreement.n == 4


def test_snippet_6_a_run_read_back_and_re_saved(tmp_path: Path) -> None:
    gem = _save(tmp_path, "gem.jsonl", ["Paris"] * 20)
    saved = open_run(gem)
    save_run(saved, tmp_path / "copy.jsonl")
    assert open_run(tmp_path / "copy.jsonl", dataset=_twenty()) == saved
    bad = tmp_path / "bad.jsonl"
    lines = gem.read_text(encoding="utf-8").splitlines()
    bad.write_text("\n".join([*lines, '{"kind": "nonsense"}']) + "\n", encoding="utf-8")
    with pytest.raises(RunFileError) as caught:
        open_run(bad)
    assert caught.value.line == len(lines) + 1 and str(bad) in str(caught.value)

