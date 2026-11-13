"""`RunRecord`, `Prereg`, `LabelRecord`, `JudgeValidationRecord`: what the 0.3.2 ledger rows asserted, as values.

Relocated from gigai 0.3.2 `tests/behaviors/evals/test_records.py`: the run row keeps its prereg (`primary_metric`
survives the JSON round trip), a score row names its grader and version with a plain float, and a run record has no
gig, manifest, step or suite field. The sqlite layout, `write_*` helpers, export, migration and `step_id` tests stay in
gigai.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from gigai_core.budget import Price
from gigai_core.dataset import DatasetRef, SliceRecipe
from gigai_core.providers import CallSettings, ProviderIdentity
from gigai_core.records import JudgeValidationRecord, LabelRecord, Prereg, RunRecord
from gigai_core.score import ScoreRecord

NOW = datetime(2026, 11, 13, 10, 0, tzinfo=UTC)
DATASET = DatasetRef(id="synthetic", version="1", sha256="b" * 64, slice_recipe=SliceRecipe(recipe="all", count=2))
IDENTITY = ProviderIdentity(provider="deterministic", endpoint="offline", adapter="deterministic:2")
PREREG = Prereg(primary_metric="exact_match", pass_rule="min_lower_bound", margin=0.05, n_planned=2, registered_at=NOW)


def _run_record(run_id: str = "run_fixture") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        dataset=DATASET,
        settings=CallSettings(model="fixture/model", temperature=0.0, top_p=1.0, seed=7, output_mode="json"),
        provider=IDENTITY,
        created_at=NOW,
        prereg=PREREG,
    )


def test_a_run_record_keeps_its_prereg_through_the_json_round_trip() -> None:
    record = _run_record()
    again = RunRecord.model_validate_json(record.model_dump_json())
    assert again == record
    assert again.prereg is not None and again.prereg.primary_metric == "exact_match"
    assert (again.settings.temperature, again.settings.top_p, again.settings.seed, again.settings.output_mode) == (0.0, 1.0, 7, "json")
    assert (again.provider.provider, again.provider.endpoint, again.provider.adapter) == ("deterministic", "offline", "deterministic:2")
    assert (again.spent_usd, again.calls_made, again.repeats, again.completed_at, again.replay_of) == (0.0, 0, 1, None, None)


def test_a_run_record_has_no_gig_fields_and_the_same_run_id_with_other_facts_is_a_different_value() -> None:
    for retired in ("gig", "manifest_digest", "step_id", "suite", "k", "schema_version"):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            RunRecord.model_validate({**_run_record().model_dump(), retired: "x"})
    assert _run_record() == _run_record()
    assert _run_record() != _run_record().model_copy(update={"tag": "other"})


def test_a_run_record_round_trips_its_costs_path_and_timestamps() -> None:
    price = Price(model="fixture/model", prompt_usd_per_token=1e-6, completion_usd_per_token=2e-6, fetched_at=NOW, source="list")
    record = _run_record().model_copy(
        update={
            "completed_at": NOW,
            "trace_path": Path("/tmp/run_fixture.jsonl"),
            "spent_usd": 0.0123,
            "calls_made": 2,
            "prices": (price,),
            "repeats": 2,
            "tag": "baseline",
            "replay_of": "run_0",
        }
    )
    again = RunRecord.model_validate_json(record.model_dump_json())
    assert again == record and again.trace_path == Path("/tmp/run_fixture.jsonl")
    with pytest.raises(ValidationError, match="trace_path"):
        RunRecord.model_validate({**_run_record().model_dump(), "trace_path": "/tmp/x.jsonl"})
    with pytest.raises(ValidationError, match="created_at"):
        RunRecord.model_validate({**_run_record().model_dump(), "created_at": datetime(2026, 11, 13, 10, 0)})


def test_a_prereg_needs_its_promise_and_takes_an_optional_guard_and_hypothesis() -> None:
    full = PREREG.model_copy(update={"guard_metric": "refusals", "hypothesis": "v2 is no worse on exact_match"})
    assert Prereg.model_validate_json(full.model_dump_json()) == full
    with pytest.raises(ValidationError, match="n_planned"):
        Prereg.model_validate({**PREREG.model_dump(), "n_planned": "2"})
    with pytest.raises(ValidationError, match="registered_at"):
        Prereg.model_validate_json('{"primary_metric": "exact_match", "pass_rule": "r", "margin": 0.0, "n_planned": 1}')


def test_a_score_record_names_its_grader_as_0_3_2_named_its_scorer() -> None:
    # 0.3.2: ScoreRecord(new_id("score"), trace.trace_id, "exact_match", "1", 1.0) and a second one at 0.0.
    for value in (1.0, 0.0):
        record = ScoreRecord(score_id="score_1", trace_id="trace_1", grader_id="exact_match", grader_version="1", kind="code", value=value, recorded_at=NOW)
        assert (record.grader_id, record.grader_version, record.value) == ("exact_match", "1", value)
        assert ScoreRecord.model_validate_json(record.model_dump_json()) == record
    assert "scorer" not in ScoreRecord.model_fields and "scorer_id" not in ScoreRecord.model_fields


def test_a_label_record_takes_the_three_label_values_only() -> None:
    label = LabelRecord(label_id="label_1", dataset_id="synthetic", item_id="q1", item_version="1", label="met", source="operator", recorded_at=NOW)
    assert LabelRecord.model_validate_json(label.model_dump_json()) == label
    assert (label.run_id, label.critique, label.rubric, label.who, label.round, label.blind, label.relabel_of) == (None,) * 7
    full = label.model_copy(update={"run_id": "run_fixture", "who": "kar", "round": 2, "blind": True, "relabel_of": "label_0", "label": "not met"})
    assert LabelRecord.model_validate_json(full.model_dump_json()) == full
    with pytest.raises(ValidationError, match="label"):
        LabelRecord.model_validate({**label.model_dump(), "label": "yes"})


def test_a_judge_validation_record_keeps_its_details_or_none() -> None:
    record = JudgeValidationRecord(
        validation_id="validation_1",
        grader_id="pairwise",
        grader_version="3",
        judge_model="judge/model",
        judge_prompt_sha256="a" * 64,
        dataset_id="synthetic",
        dataset_sha256="b" * 64,
        n_items=40,
        metric="kappa",
        value=0.71,
        details={"agreement": 0.9, "swaps": [1, 2]},
        recorded_at=NOW,
    )
    assert JudgeValidationRecord.model_validate_json(record.model_dump_json()) == record
    bare = record.model_copy(update={"details": None})
    assert JudgeValidationRecord.model_validate_json(bare.model_dump_json()).details is None
    with pytest.raises(ValidationError, match="details"):
        JudgeValidationRecord.model_validate({**record.model_dump(exclude={"details"})})
