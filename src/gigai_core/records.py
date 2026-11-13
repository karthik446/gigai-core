"""The run record and the label and judge-validation rows.

No gig, manifest, step or suite fields: gigai keeps those beside its own ledger (design section 6.2).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime

from .budget import Price
from .dataset import DatasetRef
from .models import CoreModel
from .providers import CallSettings, ProviderIdentity
from .values import JsonValue

LabelValue = Literal["met", "not met", "unknown"]


class Prereg(CoreModel):
    """What a run promised before it ran; ``hypothesis`` is the sentence the pass rule tests."""

    primary_metric: str
    pass_rule: str
    margin: float
    n_planned: int
    registered_at: AwareDatetime
    guard_metric: str | None = None
    hypothesis: str | None = None


class RunRecord(CoreModel):
    """One run: ids, what it measured, how it was asked, what it cost.

    ``trace_path`` is where the trace was written on the machine that ran it; ``open_run`` never reads it (the spans
    travel inside the samples).
    """

    run_id: str
    dataset: DatasetRef
    settings: CallSettings
    provider: ProviderIdentity
    created_at: AwareDatetime
    completed_at: AwareDatetime | None = None
    prereg: Prereg | None = None
    trace_path: Path | None = None
    spent_usd: float = 0.0
    calls_made: int = 0
    prices: tuple[Price, ...] = ()
    repeats: int = 1
    tag: str | None = None
    replay_of: str | None = None


class LabelRecord(CoreModel):
    """One human (or dataset) label of one sample."""

    label_id: str
    dataset_id: str
    item_id: str
    item_version: str
    label: LabelValue
    source: str
    recorded_at: AwareDatetime
    run_id: str | None = None
    critique: str | None = None
    rubric: str | None = None
    who: str | None = None
    round: int | None = None
    blind: bool | None = None
    relabel_of: str | None = None


class JudgeValidationRecord(CoreModel):
    """One metric of one judge validation: what makes an unvalidated judge refuse to gate."""

    validation_id: str
    grader_id: str
    grader_version: str
    judge_model: str
    judge_prompt_sha256: str
    dataset_id: str
    dataset_sha256: str
    n_items: int
    metric: str
    value: float
    details: Mapping[str, JsonValue] | None
    recorded_at: AwareDatetime
