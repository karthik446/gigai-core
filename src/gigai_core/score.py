"""A score: one grader's verdict on one sample."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime

from .models import CoreModel
from .trace import ModelExchange

GraderKind = Literal["code", "model", "human"]


class JudgeNote(CoreModel):
    """How a model judge reached a ``Score``: the model, the prompt digest, the order, and every call it made."""

    model: str
    prompt_sha256: str
    position_swapped: bool | None = None
    calls: tuple[ModelExchange, ...] = ()


class Score(CoreModel):
    """A number (0 to 1 unless the metric says otherwise) with the grader's explanation. Always finite."""

    value: float
    explanation: str | None = None
    answer: str | None = None
    judge: JudgeNote | None = None


class ScoreRecord(CoreModel):
    """The stored row: a ``Score`` plus who graded it and, for a judge, how."""

    score_id: str
    trace_id: str
    grader_id: str
    grader_version: str
    kind: GraderKind
    value: float
    explanation: str | None = None
    judge_model: str | None = None
    judge_prompt_sha256: str | None = None
    position_swapped: bool | None = None
    judge_span_ids: tuple[str, ...] = ()
    recorded_at: AwareDatetime
