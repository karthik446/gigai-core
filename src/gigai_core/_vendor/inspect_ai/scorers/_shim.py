# gigai-written shim for inspect_ai src/inspect_ai/solver/_task_state.py and src/inspect_ai/scorer/_scorer.py @ 9567b431 (MIT, Copyright (c) 2024 UK AI Security Institute; see ../LICENSE)
"""What the lifted ``match``/``includes``/``pattern`` bodies import from inspect_ai's solver and scorer registry.

A scorer body reads only ``state.output.completion``, so ``TaskState`` carries the model output and nothing else.
``scorer(metrics=...)`` registers nothing: it returns the factory unchanged (core computes metrics from the stored
score rows, not from the decorator).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from .._metric import MetricProtocol, Score
from ._target import Target


@dataclass(frozen=True)
class ModelOutput:
    """The model's answer text (inspect_ai ``ModelOutput.completion``)."""

    completion: str


@dataclass(frozen=True)
class TaskState:
    """The one field of inspect_ai's ``TaskState`` the lifted scorers read: ``output``."""

    output: ModelOutput


Scorer = Callable[[TaskState, Target], Awaitable[Score]]

ScorerFactory = TypeVar("ScorerFactory", bound=Callable[..., Scorer])
"""A function that builds a ``Scorer`` (what ``@scorer`` decorates)."""


def scorer(*, metrics: Sequence[MetricProtocol] | None = None) -> Callable[[ScorerFactory], ScorerFactory]:
    """inspect_ai's ``@scorer``: here a passthrough; ``metrics`` is ignored."""
    del metrics

    def unchanged(factory: ScorerFactory) -> ScorerFactory:
        return factory

    return unchanged
