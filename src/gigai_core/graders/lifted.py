"""Graders lifted from inspect_ai: the typed boundary over ``_vendor/inspect_ai/scorers``.

``includes``, ``match`` and ``pattern`` wrap the vendored bodies verbatim; the bodies are ``async`` upstream and never
await anything, so they are driven to completion here without an event loop. The judge-template graders
(``model_graded_qa`` and friends) arrive with the judge packet.
"""

from __future__ import annotations

from collections.abc import Awaitable, Sequence
from typing import Literal

from .._vendor.inspect_ai._metric import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    PARTIAL,
    is_finite_number,
)
from .._vendor.inspect_ai._metric import Score as VendorScore
from .._vendor.inspect_ai.scorers import includes as _includes
from .._vendor.inspect_ai.scorers import match as _match
from .._vendor.inspect_ai.scorers import pattern as _pattern
from .._vendor.inspect_ai.scorers._shim import ModelOutput, Scorer, TaskState
from .._vendor.inspect_ai.scorers._target import Target
from ..sample import Sample
from ..score import GraderKind, Score
from ..values import JsonValue
from .protocol import Grader

LIFTED_VERSION = "inspect_ai@9567b431"

MatchLocation = Literal["end", "begin", "any", "exact"]

_LETTERS: dict[str, float] = {CORRECT: 1.0, PARTIAL: 0.5, INCORRECT: 0.0, NOANSWER: 0.0}


class _Lifted:
    """A vendored scorer body as a core ``Grader``."""

    kind: GraderKind = "code"
    version = LIFTED_VERSION

    def __init__(self, id: str, body: Scorer) -> None:
        self._id = id
        self._body = body

    @property
    def id(self) -> str:
        return self._id

    def grade(self, sample: Sample, expected: JsonValue) -> Score:
        return _run(self._body, sample.output_text, _targets(expected))


def _targets(expected: JsonValue) -> list[str]:
    """The reference strings a vendored body compares against: every item of a sequence, else the value itself."""
    if isinstance(expected, Sequence) and not isinstance(expected, str | bytes):
        return [str(item) for item in expected]
    return [str(expected)]


def _complete(awaitable: Awaitable[VendorScore]) -> VendorScore:
    """Drive a body that never awaits to its result; one that does is refused rather than silently run."""
    steps = awaitable.__await__()
    try:
        next(steps)
    except StopIteration as done:
        if isinstance(done.value, VendorScore):
            return done.value
        raise RuntimeError(f"lifted body returned {type(done.value).__name__}, not a Score") from None
    steps.close()
    raise RuntimeError("lifted body awaited something; inspect_ai text scorers must not need an event loop")


def _run(body: Scorer, text: str, targets: list[str]) -> Score:
    verdict = _complete(body(TaskState(output=ModelOutput(completion=text)), Target(targets)))
    if isinstance(verdict.value, str) and verdict.value in _LETTERS:
        value = _LETTERS[verdict.value]
    elif isinstance(verdict.value, int | float) and not isinstance(verdict.value, bool):
        value = float(verdict.value)
    else:
        raise RuntimeError(f"lifted body returned an unmapped value {verdict.value!r}")
    return Score(value=value, explanation=verdict.reason, answer=verdict.answer)


def includes(*, ignore_case: bool = True) -> Grader:
    """1.0 when the reply contains one of the target strings (inspect_ai ``includes``)."""
    return _Lifted("includes", _includes(ignore_case=ignore_case))


def match(*, location: MatchLocation = "end", ignore_case: bool = True, numeric: bool | None = None) -> Grader:
    """1.0 when the reply matches a target at ``location`` (inspect_ai ``match``).

    ``numeric=None`` compares as numbers when every target is a finite number, the way 0.3.2 did; ``True``/``False``
    forces it.
    """

    def body(state: TaskState, target: Target) -> Awaitable[VendorScore]:
        compare_numbers = all(is_finite_number(item) for item in target.target) if numeric is None else numeric
        return _match(location=location, ignore_case=ignore_case, numeric=compare_numbers)(state, target)

    return _Lifted("match", body)


def pattern(regex: str, *, ignore_case: bool = True, match_all: bool = False) -> Grader:
    """1.0 when the regex's capture group(s) in the reply equal a target (inspect_ai ``pattern``)."""
    return _Lifted("pattern", _pattern(regex, ignore_case=ignore_case, match_all=match_all))
