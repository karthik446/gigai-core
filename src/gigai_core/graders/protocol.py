"""The grader contract. Kept in its own module so every grader module can import it without a cycle."""

from __future__ import annotations

from typing import Protocol

from ..sample import Sample
from ..score import GraderKind, Score
from ..values import JsonValue


class CannotGrade(Exception):
    """This grader has no verdict for this sample. ``run`` records the sample with no score and keeps going."""


class Grader(Protocol):
    """Turns one sample into one score.

    ``id`` and ``version`` are stored on every ``ScoreRecord``; bump ``version`` when the grading rule changes, so an
    old run is never compared on a new rule without notice.
    """

    @property
    def id(self) -> str:
        """The stored ``grader_id``."""
        ...

    @property
    def version(self) -> str:
        """The stored ``grader_version``: the version of the grading rule."""
        ...

    @property
    def kind(self) -> GraderKind:
        """``code``, ``model`` or ``human``."""
        ...

    def grade(self, sample: Sample, expected: JsonValue) -> Score:
        """The verdict; raises ``CannotGrade`` for no verdict."""
        ...


class Judge(Grader, Protocol):
    """A model-backed grader.

    ``grade`` renders a prompt, asks the provider the judge was built with, and reads the reply. ``Score.judge`` names
    the model and the prompt digest and carries every call made; ``run`` writes those into the trace under the
    sample's trial span and links them from ``ScoreRecord.judge_span_ids``. A judge's calls are not in ``run``'s
    up-front plan (its prompt needs the subject's reply): build the judge on ``budgeted(provider, budget)`` to count
    them. It refuses to gate until its agreement with human labels is validated (section 4.7). ``render`` and ``read``
    are public so a lesson can show the prompt and the parse without a model.
    """

    @property
    def prompt_sha256(self) -> str:
        """The digest of the judge's prompt template, stored as ``judge_prompt_sha256``."""
        ...

    def render(self, sample: Sample, expected: JsonValue, *, swapped: bool = False) -> str:
        """The prompt the judge would send for ``sample``; ``swapped`` flips the position of the two answers."""
        ...

    def read(self, reply: str) -> Score:
        """Parse the judge's reply; raises ``CannotGrade`` when it is not a verdict."""
        ...
