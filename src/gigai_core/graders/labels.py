"""Human labels as a grader, plus the labels JSONL file helpers the lessons use.

A labels file is one ``LabelRecord`` per line. ``read_labels`` validates each line through
``LabelRecord.model_validate_json`` and ports the format rules of 0.3.2's ``validate_batch`` that do not need the
store: every extra key is named (cut short, so a long key cannot carry text), a missing key is named, blank lines
are skipped, an empty file is refused, ``round`` runs from 1 to ``ROUND_MAX``, and a repeated ``label_id`` is read
once when the line is an exact repeat and refused, naming both lines, when it differs. An error never echoes a
value. The interactive labelling page, the step check and the stored-label comparisons are the application's, not
core's.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from ..records import LabelRecord
from ..sample import Sample
from ..score import GraderKind, Score
from ..values import JsonValue
from .protocol import CannotGrade, Grader

#: The answers a label can hold; ``met`` grades 1.0, ``not met`` 0.0, ``unknown`` cannot grade.
LABELS = ("met", "not met", "unknown")
#: The highest ``round`` a label may carry: a relabel is round 2, so this only bounds the number (as 0.3.2 did).
ROUND_MAX = 1000
_KEY_PREVIEW = 40


class LabelFileError(Exception):
    """A labels file this version cannot read: the file, the line number and the reason.

    The message is ``<path> line <n>: <reason>``; ``path``, ``line`` and ``reason`` are also attributes. ``line`` is
    0 when the whole file is the problem. The reason names keys and fields, never the value a line carries.
    """

    def __init__(self, path: Path, line: int, reason: str) -> None:
        """Name the file, the 1-based line (0 for the whole file) and why it could not be read."""
        super().__init__(f"{path} line {line}: {reason}")
        self.path = path
        self.line = line
        self.reason = reason


class _Labelled:
    """The newest label of each item as a ``kind="human"`` grader."""

    kind: GraderKind = "human"
    version = "1"

    def __init__(self, labels: Sequence[LabelRecord], grader_id: str) -> None:
        self._id = grader_id
        newest: dict[str, LabelRecord] = {}
        for label in labels:
            held = newest.get(label.item_id)
            if held is None or label.recorded_at >= held.recorded_at:
                newest[label.item_id] = label
        self._newest = newest

    @property
    def id(self) -> str:
        return self._id

    def grade(self, sample: Sample, expected: JsonValue) -> Score:
        label = self._newest.get(sample.item_id)
        if label is None:
            raise CannotGrade(f"no label for item {sample.item_id!r}")
        if label.label == "unknown":
            raise CannotGrade(f"the label for item {sample.item_id!r} is unknown")
        return Score(value=1.0 if label.label == "met" else 0.0, explanation=label.critique, answer=label.label)


def labelled(labels: Sequence[LabelRecord], *, grader_id: str = "human") -> Grader:
    """A ``kind="human"`` grader: 1.0 for ``met``, 0.0 for ``not met`` on the sample's item.

    The item's verdict is its newest label (by ``recorded_at``; a later label in ``labels`` wins a tie), the rule
    ``judge_agreement`` applies. Labels are matched on ``item_id`` alone: ``dataset_id``, ``item_version`` and
    ``run_id`` are not compared, so pass one dataset's labels. ``unknown`` or no label raises ``CannotGrade``. The
    score's ``answer`` is the label and its ``explanation`` the critique.

    Args:
        labels: The labels, in any order.
        grader_id: The stored ``grader_id``.

    Returns:
        The grader, version ``"1"``.
    """
    return _Labelled(labels, grader_id)


def _name(key: str) -> str:
    """A key as an error may name it: quoted, and cut short so a long key cannot carry text."""
    return repr(key if len(key) <= _KEY_PREVIEW else key[:_KEY_PREVIEW] + "...")


def _reason(exc: ValidationError) -> str:
    """Why one line is not a ``LabelRecord``: the extra keys, else the missing ones, else the first field's error."""
    errors = exc.errors(include_url=False, include_input=False)
    extra = sorted(str(error["loc"][-1]) for error in errors if error["type"] == "extra_forbidden" and error["loc"])
    if extra:
        names = ", ".join(_name(key) for key in extra)
        return f"key {names} is not in the labels format (keys: {', '.join(LabelRecord.model_fields)})"
    missing = [str(error["loc"][-1]) for error in errors if error["type"] == "missing" and error["loc"]]
    if missing:
        return f"missing key {', '.join(missing)}"
    first = errors[0]
    field = ".".join(_name(str(part)) for part in first["loc"]) if first["loc"] else "the line"
    return f"{field}: {first['msg']}"


def read_labels(path: Path) -> tuple[LabelRecord, ...]:
    """Read a labels JSONL: one ``LabelRecord.model_validate_json`` per non-blank line.

    A line that repeats an earlier one exactly is read once; a line that reuses an earlier ``label_id`` with any
    other content is refused, naming both lines (as 0.3.2 ``validate_batch`` did).

    Args:
        path: The file.

    Returns:
        The records in file order, exact repeats dropped.

    Raises:
        LabelFileError: A line is not a ``LabelRecord`` (extra or missing keys named, the field at fault named,
            values never echoed), ``round`` is outside 1 to ``ROUND_MAX``, a ``label_id`` comes back with a
            different label, or the file holds no record.
    """
    labels: list[LabelRecord] = []
    seen: dict[str, tuple[int, LabelRecord]] = {}
    with path.open("rb") as file:
        for number, raw in enumerate(file, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                label = LabelRecord.model_validate_json(line)
            except ValidationError as exc:
                raise LabelFileError(path, number, _reason(exc)) from exc
            if label.round is not None and not 1 <= label.round <= ROUND_MAX:
                raise LabelFileError(path, number, f"round must be an integer from 1 to {ROUND_MAX}")
            earlier = seen.get(label.label_id)
            if earlier is not None:
                if earlier[1] != label:
                    raise LabelFileError(path, number, f"has the label_id of line {earlier[0]} with a different label")
                continue
            seen[label.label_id] = (number, label)
            labels.append(label)
    if not labels:
        raise LabelFileError(path, 0, "no labels in the file")
    return tuple(labels)


def write_labels(path: Path, labels: Sequence[LabelRecord]) -> None:
    """Write labels as JSONL, one ``model_dump_json`` per line; a new file only.

    The lines go to a temporary file in the same directory that is renamed into place at the end, so a crash never
    leaves a half-written labels file.

    Args:
        path: The file to create.
        labels: The records, written in order.

    Raises:
        FileExistsError: ``path`` exists; ``write_labels`` never overwrites a labels file.
    """
    if path.exists():
        raise FileExistsError(f"{path} exists; write_labels never overwrites a labels file")
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False)
    try:
        with handle:
            for label in labels:
                handle.write(label.model_dump_json().encode("utf-8"))
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"{path} exists; write_labels never overwrites a labels file")
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
