"""``run``: dataset x provider x graders -> a ``Run``; and the portable file a run is saved as."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, ValidationError

from .dataset import Dataset, DatasetItem
from .models import CoreModel
from .records import RunRecord
from .sample import Sample
from .score import ScoreRecord
from .trace import Span

RUN_SCHEMA = "gigai-core-run:1"


class RunFileError(Exception):
    """A run file this version cannot open: the file, the line number, and pydantic's reason or the unknown ``schema_id``.

    The message is ``<path> line <n>: <reason>``; ``path`` and ``line`` are also attributes. ``line`` is 0 when the
    problem is the whole file (empty, saved by reference and opened without a dataset, a dataset that does not match).
    """

    def __init__(self, path: Path, line: int, reason: str) -> None:
        """Name the file, the 1-based line (0 for the whole file) and why it could not be read."""
        super().__init__(f"{path} line {line}: {reason}")
        self.path = path
        self.line = line


class Run(CoreModel):
    """One run as a value: what ``run`` returns, what ``open_run`` reads, what ``compare`` and ``gate`` take."""

    record: RunRecord
    items: Mapping[str, DatasetItem]
    samples: tuple[Sample, ...]
    scores: tuple[ScoreRecord, ...]

    def trace(self, item: str) -> Iterator[Span]:
        """The spans of every trial of ``item`` in recorded order."""
        for sample in self.samples:
            if sample.item_id == item:
                yield from sample.spans

    def _item_of(self) -> Mapping[str, str]:
        """Trace id to item id, over the samples."""
        return {sample.trace_id: sample.item_id for sample in self.samples}

    def _sampled(self) -> tuple[str, ...]:
        """Item ids in the order they were first asked."""
        return tuple(dict.fromkeys(sample.item_id for sample in self.samples))

    def _by_item(self, grader_id: str) -> Mapping[str, tuple[float, ...]]:
        """One grader's values per item, for the items where every trial got a score."""
        item_of = self._item_of()
        scored = {
            score.trace_id: score.value
            for score in self.scores
            if score.grader_id == grader_id and score.trace_id in item_of
        }
        found: dict[str, list[float]] = {}
        incomplete: set[str] = set()
        for sample in self.samples:
            if sample.trace_id in scored:
                found.setdefault(sample.item_id, []).append(scored[sample.trace_id])
            else:
                incomplete.add(sample.item_id)
        return {item: tuple(values) for item, values in found.items() if item not in incomplete}

    def values(self, grader_id: str) -> Mapping[str, float]:
        """Per-item mean of one grader's scores over the k repeats; an ungraded item is absent.

        An item is graded when every one of its trials has a score from ``grader_id``; the mean is over those trials.
        """
        return {item: sum(found) / len(found) for item, found in self._by_item(grader_id).items()}

    def ungraded(self, grader_id: str) -> tuple[str, ...]:
        """Item ids this grader raised ``CannotGrade`` for (no score): the count a report must show, not hide.

        An item asked ``k`` times is ungraded when any of its trials has no score from ``grader_id``; ``values`` and
        ``ungraded`` together cover every item that was asked.
        """
        graded = self._by_item(grader_id)
        return tuple(item for item in self._sampled() if item not in graded)

    def contested(self) -> tuple[str, ...]:
        """Item ids where two or more graders disagree about ``met`` (a score of 1.0 against less).

        Each grader's verdict on an item is its per-item mean (as ``values``); graders with no verdict on the item are
        not counted.
        """
        grader_ids = tuple(dict.fromkeys(score.grader_id for score in self.scores))
        verdicts = [self.values(grader_id) for grader_id in grader_ids]
        found: list[str] = []
        for item in self._sampled():
            means = [verdict[item] for verdict in verdicts if item in verdict]
            if any(mean == 1.0 for mean in means) and any(mean < 1.0 for mean in means):
                found.append(item)
        return tuple(found)


class RunHeaderLine(CoreModel):
    """First line of a run file: the format version and the run record."""

    kind: Literal["run"] = "run"
    schema_id: Literal["gigai-core-run:1"] = "gigai-core-run:1"
    record: RunRecord


class ItemLine(CoreModel):
    """One dataset item (omitted when the file is saved by reference)."""

    kind: Literal["item"] = "item"
    item: DatasetItem


class SampleLine(CoreModel):
    """One trial, with its spans."""

    kind: Literal["sample"] = "sample"
    sample: Sample


class ScoreLine(CoreModel):
    """One grader's verdict on one trial."""

    kind: Literal["score"] = "score"
    score: ScoreRecord


RunLine = Annotated[RunHeaderLine | ItemLine | SampleLine | ScoreLine, Field(discriminator="kind")]
RUN_LINE: TypeAdapter[RunLine] = TypeAdapter(RunLine)


def _lines(run: Run, *, inputs_by_reference: bool) -> Iterator[RunLine]:
    """The lines of the run file in order: header, items, samples, scores."""
    yield RunHeaderLine(record=run.record)
    if not inputs_by_reference:
        for item in run.items.values():
            yield ItemLine(item=item)
    for sample in run.samples:
        yield SampleLine(sample=sample)
    for score in run.scores:
        yield ScoreLine(score=score)


def save_run(run: Run, path: Path, *, inputs_by_reference: bool = True) -> None:
    """Write the portable file, the one save path.

    A ``RunHeaderLine``, then item, sample and score lines, each ``RUN_LINE.dump_json``. Refuses to overwrite an
    existing file (``FileExistsError``). The lines go to a temporary file in the same directory that is renamed into
    place at the end, so a crash never leaves a half-written run file.
    """
    if path.exists():
        raise FileExistsError(f"{path} exists; save_run never overwrites a run file")
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False)
    try:
        with handle:
            for line in _lines(run, inputs_by_reference=inputs_by_reference):
                handle.write(RUN_LINE.dump_json(line))
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"{path} exists; save_run never overwrites a run file")
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _reason(exc: ValidationError) -> str:
    """Pydantic's reason for one bad line; an unknown ``schema_id`` is named as such."""
    for error in exc.errors():
        if error["type"] == "literal_error" and error["loc"] and error["loc"][-1] == "schema_id":
            return f"unknown schema_id {error['input']!r} (this version reads {RUN_SCHEMA!r})"
    return str(exc)


def _read_lines(path: Path) -> Iterator[tuple[int, RunLine]]:
    """Every line of the file through ``RUN_LINE``, with its 1-based number; raises ``RunFileError`` at a bad line."""
    with path.open("rb") as file:
        for number, raw in enumerate(file, start=1):
            try:
                yield number, RUN_LINE.validate_json(raw.rstrip(b"\r\n"))
            except ValidationError as exc:
                raise RunFileError(path, number, _reason(exc)) from exc


def open_run(path: Path, *, dataset: Dataset | None = None) -> Run:
    """Read a saved run with no key and no store: each line through ``RUN_LINE.validate_json``.

    ``dataset`` supplies items a by-reference file does not carry; its digest must be the one the run recorded.
    Raises ``RunFileError``: a bad line (with its 1-based number), an unknown ``schema_id``, a file that does not start
    with its header, a by-reference file opened without a dataset, or a sample whose item is not there.
    """
    record: RunRecord | None = None
    file_items: dict[str, DatasetItem] = {}
    samples: list[tuple[int, Sample]] = []
    scores: list[ScoreRecord] = []
    for number, line in _read_lines(path):
        if record is None:
            if not isinstance(line, RunHeaderLine):
                raise RunFileError(path, number, f"expected the run header first, found a {line.kind!r} line")
            record = line.record
        elif isinstance(line, RunHeaderLine):
            raise RunFileError(path, number, "a second run header; a run file holds one run")
        elif isinstance(line, ItemLine):
            file_items[line.item.id] = line.item
        elif isinstance(line, SampleLine):
            samples.append((number, line.sample))
        else:
            scores.append(line.score)
    if record is None:
        raise RunFileError(path, 0, "empty file; a run file starts with its run header")
    if file_items:
        items: Mapping[str, DatasetItem] = file_items
    elif dataset is None:
        raise RunFileError(path, 0, "saved by reference (no item lines); pass dataset= to supply the items")
    elif dataset.ref.sha256 != record.dataset.sha256:
        raise RunFileError(
            path,
            0,
            f"dataset {dataset.ref.id!r} has sha256 {dataset.ref.sha256}, the run measured {record.dataset.sha256}",
        )
    else:
        items = {item.id: item for item in dataset.items}
    for number, sample in samples:
        if sample.item_id not in items:
            raise RunFileError(path, number, f"sample {sample.trace_id!r} names item {sample.item_id!r}, which is not there")
    return Run(record=record, items=items, samples=tuple(sample for _, sample in samples), scores=tuple(scores))
