"""``run``: dataset x provider x graders -> a ``Run``; and the portable file a run is saved as."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import AwareDatetime, Field, TypeAdapter, ValidationError

from .budget import Budget, budgeted
from .dataset import Dataset, DatasetItem
from .graders.builtin import parse_json_text
from .graders.protocol import CannotGrade, Grader
from .models import CoreModel
from .providers import (
    CallSettings,
    ModelProvider,
    ModelRequest,
    ProviderError,
    ProviderIdentity,
    TokenUsage,
    Turn,
    request_sha256,
)
from .records import Prereg, RunRecord
from .sample import Sample
from .score import Score, ScoreRecord
from .trace import (
    CallError,
    JsonlTrace,
    ListTrace,
    ModelCall,
    ModelSpan,
    Span,
    TraceSink,
    TrialSpan,
)
from .values import JsonValue, ParseStatus

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


def _new_id(prefix: str) -> str:
    """A fresh id: ``<prefix>_<16 hex digits of a uuid4>``."""
    return f"{prefix}_{uuid4().hex[:16]}"


def _now() -> AwareDatetime:
    return datetime.now(UTC)


class _Trial(CoreModel):
    """One planned call: the item, which repeat, and the request to send."""

    item: DatasetItem
    repeat_index: int
    request: ModelRequest


def _settings_for(item: DatasetItem, settings: CallSettings) -> CallSettings:
    """``settings`` for ``item``: in ``schema`` mode, the item's ``expected["schema"]`` wins over ``settings.json_schema``."""
    if settings.output_mode != "schema":
        return settings
    if isinstance(item.expected, dict) and "schema" in item.expected:
        schema = item.expected["schema"]
        if not isinstance(schema, dict):
            raise ValueError(f"item {item.id!r}: expected['schema'] must be a JSON object, not {type(schema).__name__}")
        return settings.model_copy(update={"json_schema": schema})
    if settings.json_schema is None:
        raise ValueError(f"item {item.id!r} asks for schema output but has no expected['schema'] and settings.json_schema is None")
    return settings


def _plan(dataset: Dataset, settings: CallSettings, system: str | None, repeats: int) -> tuple[_Trial, ...]:
    """Every request of the run in call order: items in dataset order, repeats 0..k-1 inside each."""
    trials: list[_Trial] = []
    for item in dataset.items:
        item_settings = _settings_for(item, settings)
        text = item.system if item.system is not None else system
        turns: list[Turn] = [] if text is None else [Turn(role="system", content=text)]
        turns.append(Turn(role="user", content=item.prompt))
        for repeat_index in range(repeats):
            request = ModelRequest(
                settings=item_settings,
                turns=tuple(turns),
                metadata={"item_id": item.id, "repeat_index": repeat_index},
            )
            trials.append(_Trial(item=item, repeat_index=repeat_index, request=request))
    return tuple(trials)


def _validate(dataset: Dataset, graders: Sequence[Grader], repeats: int) -> None:
    """The refusals that happen before any call: ``repeats``, duplicate grader and item ids, conversation items."""
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, not {repeats}")
    seen: set[str] = set()
    for grader in graders:
        if grader.id in seen:
            raise ValueError(f"duplicate grader id {grader.id!r}")
        seen.add(grader.id)
    item_ids = [item.id for item in dataset.items]
    duplicates = sorted({item_id for item_id in item_ids if item_ids.count(item_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate item ids {duplicates}: one trial per item needs every id once")
    scripted = [item_id for item_id, item in zip(item_ids, dataset.items, strict=True) if item.turns]
    if scripted:
        raise ValueError(f"items {scripted} have scripted turns: scripted turns need core 0.2.0")


def _failed_span(
    request: ModelRequest, exc: ProviderError, identity: ProviderIdentity, *, span_id: str, parent_id: str,
    started_at: AwareDatetime, request_turns: tuple[Turn, ...] | None,
) -> ModelSpan:
    """The ``ModelSpan`` of a call that raised: empty text, ``error`` set, the request's model as resolved."""
    completed_at = _now()
    call = ModelCall(
        identity=identity,
        target=None,
        settings=request.settings,
        resolved_model=request.settings.model,
        usage=TokenUsage(input_tokens=None, output_tokens=None, total_tokens=None),
        cost_usd=None,
        cost_status="unavailable",
        latency_ms=max(0, int((completed_at - started_at).total_seconds() * 1000)),
    )
    return ModelSpan(
        span_id=span_id,
        parent_id=parent_id,
        turn=0,
        started_at=started_at,
        completed_at=completed_at,
        call=call,
        request_sha256=request_sha256(request),
        last_turn=request.turns[-1],
        text="",
        finish_reason="error",
        error=CallError(code=exc.code, message=str(exc), retryable=exc.retryable),
        request_turns=request_turns,
    )


def _parsed(text: str, settings: CallSettings) -> tuple[JsonValue, ParseStatus]:
    """``output_parsed`` and ``parse_status`` for a succeeded call: JSON parsed in ``json``/``schema`` mode only."""
    if settings.output_mode == "text":
        return None, "not_parsed"
    parsed = parse_json_text(text)
    return parsed.value, parsed.status


def _score_record(
    score: Score, grader: Grader, trace_id: str, judge_span_ids: tuple[str, ...], recorded_at: AwareDatetime
) -> ScoreRecord:
    """The stored row of one ``Score``; a judge's note fills the judge columns."""
    note = score.judge
    return ScoreRecord(
        score_id=_new_id("score"),
        trace_id=trace_id,
        grader_id=grader.id,
        grader_version=grader.version,
        kind=grader.kind,
        value=score.value,
        explanation=score.explanation,
        judge_model=note.model if note is not None else None,
        judge_prompt_sha256=note.prompt_sha256 if note is not None else None,
        position_swapped=note.position_swapped if note is not None else None,
        judge_span_ids=judge_span_ids,
        recorded_at=recorded_at,
    )


def _grade(
    sample: Sample, expected: JsonValue, graders: Sequence[Grader], *, trial_span_id: str, sink: TraceSink
) -> tuple[tuple[ScoreRecord, ...], tuple[ModelSpan, ...]]:
    """Every grader's verdict on one succeeded sample, with the judge calls placed in the trace under its trial span.

    ``CannotGrade`` leaves no record for that grader; any other exception propagates (a grader bug is not a data
    error).
    """
    scores: list[ScoreRecord] = []
    judge_spans: list[ModelSpan] = []
    for grader in graders:
        started_at = _now()
        try:
            score = grader.grade(sample, expected)
        except CannotGrade:
            continue
        span_ids: list[str] = []
        if score.judge is not None:
            for exchange in score.judge.calls:
                completed_at = started_at + timedelta(milliseconds=exchange.call.latency_ms)
                span = ModelSpan.place(
                    exchange,
                    span_id=_new_id("span"),
                    parent_id=trial_span_id,
                    turn=0,
                    started_at=started_at,
                    completed_at=completed_at,
                )
                sink.write(span)
                judge_spans.append(span)
                span_ids.append(span.span_id)
                started_at = completed_at
        scores.append(_score_record(score, grader, sample.trace_id, tuple(span_ids), _now()))
    return tuple(scores), tuple(judge_spans)


def _trial(
    trial: _Trial,
    subject: ModelProvider,
    identity: ProviderIdentity,
    graders: Sequence[Grader],
    *,
    run_id: str,
    store_inputs: bool,
    sink: TraceSink,
) -> tuple[Sample, tuple[ScoreRecord, ...]]:
    """One trial: the call, its parse, its grades, and its spans written in order (model, judge, trial)."""
    request = trial.request
    trace_id = _new_id("trace")
    trial_span_id = _new_id("span")
    started_at = _now()
    request_turns = request.turns if store_inputs else None
    error: CallError | None = None
    try:
        result = subject.invoke(request)
    except ProviderError as exc:
        model_span = _failed_span(
            request, exc, identity, span_id=_new_id("span"), parent_id=trial_span_id, started_at=started_at,
            request_turns=request_turns,
        )
        error = model_span.error
        output_text, parsed, status = "", None, "not_parsed"
    else:
        model_span = ModelSpan(
            span_id=_new_id("span"),
            parent_id=trial_span_id,
            turn=0,
            started_at=started_at,
            completed_at=_now(),
            call=ModelCall.of(request, result, identity, target=None),
            request_sha256=request_sha256(request),
            last_turn=request.turns[-1],
            text=result.text,
            finish_reason=result.finish_reason,
            tool_calls=result.tool_calls,
            request_turns=request_turns,
        )
        output_text = result.text
        parsed, status = _parsed(result.text, request.settings)
    sink.write(model_span)
    sample = Sample(
        trace_id=trace_id,
        run_id=run_id,
        item_id=trial.item.id,
        item_version=trial.item.version,
        cluster_id=trial.item.cluster_id,
        repeat_index=trial.repeat_index,
        input=request.turns,
        output_text=output_text,
        output_parsed=parsed,
        parse_status=status,
        outcome="failed" if error is not None else "succeeded",
        error=error,
        spans=(model_span,),
        recorded_at=_now(),
    )
    scores: tuple[ScoreRecord, ...] = ()
    judge_spans: tuple[ModelSpan, ...] = ()
    if error is None:
        scores, judge_spans = _grade(sample, trial.item.expected, graders, trial_span_id=trial_span_id, sink=sink)
    completed_at = _now()
    trial_span = TrialSpan(
        span_id=trial_span_id,
        turn=0,
        started_at=started_at,
        completed_at=completed_at,
        item_id=trial.item.id,
        repeat_index=trial.repeat_index,
        outcome=sample.outcome,
        error=error,
    )
    sink.write(trial_span)
    return sample.model_copy(update={"spans": (trial_span, model_span, *judge_spans), "recorded_at": completed_at}), scores


def run(
    dataset: Dataset,
    provider: ModelProvider,
    graders: Sequence[Grader],
    *,
    settings: CallSettings,
    budget: Budget,
    system: str | None = None,
    repeats: int = 1,
    trace_dir: Path | None = None,
    store_inputs: bool = False,
    prereg: Prereg | None = None,
    tag: str | None = None,
) -> Run:
    """Ask every item ``repeats`` times, grade every answer, and return the run.

    ``budget.plan`` runs over every planned request before the first call, so a run that cannot fit its cap never
    starts and leaves no file (N18). A failing subject call is recorded as a failed sample and the run continues
    (harness error, model error and timeout are different ``CallError.code`` values). A judge's own call is not a
    sample: its ``ProviderError`` (timeout, exhausted script) propagates out of ``run`` like any other grader
    exception, as does a ``BudgetExceeded`` from any reservation; in both cases no ``Run`` is returned, the trace file
    is closed and keeps the trials completed so far. A grader that raises ``CannotGrade`` leaves the sample with no
    score for that grader. Spans are written to ``<trace_dir>/<run_id>.jsonl`` as they happen (model spans, then
    judge spans, then the trial span); with ``trace_dir=None`` nothing is written. ``RunRecord.spent_usd`` is the sum
    of every span's ``cost_usd`` (unknown costs count 0) rounded to 6 decimals, the budget's own precision. The
    caller's provider is never closed. A ``RecordedProvider`` makes the same call keyless. Core 0.2.0 adds two
    trailing keyword-only parameters, ``environment: EnvironmentFactory | None = None`` and ``limits: TurnLimits =
    TurnLimits()``, for agent and conversation items; no 0.1.0 call changes.

    Args:
        dataset: The items to ask.
        provider: What answers; every call goes through ``budgeted(provider, budget)``.
        graders: The graders, each with a distinct ``id``.
        settings: The call settings; ``output_mode="schema"`` takes ``json_schema`` or each item's
            ``expected["schema"]``.
        budget: The caps: planned once up front, then reserved per call.
        system: A system turn for every item that has no ``system`` of its own.
        repeats: How many times each item is asked (at least 1).
        trace_dir: Where the trace file goes (created when missing); ``None`` for no file.
        store_inputs: Keep every request's turns in the trace, not only the digest and the last turn.
        prereg: What the run promised before it ran.
        tag: A free label stored on the record.

    Returns:
        The run: its record, its items, one sample per trial, and one score per graded (trial, grader) pair.

    Raises:
        ValueError: ``repeats`` under 1, a duplicate grader id, a duplicate item id, an item with scripted turns, or a
            schema-mode item without a schema; all before any call.
        BudgetExceeded: the plan does not fit, or a reservation during the run was refused.
        ProviderError: a judge's own call failed (a judge timeout is not a failed sample).
    """
    _validate(dataset, graders, repeats)
    trials = _plan(dataset, settings, system, repeats)
    budget.plan([trial.request for trial in trials])
    run_id = _new_id("run")
    created_at = _now()
    file: JsonlTrace | None = None
    sink: TraceSink
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        file = JsonlTrace(trace_dir / f"{run_id}.jsonl", store_inputs=store_inputs)
        sink = file
    else:
        sink = ListTrace()
    subject = budgeted(provider, budget)
    identity = provider.identity
    samples: list[Sample] = []
    scores: list[ScoreRecord] = []
    try:
        for trial in trials:
            sample, trial_scores = _trial(
                trial, subject, identity, graders, run_id=run_id, store_inputs=store_inputs, sink=sink
            )
            samples.append(sample)
            scores.extend(trial_scores)
    finally:
        if file is not None:
            file.close()
    calls = [span.call for sample in samples for span in sample.model_spans]
    record = RunRecord(
        run_id=run_id,
        dataset=dataset.ref,
        settings=settings,
        provider=identity,
        created_at=created_at,
        completed_at=_now(),
        prereg=prereg,
        trace_path=file.path if file is not None else None,
        spent_usd=round(sum(call.cost_usd or 0.0 for call in calls), 6),
        calls_made=len(calls),
        repeats=repeats,
        tag=tag,
    )
    return Run(record=record, items={item.id: item for item in dataset.items}, samples=tuple(samples), scores=tuple(scores))
