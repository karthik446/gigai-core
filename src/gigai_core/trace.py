"""The local trace: one JSONL file per run, one ``Span`` per line, appended as each call completes."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import IO, Annotated, Literal, Protocol

from pydantic import AwareDatetime, Field, TypeAdapter, ValidationError

from .models import CoreModel
from .providers import (
    CallSettings,
    CostStatus,
    FinishReason,
    ModelRequest,
    ModelResult,
    ProviderIdentity,
    TokenUsage,
    ToolCall,
    Turn,
)
from .values import JsonValue, Outcome

TRACE_SCHEMA = "gigai-core-trace:1"


class CallError(CoreModel):
    """Why a call failed, as stored: the provider's ``code``, its message, and whether a retry could help."""

    code: str
    message: str
    retryable: bool


class ModelCall(CoreModel):
    """The facts of one model call that a trace keeps."""

    identity: ProviderIdentity
    target: str | None
    settings: CallSettings
    resolved_model: str
    usage: TokenUsage
    cost_usd: float | None
    cost_status: CostStatus
    latency_ms: int
    retry_count: int = 0
    last_error_class: str | None = None
    unsupported: tuple[str, ...] = ()

    @classmethod
    def of(
        cls, request: ModelRequest, result: ModelResult, identity: ProviderIdentity, *, target: str | None = None
    ) -> ModelCall:
        """Build the stored facts from what was sent and what came back."""
        return cls(
            identity=identity,
            target=target,
            settings=request.settings,
            resolved_model=result.resolved_model,
            usage=result.usage,
            cost_usd=result.cost_usd,
            cost_status=result.cost_status,
            latency_ms=result.latency_ms,
            retry_count=result.retry_count,
            last_error_class=result.last_error_class,
            unsupported=result.unsupported,
        )


class SpanBase(CoreModel):
    """What every span line carries. ``schema_id`` is the file format's version: a reader accepts every older value."""

    schema_id: Literal["gigai-core-trace:1"] = "gigai-core-trace:1"
    span_id: str
    turn: int
    started_at: AwareDatetime
    completed_at: AwareDatetime | None


class TrialSpan(SpanBase):
    """The root of one trial: one item asked once (repeat ``repeat_index``)."""

    kind: Literal["trial"] = "trial"
    parent_id: None = None
    item_id: str
    repeat_index: int
    outcome: Outcome | None = None
    error: CallError | None = None


class TurnSpan(SpanBase):
    """One user turn of a conversation trial; its model and tool spans are its children."""

    kind: Literal["turn"] = "turn"
    parent_id: str


class ModelExchange(CoreModel):
    """What one model call sent and got back, with no place in a trace yet.

    A judge returns these in ``Score.judge``; ``run`` places each one in the trace as a ``ModelSpan``.
    """

    call: ModelCall
    request_sha256: str
    last_turn: Turn
    text: str
    finish_reason: FinishReason
    tool_calls: tuple[ToolCall, ...] = ()
    error: CallError | None = None


class ModelSpan(SpanBase, ModelExchange):
    """One model call in a trace.

    The conversation sent is stored by digest plus its last turn, unless ``store_inputs`` was set.
    """

    kind: Literal["model_call"] = "model_call"
    parent_id: str
    request_turns: tuple[Turn, ...] | None = None

    @classmethod
    def place(
        cls,
        exchange: ModelExchange,
        *,
        span_id: str,
        parent_id: str,
        turn: int,
        started_at: AwareDatetime,
        completed_at: AwareDatetime | None,
    ) -> ModelSpan:
        """Put ``exchange`` in a trace under ``parent_id`` (``run`` does this for every judge call)."""
        return cls(
            span_id=span_id,
            parent_id=parent_id,
            turn=turn,
            started_at=started_at,
            completed_at=completed_at,
            call=exchange.call,
            request_sha256=exchange.request_sha256,
            last_turn=exchange.last_turn,
            text=exchange.text,
            finish_reason=exchange.finish_reason,
            tool_calls=exchange.tool_calls,
            error=exchange.error,
        )


class ToolSpan(SpanBase):
    """One tool call the agent loop carried out. A tool exception is recorded in ``error``, not raised."""

    kind: Literal["tool_call"] = "tool_call"
    parent_id: str
    call: ToolCall
    result: JsonValue = None
    error: CallError | None = None
    latency_ms: int = 0


Span = Annotated[TrialSpan | TurnSpan | ModelSpan | ToolSpan, Field(discriminator="kind")]
SPAN: TypeAdapter[Span] = TypeAdapter(Span)


class TraceSink(Protocol):
    """Where spans go as the run executes."""

    def write(self, span: Span) -> None:
        """Record one finished span."""
        ...


class ListTrace:
    """A sink that keeps spans in memory: the test double and the book's first trace."""

    spans: list[Span]

    def __init__(self) -> None:
        """Start with no spans."""
        self.spans = []

    def write(self, span: Span) -> None:
        """Append ``span`` to ``spans``."""
        self.spans.append(span)


class JsonlTrace:
    """Appends ``SPAN.dump_json(span)`` and a newline to ``path`` and flushes, so a crashed trial still shows its spans.

    No credential is ever written (``ProviderIdentity`` has no secret field). ``store_inputs=False`` drops
    ``request_turns`` from a ``ModelSpan`` before it is written. One lock serialises writers, so lines from concurrent
    threads never interleave. The file is opened for append on construction and closed by ``close`` (or the ``with``
    block); writing after ``close`` is a ``ValueError``.
    """

    def __init__(self, path: Path, *, store_inputs: bool = False) -> None:
        """Open ``path`` for append (created when missing)."""
        self.path = path
        self.store_inputs = store_inputs
        self._lock = threading.Lock()
        self._file: IO[bytes] | None = path.open("ab")

    def write(self, span: Span) -> None:
        """Append one line for ``span`` and flush it."""
        if isinstance(span, ModelSpan) and not self.store_inputs and span.request_turns is not None:
            span = span.model_copy(update={"request_turns": None})
        line = SPAN.dump_json(span) + b"\n"
        with self._lock:
            if self._file is None:
                raise ValueError(f"{self.path}: trace is closed")
            self._file.write(line)
            self._file.flush()

    def close(self) -> None:
        """Close the file; a second call does nothing."""
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def __enter__(self) -> JsonlTrace:
        """Use the trace as a context manager: ``close`` runs on exit."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """Close the file when the block ends, whatever happened inside it."""
        self.close()


class TraceSchemaError(Exception):
    """A trace line this version cannot open: the line number, and either an unknown ``schema_id`` or pydantic's reason.

    The message is ``<path> line <n>: <reason>``; ``path`` and ``line`` are also attributes.
    """

    def __init__(self, path: Path, line: int, reason: str) -> None:
        """Name the file, the 1-based line and why it could not be read."""
        super().__init__(f"{path} line {line}: {reason}")
        self.path = path
        self.line = line


def _reason(exc: ValidationError) -> str:
    """Pydantic's reason for one bad line; an unknown ``schema_id`` is named as such."""
    for error in exc.errors():
        if error["loc"] and error["loc"][-1] == "schema_id":
            return f"unknown schema_id {error['input']!r} (this version reads {TRACE_SCHEMA!r})"
    return str(exc)


def read_spans(path: Path) -> Iterator[Span]:
    """Read a trace file in recorded order, one ``SPAN.validate_json(line)`` per line; raises ``TraceSchemaError``.

    Spans are yielded as each line is read, so a file whose last line was cut by a crash gives back every finished span
    first and then raises for the cut line, with its 1-based number.
    """
    with path.open("rb") as file:
        for number, raw in enumerate(file, start=1):
            line = raw.rstrip(b"\r\n")
            try:
                span = SPAN.validate_json(line)
            except ValidationError as exc:
                raise TraceSchemaError(path, number, _reason(exc)) from exc
            yield span
