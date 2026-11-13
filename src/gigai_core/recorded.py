"""A provider that answers from stored spans and never calls out."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import get_args

from .providers import (
    CallCancelled,
    Capability,
    ModelRequest,
    ModelResult,
    ProviderError,
    ProviderIdentity,
    request_sha256,
)
from .trace import ModelSpan, Span

#: What a replay can carry: every request field, since the answers are already recorded.
EVERY_CAPABILITY: frozenset[Capability] = frozenset(get_args(Capability))


class _RecordedFailure(ProviderError):
    """A stored failure replayed: ``code`` and ``retryable`` are the span's ``CallError`` values."""

    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class RecordedProvider:
    """Answers each request from the ``ModelSpan`` whose ``(model, request_sha256)`` matches.

    This is how a lesson replays a run with no key (N19). A plain class: it keeps a cursor per request digest, so a
    conversation asked twice gets the first recorded answer, then the second; asked more times than recorded, it
    raises ``ProviderError`` (code ``no_recorded_answer``). A span recorded with ``error`` replays as a
    ``ProviderError`` with the stored ``code`` and ``retryable``. Only the model and the conversation take part in
    the match (``providers.request_sha256``): another temperature or budget replays the same answers. Thread-safe.
    """

    def __init__(self, spans: Sequence[Span], identity: ProviderIdentity) -> None:
        """Index the ``ModelSpan``s of ``spans`` by ``(model, request_sha256)`` in recorded order; other spans are skipped."""
        self._identity = identity
        self._answers: dict[tuple[str, str], list[ModelSpan]] = {}
        for span in spans:
            if isinstance(span, ModelSpan):
                self._answers.setdefault((span.call.settings.model, span.request_sha256), []).append(span)
        self._cursors: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    @property
    def identity(self) -> ProviderIdentity:
        """The identity given at construction (what the replayed run records)."""
        return self._identity

    @property
    def capabilities(self) -> frozenset[Capability]:
        """Every capability: a recorded answer needs nothing from a transport."""
        return EVERY_CAPABILITY

    def invoke(self, request: ModelRequest, *, cancel: threading.Event | None = None) -> ModelResult:
        """The next recorded answer for ``request``'s model and conversation.

        Raises:
            CallCancelled: ``cancel`` is set (no answer is consumed).
            ProviderError: no recorded answer is left for this request (code ``no_recorded_answer``), or the recorded
                call failed (the stored code).
        """
        if cancel is not None and cancel.is_set():
            raise CallCancelled("cancelled before the recorded answer")
        model = request.settings.model
        key = (model, request_sha256(request))
        with self._lock:
            answers = self._answers.get(key, [])
            position = self._cursors.get(key, 0)
            if position >= len(answers):
                missing = ProviderError(
                    f"no recorded answer for model {model!r} and request {key[1]} (asked {position + 1} time(s), "
                    f"recorded {len(answers)})"
                )
                missing.code = "no_recorded_answer"
                raise missing
            self._cursors[key] = position + 1
        span = answers[position]
        if span.error is not None:
            raise _RecordedFailure(span.error.code, span.error.message, retryable=span.error.retryable)
        call = span.call
        return ModelResult(
            text=span.text,
            finish_reason=span.finish_reason,
            resolved_model=call.resolved_model,
            usage=call.usage,
            cost_status=call.cost_status,
            cost_usd=call.cost_usd,
            latency_ms=call.latency_ms,
            started_at=span.started_at,
            completed_at=span.completed_at if span.completed_at is not None else span.started_at,
            tool_calls=span.tool_calls,
            retry_count=call.retry_count,
            last_error_class=call.last_error_class,
            unsupported=call.unsupported,
        )

    def close(self) -> None:
        """Nothing to release; a second call does nothing."""
