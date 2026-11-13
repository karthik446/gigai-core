"""Test doubles every core test and book lesson can use. Import as ``from tests.fakes import ScriptedProvider``."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from gigai_core.providers import (
    CallCancelled,
    Capability,
    FinishReason,
    ModelRequest,
    ModelResult,
    ProviderError,
    ProviderIdentity,
    TokenUsage,
)

Reply = str | ModelResult | Exception

SCRIPTED_IDENTITY = ProviderIdentity(provider="scripted", endpoint="memory", adapter="tests.fakes.ScriptedProvider:1")
SCRIPTED_CAPABILITIES: frozenset[Capability] = frozenset(
    {"text", "json", "schema", "system", "messages", "tools", "seed", "temperature", "top_p"}
)


def word_count(text: str) -> int:
    """The fake's token count: whitespace-separated words."""
    return len(text.split())


class ScriptedProvider:
    """A ``ModelProvider`` that answers from a script, in order.

    Each reply is a ``str`` (returned as the text of a ``ModelResult``), a ready ``ModelResult`` (returned as is) or an
    exception instance (raised). Every ``ModelRequest`` seen is kept in ``requests``. ``cancel`` already set raises
    ``CallCancelled`` before the reply is consumed. Tokens are counted as whitespace-separated words: a scripted text
    longer than the request's ``max_output_tokens`` is cut to that many words and returned with
    ``finish_reason="length"``. Once the script is used up every call raises ``ProviderError`` (``script_exhausted``).
    """

    def __init__(
        self,
        replies: Sequence[Reply],
        *,
        identity: ProviderIdentity = SCRIPTED_IDENTITY,
        capabilities: Iterable[Capability] = SCRIPTED_CAPABILITIES,
    ) -> None:
        """Script ``replies`` in order; ``identity`` and ``capabilities`` are what the provider reports."""
        self._replies = tuple(replies)
        self._identity = identity
        self._capabilities = frozenset(capabilities)
        self._lock = threading.Lock()
        self._next = 0
        self.requests: list[ModelRequest] = []
        self.closed = False

    @property
    def identity(self) -> ProviderIdentity:
        """Who this fake says it is."""
        return self._identity

    @property
    def capabilities(self) -> frozenset[Capability]:
        """The request fields this fake accepts."""
        return self._capabilities

    def invoke(self, request: ModelRequest, *, cancel: threading.Event | None = None) -> ModelResult:
        """Answer with the next scripted reply."""
        if cancel is not None and cancel.is_set():
            raise CallCancelled("cancelled before the scripted reply")
        with self._lock:
            self.requests.append(request)
            if self._next >= len(self._replies):
                exhausted = ProviderError(f"script exhausted after {len(self._replies)} replies")
                exhausted.code = "script_exhausted"
                raise exhausted
            reply = self._replies[self._next]
            self._next += 1
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, ModelResult):
            return reply
        return self._result(request, reply)

    def _result(self, request: ModelRequest, text: str) -> ModelResult:
        words = text.split()
        cap = request.settings.max_output_tokens
        finish: FinishReason = "stop"
        shown = text
        if len(words) > cap:
            words = words[:cap]
            shown = " ".join(words)
            finish = "length"
        now = datetime.now(UTC)
        input_tokens = sum(word_count(turn.content) for turn in request.turns)
        return ModelResult(
            text=shown,
            finish_reason=finish,
            resolved_model=request.settings.model,
            usage=TokenUsage(
                input_tokens=input_tokens, output_tokens=len(words), total_tokens=input_tokens + len(words)
            ),
            cost_status="unavailable",
            cost_usd=None,
            latency_ms=0,
            started_at=now,
            completed_at=now,
        )

    def close(self) -> None:
        """Mark the provider closed; idempotent."""
        self.closed = True
