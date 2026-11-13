"""The one seam between core and whatever answers a model call.

Core owns every type here. gigai implements ``ModelProvider`` over its adapters; a test or a book lesson implements it in a
few lines. Nothing here imports gigai, click, a home folder or a manifest.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

from pydantic import AwareDatetime, Field

from .canonical import sha256_hex
from .models import CoreModel
from .values import JsonValue

FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "error", "unknown"]
OutputMode = Literal["text", "json", "schema"]
CostStatus = Literal["provider_reported", "derived", "unavailable"]
Capability = Literal["text", "json", "schema", "system", "messages", "tools", "logprobs", "seed", "temperature", "top_p"]
Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(CoreModel):
    """One call the model asked for. ``arguments`` is the raw JSON text: a malformed string is itself a graded outcome."""

    id: str
    name: str
    arguments: str


class Turn(CoreModel):
    """One message of a conversation.

    ``tool_calls`` is set on an assistant turn that asked for tools; ``tool_call_id`` on a ``tool`` turn that answers one.
    """

    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


class ToolSpec(CoreModel):
    """A tool the model may call: its name, what it does, and a JSON Schema for its arguments."""

    name: str
    description: str
    parameters: Mapping[str, JsonValue]


class RetryPolicy(CoreModel):
    """Tries per call (at least one) and the first backoff in seconds (doubling, never negative)."""

    attempts: int = Field(default=1, ge=1)
    backoff_seconds: float = Field(default=0.0, ge=0.0)


class CallSettings(CoreModel):
    """The model and the knobs one call carries. Recorded unchanged on every ``ModelCall``.

    Ranges are pydantic constraints: ``max_output_tokens`` above 0, ``temperature`` 0 to 2, ``top_p`` above 0 and at most
    1, ``seed`` and ``top_logprobs`` never negative (``top_logprobs`` at most 20), ``timeout_seconds`` above 0.
    """

    model: str
    max_output_tokens: int = Field(default=1024, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    seed: int | None = None
    output_mode: OutputMode = "text"
    json_schema: Mapping[str, JsonValue] | None = None
    tool_choice: str = "auto"
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    retry: RetryPolicy = RetryPolicy()
    logprobs: bool = False
    top_logprobs: int | None = Field(default=None, ge=0, le=20)


class ModelRequest(CoreModel):
    """What one model call sends.

    ``turns`` is the whole conversation (a system instruction is its first turn, re-sent on every call). ``metadata`` is
    carried into the trace, never sent.
    """

    settings: CallSettings
    turns: tuple[Turn, ...]
    tools: tuple[ToolSpec, ...] = ()
    metadata: Mapping[str, JsonValue] = {}


class TokenUsage(CoreModel):
    """Normalised token counts (never negative; ``None`` when the provider did not say); ``raw`` is its own payload."""

    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    total_tokens: int | None = Field(ge=0)
    raw: Mapping[str, JsonValue] = {}


class ModelResult(CoreModel):
    """What one successful call returned. A failure raises ``ProviderError``; it never returns a failed result."""

    text: str
    finish_reason: FinishReason
    resolved_model: str
    usage: TokenUsage
    cost_status: CostStatus
    cost_usd: float | None = Field(ge=0.0)
    latency_ms: int = Field(ge=0)
    started_at: AwareDatetime
    completed_at: AwareDatetime
    tool_calls: tuple[ToolCall, ...] = ()
    retry_count: int = Field(default=0, ge=0)
    last_error_class: str | None = None
    unsupported: tuple[str, ...] = ()


class ProviderIdentity(CoreModel):
    """Who answered, as the trace records it. Never holds a secret."""

    provider: str
    endpoint: str
    adapter: str
    argv: tuple[str, ...] = ()


class ProviderError(Exception):
    """The provider could not complete one call; ``code`` is the stable machine-readable reason."""

    code: str = "provider_failed"
    retryable: bool = False


class AuthenticationRequired(ProviderError):
    """No usable credential."""

    code = "model_authentication_required"


class UnsupportedRequestField(ProviderError):
    """Refused before any call: the request set a field this provider cannot carry."""

    code = "model_request_field_unsupported"

    def __init__(self, fields: Sequence[str]) -> None:
        """Name the fields the provider refused; the message joins them with commas."""
        super().__init__(", ".join(fields))
        self.fields = tuple(fields)


class OutputNotJSON(ProviderError):
    """The call asked for JSON (``output_mode`` ``json`` or ``schema``) and the provider's reply was not."""

    code = "model_output_not_json"


class CallTimedOut(ProviderError):
    """The wall-time boundary around a call (``providers.bounded``)."""

    code = "model_call_timed_out"


class CallCancelled(ProviderError):
    """``cancel`` was set before or during the call."""

    code = "model_call_cancelled"


class ModelProvider(Protocol):
    """One model behind one transport: the whole seam. gigai binds its adapters to it; a fake answers from a script."""

    @property
    def identity(self) -> ProviderIdentity:
        """Who answers, as the trace records it."""
        ...

    @property
    def capabilities(self) -> frozenset[Capability]:
        """The request fields this provider can carry."""
        ...

    def invoke(self, request: ModelRequest, *, cancel: threading.Event | None = None) -> ModelResult:
        """One call; raises ``ProviderError``. ``cancel`` set raises ``CallCancelled``."""
        ...

    def close(self) -> None:
        """Release a transport this provider owns; idempotent."""
        ...


class _RequestKey(CoreModel):
    """What ``request_sha256`` digests: the model id and the conversation, nothing else."""

    model: str
    turns: tuple[Turn, ...]


def request_sha256(request: ModelRequest) -> str:
    """Digest of the model and the conversation: the key a ``RecordedProvider`` matches on.

    Lowercase hex of the SHA-256 over the JSON of ``settings.model`` and ``turns``; tools, metadata and the other
    settings do not change it, so a replay matches a run made with another temperature or budget.
    """
    key = _RequestKey(model=request.settings.model, turns=request.turns)
    return sha256_hex(key.model_dump_json().encode("utf-8"))


class _Bounded:
    """``bounded``'s wrapper: the inner provider's identity and capabilities, its ``invoke`` under a wall-time cap."""

    def __init__(self, provider: ModelProvider, max_wall_time_ms: int) -> None:
        self._provider = provider
        self._max_wall_time_ms = max_wall_time_ms

    @property
    def identity(self) -> ProviderIdentity:
        return self._provider.identity

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._provider.capabilities

    def invoke(self, request: ModelRequest, *, cancel: threading.Event | None = None) -> ModelResult:
        if cancel is not None and cancel.is_set():
            raise CallCancelled("call was cancelled before invocation")
        results: list[ModelResult] = []
        failures: list[BaseException] = []
        done = threading.Event()

        def worker() -> None:
            try:
                results.append(self._provider.invoke(request, cancel=cancel))
            except BaseException as exc:  # re-raised on the calling thread below
                failures.append(exc)
            finally:
                done.set()

        threading.Thread(target=worker, name="gigai-core-bounded-call", daemon=True).start()
        deadline = time.monotonic() + self._max_wall_time_ms / 1000
        while not done.is_set():
            if cancel is not None and cancel.is_set():
                raise CallCancelled("call was cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CallTimedOut(f"call exceeded its wall-time budget of {self._max_wall_time_ms} ms")
            done.wait(min(remaining, 0.05))
        if failures:
            raise failures[0]
        return results[0]

    def close(self) -> None:
        self._provider.close()


def bounded(provider: ModelProvider, max_wall_time_ms: int) -> ModelProvider:
    """``provider`` with every ``invoke`` under a hard caller-side wait of ``max_wall_time_ms``.

    The inner call runs on a daemon thread because a transport may not support cancellation. Past the deadline the
    caller gets ``CallTimedOut``; ``cancel`` set while waiting gets ``CallCancelled``. Either way the late result is never
    used. A ``max_wall_time_ms`` of 0 or less is a ``ValueError``.
    """
    if max_wall_time_ms <= 0:
        raise ValueError("max_wall_time_ms must be above 0")
    return _Bounded(provider, max_wall_time_ms)
