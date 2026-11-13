"""The seam: records validate and round-trip, exceptions carry codes, the scripted fake behaves like a provider."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from gigai_core.providers import (
    AuthenticationRequired,
    CallCancelled,
    CallSettings,
    CallTimedOut,
    Capability,
    ModelProvider,
    ModelRequest,
    ModelResult,
    OutputNotJSON,
    ProviderError,
    ProviderIdentity,
    TokenUsage,
    ToolSpec,
    Turn,
    UnsupportedRequestField,
    bounded,
    request_sha256,
)
from tests.fakes import SCRIPTED_IDENTITY, ScriptedProvider

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def request(text: str = "hello there", *, model: str = "fake/model", max_output_tokens: int = 1024) -> ModelRequest:
    settings = CallSettings(model=model, max_output_tokens=max_output_tokens)
    return ModelRequest(settings=settings, turns=(Turn(role="user", content=text),))


def test_scripted_provider_is_a_model_provider_and_answers_in_order() -> None:
    provider: ModelProvider = ScriptedProvider(["first", "second"])
    assert provider.identity == SCRIPTED_IDENTITY
    assert "text" in provider.capabilities
    first = provider.invoke(request("one"))
    second = provider.invoke(request("two"))
    assert (first.text, second.text) == ("first", "second")
    assert first.finish_reason == "stop"
    assert first.resolved_model == "fake/model"
    assert first.usage == TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2)
    provider.close()


def test_scripted_provider_records_requests_then_refuses_when_the_script_is_used_up() -> None:
    provider = ScriptedProvider(["only"])
    provider.invoke(request("a"))
    with pytest.raises(ProviderError) as refused:
        provider.invoke(request("b"))
    assert refused.value.code == "script_exhausted"
    assert [turn.content for seen in provider.requests for turn in seen.turns] == ["a", "b"]


def test_scripted_text_longer_than_max_output_tokens_is_cut_with_finish_reason_length() -> None:
    provider = ScriptedProvider(["one two three four five"])
    result = provider.invoke(request(max_output_tokens=3))
    assert result.text == "one two three"
    assert result.finish_reason == "length"
    assert result.usage.output_tokens == 3


def test_scripted_provider_raises_a_scripted_exception_and_honours_cancel() -> None:
    provider = ScriptedProvider([AuthenticationRequired("no key"), "unreached"])
    with pytest.raises(AuthenticationRequired):
        provider.invoke(request())
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CallCancelled):
        provider.invoke(request(), cancel=cancel)
    assert len(provider.requests) == 1


@pytest.mark.parametrize(
    "field, value",
    [
        ("max_output_tokens", 0),
        ("temperature", 2.5),
        ("temperature", -0.1),
        ("top_p", 0.0),
        ("top_p", 1.5),
        ("timeout_seconds", 0.0),
        ("top_logprobs", 21),
    ],
)
def test_call_settings_rejects_out_of_range_values(field: str, value: float) -> None:
    with pytest.raises(ValidationError) as error:
        CallSettings.model_validate({"model": "m", field: value})
    assert error.value.errors()[0]["loc"] == (field,)


def test_request_and_result_round_trip_through_json() -> None:
    settings = CallSettings(model="m", temperature=0.5, seed=7, output_mode="json")
    sent = ModelRequest(
        settings=settings,
        turns=(Turn(role="system", content="be brief"), Turn(role="user", content="hi")),
        tools=(ToolSpec(name="add", description="adds", parameters={"type": "object"}),),
        metadata={"item": "q1", "repeat": 0},
    )
    assert ModelRequest.model_validate_json(sent.model_dump_json()) == sent

    got = ModelResult(
        text="4",
        finish_reason="stop",
        resolved_model="m-2026",
        usage=TokenUsage(input_tokens=3, output_tokens=1, total_tokens=4, raw={"prompt_tokens": 3}),
        cost_status="provider_reported",
        cost_usd=0.000125,
        latency_ms=12,
        started_at=NOW,
        completed_at=NOW,
        unsupported=("top_p",),
    )
    assert ModelResult.model_validate_json(got.model_dump_json()) == got
    assert '"started_at":"2026-01-01T00:00:00Z"' in got.model_dump_json()


def test_the_provider_errors_carry_a_message_and_a_stable_code() -> None:
    unsupported = UnsupportedRequestField(["tools", "seed"])
    assert str(unsupported) == "tools, seed"
    assert unsupported.fields == ("tools", "seed")
    assert unsupported.code == "model_request_field_unsupported"
    assert str(AuthenticationRequired("no key for openrouter")) == "no key for openrouter"
    assert AuthenticationRequired.code == "model_authentication_required"
    assert OutputNotJSON.code == "model_output_not_json"
    assert CallTimedOut.code == "model_call_timed_out"
    assert CallCancelled.code == "model_call_cancelled"
    assert ProviderError.code == "provider_failed"
    assert not ProviderError("x").retryable
    for error in (unsupported, AuthenticationRequired(), OutputNotJSON(), CallTimedOut(), CallCancelled()):
        assert isinstance(error, ProviderError)


def test_request_sha256_keys_on_the_model_and_the_conversation_only() -> None:
    base = request("hi")
    same_turns_other_knobs = base.model_copy(update={"settings": CallSettings(model="fake/model", temperature=1.0)})
    assert request_sha256(base) == request_sha256(same_turns_other_knobs)
    assert request_sha256(base) != request_sha256(request("hi", model="other/model"))
    assert request_sha256(base) != request_sha256(request("bye"))
    assert len(request_sha256(base)) == 64


class _Slow:
    """A provider that takes longer than the boundary allows."""

    identity = ProviderIdentity(provider="slow", endpoint="memory", adapter="tests.test_providers._Slow:1")
    capabilities: frozenset[Capability] = frozenset({"text"})

    def invoke(self, request: ModelRequest, *, cancel: threading.Event | None = None) -> ModelResult:
        time.sleep(0.5)
        return ScriptedProvider(["late"]).invoke(request)

    def close(self) -> None:
        return None


def test_bounded_passes_a_fast_call_through_and_times_a_slow_one_out() -> None:
    fast = bounded(ScriptedProvider(["quick"]), 1_000)
    assert fast.invoke(request()).text == "quick"
    slow = bounded(_Slow(), 50)
    with pytest.raises(CallTimedOut):
        slow.invoke(request())
    with pytest.raises(ValueError):
        bounded(_Slow(), 0)
