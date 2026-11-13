"""CallBudget: the whole run refused before the first call, caps on each call, settlement, and thread safety."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest

from gigai_core.budget import (
    BudgetExceeded,
    CallBudget,
    CallCapExceeded,
    Price,
    budgeted,
)
from gigai_core.providers import (
    CallSettings,
    ModelRequest,
    ModelResult,
    ProviderError,
    TokenUsage,
    Turn,
)
from tests.fakes import ScriptedProvider

NOW = datetime(2026, 1, 1, tzinfo=UTC)
# Completion price 2**-10 per token with 1024 output tokens: each call's worst case is exactly $1.00.
PRICES = {"m": Price(model="m", prompt_usd_per_token=0.0, completion_usd_per_token=2**-10, fetched_at=NOW, source="t")}
REQUEST = ModelRequest(settings=CallSettings(model="m", max_output_tokens=1024), turns=(Turn(role="user", content="hi"),))


def reported(cost_usd: float) -> ModelResult:
    return ModelResult(
        text="",
        finish_reason="stop",
        resolved_model="m",
        usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        cost_status="provider_reported",
        cost_usd=cost_usd,
        latency_ms=1,
        started_at=NOW,
        completed_at=NOW,
    )


def test_plan_refuses_the_whole_run_before_the_first_call() -> None:
    budget = CallBudget(max_usd=2.5, prices=PRICES)
    assert budget.plan([REQUEST, REQUEST]) == 2.0
    with pytest.raises(BudgetExceeded):
        budget.plan([REQUEST, REQUEST, REQUEST])
    assert (budget.calls, budget.spent_usd) == (0, 0.0)
    with pytest.raises(BudgetExceeded):
        CallBudget(max_calls=2).plan([REQUEST, REQUEST, REQUEST])


def test_plan_counts_input_characters_in_the_worst_case() -> None:
    prices = {"m": Price(model="m", prompt_usd_per_token=0.01, completion_usd_per_token=0.0, fetched_at=NOW, source="t")}
    nine_chars = REQUEST.model_copy(update={"turns": (Turn(role="user", content="123456789"),)})
    assert CallBudget(prices=prices).plan([nine_chars]) == 0.03  # ceil(9 / 4) = 3 tokens


def test_reserve_enforces_the_call_cap_the_money_cap_and_the_per_call_cap() -> None:
    calls = CallBudget(max_calls=2)
    calls.reserve(REQUEST)
    calls.reserve(REQUEST)
    with pytest.raises(BudgetExceeded):
        calls.reserve(REQUEST)
    assert calls.calls == 2

    money = CallBudget(max_usd=1.5, prices=PRICES)
    assert money.reserve(REQUEST).estimate_usd == 1.0
    with pytest.raises(BudgetExceeded):
        money.reserve(REQUEST)
    assert money.spent_usd == 1.0

    with pytest.raises(CallCapExceeded):
        CallBudget(prices=PRICES).reserve(REQUEST, cap_usd=0.5)
    CallBudget(prices=PRICES).reserve(REQUEST, cap_usd=1.0)  # equal passes

    with pytest.raises(BudgetExceeded):
        CallBudget(max_usd=1.0).reserve(REQUEST)  # a money cap with no price is refused, not guessed


def test_settle_replaces_the_estimate_only_with_a_provider_reported_cost() -> None:
    budget = CallBudget(prices=PRICES)
    reservation = budget.reserve(REQUEST)
    budget.settle(reservation, reported(0.25))
    assert budget.spent_usd == 0.25

    second = budget.reserve(REQUEST)
    budget.settle(second, None)  # the call failed: the estimate stands
    assert budget.spent_usd == 1.25

    third = budget.reserve(REQUEST)
    budget.settle(third, reported(0.1).model_copy(update={"cost_status": "derived"}))
    assert (budget.spent_usd, budget.calls) == (2.25, 3)


def test_budgeted_reserves_before_the_call_and_settles_after_success_or_failure() -> None:
    budget = CallBudget(max_calls=2, prices=PRICES)
    inner = ScriptedProvider(["ok", ProviderError("boom")])
    provider = budgeted(inner, budget, cap_usd=1.0)
    assert provider.invoke(REQUEST).text == "ok"
    with pytest.raises(ProviderError):
        provider.invoke(REQUEST)
    assert (budget.calls, budget.spent_usd) == (2, 2.0)
    with pytest.raises(BudgetExceeded):
        provider.invoke(REQUEST)
    assert len(inner.requests) == 2  # the refused call never reached the provider


def test_two_hundred_reservations_from_eight_threads_add_up_exactly() -> None:
    budget = CallBudget(max_calls=200, prices=PRICES)

    def one(_: int) -> None:
        budget.settle(budget.reserve(REQUEST), reported(0.5))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, range(200)))
    assert (budget.calls, budget.spent_usd) == (200, 100.0)
    with pytest.raises(BudgetExceeded):
        budget.reserve(REQUEST)
