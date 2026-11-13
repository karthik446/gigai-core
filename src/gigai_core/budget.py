"""Caps on money and calls, checked before each call, and the whole run's worst case before the first."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from typing import Protocol

from pydantic import AwareDatetime, Field

from .models import CoreModel
from .providers import (
    Capability,
    ModelProvider,
    ModelRequest,
    ModelResult,
    ProviderIdentity,
)

#: The input-length rule of the worst case: one token per four characters of conversation text, rounded up.
CHARS_PER_TOKEN = 4


class BudgetExceeded(Exception):
    """A cap would be passed; raised before the call."""

    code: str = "budget_exhausted"


class CallCapExceeded(BudgetExceeded):
    """One call's worst case is over the ``cap_usd`` the caller passed."""

    code = "call_cap_exceeded"


class Price(CoreModel):
    """USD per token for one model, with when and where the number came from (stored in the run: N18)."""

    model: str
    prompt_usd_per_token: float = Field(ge=0.0)
    completion_usd_per_token: float = Field(ge=0.0)
    fetched_at: AwareDatetime
    source: str


class Reservation(CoreModel):
    """What ``Budget.reserve`` handed back: the worst-case USD charged ahead, to settle after the call."""

    estimate_usd: float = Field(ge=0.0)
    calls: int = Field(default=1, ge=0)


class Budget(Protocol):
    """Caps checked before each call. gigai implements it over ``spend.json``; core ships ``CallBudget``."""

    def plan(self, requests: Sequence[ModelRequest]) -> float:
        """Worst-case USD of ``requests``; raises ``BudgetExceeded`` when it does not fit (N18)."""
        ...

    def reserve(self, request: ModelRequest, *, cap_usd: float | None = None) -> Reservation:
        """Pre-charge one call; ``CallCapExceeded`` over ``cap_usd``, ``BudgetExceeded`` over the run's cap."""
        ...

    def settle(self, reservation: Reservation, result: ModelResult | None) -> None:
        """Replace the estimate by what the provider reported; ``None`` means the call failed."""
        ...


class CallBudget:
    """A budget with no home: at most ``max_usd`` and ``max_calls``; prices from ``prices`` (model id to price).

    A plain class: it holds a running total, which a frozen model cannot. The worst case of one call is
    ``ceil(conversation characters / 4)`` input tokens at the model's prompt price plus ``max_output_tokens`` at its
    completion price, rounded to the microdollar. A model with no price costs nothing while ``max_usd`` is ``None`` and
    is refused (``BudgetExceeded``) once a money cap is set: nothing is guessed. Counters are guarded by one lock, so
    concurrent reservations never pass a cap together.
    """

    def __init__(
        self, *, max_usd: float | None = None, max_calls: int | None = None, prices: Mapping[str, Price] | None = None
    ) -> None:
        """Set the caps (``None`` is unbounded) and the price table (``None`` is empty)."""
        self._max_usd = max_usd
        self._max_calls = max_calls
        self._prices: Mapping[str, Price] = {} if prices is None else dict(prices)
        self._lock = threading.Lock()
        self._spent_usd = 0.0
        self._calls = 0

    @property
    def spent_usd(self) -> float:
        """USD charged so far: reservations still open at their estimate, settled calls at the reported cost."""
        return self._spent_usd

    @property
    def calls(self) -> int:
        """Calls reserved so far, failed ones included."""
        return self._calls

    def _estimate(self, request: ModelRequest) -> float:
        model = request.settings.model
        price = self._prices.get(model)
        if price is None:
            if self._max_usd is None:
                return 0.0
            raise BudgetExceeded(f"no price for model {model!r}: cannot estimate a call under a max_usd cap")
        chars = sum(len(turn.content) for turn in request.turns)
        input_tokens = math.ceil(chars / CHARS_PER_TOKEN)
        output_tokens = request.settings.max_output_tokens
        return round(input_tokens * price.prompt_usd_per_token + output_tokens * price.completion_usd_per_token, 6)

    def _refuse_if_over(self, usd: float, calls: int) -> None:
        """Under the lock: raise when ``calls`` more calls or ``usd`` more dollars would pass a cap."""
        if self._max_calls is not None and self._calls + calls > self._max_calls:
            raise BudgetExceeded(
                f"{calls} more call(s) would pass max_calls={self._max_calls} (already {self._calls})"
            )
        if self._max_usd is not None and self._spent_usd + usd > self._max_usd:
            raise BudgetExceeded(
                f"${usd:.6f} does not fit the remaining ${self._max_usd - self._spent_usd:.6f} of max_usd={self._max_usd}"
            )

    def plan(self, requests: Sequence[ModelRequest]) -> float:
        """Worst-case USD of ``requests`` together; raises ``BudgetExceeded`` before any call when they cannot all fit."""
        total = round(sum(self._estimate(request) for request in requests), 6)
        with self._lock:
            self._refuse_if_over(total, len(requests))
        return total

    def reserve(self, request: ModelRequest, *, cap_usd: float | None = None) -> Reservation:
        """Charge one call's worst case ahead; ``CallCapExceeded`` over ``cap_usd`` (equal passes), else the caps."""
        estimate = self._estimate(request)
        if cap_usd is not None and estimate > cap_usd:
            raise CallCapExceeded(f"estimated ${estimate:.6f} > cap ${cap_usd:.6f}")
        with self._lock:
            self._refuse_if_over(estimate, 1)
            self._spent_usd = round(self._spent_usd + estimate, 6)
            self._calls += 1
        return Reservation(estimate_usd=estimate)

    def settle(self, reservation: Reservation, result: ModelResult | None) -> None:
        """Move the ledger from the estimate to the cost the provider reported.

        Only ``cost_status == "provider_reported"`` with a ``cost_usd`` replaces the estimate; a failed call (``None``),
        a derived or unavailable cost keeps the estimate. The call count never changes here.
        """
        if result is None or result.cost_status != "provider_reported" or result.cost_usd is None:
            return
        with self._lock:
            self._spent_usd = max(0.0, round(self._spent_usd + result.cost_usd - reservation.estimate_usd, 6))


class _Budgeted:
    """``budgeted``'s wrapper: the inner provider with each ``invoke`` reserved first and settled after."""

    def __init__(self, provider: ModelProvider, budget: Budget, cap_usd: float | None) -> None:
        self._provider = provider
        self._budget = budget
        self._cap_usd = cap_usd

    @property
    def identity(self) -> ProviderIdentity:
        return self._provider.identity

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._provider.capabilities

    def invoke(self, request: ModelRequest, *, cancel: threading.Event | None = None) -> ModelResult:
        reservation = self._budget.reserve(request, cap_usd=self._cap_usd)
        try:
            result = self._provider.invoke(request, cancel=cancel)
        except BaseException:
            self._budget.settle(reservation, None)
            raise
        self._budget.settle(reservation, result)
        return result

    def close(self) -> None:
        self._provider.close()


def budgeted(provider: ModelProvider, budget: Budget, *, cap_usd: float | None = None) -> ModelProvider:
    """``provider`` with every ``invoke`` reserved on ``budget`` first and settled after (``None`` on failure).

    ``run`` uses it for the subject provider; wrap a judge's provider on the same ``budget`` to count the judge's calls
    too. A refused reservation raises before the inner provider is asked.
    """
    return _Budgeted(provider, budget, cap_usd)
