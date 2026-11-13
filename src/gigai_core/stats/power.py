"""Power and sizing: Connor's paired-proportion sample size, minimum detectable effects, variance components, noise.

Retyped from gigai 0.3.2 ``evals/stats.py`` (stdlib only, numerics unchanged).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

from ..models import CoreModel
from .intervals import (
    Interval,
    checked_int,
    finite_scalar,
    finite_values,
    probability,
    two_sided_z,
)

_NORMAL = NormalDist()


class MinimumDetectableDifference(CoreModel):
    """The smallest accuracy gap ``n`` paired items detect; ``mde`` is None when even delta = discordance needs more."""

    n: int
    discordance: float
    alpha: float
    power: float
    mde: float | None


class MdeContinuous(CoreModel):
    """The smallest mean difference ``n`` paired items detect given the SD of the differences."""

    n: int
    sd: float
    alpha: float
    power: float
    mde: float


class MdeRow(CoreModel):
    """One ``mde_table`` row: the items needed for a (discordance, delta) pair; ``n`` is None where delta > discordance."""

    discordance: float
    delta: float
    n: int | None


class VarianceComponents(CoreModel):
    """One-way random-effects split of a score's spread; ``optimal_repeats`` is None without a cost ratio or between-item spread."""

    items: int
    repeats: int
    mean: float
    within_item: float
    between_item: float
    var_mean: float
    stderr_mean: float
    optimal_repeats: float | None


class NoiseBound(CoreModel):
    """``mean -/+ z sd / sqrt(n)``: a measured difference inside ``ci`` is within run-to-run noise."""

    mean: float
    sd: float
    n: int
    half_width: float
    ci: Interval


def _power_z(alpha: float, power: float) -> tuple[float, float]:
    alpha = probability(alpha, "alpha")
    power = probability(power, "power")
    return _NORMAL.inv_cdf(1.0 - alpha / 2.0), _NORMAL.inv_cdf(power)


def _connor_n(discordance: float, delta: float, z_alpha: float, z_beta: float) -> float:
    return (z_alpha * math.sqrt(discordance) + z_beta * math.sqrt(discordance - delta * delta)) ** 2 / (delta * delta)


def mde_sample_size(discordance: float, delta: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """How many paired items a run needs to detect a true accuracy gap `delta` at the given discordance.

    Connor 1987 paired-proportion sample size, n = ceil((z_{alpha/2} sqrt(p_d) + z_beta
    sqrt(p_d - delta^2))^2 / delta^2), two-sided; needs 0 < delta <= p_d (Connor 1987,
    https://doi.org/10.2307/2531961, fetched 2026-11-09; D §3.8a).

    Args:
        discordance: Expected share of discordant pairs, in [0, 1].
        delta: The true accuracy gap to detect, in (0, discordance].
        alpha: Two-sided significance level in (0, 1).
        power: Target power in (0, 1).

    Returns:
        The number of paired items, rounded up.
    """
    discordance = probability(discordance, "discordance", open_interval=False)
    delta = finite_scalar(delta, "delta")
    if not 0.0 < delta <= discordance:
        raise ValueError(f"need 0 < delta <= discordance, got delta={delta}, discordance={discordance}")
    z_alpha, z_beta = _power_z(alpha, power)
    return math.ceil(_connor_n(discordance, delta, z_alpha, z_beta))


def minimum_detectable_difference(
    n: int, discordance: float, alpha: float = 0.05, power: float = 0.80
) -> MinimumDetectableDifference:
    """The smallest true accuracy gap a paired run of `n` items can detect at the given discordance.

    Inverts Connor 1987 (as `mde_sample_size`, without the ceiling) by bisection over
    0 < delta <= p_d; `mde` is None when even delta = p_d needs more than `n` items
    (https://doi.org/10.2307/2531961, fetched 2026-11-09; D §3.8).

    Args:
        n: Paired items in the run (positive).
        discordance: Expected share of discordant pairs, in (0, 1].
        alpha: Two-sided significance level in (0, 1).
        power: Target power in (0, 1).

    Returns:
        The inputs with the minimum detectable difference.
    """
    n = checked_int(n, "n", minimum=1)
    discordance = probability(discordance, "discordance", open_interval=False)
    if discordance == 0.0:
        raise ValueError("discordance must be positive")
    z_alpha, z_beta = _power_z(alpha, power)
    alpha, power = probability(alpha, "alpha"), probability(power, "power")
    if _connor_n(discordance, discordance, z_alpha, z_beta) > n:
        return MinimumDetectableDifference(n=n, discordance=discordance, alpha=alpha, power=power, mde=None)
    low, high = 0.0, discordance  # needed n(low) > n >= needed n(high); n(delta) falls as delta grows
    for _ in range(200):
        middle = (low + high) / 2.0
        if middle in (low, high):
            break
        if _connor_n(discordance, middle, z_alpha, z_beta) > n:
            low = middle
        else:
            high = middle
    return MinimumDetectableDifference(n=n, discordance=discordance, alpha=alpha, power=power, mde=high)


def mde_continuous(n: int, sd: float, alpha: float = 0.05, power: float = 0.80) -> MdeContinuous:
    """The smallest true mean difference a paired run of `n` items can detect, given the SD of the differences.

    MDE = (z_{alpha/2} + z_beta) sd / sqrt(n), two-sided (D §3.8; Miller 2024 §5,
    https://arxiv.org/abs/2411.00640, fetched 2026-11-09).

    Args:
        n: Paired items in the run (positive).
        sd: Standard deviation of the paired differences (non-negative).
        alpha: Two-sided significance level in (0, 1).
        power: Target power in (0, 1).

    Returns:
        The inputs with the minimum detectable effect.
    """
    n = checked_int(n, "n", minimum=1)
    sd = finite_scalar(sd, "sd")
    if sd < 0.0:
        raise ValueError(f"sd must be non-negative, got {sd}")
    z_alpha, z_beta = _power_z(alpha, power)
    alpha, power = probability(alpha, "alpha"), probability(power, "power")
    return MdeContinuous(n=n, sd=sd, alpha=alpha, power=power, mde=(z_alpha + z_beta) * sd / math.sqrt(n))


def mde_table(
    discordances: Sequence[float], deltas: Sequence[float], alpha: float = 0.05, power: float = 0.80
) -> tuple[MdeRow, ...]:
    """The items needed for every (discordance, gap) pair, so a suite's size can be checked before running.

    One row per pair from `mde_sample_size`; `n` is None where delta > p_d (B cannot win more pairs
    than are discordant). `alpha`, `power` and every value are checked up front, even when a list is
    empty (empty input gives no rows): discordances must lie in [0, 1], deltas must be positive.
    Connor 1987, https://doi.org/10.2307/2531961 (fetched 2026-11-09); D §3.8a.

    Args:
        discordances: Expected shares of discordant pairs, each in [0, 1].
        deltas: Accuracy gaps to detect, each positive.
        alpha: Two-sided significance level in (0, 1).
        power: Target power in (0, 1).

    Returns:
        One row per (discordance, delta) pair, discordances outer, deltas inner.
    """
    _power_z(alpha, power)
    checked_discordances = [
        probability(value, f"discordances[{index}]", open_interval=False)
        for index, value in enumerate(finite_values(discordances, "discordances"))
    ]
    checked_deltas = finite_values(deltas, "deltas")
    for index, delta in enumerate(checked_deltas):
        if delta <= 0.0:
            raise ValueError(f"deltas[{index}] must be positive, got {delta}")
    rows: list[MdeRow] = []
    for discordance in checked_discordances:
        for delta in checked_deltas:
            n = mde_sample_size(discordance, delta, alpha, power) if delta <= discordance else None
            rows.append(MdeRow(discordance=discordance, delta=delta, n=n))
    return tuple(rows)


def variance_components(scores: Sequence[Sequence[float]], cost_ratio: float | None = None) -> VarianceComponents:
    """How much of a score's spread comes from which items were asked versus from re-running the same item.

    `scores[i][r]` = item i, repeat r (every item the same k >= 2 repeats, n >= 2 items). One-way
    random effects: within_item = sum_i sum_r (y_ir - mean_i)^2 / (n (k - 1)); between_item =
    max(0, var_i(mean_i) - within_item / k) (sample variance over items; truncated at 0);
    var_mean = (between_item + within_item / k) / n. With `cost_ratio` c = the cost of one NEW item in
    units of one repeat (c = 10: a new item costs as much as ten re-runs), optimal_repeats =
    sqrt(c within_item / between_item), the repeats per item that minimise var_mean for a fixed
    budget; None when between_item = 0 (D-measurement §3.5: c ≈ 10 there gives k* ≈ 2.6).
    Miller 2024, https://arxiv.org/abs/2411.00640; Bouthillier et al. 2021,
    https://arxiv.org/abs/2103.03098 (both fetched 2026-11-09).

    Args:
        scores: Per item, the score of every repeat.
        cost_ratio: Cost of one new item in repeats (positive), or None to skip ``optimal_repeats``.

    Returns:
        The variance split with the standard error of the grand mean.
    """
    matrix = [finite_values(row, f"scores[{i}]") for i, row in enumerate(scores)]
    n = len(matrix)
    if n < 2:
        raise ValueError("need at least 2 items")
    k = len(matrix[0])
    if k < 2 or any(len(row) != k for row in matrix):
        raise ValueError("every item needs the same number (>= 2) of repeats")
    if cost_ratio is not None:
        cost_ratio = finite_scalar(cost_ratio, "cost_ratio")
        if cost_ratio <= 0.0:
            raise ValueError(f"cost_ratio must be positive, got {cost_ratio}")
    item_means = [math.fsum(row) / k for row in matrix]
    grand = math.fsum(item_means) / n
    within = math.fsum((y - mean) ** 2 for row, mean in zip(matrix, item_means) for y in row) / (n * (k - 1))
    between_raw = math.fsum((mean - grand) ** 2 for mean in item_means) / (n - 1) - within / k
    between = max(between_raw, 0.0)
    var_mean = (between + within / k) / n
    optimal = math.sqrt(cost_ratio * within / between) if cost_ratio is not None and between > 0.0 else None
    return VarianceComponents(
        items=n,
        repeats=k,
        mean=grand,
        within_item=within,
        between_item=between,
        var_mean=var_mean,
        stderr_mean=math.sqrt(var_mean),
        optimal_repeats=optimal,
    )


def noise_bound(mean: float, sd: float, n: int, level: float = 0.95) -> NoiseBound:
    """How large a mean difference must be before it stands out from run-to-run noise.

    upper = mean + z sd / sqrt(n) (z = 1.96 at 95 %); a measured difference below `upper` is within
    noise. F-read `:93` (0.167 + 1.96 x 0.595 / sqrt(48) = 0.335); D §3.5, Miller 2024,
    https://arxiv.org/abs/2411.00640 (fetched 2026-11-09).

    Args:
        mean: The mean run-to-run difference.
        sd: Its standard deviation (non-negative).
        n: Runs behind the estimate (positive).
        level: Confidence level in (0, 1).

    Returns:
        The half-width with the noise interval.
    """
    mean = finite_scalar(mean, "mean")
    sd = finite_scalar(sd, "sd")
    if sd < 0.0:
        raise ValueError(f"sd must be non-negative, got {sd}")
    n = checked_int(n, "n", minimum=1)
    half_width = two_sided_z(level) * sd / math.sqrt(n)
    return NoiseBound(
        mean=mean,
        sd=sd,
        n=n,
        half_width=half_width,
        ci=Interval(lower=mean - half_width, upper=mean + half_width, level=level),
    )
