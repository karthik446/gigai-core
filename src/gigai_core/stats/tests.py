"""Paired tests and corrections: exact McNemar, Wilcoxon signed-rank, Holm, Benjamini-Hochberg and TOST.

Retyped from gigai 0.3.2 ``evals/stats.py`` (stdlib only, numerics unchanged).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist
from typing import Literal

from ..models import CoreModel
from .intervals import (
    Interval,
    checked_int,
    finite_scalar,
    finite_values,
    paired_differences,
    probability,
)

WILCOXON_EXACT_MAX_N = 20
"""Largest number of non-zero differences for which the Wilcoxon p-value is exact (2^n sign assignments)."""

_NORMAL = NormalDist()


class McNemarExact(CoreModel):
    """Exact McNemar on the discordant pairs; ``significant`` gates on ``p_exact`` only."""

    b: int
    c: int
    discordant: int
    p_exact: float
    p_mid: float
    alpha: float
    significant: bool


class WilcoxonSignedRank(CoreModel):
    """Two-sided Wilcoxon signed-rank on paired differences; ``z`` is set by the normal approximation only."""

    n: int
    zeros: int
    w_plus: float
    w_minus: float
    statistic: float
    p_value: float
    method: Literal["exact", "normal"]
    z: float | None


class Holm(CoreModel):
    """Holm step-down decisions and adjusted p-values, in input order."""

    reject: tuple[bool, ...]
    adjusted: tuple[float, ...]
    alpha: float


class BenjaminiHochberg(CoreModel):
    """Benjamini-Hochberg step-up decisions and adjusted p-values, in input order."""

    reject: tuple[bool, ...]
    adjusted: tuple[float, ...]
    alpha: float


class Tost(CoreModel):
    """Two one-sided tests; ``ci`` is the (1 - 2 alpha) interval that must lie inside (-margin, margin)."""

    difference: float
    stderr: float
    margin: float
    alpha: float
    p_lower: float
    p_upper: float
    p_value: float
    ci: Interval
    equivalent: bool


def mcnemar_exact(b: int, c: int, alpha: float = 0.05) -> McNemarExact:
    """Whether the items that flipped between two arms flipped one way more often than chance.

    `b` = pairs where arm A was right and arm B wrong, `c` = A wrong and B right. Exact two-sided
    p = min(1, 2 * P(X <= min(b, c))) with X ~ Binomial(b + c, 1/2) (McNemar 1947,
    https://doi.org/10.1007/BF02295996); mid-p = min(1, 2 * (P(X <= min) - P(X = min) / 2))
    (Fagerland, Lydersen & Laake 2013, https://doi.org/10.1186/1471-2288-13-91; both fetched
    2026-11-09). `significant` gates on the exact p only; the mid-p is reported, not gated on.

    Args:
        b: Pairs where A was right and B wrong.
        c: Pairs where A was wrong and B right.
        alpha: Significance level in (0, 1).

    Returns:
        The exact and mid p-values with the decision.
    """
    b = checked_int(b, "b", minimum=0)
    c = checked_int(c, "c", minimum=0)
    alpha = probability(alpha, "alpha")
    discordant = b + c
    smaller = min(b, c)
    if discordant == 0:
        p_exact = p_mid = 1.0
    else:
        total = 2**discordant
        tail = sum(math.comb(discordant, i) for i in range(smaller + 1))
        at_smaller = math.comb(discordant, smaller)
        p_exact = min(1.0, 2 * tail / total)
        p_mid = min(1.0, (2 * tail - at_smaller) / total)
    return McNemarExact(
        b=b,
        c=c,
        discordant=discordant,
        p_exact=p_exact,
        p_mid=p_mid,
        alpha=alpha,
        significant=p_exact <= alpha,
    )


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> WilcoxonSignedRank:
    """Whether paired continuous scores moved one way, using ranks so a few large swings cannot dominate.

    d = b - a; zero differences are dropped (Wilcoxon's rule) and tied |d| share their mean rank.
    Rule: with n <= `WILCOXON_EXACT_MAX_N` non-zero differences the p-value is exact (all 2^n sign
    assignments of the observed ranks, ties included); above that it uses the normal approximation
    with mean n(n+1)/4, variance n(n+1)(2n+1)/24 - sum(t^3 - t)/48 over tie groups of size t, and a
    0.5 continuity correction. Two-sided. Wilcoxon 1945, https://doi.org/10.2307/3001968; Demšar
    2006, https://jmlr.org/papers/v7/demsar06a.html (both fetched 2026-11-09).

    Args:
        a: Scores of arm A.
        b: Scores of arm B, paired item for item with ``a``.

    Returns:
        The rank sums, the statistic ``min(w_plus, w_minus)`` and the two-sided p-value.
    """
    differences = paired_differences(a, b)
    nonzero = [d for d in differences if d != 0.0]
    n = len(nonzero)
    zeros = len(differences) - n
    if n == 0:
        return WilcoxonSignedRank(
            n=0, zeros=zeros, w_plus=0.0, w_minus=0.0, statistic=0.0, p_value=1.0, method="exact", z=None
        )

    order = sorted(range(n), key=lambda i: abs(nonzero[i]))
    doubled = [0] * n  # 2 x mid-rank: always an integer
    tie_sizes: list[int] = []
    start = 0
    while start < n:
        end = start
        while end + 1 < n and abs(nonzero[order[end + 1]]) == abs(nonzero[order[start]]):
            end += 1
        for position in range(start, end + 1):
            doubled[order[position]] = (start + 1) + (end + 1)
        tie_sizes.append(end - start + 1)
        start = end + 1

    plus2 = sum(r for r, d in zip(doubled, nonzero) if d > 0)
    total2 = n * (n + 1)
    w_plus, w_minus = plus2 / 2.0, (total2 - plus2) / 2.0

    method: Literal["exact", "normal"]
    z: float | None
    if n <= WILCOXON_EXACT_MAX_N:
        counts = [0] * (total2 + 1)
        counts[0] = 1
        for rank2 in doubled:
            for s in range(total2, rank2 - 1, -1):
                counts[s] += counts[s - rank2]
        low = min(plus2, total2 - plus2)
        p_value = min(1.0, 2 * sum(counts[: low + 1]) / 2**n)
        method, z = "exact", None
    else:
        mean = n * (n + 1) / 4.0
        variance = n * (n + 1) * (2 * n + 1) / 24.0 - sum(t**3 - t for t in tie_sizes) / 48.0
        gap = w_plus - mean
        z = math.copysign(max(abs(gap) - 0.5, 0.0), gap) / math.sqrt(variance)
        p_value = min(1.0, 2.0 * _NORMAL.cdf(-abs(z)))
        method = "normal"
    return WilcoxonSignedRank(
        n=n,
        zeros=zeros,
        w_plus=w_plus,
        w_minus=w_minus,
        statistic=min(w_plus, w_minus),
        p_value=p_value,
        method=method,
        z=z,
    )


def _checked_p_values(p_values: Sequence[float], alpha: float) -> tuple[list[float], float]:
    values = finite_values(p_values, "p_values")
    for index, value in enumerate(values):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"p_values[{index}] = {value} is not a probability")
    return values, probability(alpha, "alpha")


def holm(p_values: Sequence[float], alpha: float = 0.05) -> Holm:
    """Which of several confirmatory tests still count as wins once the number of tests is paid for.

    Holm step-down (controls the family-wise error rate under any dependence): sort p ascending,
    reject p_(j) while p_(j) <= alpha / (m - j + 1), stop at the first failure. Adjusted p_(j) =
    max over i <= j of min(1, (m - i + 1) p_(i)). Results are in input order. Holm 1979,
    https://www.jstor.org/stable/4615733 (fetched 2026-11-09).

    Args:
        p_values: The raw p-values, each in [0, 1].
        alpha: Family-wise error rate in (0, 1).

    Returns:
        Decisions and adjusted p-values in input order.
    """
    values, alpha = _checked_p_values(p_values, alpha)
    m = len(values)
    order = sorted(range(m), key=lambda i: values[i])
    reject = [False] * m
    adjusted = [0.0] * m
    running = 0.0
    stopped = False
    for j, i in enumerate(order):
        running = max(running, min(1.0, (m - j) * values[i]))
        adjusted[i] = running
        if not stopped and values[i] <= alpha / (m - j):
            reject[i] = True
        else:
            stopped = True
    return Holm(reject=tuple(reject), adjusted=tuple(adjusted), alpha=alpha)


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> BenjaminiHochberg:
    """Which exploratory results to follow up while keeping the expected share of false wins at alpha.

    Benjamini-Hochberg step-up (controls the false discovery rate for independent or positively
    dependent tests): reject every p_(j) with j <= max{j : p_(j) <= j alpha / m}. Adjusted p_(j) =
    min over i >= j of min(1, m p_(i) / i). Results are in input order. Benjamini & Hochberg 1995,
    https://www.jstor.org/stable/2346101 (fetched 2026-11-09).

    Args:
        p_values: The raw p-values, each in [0, 1].
        alpha: False discovery rate in (0, 1).

    Returns:
        Decisions and adjusted p-values in input order.
    """
    values, alpha = _checked_p_values(p_values, alpha)
    m = len(values)
    order = sorted(range(m), key=lambda i: values[i])
    cutoff = 0
    for j, i in enumerate(order, start=1):
        if values[i] <= j * alpha / m:
            cutoff = j
    reject = [False] * m
    adjusted = [0.0] * m
    running = 1.0
    for j in range(m, 0, -1):
        i = order[j - 1]
        running = min(running, m * values[i] / j)
        adjusted[i] = running
        reject[i] = j <= cutoff
    return BenjaminiHochberg(reject=tuple(reject), adjusted=tuple(adjusted), alpha=alpha)


def tost(difference: float, stderr: float, margin: float, alpha: float = 0.05) -> Tost:
    """Whether two arms are shown to differ by less than `margin` ("no regression"), not merely "not shown different".

    Two one-sided z tests: p_lower = P(Z >= (d + margin) / se), p_upper = P(Z <= (d - margin) / se);
    equivalent when max(p_lower, p_upper) <= alpha, i.e. the (1 - 2 alpha) interval d -/+ z se lies
    inside (-margin, margin). Normal theory: pass a paired-difference SE from a large enough sample.
    Schuirmann 1987, https://doi.org/10.1007/BF01068419; Lakens 2017,
    https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/ (both fetched 2026-11-09).

    Args:
        difference: The observed mean difference B - A.
        stderr: Its standard error (positive).
        margin: The equivalence margin (positive).
        alpha: Significance level of each one-sided test, in (0, 0.5).

    Returns:
        The two one-sided p-values, their maximum, the (1 - 2 alpha) interval and the decision.
    """
    difference = finite_scalar(difference, "difference")
    stderr = finite_scalar(stderr, "stderr")
    margin = finite_scalar(margin, "margin")
    alpha = probability(alpha, "alpha")
    if stderr <= 0.0 or margin <= 0.0:
        raise ValueError(f"stderr and margin must be positive, got stderr={stderr}, margin={margin}")
    if alpha >= 0.5:
        raise ValueError(f"alpha must be below 0.5, got {alpha}")
    p_lower = _NORMAL.cdf(-(difference + margin) / stderr)
    p_upper = _NORMAL.cdf((difference - margin) / stderr)
    critical = _NORMAL.inv_cdf(1.0 - alpha)
    p_value = max(p_lower, p_upper)
    return Tost(
        difference=difference,
        stderr=stderr,
        margin=margin,
        alpha=alpha,
        p_lower=p_lower,
        p_upper=p_upper,
        p_value=p_value,
        ci=Interval(lower=difference - critical * stderr, upper=difference + critical * stderr, level=1.0 - 2.0 * alpha),
        equivalent=p_value <= alpha,
    )
