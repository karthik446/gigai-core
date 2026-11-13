"""Confidence intervals: Wilson, the paired cluster bootstrap, and the input checks every stats module shares.

Retyped from gigai 0.3.2 ``evals/stats.py`` (stdlib only). Wilson intervals are not re-implemented: ``wilson_interval``
calls the numpy-free lift of inspect_ai in ``gigai_core._vendor.inspect_ai`` (see its DIFF.md).

Non-finite input is refused: the vendored metrics give silently wrong answers on NaN or inf (DIFF.md, "Behaviour
differences from upstream"), so every function raises ``NonFiniteScoreError`` on a NaN or infinite value before
computing anything, and every result is a frozen ``CoreModel`` whose floats are finite.
"""

from __future__ import annotations

import math
import random
from collections.abc import Hashable, Mapping, Sequence
from statistics import NormalDist, pstdev

from gigai_core._vendor.inspect_ai._metric import SampleScore, Score, Value
from gigai_core._vendor.inspect_ai.std import ci_wilson

from ..models import CoreModel

MIN_CLUSTERS = 10
"""Fewest clusters a clustered estimate accepts (SPIKE §6 / D §3.4; a carried guess, `[INFERENCE]`)."""

MAX_BOOTSTRAP_RESAMPLES = 10_000
"""Upper bound on bootstrap resamples (PLAN V3 risk 2: pure-Python speed)."""

_NORMAL = NormalDist()


class NonFiniteScoreError(ValueError):
    """A score, p-value or summary statistic was NaN or infinite; drop or reject it before calling."""


class TooFewClustersError(ValueError):
    """A clustered estimate was asked for with fewer than `MIN_CLUSTERS` clusters."""


class Interval(CoreModel):
    """A confidence interval.

    ``stderr`` and ``resamples`` are set by the bootstrap only.
    """

    lower: float
    upper: float
    level: float
    stderr: float | None = None
    resamples: int | None = None


class WilsonInterval(CoreModel):
    """One arm's accuracy with its Wilson interval; ``half_width`` is half the interval length."""

    successes: int
    n: int
    proportion: float
    half_width: float
    ci: Interval


class ClusterBootstrap(CoreModel):
    """The mean difference B - A with its percentile interval from resampling whole clusters.

    ``ci.stderr`` is the population SD of the resampled means and ``ci.resamples`` the number of resamples.
    """

    difference: float
    ci: Interval
    seed: int
    items: int
    clusters: int


def finite_values(values: Sequence[float], name: str) -> list[float]:
    """Every value as a float, or ``NonFiniteScoreError`` naming the first NaN or infinite one.

    Args:
        values: The numbers to check.
        name: How the error names the argument (``name[index]``).

    Returns:
        The values as floats, in order.
    """
    out: list[float] = []
    for index, value in enumerate(values):
        number = float(value)
        if not math.isfinite(number):
            raise NonFiniteScoreError(f"{name}[{index}] is {number!r}; drop or reject non-finite values first")
        out.append(number)
    return out


def finite_scalar(value: float, name: str) -> float:
    """``value`` as a float, or ``NonFiniteScoreError`` when it is NaN or infinite.

    Args:
        value: The number to check.
        name: How the error names the argument.

    Returns:
        The value as a float.
    """
    number = float(value)
    if not math.isfinite(number):
        raise NonFiniteScoreError(f"{name} is {number!r}; drop or reject non-finite values first")
    return number


def two_sided_z(level: float) -> float:
    """The two-sided normal critical value for a confidence ``level`` in (0, 1).

    Args:
        level: The confidence level, e.g. ``0.95``.

    Returns:
        ``z`` such that the central ``level`` share of the standard normal lies within ``-z..z``.
    """
    level = finite_scalar(level, "level")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    return -_NORMAL.inv_cdf((1.0 - level) / 2.0)


def probability(value: float, name: str, *, open_interval: bool = True) -> float:
    """``value`` as a finite float inside (0, 1), or [0, 1] when ``open_interval`` is false.

    Args:
        value: The number to check.
        name: How the error names the argument.
        open_interval: Whether the bounds 0 and 1 are excluded.

    Returns:
        The value as a float.
    """
    number = finite_scalar(value, name)
    inside = 0.0 < number < 1.0 if open_interval else 0.0 <= number <= 1.0
    if not inside:
        bounds = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{name} must be in {bounds}, got {number}")
    return number


def paired_differences(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """The item-wise differences ``b - a`` of two finite, equally long score sequences.

    Args:
        a: Scores of arm A.
        b: Scores of arm B, paired item for item with ``a``.

    Returns:
        ``b[i] - a[i]`` for every item.
    """
    left, right = finite_values(a, "a"), finite_values(b, "b")
    if len(left) != len(right):
        raise ValueError(f"a and b must pair item for item, got {len(left)} and {len(right)} values")
    return [y - x for x, y in zip(left, right)]


def integer(value: object, name: str, *, allow_whole_float: bool = False) -> int:
    """``value`` as an int, or ``TypeError`` for a bool, a string or a float that is not a whole number.

    A float goes through the finite check first (``NonFiniteScoreError`` on NaN or inf); a whole-number float is
    accepted only when ``allow_whole_float`` is set.

    Args:
        value: The untrusted value.
        name: How the error names the argument.
        allow_whole_float: Whether ``10.0`` counts as ``10``.

    Returns:
        The integer.
    """
    if isinstance(value, float):
        finite_scalar(value, name)
        if allow_whole_float and value.is_integer():
            return int(value)
        kind = "a whole number" if allow_whole_float else "an integer"
        raise TypeError(f"{name} must be {kind}, got {value!r}")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    return value


def checked_int(value: object, name: str, *, minimum: int) -> int:
    """``value`` as an int at least ``minimum``, or ``ValueError`` (a bool is not a count).

    Args:
        value: The untrusted value.
        name: How the error names the argument.
        minimum: ``0`` for a non-negative count, ``1`` for a positive one.

    Returns:
        The integer.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        kind = "positive" if minimum > 0 else "non-negative"
        raise ValueError(f"{name} must be a {kind} integer, got {value!r}")
    return value


def require_min_clusters(n_clusters: int) -> None:
    """Refuse a clustered estimate on fewer than `MIN_CLUSTERS` clusters, where its CI is too narrow to trust.

    With 2 clusters the clustered interval came out narrower than Wilson (B-landscape H1, SPIKE §6);
    Miller 2024, "Adding Error Bars to Evals", https://arxiv.org/abs/2411.00640 (fetched 2026-11-09),
    Appendix A. The threshold 10 is `[INFERENCE]`.

    Args:
        n_clusters: The number of clusters (a whole-number float is accepted as a count).

    Raises:
        TooFewClustersError: Below the threshold.
        NonFiniteScoreError: On NaN or inf.
        TypeError: On anything that is not a whole number.
    """
    count = integer(n_clusters, "n_clusters", allow_whole_float=True)
    if count < MIN_CLUSTERS:
        raise TooFewClustersError(f"{count} clusters; clustered estimates need at least {MIN_CLUSTERS}")


def _bound(interval: Value, key: str) -> float:
    """One bound of the vendored ``ci_wilson`` mapping as a finite float (the typed boundary over ``_vendor``)."""
    if not isinstance(interval, Mapping):
        raise TypeError(f"ci_wilson returned {type(interval).__name__}, expected a mapping")
    value = interval.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"ci_wilson {key!r} is {value!r}, expected a number")
    return finite_scalar(value, f"ci_wilson[{key!r}]")


def wilson_interval(successes: int, n: int, level: float = 0.95) -> WilsonInterval:
    """How uncertain one arm's accuracy is, with bounds that stay sensible near 0 % and 100 %.

    Calls the vendored `ci_wilson` (Wilson 1927; Brown, Cai & DasGupta 2001,
    https://doi.org/10.1214/ss/1009213286, fetched 2026-11-09), then pins the edges: the lower
    bound is exactly 0.0 when ``successes == 0`` and the upper bound exactly 1.0 when
    ``successes == n`` (the vendored ``centre - half_width`` leaves a rounding residue such as
    2.8e-17 on some platforms and sizes), and both bounds are clamped to [0, 1]. ``half_width`` is half the length
    of the final interval.

    Args:
        successes: Items the arm got right.
        n: Items in the arm (``n >= 1``, ``0 <= successes <= n``).
        level: Confidence level in (0, 1).

    Returns:
        The proportion with its Wilson interval and half-width.
    """
    successes = integer(successes, "successes")
    n = integer(n, "n")
    if n < 1 or not 0 <= successes <= n:
        raise ValueError(f"need n >= 1 and 0 <= successes <= n, got successes={successes}, n={n}")
    two_sided_z(level)
    scores = [SampleScore(score=Score(value=1.0 if i < successes else 0.0), sample_id=i) for i in range(n)]
    interval = ci_wilson(level=level)(scores)
    lower = 0.0 if successes == 0 else min(max(_bound(interval, "lower"), 0.0), 1.0)
    upper = 1.0 if successes == n else min(max(_bound(interval, "upper"), 0.0), 1.0)
    return WilsonInterval(
        successes=successes,
        n=n,
        proportion=successes / n,
        half_width=(upper - lower) / 2.0,
        ci=Interval(lower=lower, upper=upper, level=level),
    )


def _quantile(values: list[float], q: float) -> float:
    """Linear-interpolation quantile (numpy default, Hyndman-Fan type 7; the vendored ``std._quantile``)."""
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def paired_cluster_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    clusters: Sequence[Hashable],
    *,
    resamples: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> ClusterBootstrap:
    """How uncertain the mean difference B - A is when items come in clusters (documents, postings).

    Resamples whole clusters with replacement, never single items, and recomputes the item-weighted
    mean of d = b - a; the interval is the percentile interval, ``ci.stderr`` the population SD of the
    resampled means. Seeded `random.Random(seed)` (`seed` must be an int; NaN or inf raises
    `NonFiniteScoreError`, other non-ints `TypeError`); `resamples` capped at `MAX_BOOTSTRAP_RESAMPLES`;
    refuses fewer than `MIN_CLUSTERS` clusters. Efron & Tibshirani 1993 (bootstrap); cluster
    resampling: Miller 2024, https://arxiv.org/abs/2411.00640; resampling methods: Ernst 2004,
    https://doi.org/10.1214/088342304000000396 (both fetched 2026-11-09). Ernst 2004 is about
    permutation tests; using it for the bootstrap rests on the same resample-the-exchangeable-units
    logic `[INFERENCE]`.

    Args:
        a: Scores of arm A.
        b: Scores of arm B, paired item for item with ``a``.
        clusters: The cluster label of every item.
        resamples: Bootstrap resamples, in ``[1, MAX_BOOTSTRAP_RESAMPLES]``.
        seed: Seed of the resampling generator.
        level: Confidence level in (0, 1).

    Returns:
        The mean difference with its percentile interval.
    """
    differences = paired_differences(a, b)
    if len(clusters) != len(differences):
        raise ValueError(f"clusters must label every item, got {len(clusters)} labels for {len(differences)} items")
    seed = integer(seed, "seed")
    if type(resamples) is not int or not 1 <= resamples <= MAX_BOOTSTRAP_RESAMPLES:
        raise ValueError(f"resamples must be an integer in [1, {MAX_BOOTSTRAP_RESAMPLES}], got {resamples!r}")
    level = probability(level, "level")
    tail = (1.0 - level) / 2.0

    index: dict[Hashable, int] = {}
    sums: list[float] = []
    sizes: list[int] = []
    for label, difference in zip(clusters, differences):
        slot = index.setdefault(label, len(sums))
        if slot == len(sums):
            sums.append(0.0)
            sizes.append(0)
        sums[slot] += difference
        sizes[slot] += 1
    require_min_clusters(len(sums))

    rng = random.Random(seed)
    cluster_ids = range(len(sums))
    means: list[float] = []
    for _ in range(resamples):
        picks = rng.choices(cluster_ids, k=len(sums))
        means.append(sum(sums[i] for i in picks) / sum(sizes[i] for i in picks))
    return ClusterBootstrap(
        difference=math.fsum(differences) / len(differences),
        ci=Interval(
            lower=_quantile(means, tail),
            upper=_quantile(means, 1.0 - tail),
            level=level,
            stderr=pstdev(means),
            resamples=resamples,
        ),
        seed=seed,
        items=len(differences),
        clusters=len(sums),
    )
