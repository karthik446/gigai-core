"""Agreement: Cohen's kappa, Krippendorff's alpha (vendored) and the Hanley-McNeil AUC.

Retyped from gigai 0.3.2 ``evals/stats.py``; ``cohen_kappa`` takes the two raters' label sequences (DESIGN 4.7)
and builds the 2 x 2 table itself, the math is unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from gigai_core._vendor.inspect_ai._metric import SampleScore, Score
from gigai_core._vendor.inspect_ai.krippendorff import KrippendorffLevel
from gigai_core._vendor.inspect_ai.krippendorff import (
    krippendorff_alpha as _vendored_krippendorff_alpha,
)

from ..models import CoreModel
from .intervals import Interval, NonFiniteScoreError, finite_values, two_sided_z


class Kappa(CoreModel):
    """Cohen's kappa with its large-sample interval, the SE under kappa = 0, and the prevalence-free PABAK."""

    kappa: float
    se: float
    ci: Interval
    se_null: float
    observed_agreement: float
    chance_agreement: float
    pabak: float
    n: int


class KrippendorffAlpha(CoreModel):
    """Krippendorff's alpha over ``units`` rated units; ``alpha`` is None when no unit had two ratings."""

    alpha: float | None
    units: int
    level: KrippendorffLevel


class Auc(CoreModel):
    """The Mann-Whitney AUC with its Hanley-McNeil standard error and interval."""

    auc: float
    se: float
    ci: Interval
    n_positive: int
    n_negative: int


def _label(value: object, name: str, index: int) -> int:
    """One 0/1 verdict as an int, or ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError(f"{name}[{index}] must be 0 or 1, got {value!r}")
    return value


def cohen_kappa(human: Sequence[int], judge: Sequence[int], *, level: float = 0.95) -> Kappa:
    """How far a judge agrees with a human beyond chance, with an interval and the prevalence-free PABAK.

    ``human[i]`` and ``judge[i]`` are the two raters' 0/1 verdicts on item i; the 2 x 2 table has
    ``table[h][j]`` items the human put in h and the judge in j.
    kappa = (p_o - p_e) / (1 - p_e) with p_e = sum_i r_i c_i (Cohen 1960,
    https://doi.org/10.1177/001316446002000104). `se` is the Fleiss-Cohen-Everitt large-sample SE
    for kappa != 0 (Fleiss, Cohen & Everitt 1969, https://doi.org/10.1037/h0028106):
    var = [sum_i p_ii (1 - (r_i + c_i)(1 - kappa))^2 + (1 - kappa)^2 sum_{i != j} p_ij (c_i + r_j)^2
    - (kappa - p_e (1 - kappa))^2] / (n (1 - p_e)^2); the interval is kappa -/+ z se clipped to [-1, 1].
    `se_null` is their SE under kappa = 0: [p_e + p_e^2 - sum_i r_i c_i (r_i + c_i)] / (n (1 - p_e)^2).
    PABAK = 2 p_o - 1 for 2 x 2 (Byrt, Bishop & Carlin 1993, https://doi.org/10.1016/0895-4356(93)90018-V,
    fetched 2026-11-09); for k categories (k p_o - 1) / (k - 1) `[INFERENCE]` (the same idea: chance = 1/k).

    Args:
        human: The human's 0/1 verdict per item.
        judge: The judge's 0/1 verdict per item, paired with ``human``.
        level: Confidence level of the interval, in (0, 1).

    Returns:
        Kappa with its interval; ``n`` is the number of pairs.

    Raises:
        ValueError: On a label that is not 0 or 1, unequal lengths, no pairs, or when both raters
            used one category only (chance agreement 1, kappa undefined).
    """
    if len(human) != len(judge):
        raise ValueError(f"human and judge must pair item for item, got {len(human)} and {len(judge)} verdicts")
    k = 2
    table = [[0] * k for _ in range(k)]
    for index, (h, j) in enumerate(zip(human, judge)):
        table[_label(h, "human", index)][_label(j, "judge", index)] += 1
    n = sum(sum(row) for row in table)
    if n == 0:
        raise ValueError("table is empty")
    z = two_sided_z(level)
    p = [[count / n for count in row] for row in table]
    rows = [sum(p[i]) for i in range(k)]
    cols = [sum(p[i][j] for i in range(k)) for j in range(k)]
    p_o = sum(p[i][i] for i in range(k))
    p_e = sum(rows[i] * cols[i] for i in range(k))
    if p_e >= 1.0:
        raise ValueError("chance agreement is 1 (both raters used one category); kappa is undefined")
    kappa = (p_o - p_e) / (1.0 - p_e)
    spread = 1.0 - kappa
    diagonal = sum(p[i][i] * (1.0 - (rows[i] + cols[i]) * spread) ** 2 for i in range(k))
    off = spread**2 * sum(p[i][j] * (cols[i] + rows[j]) ** 2 for i in range(k) for j in range(k) if i != j)
    variance = (diagonal + off - (kappa - p_e * spread) ** 2) / (n * (1.0 - p_e) ** 2)
    se = math.sqrt(max(variance, 0.0))
    null_variance = (p_e + p_e**2 - sum(rows[i] * cols[i] * (rows[i] + cols[i]) for i in range(k))) / (
        n * (1.0 - p_e) ** 2
    )
    return Kappa(
        kappa=kappa,
        se=se,
        ci=Interval(lower=max(kappa - z * se, -1.0), upper=min(kappa + z * se, 1.0), level=level),
        se_null=math.sqrt(max(null_variance, 0.0)),
        observed_agreement=p_o,
        chance_agreement=p_e,
        pabak=(k * p_o - 1.0) / (k - 1),
        n=n,
    )


def krippendorff_alpha(
    units: Sequence[Sequence[str | int | float | None]], level: KrippendorffLevel = "nominal"
) -> KrippendorffAlpha:
    """How far several raters agree beyond chance, allowing missing ratings (None) per unit.

    Calls the vendored `krippendorff_alpha` (Krippendorff 2011, "Computing Krippendorff's Alpha-Reliability",
    https://repository.upenn.edu/items/034a6030-c584-4d14-9d3d-7b7e8d16df20, fetched 2026-11-09). Units with
    fewer than two non-missing ratings do not count; `alpha` is None when no unit counts.

    Args:
        units: Per unit, one rating per rater (``None`` for a missing rating).
        level: The measurement level of the ratings.

    Returns:
        Alpha, the number of units that counted, and the level.
    """
    rated: list[list[str | int | float]] = []
    for unit_index, unit in enumerate(units):
        ratings = [rating for rating in unit if rating is not None]
        for rating in ratings:
            if isinstance(rating, float) and not math.isfinite(rating):
                raise NonFiniteScoreError(f"units[{unit_index}] holds {rating!r}; drop or reject non-finite ratings first")
        if len(ratings) >= 2:
            rated.append(ratings)
    if not rated:
        return KrippendorffAlpha(alpha=None, units=0, level=level)
    scores = [SampleScore(score=Score(value=ratings), sample_id=i) for i, ratings in enumerate(rated)]
    alpha = _vendored_krippendorff_alpha(level=level)(scores)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError(f"krippendorff_alpha returned {alpha!r}, expected a number")
    if not math.isfinite(alpha):
        raise NonFiniteScoreError(f"krippendorff_alpha is {alpha!r} on these units")
    return KrippendorffAlpha(alpha=float(alpha), units=len(rated), level=level)


def auc_hanley_mcneil(positive_scores: Sequence[float], negative_scores: Sequence[float], level: float = 0.95) -> Auc:
    """How well a score ranks positives above negatives (AUC), with the Hanley-McNeil standard error.

    AUC = share of (positive, negative) pairs ranked right, ties counting 1/2 (Mann-Whitney).
    SE^2 = [A(1 - A) + (n1 - 1)(Q1 - A^2) + (n0 - 1)(Q2 - A^2)] / (n1 n0), Q1 = A / (2 - A),
    Q2 = 2 A^2 / (1 + A); interval A -/+ z SE clipped to [0, 1]. Hanley & McNeil 1982,
    https://pubmed.ncbi.nlm.nih.gov/7063747/ (fetched 2026-11-09).

    Args:
        positive_scores: Scores of the positive items.
        negative_scores: Scores of the negative items.
        level: Confidence level of the interval, in (0, 1).

    Returns:
        The AUC with its standard error and interval.
    """
    positives = finite_values(positive_scores, "positive_scores")
    negatives = finite_values(negative_scores, "negative_scores")
    if not positives or not negatives:
        raise ValueError("need at least one positive and one negative score")
    z = two_sided_z(level)
    n1, n0 = len(positives), len(negatives)
    wins = sum(1.0 if s > t else 0.5 if s == t else 0.0 for s in positives for t in negatives)
    auc = wins / (n1 * n0)
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    variance = (auc * (1.0 - auc) + (n1 - 1) * (q1 - auc * auc) + (n0 - 1) * (q2 - auc * auc)) / (n1 * n0)
    se = math.sqrt(max(variance, 0.0))
    return Auc(
        auc=auc,
        se=se,
        ci=Interval(lower=max(auc - z * se, 0.0), upper=min(auc + z * se, 1.0), level=level),
        n_positive=n1,
        n_negative=n0,
    )
