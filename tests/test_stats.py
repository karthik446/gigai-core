"""Typed statistics (`gigai_core.stats`) on closed-form fixtures.

Every expected number is derived by hand in the comment next to it; nothing here imports scipy,
statsmodels or numpy. Relocated from gigai 0.3.2 `tests/behaviors/evals/test_stats.py`, with the
dict lookups turned into attribute reads, plus the frozen-model, degenerate-input and pass@k tests.
"""

from __future__ import annotations

import importlib
import inspect
import math
import sys
import time
from collections.abc import Callable
from fractions import Fraction
from statistics import NormalDist, pstdev
from typing import cast

import pytest
from pydantic import ValidationError

import gigai_core.stats as stats
from gigai_core.models import CoreModel
from gigai_core.stats import (
    Auc,
    BenjaminiHochberg,
    ClusterBootstrap,
    Holm,
    Interval,
    Kappa,
    KrippendorffAlpha,
    McNemarExact,
    MdeContinuous,
    MdeRow,
    MinimumDetectableDifference,
    NoiseBound,
    NonFiniteScoreError,
    PassK,
    TooFewClustersError,
    Tost,
    VarianceComponents,
    WilcoxonSignedRank,
    WilsonInterval,
)

EXACT = 1e-12
Z95 = NormalDist().inv_cdf(0.975)  # 1.959964...


def _labels(table: list[list[int]]) -> tuple[list[int], list[int]]:
    """The (human, judge) 0/1 label sequences whose 2 x 2 confusion table is `table[human][judge]`."""
    human = [h for h in (0, 1) for j in (0, 1) for _ in range(table[h][j])]
    judge = [j for h in (0, 1) for j in (0, 1) for _ in range(table[h][j])]
    return human, judge


# --- McNemar ------------------------------------------------------------------------------


def test_exact_mcnemar_b3_c9_gives_p_0_146_and_the_mid_p() -> None:
    result = stats.mcnemar_exact(3, 9)

    # m = 12 discordant pairs, X ~ Bin(12, 1/2): P(X <= 3) = (1 + 12 + 66 + 220) / 4096 = 299 / 4096.
    assert result.p_exact == pytest.approx(2 * 299 / 4096, abs=EXACT)
    assert abs(result.p_exact - 0.146) <= 5e-4  # PLAN acceptance: 0.14599...
    # mid-p = 2 * (P(X <= 3) - P(X = 3) / 2) = 2 * (299 - 110) / 4096 = 378 / 4096 = 0.09228515625.
    assert result.p_mid == pytest.approx(378 / 4096, abs=EXACT)
    assert result.discordant == 12
    assert result.significant is False  # gate on the exact p (0.146 > 0.05), not the mid-p
    assert stats.mcnemar_exact(9, 3).p_exact == result.p_exact


def test_exact_mcnemar_edge_cases() -> None:
    # No discordant pairs: no evidence either way.
    assert stats.mcnemar_exact(0, 0).p_exact == 1.0
    # b = c = 5, m = 10: 2 * P(X <= 5) = 2 * 638 / 1024 > 1 -> capped at 1; mid-p = 2 * (638 - 126) / 1024 = 1.
    balanced = stats.mcnemar_exact(5, 5)
    assert (balanced.p_exact, balanced.p_mid) == (1.0, 1.0)
    # b = 0, c = 10: 2 * 1 / 1024 = 0.001953125 -> significant; mid-p = (2 - 1) / 1024.
    lopsided = stats.mcnemar_exact(0, 10)
    assert lopsided.p_exact == pytest.approx(2 / 1024, abs=EXACT)
    assert lopsided.p_mid == pytest.approx(1 / 1024, abs=EXACT)
    assert lopsided.significant is True
    with pytest.raises(ValueError):
        stats.mcnemar_exact(-1, 3)
    with pytest.raises(ValueError):
        stats.mcnemar_exact(cast(int, True), 3)  # a bool is not a count; cast reaches the runtime check


# --- Wilson and noise bound (F-read :93-94) -------------------------------------------------


def test_wilson_half_width_at_n60_p085_is_0_090() -> None:
    result = stats.wilson_interval(51, 60)

    assert result.proportion == 0.85
    assert abs(result.half_width - 0.090) <= 0.001  # F-read :94
    # Closed form: z / (1 + z^2/n) * sqrt(p(1-p)/n + z^2/(4 n^2)) = 1.95996 / 1.06402 * 0.048906 = 0.090087.
    n, p = 60, 0.85
    closed = Z95 / (1 + Z95**2 / n) * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n))
    assert result.half_width == pytest.approx(closed, abs=EXACT)
    assert result.ci.level == 0.95
    assert result.ci.upper - result.ci.lower == pytest.approx(2 * closed, abs=EXACT)
    assert (result.ci.stderr, result.ci.resamples) == (None, None)
    with pytest.raises(ValueError):
        stats.wilson_interval(61, 60)
    with pytest.raises(TypeError):
        stats.wilson_interval(cast(int, 5.0), 60)  # cast reaches the runtime check on an untyped caller


def test_noise_bound_is_0_335() -> None:
    result = stats.noise_bound(0.167, 0.595, 48)

    # F-read :93: 0.167 + 1.96 * 0.595 / sqrt(48) = 0.167 + 0.16832 = 0.33532.
    assert abs(result.ci.upper - 0.335) <= 0.001
    assert result.half_width == pytest.approx(Z95 * 0.595 / math.sqrt(48), abs=EXACT)
    assert result.ci.lower == pytest.approx(0.167 - result.half_width, abs=EXACT)


# --- Holm and Benjamini-Hochberg ------------------------------------------------------------


def test_holm_on_the_plan_fixture_rejects_all_three() -> None:
    # PLAN acceptance says "rejects the first two"; that is wrong for Holm. Thresholds alpha / (m - j + 1):
    # 0.01 <= 0.05/3 = 0.0167 yes; 0.02 <= 0.05/2 = 0.025 yes; 0.03 <= 0.05/1 = 0.05 yes -> all three.
    # Adjusted: 3*0.01 = 0.03; max(0.03, 2*0.02) = 0.04; max(0.04, 1*0.03) = 0.04.
    result = stats.holm([0.01, 0.02, 0.03])
    assert result.reject == (True, True, True)
    assert result.adjusted == pytest.approx((0.03, 0.04, 0.04), abs=EXACT)


def test_holm_rejects_exactly_the_first_two() -> None:
    # 0.01 <= 0.0167, 0.02 <= 0.025, 0.06 > 0.05 -> first two. Input order is kept.
    assert stats.holm([0.01, 0.02, 0.06]).reject == (True, True, False)
    assert stats.holm([0.06, 0.01, 0.02]).reject == (False, True, True)


def test_benjamini_hochberg_step_up_rescues_a_value_holm_stops_at() -> None:
    p = [0.01, 0.028, 0.035, 0.6]
    # BH thresholds j * 0.05 / 4: 0.0125, 0.025, 0.0375, 0.05. j=1 passes, j=2 (0.028 > 0.025) fails,
    # j=3 (0.035 <= 0.0375) passes, j=4 fails -> largest passing j is 3 -> reject the first three.
    bh = stats.benjamini_hochberg(p)
    assert bh.reject == (True, True, True, False)
    # Adjusted, from the top: 0.6*4/4 = 0.6; 0.035*4/3 = 0.046667; min(0.028*4/2 = 0.056, 0.046667); min(0.04, ...).
    assert bh.adjusted == pytest.approx((0.04, 0.035 * 4 / 3, 0.035 * 4 / 3, 0.6), abs=EXACT)
    # Holm on the same p: 0.01 <= 0.0125 yes; 0.028 > 0.05/3 = 0.0167 -> stop.
    assert stats.holm(p).reject == (True, False, False, False)
    with pytest.raises(ValueError):
        stats.holm([0.01, 1.5])


# --- TOST ----------------------------------------------------------------------------------


def test_tost_margin_0_05_passes_at_se_0_01_and_fails_at_se_0_05() -> None:
    narrow = stats.tost(0.0, 0.01, 0.05)
    # (0 + 0.05) / 0.01 = 5 -> p = P(Z >= 5) = 2.87e-7 on both sides; 90 % CI = -/+ 1.6449 * 0.01 inside -/+ 0.05.
    assert narrow.equivalent is True
    assert narrow.p_value == pytest.approx(NormalDist().cdf(-5.0), abs=EXACT)
    assert narrow.ci.level == pytest.approx(0.90, abs=EXACT)

    wide = stats.tost(0.0, 0.05, 0.05)
    # (0 + 0.05) / 0.05 = 1 -> p = P(Z >= 1) = 0.158655 > 0.05; 90 % CI = -/+ 0.0822 crosses the margin.
    assert wide.equivalent is False
    assert wide.p_value == pytest.approx(0.158655, abs=1e-6)
    assert wide.ci.upper == pytest.approx(NormalDist().inv_cdf(0.95) * 0.05, abs=EXACT)


def test_tost_checks_both_one_sided_tests_away_from_zero() -> None:
    # d = 0.04, se 0.01, margin 0.05: p_lower = P(Z >= (0.04 + 0.05) / 0.01 = 9) ~ 1e-19, but
    # p_upper = P(Z <= (0.04 - 0.05) / 0.01 = -1) = 0.158655 -> TOST p = max = 0.158655, not equivalent
    # (90 % CI 0.04 + 0.0164 = 0.0564 crosses +0.05). Dropping the upper test would call this equivalent.
    high = stats.tost(0.04, 0.01, 0.05)
    assert high.p_upper == pytest.approx(NormalDist().cdf(-1.0), abs=EXACT)
    assert high.p_value == pytest.approx(0.158655, abs=1e-6)
    assert high.equivalent is False
    # Mirror image: d = -0.04 fails on the lower side.
    low = stats.tost(-0.04, 0.01, 0.05)
    assert low.p_lower == pytest.approx(NormalDist().cdf(-1.0), abs=EXACT)
    assert low.equivalent is False
    # A wider margin passes the same d: (0.04 - 0.08) / 0.01 = -4 -> p = P(Z <= -4) = 3.2e-5.
    assert stats.tost(0.04, 0.01, 0.08).equivalent is True
    for margin in (0.0, -0.05):
        with pytest.raises(ValueError):
            stats.tost(0.0, 0.01, margin)
    with pytest.raises(ValueError):
        stats.tost(0.0, 0.01, 0.05, alpha=0.5)


# --- Cohen's kappa, Fleiss-Cohen SE, PABAK ---------------------------------------------------


def test_kappa_at_90_percent_agreement_and_50_percent_prevalence_is_0_8_with_the_fleiss_cohen_se() -> None:
    # 100 items: 45 both-0, 5 human-0/judge-1, 5 human-1/judge-0, 45 both-1.
    human, judge = _labels([[45, 5], [5, 45]])
    result = stats.cohen_kappa(human, judge)

    assert result.n == 100
    # p_o = 0.9, marginals all 0.5 -> p_e = 0.5, kappa = (0.9 - 0.5) / 0.5 = 0.8.
    assert result.observed_agreement == pytest.approx(0.9, abs=EXACT)
    assert result.chance_agreement == pytest.approx(0.5, abs=EXACT)
    assert result.kappa == pytest.approx(0.8, abs=EXACT)
    # Fleiss-Cohen-Everitt 1969 non-null variance, n = 100, 1 - kappa = 0.2, r_i + c_i = 1:
    #   diagonal  sum p_ii (1 - (r_i + c_i)(1 - kappa))^2 = 0.9 * 0.8^2          = 0.576
    #   off-diag  (1 - kappa)^2 sum_{i!=j} p_ij (c_i + r_j)^2 = 0.04 * (0.05 + 0.05) = 0.004
    #   minus     (kappa - p_e (1 - kappa))^2 = (0.8 - 0.1)^2                      = 0.49
    #   var = (0.576 + 0.004 - 0.49) / (100 * 0.5^2) = 0.09 / 25 = 0.0036 -> SE = 0.06.
    assert result.se == pytest.approx(0.06, abs=EXACT)
    assert result.ci.lower == pytest.approx(0.8 - Z95 * 0.06, abs=EXACT)
    assert result.ci.upper == pytest.approx(0.8 + Z95 * 0.06, abs=EXACT)
    assert result.ci.level == 0.95
    # Null SE: (p_e + p_e^2 - sum r_i c_i (r_i + c_i)) / (n (1 - p_e)^2) = (0.5 + 0.25 - 0.5) / 25 = 0.01 -> 0.1.
    assert result.se_null == pytest.approx(0.1, abs=EXACT)
    assert result.pabak == pytest.approx(0.8, abs=EXACT)  # 2 * 0.9 - 1
    # The pairing is what counts, not the order the items come in.
    assert stats.cohen_kappa(human[::-1], judge[::-1]) == result


def test_pabak_shows_the_prevalence_effect_kappa_hides() -> None:
    result = stats.cohen_kappa(*_labels([[80, 5], [5, 10]]))

    # Same 90 % agreement at 85 % prevalence: p_e = 0.85^2 + 0.15^2 = 0.745,
    # kappa = (0.9 - 0.745) / 0.255 = 31 / 51 = 0.6078; PABAK = 2 * 0.9 - 1 = 0.8 is unchanged.
    assert result.kappa == pytest.approx(31 / 51, abs=EXACT)
    assert result.pabak == pytest.approx(0.8, abs=EXACT)
    # Perfect agreement: kappa 1, variance (1 + 0 - 1) / ... = 0.
    perfect = stats.cohen_kappa(*_labels([[30, 0], [0, 20]]))
    assert (perfect.kappa, perfect.se) == (pytest.approx(1.0, abs=EXACT), pytest.approx(0.0, abs=1e-9))


def test_kappa_refuses_degenerate_or_malformed_labels() -> None:
    # Both raters used one category: p_e = 1 and kappa is undefined (A3b's Agreement maps this to None).
    with pytest.raises(ValueError, match="chance agreement is 1"):
        stats.cohen_kappa([1, 1, 1], [1, 1, 1])
    # One rater used one category: kappa is defined (0 here: p_o = p_e = 0.5) and the interval is clipped.
    one_sided = stats.cohen_kappa([1, 1, 1, 1], [0, 1, 0, 1])
    assert one_sided.kappa == pytest.approx(0.0, abs=EXACT)
    assert -1.0 <= one_sided.ci.lower <= one_sided.ci.upper <= 1.0
    with pytest.raises(ValueError, match="table is empty"):
        stats.cohen_kappa([], [])
    with pytest.raises(ValueError, match="pair item for item"):
        stats.cohen_kappa([0, 1], [0])
    with pytest.raises(ValueError, match=r"judge\[1\] must be 0 or 1"):
        stats.cohen_kappa([0, 1], [0, 2])
    with pytest.raises(ValueError, match=r"human\[0\] must be 0 or 1"):
        stats.cohen_kappa([cast(int, True), 1], [0, 1])  # cast reaches the runtime check on an untyped caller


# --- MDE (D §3.8a) and the continuous MDE (F-read :91) ---------------------------------------


@pytest.mark.parametrize(
    ("discordance", "delta", "n"),
    [(0.10, 0.10, 77), (0.15, 0.10, 116), (0.20, 0.05, 626), (0.20, 0.10, 155), (0.30, 0.10, 234)],
)
def test_mde_sample_size_reproduces_the_plan_rows(discordance: float, delta: float, n: int) -> None:
    # e.g. 0.10/0.10: (1.959964 * sqrt(0.1) + 0.841621 * sqrt(0.1 - 0.01))^2 / 0.01 = 0.872281^2 / 0.01 = 76.09 -> 77.
    assert stats.mde_sample_size(discordance, delta) == n


def test_mde_table_reproduces_d_3_8a_whole() -> None:
    deltas = [0.03, 0.05, 0.10, 0.15, 0.20]
    expected: dict[float, list[int | None]] = {  # D-ablation-stats.md:131-137; None where delta > p_d
        0.05: [434, 155, None, None, None],
        0.10: [870, 312, 77, None, None],
        0.15: [1306, 469, 116, 50, None],
        0.20: [1742, 626, 155, 68, 37],
        0.30: [2614, 940, 234, 103, 57],
    }
    rows = stats.mde_table(list(expected), deltas)
    got: dict[float, list[int | None]] = {}
    for row in rows:
        got.setdefault(row.discordance, []).append(row.n)
    assert got == expected


def test_minimum_detectable_difference_inverts_connor() -> None:
    # Continuous n(0.10) at p_d 0.10 is 76.09 < 77, so the MDE at n = 77 is just under 0.10 and needs 77 items.
    result = stats.minimum_detectable_difference(77, 0.10)
    assert result.mde is not None
    assert 0.099 < result.mde < 0.10
    assert stats.mde_sample_size(0.10, result.mde) == 77
    # Even delta = p_d = 0.10 needs 77 items, so 10 items detect nothing.
    assert stats.minimum_detectable_difference(10, 0.10).mde is None
    with pytest.raises(ValueError):
        stats.minimum_detectable_difference(0, 0.10)


def test_mde_table_validates_every_argument_up_front() -> None:
    # No row has 0 < delta <= p_d, so no row reaches mde_sample_size; alpha and power are still checked.
    with pytest.raises(NonFiniteScoreError):
        stats.mde_table([0.1], [0.2], alpha=math.nan)
    with pytest.raises(NonFiniteScoreError):
        stats.mde_table([], [], power=math.inf)
    with pytest.raises(NonFiniteScoreError):
        stats.mde_table([], [0.05, math.nan])  # deltas checked even with no discordances
    with pytest.raises(ValueError):
        stats.mde_table([0.1], [-0.05])
    with pytest.raises(ValueError):
        stats.mde_table([1.5], [0.05])
    assert stats.mde_table([], [0.05]) == ()


def test_mde_continuous_reproduces_f_read_verdict_mde() -> None:
    # F-read :91: Var = 0.0906 + 2 * 0.0605 / k, MDE = 2.80 * sqrt(Var) / sqrt(48) -> k=1 0.1859, k=3 0.1462.
    for k, expected in ((1, 0.1859), (3, 0.1462)):
        sd = math.sqrt(0.0906 + 2 * 0.0605 / k)
        assert abs(stats.mde_continuous(48, sd).mde - expected) <= 0.001  # 2.8016 here vs F-read's rounded 2.80


# --- Minimum-cluster guard --------------------------------------------------------------------


def _clustered(n_clusters: int, per: int = 2) -> tuple[list[float], list[float], list[int]]:
    a = [0.0] * (n_clusters * per)
    b = [float(i % per) for i in range(n_clusters * per)]
    return a, b, [i // per for i in range(n_clusters * per)]


def test_guard_raises_on_9_clusters_and_passes_on_10() -> None:
    assert stats.MIN_CLUSTERS == 10
    with pytest.raises(TooFewClustersError):
        stats.require_min_clusters(9)
    stats.require_min_clusters(10)
    with pytest.raises(TooFewClustersError):
        stats.paired_cluster_bootstrap(*_clustered(9), resamples=10)
    assert stats.paired_cluster_bootstrap(*_clustered(10), resamples=10).clusters == 10
    stats.require_min_clusters(cast(int, 10.0))  # a whole-number float is a count (cast: untyped caller)
    for bad in (9.5, "10", True):
        with pytest.raises(TypeError):
            stats.require_min_clusters(cast(int, bad))  # cast reaches the runtime check on an untyped caller


# --- Paired cluster bootstrap -------------------------------------------------------------------


def test_cluster_bootstrap_resamples_clusters_not_items() -> None:
    # Every cluster holds d = [0, 1]: each cluster's sum is 1 over 2 items, so any resample of whole clusters
    # has mean exactly 10 / 20 = 0.5 and SE exactly 0. An item-level bootstrap would see SE ~ 0.5 / sqrt(20) = 0.11.
    result = stats.paired_cluster_bootstrap(*_clustered(10), resamples=500, seed=3)
    assert (result.difference, result.ci.lower, result.ci.upper, result.ci.stderr) == (0.5, 0.5, 0.5, 0.0)
    assert (result.items, result.clusters, result.seed) == (20, 10, 3)
    assert (result.ci.level, result.ci.resamples) == (0.95, 500)


def test_cluster_bootstrap_se_matches_the_cluster_mean_formula() -> None:
    # 10 clusters of 3 items, cluster mean differences 0.0, 0.1, ..., 0.9 (items m - 0.05, m, m + 0.05).
    # Equal sizes -> the resampled mean is the mean of 10 resampled cluster means, whose SE tends to
    # pstdev(cluster means) / sqrt(10) = sqrt(0.0825) / sqrt(10) = 0.090830 (Monte Carlo SE of that at
    # B = 10,000 is about 0.0908 / sqrt(2 * 10,000) = 6.4e-4; tolerance 3e-3 ~ 5 MC SEs).
    means = [i / 10 for i in range(10)]
    b = [m + offset for m in means for offset in (-0.05, 0.0, 0.05)]
    a = [0.0] * len(b)
    clusters = [i // 3 for i in range(len(b))]
    result = stats.paired_cluster_bootstrap(a, b, clusters, resamples=10_000, seed=0)

    assert result.difference == pytest.approx(0.45, abs=EXACT)
    assert result.ci.stderr is not None
    assert abs(result.ci.stderr - pstdev(means) / math.sqrt(10)) <= 3e-3
    assert result.ci.lower < 0.45 < result.ci.upper
    assert abs((result.ci.upper - result.ci.lower) / 2 - Z95 * 0.090830) <= 0.02  # ~normal percentile interval
    assert stats.paired_cluster_bootstrap(a, b, clusters, resamples=10_000, seed=0) == result  # seeded
    assert stats.paired_cluster_bootstrap(a, b, clusters, resamples=10_000, seed=1).ci.stderr != result.ci.stderr


def test_cluster_bootstrap_caps_resamples_and_checks_shapes() -> None:
    a, b, clusters = _clustered(10)
    assert stats.MAX_BOOTSTRAP_RESAMPLES == 10_000
    for bad in (0, 10_001):
        with pytest.raises(ValueError):
            stats.paired_cluster_bootstrap(a, b, clusters, resamples=bad)
    for not_an_int in (10.0, True, "10"):
        with pytest.raises(ValueError, match="resamples must be an integer"):
            # cast reaches the runtime check on an untyped caller
            stats.paired_cluster_bootstrap(a, b, clusters, resamples=cast(int, not_an_int))
    with pytest.raises(ValueError):
        stats.paired_cluster_bootstrap(a, b[:-1], clusters)
    with pytest.raises(ValueError):
        stats.paired_cluster_bootstrap(a, b, clusters[:-1])
    for bad_seed in (1.5, "0", None):
        with pytest.raises(TypeError):
            # cast reaches the runtime check on an untyped caller
            stats.paired_cluster_bootstrap(a, b, clusters, resamples=10, seed=cast(int, bad_seed))


def test_cluster_bootstrap_speed_300_items_30_clusters_2000_resamples() -> None:
    # PLAN V3 risk 2 guesses < 5 s.
    a = [0.0] * 300
    b = [((i * 37) % 11) / 10 for i in range(300)]
    clusters = [i // 10 for i in range(300)]
    started = time.perf_counter()
    stats.paired_cluster_bootstrap(a, b, clusters, resamples=2000, seed=0)
    assert time.perf_counter() - started < 5.0


# --- Wilcoxon signed-rank -----------------------------------------------------------------------


def test_wilcoxon_exact_small_n() -> None:
    # d = 1..5 all positive: W+ = 15, W- = 0; only the empty set has rank sum <= 0 -> p = 2 * 1 / 32 = 0.0625.
    up = stats.wilcoxon_signed_rank([0.0] * 5, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert (up.method, up.w_plus, up.w_minus, up.statistic, up.z) == ("exact", 15.0, 0.0, 0.0, None)
    assert up.p_value == pytest.approx(0.0625, abs=EXACT)
    # d = -1, 2, 3, 4, 5: W- = 1; subsets of {1..5} with sum <= 1 are {} and {1} -> p = 2 * 2 / 32 = 0.125.
    assert stats.wilcoxon_signed_rank([0.0] * 5, [-1.0, 2.0, 3.0, 4.0, 5.0]).p_value == pytest.approx(0.125, abs=EXACT)
    # Zero differences are dropped (Wilcoxon's rule): same p, zeros counted.
    padded = stats.wilcoxon_signed_rank([0.0] * 7, [1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0])
    assert (padded.n, padded.zeros, padded.p_value) == (5, 2, up.p_value)


def test_wilcoxon_exact_with_tied_ranks() -> None:
    # d = 1, 1, 2, -3: |d| ranks 1.5, 1.5, 3, 4 (doubled 3, 3, 6, 8); W+ = 1.5 + 1.5 + 3 = 6, W- = 4.
    # The 16 sign assignments give W+ (doubled) sums 0,3,3,6,6,8,9,9,11,11,12,14,14,17,17,20;
    # six are <= min(12, 8) = 8 -> p = 2 * 6 / 16 = 0.75.
    result = stats.wilcoxon_signed_rank([0.0] * 4, [1.0, 1.0, 2.0, -3.0])
    assert (result.w_plus, result.w_minus, result.method) == (6.0, 4.0, "exact")
    assert result.p_value == pytest.approx(0.75, abs=EXACT)


def test_wilcoxon_normal_approximation_with_continuity_and_ties() -> None:
    # n = 25 distinct positive d: W+ = 325, mean 25*26/4 = 162.5, var 25*26*51/24 = 1381.25;
    # z = (325 - 162.5 - 0.5) / sqrt(1381.25) = 162 / 37.1652 = 4.35892.
    distinct = stats.wilcoxon_signed_rank([0.0] * 25, [float(i) for i in range(1, 26)])
    assert distinct.method == "normal"
    assert distinct.z == pytest.approx(162 / math.sqrt(1381.25), abs=EXACT)
    assert distinct.p_value == pytest.approx(2 * NormalDist().cdf(-162 / math.sqrt(1381.25)), abs=EXACT)
    # Ties: d = ten 1s and fifteen 2s: mid-ranks 5.5 and 18, W+ = 55 + 270 = 325 again;
    # tie correction (10^3 - 10 + 15^3 - 15) / 48 = 4350 / 48 = 90.625 -> var 1290.625, z = 162 / 35.9253.
    tied = stats.wilcoxon_signed_rank([0.0] * 25, [1.0] * 10 + [2.0] * 15)
    assert tied.w_plus == 325.0
    assert tied.z == pytest.approx(162 / math.sqrt(1290.625), abs=EXACT)


def test_wilcoxon_switches_from_exact_to_normal_after_20() -> None:
    assert stats.WILCOXON_EXACT_MAX_N == 20
    assert stats.wilcoxon_signed_rank([0.0] * 20, [float(i) for i in range(1, 21)]).method == "exact"
    assert stats.wilcoxon_signed_rank([0.0] * 21, [float(i) for i in range(1, 22)]).method == "normal"
    assert stats.wilcoxon_signed_rank([1.0, 2.0], [1.0, 2.0]).p_value == 1.0  # all zero differences


# --- Hanley-McNeil AUC SE -----------------------------------------------------------------------


def test_auc_hanley_mcneil_on_a_hand_fixture_with_a_tie() -> None:
    # Positives 2, 3, 4 vs negatives 1, 2: pairs (2,1) 1, (2,2) 1/2, the other four 1 -> A = 5.5 / 6 = 11/12.
    # Q1 = A / (2 - A) = 11/13; Q2 = 2 A^2 / (1 + A) = 121/138;
    # SE^2 = [11/144 + 2 (11/13 - 121/144) + 1 (121/138 - 121/144)] / 6 = (0.0763889 + 0.0117521 + 0.0365338) / 6
    #      = 0.0207791 -> SE = 0.144150.
    a = Fraction(11, 12)
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = math.sqrt((a * (1 - a) + 2 * (q1 - a * a) + 1 * (q2 - a * a)) / 6)
    result = stats.auc_hanley_mcneil([2.0, 3.0, 4.0], [1.0, 2.0])
    assert result.auc == pytest.approx(11 / 12, abs=EXACT)
    assert result.se == pytest.approx(se, abs=EXACT)
    assert abs(result.se - 0.144150) <= 5e-7
    assert result.ci.upper == 1.0  # 0.9167 + 1.96 * 0.1441 clipped to 1
    assert (result.n_positive, result.n_negative) == (3, 2)
    with pytest.raises(ValueError):
        stats.auc_hanley_mcneil([], [1.0])


# --- Variance components over repeats -----------------------------------------------------------


def test_variance_components_on_a_hand_fixture() -> None:
    # Items x repeats [[0, 1], [1, 1], [0, 0]]: item means 0.5, 1, 0; grand mean 0.5.
    # within = (0.25 + 0.25) / (3 * 1) = 1/6; var of item means (n - 1) = 0.5 / 2 = 0.25;
    # between = 0.25 - (1/6) / 2 = 1/6; var_mean = (1/6 + 1/12) / 3 = 1/12 (= 0.25 / 3);
    # optimal repeats at cost ratio 10 = sqrt(10 * (1/6) / (1/6)) = sqrt(10).
    result = stats.variance_components([[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]], cost_ratio=10.0)
    assert result.within_item == pytest.approx(1 / 6, abs=EXACT)
    assert result.between_item == pytest.approx(1 / 6, abs=EXACT)
    assert result.var_mean == pytest.approx(1 / 12, abs=EXACT)
    assert result.mean == 0.5
    assert result.optimal_repeats == pytest.approx(math.sqrt(10), abs=EXACT)
    assert (result.items, result.repeats) == (3, 2)


def test_variance_components_truncates_a_negative_between_item_estimate() -> None:
    # [[0, 1], [1, 0]]: item means 0.5, 0.5 -> var 0; within = 1 / (2 * 1) = 0.5; between = 0 - 0.25 -> 0.
    result = stats.variance_components([[0.0, 1.0], [1.0, 0.0]], cost_ratio=10.0)
    assert result.between_item == 0.0
    assert result.var_mean == pytest.approx(0.125, abs=EXACT)  # (0 + 0.5 / 2) / 2
    assert result.optimal_repeats is None
    with pytest.raises(ValueError):
        stats.variance_components([[0.0, 1.0], [1.0]])


def test_optimal_repeats_takes_cost_ratio_as_new_item_cost_in_repeats() -> None:
    # D-measurement §3.5: s_run^2 = 0.0605 (within), s_het^2 = 0.0906 (between); a new posting costs about
    # ten runs, so c = 10 and k* = sqrt(10 * 0.0605 / 0.0906) = 2.584 ~ 2.6 -> 3 repeats.
    # Two items, two repeats: rows [-s - e, -s + e] and [s - e, s + e] give within = 2 e^2 and
    # between = 2 s^2 - within / 2, so e^2 = 0.0605 / 2 and s^2 = (0.0906 + 0.0605 / 2) / 2.
    e, s = math.sqrt(0.0605 / 2), math.sqrt((0.0906 + 0.0605 / 2) / 2)
    scores = [[-s - e, -s + e], [s - e, s + e]]
    result = stats.variance_components(scores, cost_ratio=10.0)
    assert result.within_item == pytest.approx(0.0605, abs=1e-12)
    assert result.between_item == pytest.approx(0.0906, abs=1e-12)
    assert result.optimal_repeats is not None
    assert result.optimal_repeats == pytest.approx(math.sqrt(10 * 0.0605 / 0.0906), abs=1e-9)
    assert round(result.optimal_repeats, 1) == 2.6
    # The reciprocal reading (c = cost of a repeat relative to an item = 0.1) would give 0.26 repeats.
    assert stats.variance_components(scores, cost_ratio=0.1).optimal_repeats == pytest.approx(0.2584, abs=1e-4)


# --- Krippendorff wrapper calls the vendored metric -----------------------------------------------


def test_krippendorff_wrapper_calls_the_vendored_metric_and_drops_missing_ratings() -> None:
    units: list[list[str | None]] = [["a", "a", "b"], ["b", "b", "b"], ["a", "b", "a"], ["c", "c", "c"], ["a", "a", "a"]]
    # S-smoke / test_stats_lifted.py: nominal alpha = 43/71 = 0.605634.
    assert stats.krippendorff_alpha(units).alpha == pytest.approx(43 / 71, abs=EXACT)
    with_missing = stats.krippendorff_alpha(units + [[None, "a"]])  # one rating left: the unit does not count
    assert (with_missing.alpha, with_missing.units) == (pytest.approx(43 / 71, abs=EXACT), 5)
    assert stats.krippendorff_alpha([[None, None]]) == KrippendorffAlpha(alpha=None, units=0, level="nominal")
    assert stats.krippendorff_alpha([[1, 2], [2, 3]], level="interval").level == "interval"


# --- pass@k and pass^k (new in core) ------------------------------------------------------------


def test_pass_at_k_and_pass_hat_k_match_the_unbiased_estimators() -> None:
    # n = 10, c = 3, k = 2: pass@k = 1 - C(7, 2) / C(10, 2) = 1 - 21 / 45 = 8 / 15; pass^k = C(3, 2) / C(10, 2) = 1 / 15.
    assert stats.pass_at_k(10, 3, 2) == pytest.approx(float(Fraction(8, 15)), abs=EXACT)
    assert stats.pass_hat_k(10, 3, 2) == pytest.approx(float(Fraction(1, 15)), abs=EXACT)
    # The product form of Chen et al. 2021 gives the same number: 1 - prod_{i = n - c + 1..n} (1 - k / i).
    product = 1.0 - math.prod(1 - 2 / i for i in range(8, 11))
    assert stats.pass_at_k(10, 3, 2) == pytest.approx(product, abs=EXACT)
    # n = 5, c = 4, k = 3: only one failure, so some k-subsets still all pass: pass^k = C(4, 3) / C(5, 3) = 4 / 10.
    assert stats.pass_hat_k(5, 4, 3) == pytest.approx(0.4, abs=EXACT)
    result = stats.pass_k(10, 3, 2)
    assert result == PassK(n=10, c=3, k=2, pass_at_k=stats.pass_at_k(10, 3, 2), pass_hat_k=stats.pass_hat_k(10, 3, 2))


def test_pass_k_edges_are_exact() -> None:
    for n, k in ((1, 1), (7, 3), (20, 20)):
        assert stats.pass_at_k(n, 0, k) == 0.0
        assert stats.pass_hat_k(n, 0, k) == 0.0
        assert stats.pass_at_k(n, n, k) == 1.0
        assert stats.pass_hat_k(n, n, k) == 1.0
    # k = 1: both are the pass rate c / n.
    for n, c in ((4, 1), (10, 3), (9, 9)):
        assert stats.pass_at_k(n, c, 1) == pytest.approx(c / n, abs=EXACT)
        assert stats.pass_hat_k(n, c, 1) == pytest.approx(c / n, abs=EXACT)
    # More failures than k - 1 leave some all-fail subsets; fewer make pass@k exactly 1 (n = 6, c = 5, k = 2).
    assert stats.pass_at_k(6, 5, 2) == 1.0
    # pass@k grows and pass^k shrinks with k.
    at = [stats.pass_at_k(12, 5, k) for k in range(1, 13)]
    hat = [stats.pass_hat_k(12, 5, k) for k in range(1, 13)]
    assert at == sorted(at) and hat == sorted(hat, reverse=True)
    assert all(0.0 <= value <= 1.0 for value in at + hat)


@pytest.mark.parametrize(
    ("n", "c", "k", "argument"),
    [(5, 2, 6, "k"), (5, 6, 2, "c"), (5, 2, 0, "k"), (0, 0, 1, "n"), (5, -1, 1, "c")],
)
def test_pass_k_names_the_offending_argument(n: int, c: int, k: int, argument: str) -> None:
    for function in (stats.pass_at_k, stats.pass_hat_k, stats.pass_k):
        with pytest.raises(ValueError, match=argument):
            function(n, c, k)


# --- Non-finite input is refused (V3a DIFF.md: callers MUST drop or reject NaN/inf) ---------------


NON_FINITE = [math.nan, math.inf, -math.inf]


def _calls_with(bad: float) -> dict[str, Callable[[], object]]:
    a, b, clusters = _clustered(10)
    human, judge = _labels([[5, 1], [1, 5]])
    return {
        "paired_cluster_bootstrap": lambda: stats.paired_cluster_bootstrap(a, b[:-1] + [bad], clusters, resamples=10),
        "wilcoxon_signed_rank": lambda: stats.wilcoxon_signed_rank([bad, 0.0], [0.0, 1.0]),
        "holm": lambda: stats.holm([0.01, bad]),
        "benjamini_hochberg": lambda: stats.benjamini_hochberg([bad, 0.01]),
        "tost": lambda: stats.tost(bad, 0.01, 0.05),
        "tost_stderr": lambda: stats.tost(0.0, bad, 0.05),
        "tost_margin": lambda: stats.tost(0.0, 0.01, bad),
        "tost_alpha": lambda: stats.tost(0.0, 0.01, 0.05, alpha=bad),
        # cast: a float count only reaches require_min_clusters from an untyped caller
        "require_min_clusters": lambda: stats.require_min_clusters(cast(int, bad)),
        "paired_cluster_bootstrap_seed": lambda: stats.paired_cluster_bootstrap(
            a, b, clusters, resamples=10, seed=cast(int, bad)
        ),
        "mde_table_alpha_no_valid_row": lambda: stats.mde_table([0.1], [0.2], alpha=bad),
        "auc_hanley_mcneil": lambda: stats.auc_hanley_mcneil([0.9, bad], [0.1]),
        "variance_components": lambda: stats.variance_components([[0.0, bad], [1.0, 1.0]]),
        "noise_bound": lambda: stats.noise_bound(0.1, bad, 48),
        "mde_continuous": lambda: stats.mde_continuous(48, bad),
        "mde_table": lambda: stats.mde_table([0.1, bad], [0.05]),
        "mde_sample_size": lambda: stats.mde_sample_size(0.1, bad),
        "minimum_detectable_difference": lambda: stats.minimum_detectable_difference(48, bad),
        "mcnemar_exact_alpha": lambda: stats.mcnemar_exact(3, 9, alpha=bad),
        "krippendorff_alpha": lambda: stats.krippendorff_alpha([[0.1, bad], [0.2, 0.3]], level="interval"),
        "wilson_interval_level": lambda: stats.wilson_interval(5, 10, level=bad),
        "cohen_kappa_level": lambda: stats.cohen_kappa(human, judge, level=bad),
    }


@pytest.mark.parametrize("bad", NON_FINITE, ids=["nan", "inf", "-inf"])
def test_every_public_function_refuses_non_finite_input(bad: float) -> None:
    assert ValueError in NonFiniteScoreError.__mro__
    for name, call in _calls_with(bad).items():
        with pytest.raises(NonFiniteScoreError):
            call()
        assert name  # names the failing call in the traceback


# --- Frozen, finite result models ------------------------------------------------------------------


def _one_of_each() -> list[CoreModel]:
    a, b, clusters = _clustered(10)
    wilson = stats.wilson_interval(51, 60)
    return [
        wilson.ci,
        wilson,
        stats.paired_cluster_bootstrap(a, b, clusters, resamples=20),
        stats.mcnemar_exact(3, 9),
        stats.wilcoxon_signed_rank([0.0] * 25, [float(i) for i in range(1, 26)]),
        stats.holm([0.01, 0.2]),
        stats.benjamini_hochberg([0.01, 0.2]),
        stats.tost(0.0, 0.01, 0.05),
        stats.cohen_kappa(*_labels([[45, 5], [5, 45]])),
        stats.krippendorff_alpha([["a", "a"], ["a", "b"], ["b", "b"]]),
        stats.auc_hanley_mcneil([2.0, 3.0], [1.0]),
        stats.minimum_detectable_difference(77, 0.1),
        stats.mde_continuous(48, 0.46),
        stats.mde_table([0.1], [0.05, 0.2])[0],
        stats.variance_components([[0.0, 1.0], [1.0, 1.0]], cost_ratio=2.0),
        stats.noise_bound(0.167, 0.595, 48),
        stats.pass_k(10, 3, 2),
    ]


RESULT_TYPES: tuple[type[CoreModel], ...] = (
    Interval,
    WilsonInterval,
    ClusterBootstrap,
    McNemarExact,
    WilcoxonSignedRank,
    Holm,
    BenjaminiHochberg,
    Tost,
    Kappa,
    KrippendorffAlpha,
    Auc,
    MinimumDetectableDifference,
    MdeContinuous,
    MdeRow,
    VarianceComponents,
    NoiseBound,
    PassK,
)


@pytest.mark.parametrize("result", _one_of_each(), ids=lambda result: type(result).__name__)
def test_every_result_is_a_frozen_finite_core_model(result: CoreModel) -> None:
    assert isinstance(result, RESULT_TYPES)
    model = type(result)
    fields = result.model_dump()
    first = next(iter(fields))
    with pytest.raises(ValidationError):  # frozen
        setattr(result, first, fields[first])
    with pytest.raises(ValidationError):  # extra forbidden
        model.model_validate({**fields, "bogus": 1})
    float_fields = [name for name, value in fields.items() if isinstance(value, float)]
    for name in float_fields:
        for bad in NON_FINITE:
            with pytest.raises(ValidationError):  # no NaN or inf can be stored
                model.model_validate({**fields, name: bad})
    assert model.model_validate_json(result.model_dump_json()) == result


def test_each_result_type_is_produced_exactly_once() -> None:
    assert {type(result) for result in _one_of_each()} == set(RESULT_TYPES)
    assert len(RESULT_TYPES) == 17


def test_degenerate_inputs_give_none_fields_not_nan() -> None:
    # Zero spread: SE 0, interval collapsed, nothing non-finite.
    assert stats.variance_components([[1.0, 1.0], [1.0, 1.0]], cost_ratio=10.0).optimal_repeats is None
    assert stats.noise_bound(0.0, 0.0, 3).ci == Interval(lower=0.0, upper=0.0, level=0.95)
    assert stats.wilson_interval(0, 5).ci.lower == 0.0
    assert stats.wilson_interval(5, 5).ci.upper == 1.0
    assert stats.wilcoxon_signed_rank([1.0], [1.0]) == WilcoxonSignedRank(
        n=0, zeros=1, w_plus=0.0, w_minus=0.0, statistic=0.0, p_value=1.0, method="exact", z=None
    )
    assert stats.krippendorff_alpha([]).alpha is None
    assert stats.minimum_detectable_difference(1, 0.01).mde is None
    assert stats.holm([]) == Holm(reject=(), adjusted=(), alpha=0.05)
    assert stats.benjamini_hochberg([]) == BenjaminiHochberg(reject=(), adjusted=(), alpha=0.05)


# --- Docstrings and the public API -----------------------------------------------------------------


def _public_functions() -> dict[str, Callable[..., object]]:
    return {
        name: obj
        for name, obj in vars(stats).items()
        if inspect.isfunction(obj) and not name.startswith("_") and obj.__module__.startswith(stats.__name__ + ".")
    }


def test_every_public_function_has_a_docstring_with_a_citation() -> None:
    functions = _public_functions()
    assert len(functions) == 20  # 17 from 0.3.2 + pass_at_k, pass_hat_k, pass_k
    for name, function in functions.items():
        doc = inspect.getdoc(function) or ""
        assert doc.splitlines()[0].endswith("."), name
        assert "https://" in doc, name


def test_all_lists_exactly_the_public_api() -> None:
    functions = set(_public_functions())
    constants = {"MIN_CLUSTERS", "MAX_BOOTSTRAP_RESAMPLES", "WILCOXON_EXACT_MAX_N"}
    exceptions = {"NonFiniteScoreError", "TooFewClustersError"}
    models = {model.__name__ for model in RESULT_TYPES}
    assert len(stats.__all__) == len(set(stats.__all__)) == 42
    assert set(stats.__all__) == functions | constants | exceptions | models


def test_stats_imports_without_numpy(block_numpy: None, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [name for name in sys.modules if name == "gigai_core.stats" or name.startswith("gigai_core.stats.")]:
        monkeypatch.delitem(sys.modules, name)
    fresh = importlib.import_module("gigai_core.stats")
    assert sys.modules["numpy"] is None
    assert fresh.wilson_interval(51, 60).proportion == 0.85
