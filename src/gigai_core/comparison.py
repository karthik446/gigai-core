"""``compare``: two runs of one dataset paired item by item, with the paired tests and their corrections.

A comparison is only ever of B against A on the items both runs asked. It refuses (``ComparisonRefused``) when the
two runs measured different things: another dataset, other items, other clusters, another k, a grader at another
version, or too few clusters for a clustered interval. An item one grader could not grade drops out of both arms
for that metric and is counted, not hidden.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Literal

from .dataset import DatasetItem
from .models import CoreModel
from .records import Prereg
from .run import Run
from .stats import (
    MIN_CLUSTERS,
    Interval,
    WilcoxonSignedRank,
    benjamini_hochberg,
    holm,
    mcnemar_exact,
    mde_continuous,
    minimum_detectable_difference,
    noise_bound,
    paired_cluster_bootstrap,
    wilcoxon_signed_rank,
)

Correction = Literal["holm", "bh", "none"]
Test = Literal["mcnemar", "wilcoxon"]
RefusalCode = Literal[
    "different_dataset",
    "different_k",
    "duplicate_trial",
    "unequal_repeats",
    "different_items",
    "different_clusters",
    "too_few_clusters",
    "no_shared_metric",
    "metric_missing",
    "ambiguous_metric",
    "grader_version",
    "no_paired_items",
]
#: The seed of every bootstrap interval a comparison reports, so two reads of the same runs print the same CI.
BOOTSTRAP_SEED = 0
#: The family-wise level the prereg family is corrected at.
ALPHA = 0.05

__all__ = [
    "ALPHA",
    "BOOTSTRAP_SEED",
    "MIN_CLUSTERS",
    "Comparison",
    "ComparisonRefused",
    "Correction",
    "McNemar",
    "MetricComparison",
    "Pairing",
    "RefusalCode",
    "Test",
    "compare",
    "in_slice",
    "pair",
]


class ComparisonRefused(Exception):
    """Two runs that cannot be compared: ``code`` names the mismatch, the message says it in words.

    The message starts with ``cannot compare:`` as 0.3.2 printed it; a gate turns it into ``cannot_compare``.
    """

    def __init__(self, code: RefusalCode, reason: str) -> None:
        """Name the mismatch (``code``) and spell it out (``reason``)."""
        super().__init__(f"cannot compare: {reason}")
        self.code: RefusalCode = code
        self.reason = reason


class McNemar(CoreModel):
    """Exact McNemar on the discordant pairs: ``b`` A right and B wrong, ``c`` A wrong and B right."""

    b: int
    c: int
    p_exact: float
    p_mid: float


class MetricComparison(CoreModel):
    """One metric of B against A on the paired items.

    ``items`` are the items both graders scored; ``ungraded_a``/``ungraded_b`` the items asked that each arm's
    grader could not grade. ``ci`` is the seeded cluster bootstrap of the difference, None when fewer than
    ``MIN_CLUSTERS`` clusters were graded. ``p_adjusted`` and ``reject`` are set for the prereg family only.
    ``minimum_detectable`` is the MDE at this n (None when no item flipped); ``noise_bound`` is the upper edge of the
    repeat-noise band, set when the runs repeated each item (k >= 2).
    """

    metric: str
    items: int
    clusters: int
    ungraded_a: int
    ungraded_b: int
    mean_a: float
    mean_b: float
    difference: float
    test: Test
    p: float
    ci: Interval | None
    mcnemar: McNemar | None = None
    wilcoxon: WilcoxonSignedRank | None = None
    p_adjusted: float | None = None
    reject: bool | None = None
    minimum_detectable: float | None = None
    noise_bound: float | None = None


class Pairing(CoreModel):
    """The facts ``pair`` established before any statistic: which items, clusters and graders the runs share.

    ``graders`` are the grader ids both runs scored; ``shown`` the ones a comparison reports; ``family`` the
    pre-registered ones that share one correction.
    """

    run_a: str
    run_b: str
    dataset_sha256: str
    slice_name: str | None
    items: tuple[str, ...]
    clusters: Mapping[str, str]
    repeats: int
    graders: tuple[str, ...]
    shown: tuple[str, ...]
    family: tuple[str, ...]


class Comparison(CoreModel):
    """Run B against run A: the pairing facts and every shown metric."""

    run_a: str
    run_b: str
    dataset_sha256: str
    slice_name: str | None
    items: int
    clusters: int
    correction: Correction
    family: tuple[str, ...]
    metrics: tuple[MetricComparison, ...]


def in_slice(item: DatasetItem, name: str) -> bool:
    """Whether ``item`` belongs to slice ``name``: its ``slice_name`` is ``name`` or ``name`` is one of its tags."""
    return item.slice_name == name or name in item.tags


def _trials(run: Run) -> dict[str, list[int]]:
    """Repeat indexes sampled per item, refusing a duplicate trial or an item repeated unlike the others."""
    found: dict[str, list[int]] = {}
    for sample in run.samples:
        repeats = found.setdefault(sample.item_id, [])
        if sample.repeat_index in repeats:
            raise ComparisonRefused(
                "duplicate_trial",
                f"run {run.record.run_id} records 2 samples for item {sample.item_id} repeat {sample.repeat_index}",
            )
        repeats.append(sample.repeat_index)
    counts = {len(repeats) for repeats in found.values()}
    if len(counts) > 1:
        raise ComparisonRefused(
            "unequal_repeats",
            f"run {run.record.run_id} repeats its items unequally ({min(counts)} to {max(counts)})",
        )
    return found


def _clusters(run: Run) -> dict[str, str]:
    """The cluster of every sampled item, as its samples recorded it."""
    return {sample.item_id: sample.cluster_id for sample in run.samples}


def _versions(run: Run, grader_id: str) -> tuple[str, ...]:
    return tuple(sorted({score.grader_version for score in run.scores if score.grader_id == grader_id}))


def _resolve(name: str, graders: Sequence[str]) -> str:
    """The shared grader ``name`` names: its id, or the one id whose last ``:`` part is ``name``."""
    if name in graders:
        return name
    found = [grader for grader in graders if grader.rpartition(":")[2] == name]
    if not found:
        raise ComparisonRefused("metric_missing", f"metric '{name}' is not scored in both runs")
    if len(found) > 1:
        raise ComparisonRefused("ambiguous_metric", f"metric '{name}' is ambiguous ({', '.join(found)})")
    return found[0]


def _family(prereg: Prereg | None, graders: Sequence[str]) -> tuple[str, ...]:
    """The shared graders the prereg names (primary, then guard); a registered metric not scored in both is absent."""
    if prereg is None:
        return ()
    found: list[str] = []
    for name in (prereg.primary_metric, prereg.guard_metric):
        if name is None:
            continue
        try:
            grader = _resolve(name, graders)
        except ComparisonRefused:
            continue
        if grader not in found:
            found.append(grader)
    return tuple(found)


def pair(
    a: Run,
    b: Run,
    metrics: Sequence[str] = (),
    *,
    slice_name: str | None = None,
    prereg: Prereg | None = None,
    ignore_grader_version: bool = False,
) -> Pairing:
    """Establish what two runs share, or refuse.

    Args:
        a: The baseline run.
        b: The candidate run.
        metrics: Grader ids (or their last ``:`` part) to show; empty shows every grader both runs scored.
        slice_name: Keep only the items of this slice (``in_slice``).
        prereg: The pre-registration whose primary and guard metrics form the correction family; defaults to the
            candidate's, then the baseline's.
        ignore_grader_version: Compare a grader scored at different versions anyway (the opt-in; off by default).

    Returns:
        The pairing: items, clusters, repeats, shared graders, the shown ones and the family.

    Raises:
        ComparisonRefused: The runs measured different things; ``code`` says what.
    """
    if a.record.dataset.sha256 != b.record.dataset.sha256:
        raise ComparisonRefused(
            "different_dataset",
            f"dataset sha256 differs (A {a.record.dataset.sha256}, B {b.record.dataset.sha256})",
        )
    if a.record.repeats != b.record.repeats:
        raise ComparisonRefused("different_k", f"k differs (A {a.record.repeats}, B {b.record.repeats})")
    trials_a, trials_b = _trials(a), _trials(b)
    if set(trials_a) != set(trials_b):
        raise ComparisonRefused(
            "different_items", f"the runs asked different items (A {len(trials_a)}, B {len(trials_b)})"
        )
    clusters_a, clusters_b = _clusters(a), _clusters(b)
    moved = sum(1 for item in trials_a if clusters_a[item] != clusters_b[item])
    if moved:
        raise ComparisonRefused("different_clusters", f"the cluster key differs ({moved} item(s) in a different cluster)")
    items = sorted(trials_a)
    if slice_name is not None:
        items = [item for item in items if item in b.items and in_slice(b.items[item], slice_name)]
    clusters = {item: clusters_b[item] for item in items}
    distinct = len(set(clusters.values()))
    if distinct < MIN_CLUSTERS:
        where = f"slice '{slice_name}' has {distinct} clusters" if slice_name is not None else f"{distinct} clusters"
        raise ComparisonRefused(
            "too_few_clusters", f"{where}, fewer than the {MIN_CLUSTERS} a paired comparison needs"
        )
    graders = tuple(sorted({score.grader_id for score in a.scores} & {score.grader_id for score in b.scores}))
    if not graders:
        raise ComparisonRefused("no_shared_metric", "no metric is scored in both runs")
    shown = tuple(_resolve(name, graders) for name in metrics) if metrics else graders
    family = _family(prereg or b.record.prereg or a.record.prereg, graders)
    if not ignore_grader_version:
        for grader in dict.fromkeys(shown + family):
            versions_a, versions_b = _versions(a, grader), _versions(b, grader)
            if versions_a != versions_b:
                raise ComparisonRefused(
                    "grader_version",
                    f"grader {grader} version differs (A {', '.join(versions_a)}, B {', '.join(versions_b)})",
                )
    return Pairing(
        run_a=a.record.run_id,
        run_b=b.record.run_id,
        dataset_sha256=b.record.dataset.sha256,
        slice_name=slice_name,
        items=tuple(items),
        clusters=clusters,
        repeats=b.record.repeats,
        graders=graders,
        shown=shown,
        family=family,
    )


def _measure(pairing: Pairing, a: Run, b: Run, grader: str) -> MetricComparison:
    """One grader's paired statistics on the items both arms graded."""
    values_a, values_b = a.values(grader), b.values(grader)
    paired = [item for item in pairing.items if item in values_a and item in values_b]
    if not paired:
        raise ComparisonRefused("no_paired_items", f"grader {grader} graded no item in both runs")
    xs = [values_a[item] for item in paired]
    ys = [values_b[item] for item in paired]
    diffs = [y - x for x, y in zip(xs, ys, strict=True)]
    n = len(paired)
    cluster_ids = [pairing.clusters[item] for item in paired]
    n_clusters = len(set(cluster_ids))
    mean_a, mean_b = statistics.fmean(xs), statistics.fmean(ys)
    difference = mean_b - mean_a
    sd = statistics.stdev(diffs) if n >= 2 else 0.0

    test: Test
    mcnemar: McNemar | None = None
    wilcoxon: WilcoxonSignedRank | None = None
    minimum_detectable: float | None
    if all(value in (0.0, 1.0) for value in xs + ys):
        test = "mcnemar"
        flips_b = sum(1 for x, y in zip(xs, ys, strict=True) if x == 1.0 and y == 0.0)
        flips_c = sum(1 for x, y in zip(xs, ys, strict=True) if x == 0.0 and y == 1.0)
        exact = mcnemar_exact(flips_b, flips_c, ALPHA)
        mcnemar = McNemar(b=exact.b, c=exact.c, p_exact=exact.p_exact, p_mid=exact.p_mid)
        p = exact.p_exact
        discordance = exact.discordant / n
        minimum_detectable = None if discordance == 0.0 else minimum_detectable_difference(n, discordance).mde
    else:
        test = "wilcoxon"
        wilcoxon = wilcoxon_signed_rank(xs, ys)
        p = wilcoxon.p_value
        minimum_detectable = mde_continuous(n, sd).mde
    ci = paired_cluster_bootstrap(xs, ys, cluster_ids, seed=BOOTSTRAP_SEED).ci if n_clusters >= MIN_CLUSTERS else None
    bound = noise_bound(difference, sd, n).ci.upper if pairing.repeats >= 2 else None
    return MetricComparison(
        metric=grader,
        items=n,
        clusters=n_clusters,
        ungraded_a=sum(1 for item in pairing.items if item not in values_a),
        ungraded_b=sum(1 for item in pairing.items if item not in values_b),
        mean_a=mean_a,
        mean_b=mean_b,
        difference=difference,
        test=test,
        p=p,
        ci=ci,
        mcnemar=mcnemar,
        wilcoxon=wilcoxon,
        minimum_detectable=minimum_detectable,
        noise_bound=bound,
    )


def compare(
    a: Run,
    b: Run,
    metrics: Sequence[str] = (),
    *,
    slice_name: str | None = None,
    correction: Correction = "holm",
    prereg: Prereg | None = None,
    ignore_grader_version: bool = False,
) -> Comparison:
    """Compare run ``b`` against run ``a`` item by item.

    Binary metrics (every per-item value 0 or 1) get the exact McNemar test and Connor's MDE; others the Wilcoxon
    signed-rank test and the continuous MDE. Every metric gets the seeded cluster bootstrap interval of the
    difference. The prereg family (primary and guard metric) is corrected together by ``correction`` at ``ALPHA``;
    ``metrics`` picks what is shown, never the family, so a shown family metric keeps its corrected p.

    Args:
        a: The baseline run.
        b: The candidate run.
        metrics: Grader ids (or their last ``:`` part) to show; empty shows every grader both runs scored.
        slice_name: Compare only the items of this slice.
        correction: ``holm`` (family-wise), ``bh`` (false discovery rate) or ``none``.
        prereg: The pre-registration that names the family; defaults to the candidate's, then the baseline's.
        ignore_grader_version: Compare a grader scored at different versions anyway.

    Returns:
        The comparison with one ``MetricComparison`` per shown metric, in grader id order.

    Raises:
        ComparisonRefused: The runs measured different things; ``code`` says what.
    """
    pairing = pair(a, b, metrics, slice_name=slice_name, prereg=prereg, ignore_grader_version=ignore_grader_version)
    measured = {grader: _measure(pairing, a, b, grader) for grader in dict.fromkeys(pairing.shown + pairing.family)}
    if pairing.family and correction != "none":
        p_values = [measured[grader].p for grader in pairing.family]
        corrected = holm(p_values, ALPHA) if correction == "holm" else benjamini_hochberg(p_values, ALPHA)
        for grader, adjusted, reject in zip(pairing.family, corrected.adjusted, corrected.reject, strict=True):
            measured[grader] = measured[grader].model_copy(update={"p_adjusted": adjusted, "reject": reject})
    return Comparison(
        run_a=pairing.run_a,
        run_b=pairing.run_b,
        dataset_sha256=pairing.dataset_sha256,
        slice_name=slice_name,
        items=len(pairing.items),
        clusters=len(set(pairing.clusters.values())),
        correction=correction,
        family=pairing.family,
        metrics=tuple(measured[grader] for grader in pairing.shown),
    )
