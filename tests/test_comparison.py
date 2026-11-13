"""``compare(a, b)``: two runs paired by item, the paired tests against their closed forms, and the refusals that
keep two runs that measured different things from being compared at all.

The runs are built by hand (no model call): ``fake_run`` records samples and scores with chosen values, so each
statistic can be checked against its closed form. Ported from 0.3.2 ``tests/behaviors/evals/test_compare.py``.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import pytest

from gigai_core.comparison import (
    MIN_CLUSTERS,
    Comparison,
    ComparisonRefused,
    MetricComparison,
    compare,
    pair,
)
from gigai_core.dataset import Dataset, DatasetItem
from gigai_core.providers import CallSettings, ProviderIdentity, Turn
from gigai_core.records import Prereg, RunRecord
from gigai_core.run import Run
from gigai_core.sample import Sample
from gigai_core.score import ScoreRecord
from gigai_core.stats import mde_continuous, paired_cluster_bootstrap

NOW = datetime(2026, 11, 13, 10, 0, tzinfo=UTC)
IDENTITY = ProviderIdentity(provider="scripted", endpoint="memory", adapter="tests:1")
SETTINGS = CallSettings(model="some/model")
ITEMS = tuple(f"r{i:02d}" for i in range(1, 13))
#: r11 and r12 share the clusters of r01 and r02: 12 items, 10 clusters.
CLUSTERS: Mapping[str, str] = {item: f"c{(n % 10) + 1:02d}" for n, item in enumerate(ITEMS)}
#: core = r01-r10 (10 clusters), easy = r01-r06 (6), hard = r07-r12 (6).
TAGS: Mapping[str, tuple[str, ...]] = {
    item: (("core",) if n < 10 else ()) + (("easy",) if n < 6 else ("hard",)) for n, item in enumerate(ITEMS)
}
MATCH = "match"
MARKER = "marker"
#: match with b = 3 (A right, B wrong: r01-r03) and c = 9 (A wrong, B right: r04-r12).
ACC_A: Mapping[str, Sequence[float | None]] = {item: (1.0 if n < 3 else 0.0,) for n, item in enumerate(ITEMS)}
ACC_B: Mapping[str, Sequence[float | None]] = {item: (0.0 if n < 3 else 1.0,) for n, item in enumerate(ITEMS)}
SCORE_A: Mapping[str, Sequence[float | None]] = {item: (0.5,) for item in ITEMS}
SCORE_B: Mapping[str, Sequence[float | None]] = {item: (0.5 + 0.05 * (n % 4),) for n, item in enumerate(ITEMS)}
PREREG = Prereg(
    primary_metric=MATCH, pass_rule="ci_lower >= 0", margin=0.0, n_planned=12, registered_at=NOW, guard_metric=MARKER
)
Scores = Mapping[str, Mapping[str, Sequence[float | None]]]


def dataset(
    items: Sequence[str] = ITEMS,
    *,
    clusters: Mapping[str, str] | None = None,
    licences: Mapping[str, str] | None = None,
    prompt: str = "say",
) -> Dataset:
    """The dataset of ``items``; ``prompt`` changes its digest, ``licences`` marks held-out items."""
    clusters = CLUSTERS if clusters is None else clusters
    return Dataset.of(
        "release",
        [
            DatasetItem(
                id=item,
                cluster_id=clusters.get(item, item),
                prompt=f"{prompt} {item}",
                expected=item,
                tags=TAGS.get(item, ()),
                licence=(licences or {}).get(item),
            )
            for item in items
        ],
    )


def fake_run(
    scores: Scores,
    *,
    run_id: str = "run_a",
    data: Dataset | None = None,
    versions: Mapping[str, str] | None = None,
    prereg: Prereg | None = PREREG,
    repeats: int | None = None,
    trials: Mapping[str, int] | None = None,
) -> Run:
    """A finished run whose grader ``g`` gave item ``i`` repeat ``r`` the value ``scores[g][i][r]``.

    ``None`` (or a missing repeat) is ``CannotGrade``: no score row, so the item is ungraded by that grader. The
    items asked are those of ``data`` (default: the 12 of ``dataset()``); ``repeats`` is k (default: the longest
    value list); ``trials`` overrides the sample count of single items.
    """
    data = dataset() if data is None else data
    k = repeats or max((len(values) for by_item in scores.values() for values in by_item.values()), default=1)
    record = RunRecord(
        run_id=run_id,
        dataset=data.ref,
        settings=SETTINGS,
        provider=IDENTITY,
        created_at=NOW,
        completed_at=NOW,
        prereg=prereg,
        repeats=k,
    )
    samples: list[Sample] = []
    rows: list[ScoreRecord] = []
    for item in data.items:
        for repeat in range((trials or {}).get(item.id, k)):
            trace_id = f"{run_id}-{item.id}-{repeat}"
            samples.append(
                Sample(
                    trace_id=trace_id,
                    run_id=run_id,
                    item_id=item.id,
                    item_version=item.version,
                    cluster_id=item.cluster_id,
                    repeat_index=repeat,
                    input=(Turn(role="user", content=item.prompt),),
                    output_text="the output",
                    output_parsed="the output",
                    parse_status="not_parsed",
                    outcome="succeeded",
                    error=None,
                    recorded_at=NOW,
                )
            )
            for grader_id, by_item in scores.items():
                values = by_item.get(item.id, ())
                value = values[repeat] if repeat < len(values) else None
                if value is None:
                    continue
                rows.append(
                    ScoreRecord(
                        score_id=f"score-{grader_id}-{trace_id}",
                        trace_id=trace_id,
                        grader_id=grader_id,
                        grader_version=(versions or {}).get(grader_id, "1"),
                        kind="code",
                        value=value,
                        recorded_at=NOW,
                    )
                )
    return Run(record=record, items={item.id: item for item in data.items}, samples=tuple(samples), scores=tuple(rows))


def pair_of_runs() -> tuple[Run, Run]:
    return fake_run({MATCH: ACC_A, MARKER: SCORE_A}), fake_run({MATCH: ACC_B, MARKER: SCORE_B}, run_id="run_b")


def connor_mde(n: int, discordance: float) -> float:
    """Connor 1987 solved for delta by bisection, independently of ``gigai_core.stats``."""
    z = statistics.NormalDist()
    z_alpha, z_beta = z.inv_cdf(0.975), z.inv_cdf(0.80)
    low, high = 1e-9, discordance
    for _ in range(200):
        mid = (low + high) / 2
        needed = (z_alpha * math.sqrt(discordance) + z_beta * math.sqrt(discordance - mid**2)) ** 2 / mid**2
        low, high = (mid, high) if needed > n else (low, mid)
    return high


def seeded_diffs(n: int = 48, mean: float = 0.167, sd: float = 0.595) -> list[float]:
    rng = random.Random(7)
    raw = [rng.gauss(0.0, 1.0) for _ in range(n)]
    centre, spread = statistics.fmean(raw), statistics.stdev(raw)
    return [mean + sd * (value - centre) / spread for value in raw]


def by_metric(comparison: Comparison) -> Mapping[str, MetricComparison]:
    return {metric.metric: metric for metric in comparison.metrics}


# --- the closed forms ---------------------------------------------------------------------------------------------


def test_binary_metrics_use_exact_mcnemar_and_report_the_mde_at_the_runs_n() -> None:
    a, b = pair_of_runs()
    result = compare(a, b, [MATCH])
    (accuracy,) = result.metrics

    assert (result.items, result.clusters, result.dataset_sha256) == (12, 10, a.record.dataset.sha256)
    assert (result.run_a, result.run_b, result.correction, result.family) == ("run_a", "run_b", "holm", (MATCH, MARKER))
    assert accuracy.test == "mcnemar" and accuracy.mcnemar is not None and accuracy.wilcoxon is None
    assert (accuracy.mcnemar.b, accuracy.mcnemar.c) == (3, 9)
    assert accuracy.p == pytest.approx(2 * 299 / 4096, abs=1e-12)
    assert accuracy.p == pytest.approx(0.146, abs=0.001)
    assert (accuracy.mean_a, accuracy.mean_b, accuracy.difference) == (0.25, 0.75, 0.5)
    assert (accuracy.items, accuracy.clusters, accuracy.ungraded_a, accuracy.ungraded_b) == (12, 10, 0, 0)
    assert accuracy.minimum_detectable == pytest.approx(connor_mde(12, 1.0), abs=1e-6)  # discordance (b + c) / n = 1
    assert accuracy.ci is not None and accuracy.ci.lower < 0.5 < accuracy.ci.upper and accuracy.ci.level == 0.95
    expected = paired_cluster_bootstrap([ACC_A[i][0] or 0.0 for i in ITEMS], [ACC_B[i][0] or 0.0 for i in ITEMS], [CLUSTERS[i] for i in ITEMS])
    assert accuracy.ci == expected.ci
    assert accuracy.noise_bound is None  # k = 1 on both sides


def test_continuous_metrics_use_wilcoxon_and_with_k_2_a_noise_bound_of_0_335() -> None:
    diffs = seeded_diffs()
    assert (statistics.fmean(diffs), statistics.stdev(diffs)) == (pytest.approx(0.167), pytest.approx(0.595))
    items = [f"r{i:02d}" for i in range(1, 49)]
    alone = dataset(items, clusters={})  # no shared cluster: each item is its own
    a = fake_run({MARKER: {item: (1.0, 1.0) for item in items}}, data=alone, prereg=None)
    b = fake_run({MARKER: {item: (1.0 + d, 1.0 + d) for item, d in zip(items, diffs, strict=True)}}, run_id="run_b", data=alone, prereg=None)

    result = compare(a, b, [MARKER])
    (score,) = result.metrics

    assert score.noise_bound == pytest.approx(0.335, abs=0.01)
    assert score.noise_bound == pytest.approx(0.167 + 1.959964 * 0.595 / math.sqrt(48), abs=1e-6)
    assert score.test == "wilcoxon" and score.wilcoxon is not None and score.wilcoxon.n == 48 and score.mcnemar is None
    assert score.minimum_detectable == mde_continuous(48, statistics.stdev(diffs)).mde
    assert score.difference == pytest.approx(0.167)
    assert (result.items, result.clusters, result.family) == (48, 48, ())


# --- corrections, slices, ungraded items --------------------------------------------------------------------------


def test_holm_corrects_the_preregistered_metrics_and_bh_on_request_others_stay_raw() -> None:
    extra = {item: (0.0,) for item in ITEMS}
    moved = [0.1, 0.2, 0.3, -0.15, 0.25, 0.05, -0.1, 0.35, -0.2, -0.05, 0.4, 0.12]  # Wilcoxon p = 0.1157
    a = fake_run({MATCH: ACC_A, MARKER: {i: (0.0,) for i in ITEMS}, "extra": extra})
    b = fake_run({MATCH: ACC_B, MARKER: {i: (d,) for i, d in zip(ITEMS, moved, strict=True)}, "extra": {i: (1.0,) for i in ITEMS}}, run_id="run_b")

    held = {m.metric: m for m in compare(a, b).metrics}
    bh = {m.metric: m for m in compare(a, b, correction="bh").metrics}

    assert held[MARKER].p == pytest.approx(0.11572265625)
    assert [held[MATCH].p_adjusted, held[MARKER].p_adjusted] == pytest.approx([0.2314453125, 0.2314453125])
    assert [held[MATCH].reject, held[MARKER].reject] == [False, False]
    assert [bh[MATCH].p_adjusted, bh[MARKER].p_adjusted] == pytest.approx([2 * 299 / 4096, 2 * 299 / 4096])
    assert held["extra"].p_adjusted is None and held["extra"].reject is None and held["extra"].p < 0.01
    assert compare(a, b).family == (MATCH, MARKER) and [m.metric for m in compare(a, b).metrics] == ["extra", MARKER, MATCH]
    # metrics picks what is shown, not the family: the one shown metric keeps its family-corrected p.
    shown = compare(a, b, [MARKER])
    assert [m.metric for m in shown.metrics] == [MARKER] and shown.family == (MATCH, MARKER)
    assert shown.metrics[0].p_adjusted == pytest.approx(0.2314453125)
    (alone,) = compare(a, b, ["extra"]).metrics
    assert alone.p_adjusted is None
    # correction "none" reports the family but corrects nothing.
    raw = compare(a, b, correction="none")
    assert raw.family == (MATCH, MARKER) and all(m.p_adjusted is None and m.reject is None for m in raw.metrics)


def test_without_a_prereg_no_metric_is_corrected_and_an_explicit_prereg_names_the_family() -> None:
    a = fake_run({MATCH: ACC_A, MARKER: SCORE_A}, prereg=None)
    b = fake_run({MATCH: ACC_B, MARKER: SCORE_B}, run_id="run_b", prereg=None)

    result = compare(a, b)
    assert result.family == () and all(m.p_adjusted is None for m in result.metrics)

    explicit = compare(a, b, prereg=PREREG.model_copy(update={"guard_metric": None}))
    assert explicit.family == (MATCH,)
    assert by_metric(explicit)[MATCH].p_adjusted == pytest.approx(2 * 299 / 4096) and by_metric(explicit)[MATCH].reject is False
    assert by_metric(explicit)[MATCH].p == by_metric(result)[MATCH].p
    assert by_metric(explicit)[MARKER].p_adjusted is None


def test_a_slice_keeps_its_items_and_a_slice_under_10_clusters_is_refused() -> None:
    a, b = pair_of_runs()

    core = compare(a, b, [MATCH], slice_name="core")  # tag core: r01-r10
    assert (core.items, core.clusters, core.slice_name) == (10, 10, "core")
    assert core.metrics[0].mcnemar is not None and (core.metrics[0].mcnemar.b, core.metrics[0].mcnemar.c) == (3, 7)
    with pytest.raises(ComparisonRefused, match=r"^cannot compare: slice 'easy' has 6 clusters, fewer than the 10 a paired comparison needs$") as refused:
        compare(a, b, slice_name="easy")
    assert refused.value.code == "too_few_clusters"
    with pytest.raises(ComparisonRefused, match=r"^cannot compare: slice 'nope' has 0 clusters") as empty:
        compare(a, b, slice_name="nope")
    assert empty.value.code == "too_few_clusters"


def test_an_item_a_grader_could_not_grade_drops_out_of_both_arms_and_is_counted() -> None:
    a = fake_run({MATCH: {**ACC_A, "r04": (None,)}, MARKER: SCORE_A})
    b = fake_run({MATCH: {**ACC_B, "r05": (None,), "r06": (None,)}, MARKER: SCORE_B}, run_id="run_b")

    result = compare(a, b)
    marker, accuracy = result.metrics  # grader id order

    assert (result.items, result.clusters) == (12, 10)  # the pairing: every item both runs asked
    assert (accuracy.items, accuracy.ungraded_a, accuracy.ungraded_b) == (9, 1, 2)
    assert accuracy.mcnemar is not None and (accuracy.mcnemar.b, accuracy.mcnemar.c) == (3, 6)
    assert accuracy.clusters == 7 and accuracy.ci is None  # under MIN_CLUSTERS once graded: no interval, no refusal
    assert (marker.items, marker.ungraded_a, marker.ungraded_b, marker.clusters) == (12, 0, 0, 10) and marker.ci is not None
    assert MIN_CLUSTERS == 10


def test_an_item_graded_on_some_repeats_only_is_ungraded_not_a_refusal() -> None:
    a = fake_run({MATCH: {i: (v[0], v[0]) for i, v in ACC_A.items()}})
    b = fake_run({MATCH: {**{i: (v[0], v[0]) for i, v in ACC_B.items()}, "r04": (1.0,)}}, run_id="run_b")

    (accuracy,) = compare(a, b).metrics

    assert (accuracy.items, accuracy.ungraded_a, accuracy.ungraded_b) == (11, 0, 1)
    assert accuracy.noise_bound is not None  # k = 2


# --- refusals: runs that measured different things ----------------------------------------------------------------

def _refusal(case: str) -> tuple[Run, Run, list[str]]:
    a = fake_run({MATCH: ACC_A, MARKER: SCORE_A})
    if case == "dataset":
        return a, fake_run({MATCH: ACC_B}, run_id="run_b", data=dataset(prompt="ask")), []
    if case == "version":
        return a, fake_run({MATCH: ACC_B}, run_id="run_b", versions={MATCH: "2"}), []
    if case == "family version":  # not asked for, but pre-registered: its version is checked as well
        return a, fake_run({MATCH: ACC_B, MARKER: SCORE_B}, run_id="run_b", versions={MARKER: "2"}), [MATCH]
    if case == "cluster":  # the dataset digest covers cluster ids, so only a sample can disagree with it
        b = fake_run({MATCH: ACC_B}, run_id="run_b")
        moved = tuple(s.model_copy(update={"cluster_id": "c09"}) if s.item_id == "r12" else s for s in b.samples)
        return a, b.model_copy(update={"samples": moved}), []
    if case == "items":  # r12 was never asked in B
        return a, fake_run({MATCH: ACC_B}, run_id="run_b", trials={"r12": 0}), []
    if case == "k":
        return a, fake_run({MATCH: {i: (v[0], v[0]) for i, v in ACC_B.items()}}, run_id="run_b"), []
    if case == "repeats":
        return a, fake_run({MATCH: ACC_B}, run_id="run_b", trials={"r04": 2}), []
    if case == "duplicate":
        b = fake_run({MATCH: ACC_B}, run_id="run_b")
        twin = b.samples[2].model_copy(update={"trace_id": "trace_twin"})
        return a, b.model_copy(update={"samples": (*b.samples, twin)}), []
    if case == "metric":
        return a, fake_run({MATCH: ACC_B}, run_id="run_b"), [MARKER]
    if case == "ambiguous":
        two = {"x:" + MATCH: ACC_A, "y:" + MATCH: ACC_A}
        return fake_run(two, prereg=None), fake_run(two, run_id="run_b", prereg=None), [MATCH]
    if case == "none shared":
        return a, fake_run({"other": ACC_B}, run_id="run_b"), []
    if case == "nothing paired":  # A graded r01-r06 only, B r07-r12 only: match has no item both arms graded
        first = fake_run({MATCH: {i: (v[0] if n < 6 else None,) for n, (i, v) in enumerate(ACC_A.items())}})
        second = fake_run({MATCH: {i: (v[0] if n >= 6 else None,) for n, (i, v) in enumerate(ACC_B.items())}}, run_id="run_b")
        return first, second, []
    nine = dataset(clusters={item: f"c{min(n, 8):02d}" for n, item in enumerate(ITEMS)})
    return fake_run({MATCH: ACC_A}, data=nine), fake_run({MATCH: ACC_B}, run_id="run_b", data=nine), []


REFUSALS = {
    "dataset": ("different_dataset", r"^cannot compare: dataset sha256 differs \(A [0-9a-f]+, B [0-9a-f]+\)$"),
    "version": ("grader_version", r"^cannot compare: grader match version differs \(A 1, B 2\)$"),
    "family version": ("grader_version", r"^cannot compare: grader marker version differs \(A 1, B 2\)$"),
    "cluster": ("different_clusters", r"^cannot compare: the cluster key differs \(1 item\(s\) in a different cluster\)$"),
    "items": ("different_items", r"^cannot compare: the runs asked different items \(A 12, B 11\)$"),
    "k": ("different_k", r"^cannot compare: k differs \(A 1, B 2\)$"),
    "repeats": ("unequal_repeats", r"^cannot compare: run run_b repeats its items unequally \(1 to 2\)$"),
    "duplicate": ("duplicate_trial", r"^cannot compare: run run_b records 2 samples for item r03 repeat 0$"),
    "metric": ("metric_missing", r"^cannot compare: metric 'marker' is not scored in both runs$"),
    "ambiguous": ("ambiguous_metric", r"^cannot compare: metric 'match' is ambiguous \(x:match, y:match\)$"),
    "none shared": ("no_shared_metric", r"^cannot compare: no metric is scored in both runs$"),
    "nothing paired": ("no_paired_items", r"^cannot compare: grader match graded no item in both runs$"),
    "clusters": ("too_few_clusters", r"^cannot compare: 9 clusters, fewer than the 10 a paired comparison needs$"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_runs_that_measured_different_things_are_refused(case: str) -> None:
    a, b, metrics = _refusal(case)
    code, message = REFUSALS[case]

    with pytest.raises(ComparisonRefused, match=message) as refused:
        compare(a, b, metrics)

    assert refused.value.code == code
    assert refused.value.reason == str(refused.value).removeprefix("cannot compare: ")


def test_a_grader_version_mismatch_is_compared_only_with_the_opt_in() -> None:
    a = fake_run({MATCH: ACC_A})
    b = fake_run({MATCH: ACC_B}, run_id="run_b", versions={MATCH: "2"})

    with pytest.raises(ComparisonRefused):
        compare(a, b)
    (accuracy,) = compare(a, b, ignore_grader_version=True).metrics
    assert accuracy.mcnemar is not None and (accuracy.mcnemar.b, accuracy.mcnemar.c) == (3, 9)


def test_a_metric_name_resolves_to_the_one_grader_whose_last_part_it_is() -> None:
    a = fake_run({"suite:" + MATCH: ACC_A}, prereg=None)
    b = fake_run({"suite:" + MATCH: ACC_B}, run_id="run_b", prereg=None)

    pairing = pair(a, b, [MATCH])

    assert pairing.shown == ("suite:match",) and pairing.graders == ("suite:match",) and pairing.family == ()
    assert (len(pairing.items), len(set(pairing.clusters.values())), pairing.repeats) == (12, 10, 1)
    assert compare(a, b, [MATCH]).metrics[0].metric == "suite:match"


def test_a_prereg_metric_not_scored_in_both_runs_is_outside_the_family() -> None:
    a = fake_run({MATCH: ACC_A})  # prereg names marker too, but neither run scored it
    b = fake_run({MATCH: ACC_B}, run_id="run_b")

    assert compare(a, b).family == (MATCH,)


def test_a_comparison_is_a_frozen_value_and_never_touches_the_runs() -> None:
    a, b = pair_of_runs()
    before = (a.model_dump(), b.model_dump())

    result = compare(a, b)

    assert (a.model_dump(), b.model_dump()) == before
    with pytest.raises(Exception, match="frozen"):
        result.items = 0  # type: ignore[misc]
