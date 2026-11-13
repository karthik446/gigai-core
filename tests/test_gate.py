"""The gate: typed rules over a candidate and a baseline, a verdict with every failing rule's numbers.

Ported from 0.3.2 ``tests/behaviors/evals/test_gate.py``: the min-CI-lower and no-worse rules with their
reasons, the held-out share refusal, and the pass / fail / cannot-compare exit codes. Dropped: the CLI, the
store, the ci snapshot gate and the ``regated`` manifest, which core does not have.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter

from gigai_core.gate import (
    GateResult,
    GateRule,
    HeldOutShare,
    MinLowerBound,
    NoWorseThan,
    RuleVerdict,
    gate,
    held_out_share,
)
from gigai_core.run import Run
from gigai_core.stats import wilson_interval
from tests.test_comparison import (
    ACC_A,
    ACC_B,
    ITEMS,
    MARKER,
    MATCH,
    SCORE_A,
    SCORE_B,
    dataset,
    fake_run,
)

PRIVATE = {item: "private" for item in ITEMS[:6]}


def candidate_and_baseline() -> tuple[Run, Run]:
    """Candidate: match 9/12 (r04-r12); baseline: 3/12 (r01-r03). Half the items are held out."""
    data = dataset(licences=PRIVATE)
    candidate = fake_run({MATCH: ACC_B, MARKER: SCORE_B}, run_id="candidate", data=data)
    baseline = fake_run({MATCH: ACC_A, MARKER: SCORE_A}, run_id="baseline", data=data)
    return candidate, baseline


def test_a_candidate_that_meets_every_rule_passes_with_exit_code_0() -> None:
    candidate, baseline = candidate_and_baseline()
    rules = [MinLowerBound(metric=MATCH, floor=0.4), NoWorseThan(metric=MATCH, margin=0.05), HeldOutShare(minimum=0.5)]

    verdict = gate(candidate, baseline, rules)

    assert verdict.status == "pass" and verdict.passed and verdict.exit_code == 0 and verdict.reason is None
    assert verdict.failed == () and [v.rule for v in verdict.verdicts] == rules
    assert all(v.passed and v.reasons == () for v in verdict.verdicts)
    assert verdict.verdicts[1].measured is not None and verdict.verdicts[1].measured.difference == pytest.approx(0.5)
    assert verdict.verdicts[0].measured is None and verdict.verdicts[2].measured is None


def test_every_failing_rule_is_reported_with_its_numbers_and_exit_code_1() -> None:
    baseline, candidate = candidate_and_baseline()  # the other way round: the candidate is the worse run
    rules = [MinLowerBound(metric=MATCH, floor=0.6), NoWorseThan(metric=MATCH, margin=0.05), HeldOutShare(minimum=0.5)]

    verdict = gate(candidate, baseline, rules)

    assert verdict.status == "fail" and not verdict.passed and verdict.exit_code == 1 and verdict.reason is None
    lower = wilson_interval(3, 12).ci.lower
    assert [v.rule for v in verdict.failed] == rules[:2]
    assert verdict.failed[0].rule == MinLowerBound(metric=MATCH, floor=0.6)
    assert verdict.failed[0].reasons == (f"match: Wilson lower {lower:+.4f} < floor +0.6000",)
    assert verdict.failed[1].rule == NoWorseThan(metric=MATCH, margin=0.05)  # DESIGN 4.9: the rule, with defaults
    assert verdict.failed[1].reasons == ("match: difference -0.5000 < -0.0500",)
    assert verdict.failed[1].measured is not None and verdict.failed[1].measured.difference == pytest.approx(-0.5)
    assert verdict.verdicts[2].passed


def test_no_worse_than_can_hold_the_interval_lower_end_instead_of_the_point() -> None:
    baseline, candidate = candidate_and_baseline()
    rule = NoWorseThan(metric=MATCH, margin=0.05, bound="ci_lower")

    (fails,) = gate(candidate, baseline, [rule]).verdicts
    assert not fails.passed and fails.measured is not None and fails.measured.ci is not None
    assert fails.reasons == (f"match: CI lower {fails.measured.ci.lower:+.4f} < -0.0500",)

    # A wide margin the interval stays inside passes on the interval too.
    (passes,) = gate(candidate, baseline, [NoWorseThan(metric=MATCH, margin=1.0, bound="ci_lower")]).verdicts
    assert passes.passed and passes.reasons == ()

    # Under MIN_CLUSTERS once ungraded items drop out there is no interval: the interval bound fails, the point holds.
    short = fake_run(
        {MATCH: {**ACC_B, "r04": (None,), "r05": (None,), "r06": (None,)}}, run_id="candidate", data=dataset(licences=PRIVATE)
    )
    by_interval, by_point = gate(
        short, baseline, [rule, NoWorseThan(metric=MATCH, margin=0.05)]
    ).verdicts
    assert by_interval.reasons == ("match: no interval over 7 clusters; the bound needs MIN_CLUSTERS",)
    assert by_point.passed and by_point.measured is not None and by_point.measured.clusters == 7


def test_without_a_baseline_a_paired_rule_fails_and_the_others_still_apply() -> None:
    candidate, _ = candidate_and_baseline()

    verdict = gate(candidate, None, [MinLowerBound(metric=MATCH, floor=0.4), NoWorseThan(metric=MATCH, margin=0.05)])

    assert verdict.status == "fail" and verdict.exit_code == 1
    assert verdict.verdicts[0].passed
    assert verdict.failed[0].reasons == ("no baseline run to pair match with",) and verdict.failed[0].measured is None
    assert gate(candidate, None, [MinLowerBound(metric=MATCH, floor=0.4)]).passed


def test_a_refused_comparison_cannot_compare_with_exit_code_2_and_every_rule_is_still_reported() -> None:
    candidate, _ = candidate_and_baseline()
    other = fake_run({MATCH: ACC_A}, run_id="baseline", data=dataset(prompt="ask"))
    rules = [MinLowerBound(metric=MATCH, floor=0.9), NoWorseThan(metric=MATCH, margin=0.05), HeldOutShare(minimum=0.5)]

    verdict = gate(candidate, other, rules)

    assert verdict.status == "cannot_compare" and verdict.exit_code == 2 and not verdict.passed
    assert verdict.reason is not None and verdict.reason.startswith("cannot compare: dataset sha256 differs (A ")
    assert len(verdict.verdicts) == 3 and [v.passed for v in verdict.verdicts] == [False, False, True]
    assert verdict.verdicts[1].reasons == (verdict.reason,) and verdict.verdicts[1].measured is None


def test_too_little_held_out_data_cannot_compare_and_the_share_is_public() -> None:
    candidate, baseline = candidate_and_baseline()
    public = fake_run({MATCH: ACC_B}, run_id="candidate")
    assert held_out_share(candidate) == 0.5 and held_out_share(public) == 0.0

    verdict = gate(public, None, [HeldOutShare(minimum=0.5), MinLowerBound(metric=MATCH, floor=0.4)])
    assert verdict.status == "cannot_compare" and verdict.exit_code == 2
    assert verdict.reason == "held-out share 0.00 < 0.50" and verdict.failed[0].reasons == (verdict.reason,)
    assert verdict.verdicts[1].passed

    # A comparison refusal is reported before an unmet share when both occur: rule order.
    other = fake_run({MATCH: ACC_A}, run_id="baseline", data=dataset(prompt="ask"))
    both = gate(public, other, [NoWorseThan(metric=MATCH, margin=0.0), HeldOutShare(minimum=0.5)])
    assert both.reason is not None and both.reason.startswith("cannot compare:") and len(both.failed) == 2
    assert gate(candidate, baseline, [HeldOutShare(minimum=0.5)]).passed


def test_min_lower_bound_names_a_slice_a_grader_by_its_name_and_refuses_what_is_not_a_proportion() -> None:
    candidate, _ = candidate_and_baseline()

    (easy,) = gate(candidate, None, [MinLowerBound(metric=MATCH, floor=0.4, slice_name="easy")]).verdicts  # 3 of 6
    lower = wilson_interval(3, 6).ci.lower
    assert easy.reasons == (f"match in slice 'easy': Wilson lower {lower:+.4f} < floor +0.4000",)
    (hard,) = gate(candidate, None, [MinLowerBound(metric=MATCH, floor=0.4, slice_name="hard")]).verdicts  # 6 of 6
    assert hard.passed
    (empty,) = gate(candidate, None, [MinLowerBound(metric=MATCH, floor=0.4, slice_name="nope")]).verdicts
    assert empty.reasons == ("match in slice 'nope': no graded item",)

    (unknown,) = gate(candidate, None, [MinLowerBound(metric="nope", floor=0.4)]).verdicts
    assert unknown.reasons == ("nope is not graded on the candidate",)
    (scored,) = gate(candidate, None, [MinLowerBound(metric=MARKER, floor=0.4)]).verdicts
    assert scored.reasons == ("marker is not 0/1 per item; a Wilson lower bound needs a binary metric",)

    named = fake_run({"x:" + MATCH: ACC_B}, run_id="candidate")
    (by_name,) = gate(named, None, [MinLowerBound(metric=MATCH, floor=0.4)]).verdicts
    assert by_name.passed

    both = fake_run({"x:" + MATCH: ACC_B, "y:" + MATCH: ACC_B}, run_id="candidate")
    (ambiguous,) = gate(both, None, [MinLowerBound(metric=MATCH, floor=0.4)]).verdicts
    assert ambiguous.reasons == ("match is ambiguous (x:match, y:match)",)


def test_a_slice_with_10_clusters_is_gated_on_its_own_and_a_smaller_one_cannot_compare() -> None:
    candidate, baseline = candidate_and_baseline()
    core = NoWorseThan(metric=MATCH, margin=0.0, slice_name="core", bound="ci_lower")

    verdict = gate(candidate, baseline, [core, NoWorseThan(metric=MATCH, margin=0.0, slice_name="core")])

    on_interval, on_point = verdict.verdicts
    assert verdict.status == "fail" and on_point.passed
    assert on_interval.reasons == ("match in slice 'core': CI lower -0.2000 < +0.0000",)  # 0.3.2's number, no "-0.0000"
    assert on_interval.measured is not None and (on_interval.measured.items, on_interval.measured.clusters) == (10, 10)
    assert on_point.measured is not None and on_point.measured.difference == pytest.approx(0.4)

    small = gate(candidate, baseline, [NoWorseThan(metric=MATCH, margin=0.0, slice_name="easy")])
    assert small.status == "cannot_compare" and small.exit_code == 2
    assert small.reason == "cannot compare: slice 'easy' has 6 clusters, fewer than the 10 a paired comparison needs"


def test_a_bound_exactly_at_its_floor_passes() -> None:
    baseline, candidate = candidate_and_baseline()  # candidate 3/12 against 9/12: difference -0.5
    (point,) = gate(candidate, baseline, [NoWorseThan(metric=MATCH, margin=0.05)]).verdicts
    (interval,) = gate(candidate, baseline, [NoWorseThan(metric=MATCH, margin=0.05, bound="ci_lower")]).verdicts
    assert point.measured is not None and interval.measured is not None and interval.measured.ci is not None
    assert not point.passed and not interval.passed

    at_point = gate(candidate, baseline, [NoWorseThan(metric=MATCH, margin=-point.measured.difference)])
    at_lower = gate(
        candidate, baseline, [NoWorseThan(metric=MATCH, margin=-interval.measured.ci.lower, bound="ci_lower")]
    )
    assert at_point.passed and at_lower.passed
    assert not gate(candidate, baseline, [NoWorseThan(metric=MATCH, margin=-point.measured.difference - 1e-9)]).passed

    lower = wilson_interval(3, 12).ci.lower
    assert gate(candidate, None, [MinLowerBound(metric=MATCH, floor=lower)]).passed
    assert not gate(candidate, None, [MinLowerBound(metric=MATCH, floor=lower + 1e-9)]).passed


def test_rules_and_verdicts_are_data_that_save_and_load() -> None:
    candidate, baseline = candidate_and_baseline()
    rules: list[GateRule] = [
        MinLowerBound(metric=MATCH, floor=0.6, slice_name="core"),
        NoWorseThan(metric=MATCH, margin=0.05, bound="ci_lower"),
        HeldOutShare(minimum=0.5),
    ]
    adapter = TypeAdapter(list[GateRule])

    assert adapter.validate_json(adapter.dump_json(rules)) == rules
    assert TypeAdapter(GateRule).validate_python({"kind": "held_out_share", "minimum": 0.5}) == HeldOutShare(minimum=0.5)
    with pytest.raises(ValueError, match="kind"):
        TypeAdapter(GateRule).validate_python({"metric": MATCH, "floor": 0.5})

    verdict = gate(candidate, baseline, rules)
    loaded = GateResult.model_validate_json(verdict.model_dump_json())
    assert loaded == verdict and loaded.exit_code == verdict.exit_code and loaded.failed == verdict.failed
    assert all(isinstance(v, RuleVerdict) for v in loaded.verdicts)
