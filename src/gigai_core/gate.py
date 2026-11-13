"""A gate: typed rules over a candidate run and, when a rule needs one, a baseline.

Rules are data: a ``GateRule`` saves and loads (``kind`` is the discriminator), so a gate can be pre-registered
and re-applied later. The gate never calls a model; it reads the runs it is given. Its verdict maps to a process
exit status: 0 the gate passed, 1 a rule failed, 2 the runs could not be compared (the paired comparison refused,
or too little of the dataset is held out for the verdict to mean anything).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from pydantic import Field

from .comparison import ComparisonRefused, MetricComparison, compare, in_slice
from .models import CoreModel
from .run import Run
from .stats import wilson_interval

HELD_OUT_LICENCE = "private"


class MinLowerBound(CoreModel):
    """The Wilson lower bound of ``metric`` on the candidate must reach ``floor`` (optionally within one slice).

    ``metric`` is a grader id, or the part after the last ``:`` when that names one grader of the run. The metric
    has to be 0/1 per item: a Wilson interval is a bound on a proportion. The per-item value is the mean over the
    k repeats, so at k >= 2 an item whose repeats disagree is refused as not 0/1 per item.
    """

    kind: Literal["min_lower_bound"] = "min_lower_bound"
    metric: str
    floor: float
    slice_name: str | None = None


class NoWorseThan(CoreModel):
    """The candidate may be at most ``margin`` below the baseline on ``metric`` (paired, with its interval).

    ``bound`` picks what is held to ``-margin``: the paired point difference (``point``, the default) or the lower
    end of its cluster-bootstrap interval (``ci_lower``, which needs ``MIN_CLUSTERS`` graded clusters).
    """

    kind: Literal["no_worse_than"] = "no_worse_than"
    metric: str
    margin: float
    slice_name: str | None = None
    bound: Literal["point", "ci_lower"] = "point"


class HeldOutShare(CoreModel):
    """At least ``minimum`` of the candidate's items must be held out, or the gate refuses to call itself release-grade."""

    kind: Literal["held_out_share"] = "held_out_share"
    minimum: float


GateRule = Annotated[MinLowerBound | NoWorseThan | HeldOutShare, Field(discriminator="kind")]


class RuleVerdict(CoreModel):
    """One rule applied: whether it passed, why not, and the numbers behind it."""

    rule: GateRule
    passed: bool
    reasons: tuple[str, ...]
    measured: MetricComparison | None = None


class GateResult(CoreModel):
    """The verdict. ``failed`` lists every rule that did not pass with its numbers.

    ``reason`` is set when the status is ``cannot_compare``: the first refusal, in rule order.
    """

    status: Literal["pass", "fail", "cannot_compare"]
    verdicts: tuple[RuleVerdict, ...]
    reason: str | None = None

    @property
    def passed(self) -> bool:
        """True when every rule passed."""
        return self.status == "pass"

    @property
    def failed(self) -> tuple[RuleVerdict, ...]:
        """The verdicts that did not pass, in rule order."""
        return tuple(verdict for verdict in self.verdicts if not verdict.passed)

    @property
    def exit_code(self) -> int:
        """0 pass, 1 fail, 2 cannot compare: what a CLI maps to its exit status."""
        return {"pass": 0, "fail": 1, "cannot_compare": 2}[self.status]


def held_out_share(run: Run) -> float:
    """The share of the run's dataset items whose licence is ``private``; 0.0 for a run without items."""
    if not run.items:
        return 0.0
    return sum(item.licence == HELD_OUT_LICENCE for item in run.items.values()) / len(run.items)


def gate(candidate: Run, baseline: Run | None, rules: Sequence[GateRule]) -> GateResult:
    """Apply every rule to ``candidate`` (paired to ``baseline`` when a rule needs one). Never calls a model.

    Every rule is applied and reported, even after one fails. A ``NoWorseThan`` rule without a baseline fails
    (the gate cannot show the candidate is no worse); a ``NoWorseThan`` rule whose paired comparison refuses, or
    a ``HeldOutShare`` rule that is not met, makes the status ``cannot_compare``. That includes a ``NoWorseThan``
    rule on a slice under ``MIN_CLUSTERS``: naming the slice in a rule asks for a verdict on it, so a slice too
    small to pair refuses rather than being reported without gating.

    Args:
        candidate: The run under judgement.
        baseline: The run a ``NoWorseThan`` rule pairs it with; ``None`` when there is none.
        rules: The rules, in the order they are reported.

    Returns:
        The verdict over every rule.
    """
    verdicts: list[RuleVerdict] = []
    refusals: list[str] = []
    for rule in rules:
        if isinstance(rule, MinLowerBound):
            verdicts.append(_min_lower_bound(candidate, rule))
        elif isinstance(rule, NoWorseThan):
            verdict, refused = _no_worse_than(candidate, baseline, rule)
            verdicts.append(verdict)
            refusals.extend(refused)
        else:
            share = held_out_share(candidate)
            reasons = (f"held-out share {share:.2f} < {rule.minimum:.2f}",) if share < rule.minimum else ()
            verdicts.append(RuleVerdict(rule=rule, passed=not reasons, reasons=reasons))
            refusals.extend(reasons)
    if refusals:
        status: Literal["pass", "fail", "cannot_compare"] = "cannot_compare"
    elif any(not verdict.passed for verdict in verdicts):
        status = "fail"
    else:
        status = "pass"
    return GateResult(status=status, verdicts=tuple(verdicts), reason=refusals[0] if refusals else None)


def _where(slice_name: str | None) -> str:
    return f" in slice '{slice_name}'" if slice_name is not None else ""


def _grader(run: Run, metric: str) -> str | tuple[str, ...]:
    """The grader id ``metric`` names in ``run``: the id itself, or the one id whose last ``:`` part is ``metric``.

    Returns the ids that match (none, or several) when ``metric`` does not name exactly one grader.
    """
    graders = {score.grader_id for score in run.scores}
    if metric in graders:
        return metric
    named = sorted(grader for grader in graders if grader.rpartition(":")[2] == metric)
    return named[0] if len(named) == 1 else tuple(named)


def _min_lower_bound(candidate: Run, rule: MinLowerBound) -> RuleVerdict:
    grader = _grader(candidate, rule.metric)
    if isinstance(grader, tuple):
        if grader:
            return RuleVerdict(rule=rule, passed=False, reasons=(f"{rule.metric} is ambiguous ({', '.join(grader)})",))
        return RuleVerdict(rule=rule, passed=False, reasons=(f"{rule.metric} is not graded on the candidate",))
    values: Mapping[str, float] = {
        item: value
        for item, value in candidate.values(grader).items()
        if rule.slice_name is None or (item in candidate.items and in_slice(candidate.items[item], rule.slice_name))
    }
    if not values:
        return RuleVerdict(rule=rule, passed=False, reasons=(f"{rule.metric}{_where(rule.slice_name)}: no graded item",))
    if any(value not in (0.0, 1.0) for value in values.values()):
        reason = f"{rule.metric} is not 0/1 per item; a Wilson lower bound needs a binary metric"
        return RuleVerdict(rule=rule, passed=False, reasons=(reason,))
    lower = wilson_interval(sum(value == 1.0 for value in values.values()), len(values)).ci.lower
    reasons = ()
    if lower < rule.floor:
        reasons = (f"{rule.metric}{_where(rule.slice_name)}: Wilson lower {lower:+.4f} < floor {rule.floor:+.4f}",)
    return RuleVerdict(rule=rule, passed=not reasons, reasons=reasons)


def _no_worse_than(candidate: Run, baseline: Run | None, rule: NoWorseThan) -> tuple[RuleVerdict, tuple[str, ...]]:
    if baseline is None:
        return RuleVerdict(rule=rule, passed=False, reasons=(f"no baseline run to pair {rule.metric} with",)), ()
    try:
        comparison = compare(baseline, candidate, [rule.metric], slice_name=rule.slice_name)
    except ComparisonRefused as refused:
        return RuleVerdict(rule=rule, passed=False, reasons=(str(refused),)), (str(refused),)
    (measured,) = comparison.metrics
    floor = 0.0 - rule.margin
    where = _where(rule.slice_name)
    reasons: tuple[str, ...] = ()
    if rule.bound == "point":
        if measured.difference < floor:
            reasons = (f"{rule.metric}{where}: difference {measured.difference:+.4f} < {floor:+.4f}",)
    elif measured.ci is None:
        reasons = (f"{rule.metric}{where}: no interval over {measured.clusters} clusters; the bound needs MIN_CLUSTERS",)
    elif measured.ci.lower < floor:
        reasons = (f"{rule.metric}{where}: CI lower {measured.ci.lower:+.4f} < {floor:+.4f}",)
    return RuleVerdict(rule=rule, passed=not reasons, reasons=reasons, measured=measured), ()
