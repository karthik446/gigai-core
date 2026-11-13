"""Judge agreement against human labels and the pass rule.

Relocated from 0.3.2 ``tests/behaviors/evals/test_judge_validation.py``: ``test_agreement_reports_kappa...`` and
``test_pass_rule_needs_126_units...``. The rest of that file (the CLI, the store, the vendored templates, the
swap run) is not core's.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from gigai_core.agreement import (
    MIN_KAPPA,
    MIN_SWAP,
    MIN_UNITS,
    Agreement,
    JudgeNotValidated,
    agreement,
    judge_agreement,
    validation_failures,
    validation_passes,
)
from gigai_core.records import LabelRecord, LabelValue
from tests.test_comparison import ACC_B, ITEMS, NOW, fake_run

JUDGE = "judge:includes"


def label(item: str, value: LabelValue, *, at_hours: int = 0, label_id: str | None = None) -> LabelRecord:
    return LabelRecord(
        label_id=label_id or f"label-{item}-{value}-{at_hours}",
        dataset_id="release",
        item_id=item,
        item_version="1",
        label=value,
        source="human",
        recorded_at=NOW + timedelta(hours=at_hours),
    )


def test_agreement_reports_kappa_its_interval_and_the_rates() -> None:
    pairs = [(1, 1)] * 40 + [(0, 0)] * 40 + [(1, 0)] * 10 + [(0, 1)] * 10
    report = agreement(pairs)
    assert report.n == 100 and report.pairs == tuple(pairs)
    assert report.kappa is not None and report.kappa.kappa == pytest.approx(0.6)
    assert report.kappa.ci.lower < 0.6 < report.kappa.ci.upper
    assert (report.tpr, report.tnr, report.observed_agreement) == (0.8, 0.8, 0.8)
    assert report.kappa_undefined is None
    # both raters using one category: kappa is undefined, the rates still read
    degenerate = agreement([(1, 1)] * 5)
    assert (degenerate.kappa, degenerate.tpr, degenerate.tnr, degenerate.observed_agreement) == (None, 1.0, None, 1.0)
    assert degenerate.kappa_undefined == "chance agreement is 1 (both raters used one category); kappa is undefined"
    empty = agreement([])
    assert (empty.n, empty.kappa, empty.observed_agreement, empty.tpr, empty.tnr) == (0, None, None, None, None)
    assert empty.kappa_undefined == "table is empty"


def test_pass_rule_needs_126_units_kappa_0_6_and_swap_0_9() -> None:
    assert (MIN_UNITS, MIN_KAPPA, MIN_SWAP) == (126, 0.6, 0.9)
    assert validation_passes(kappa=0.6, n=126, swap_consistency=None)
    assert validation_passes(kappa=0.6, n=126, swap_consistency=0.9)
    assert validation_passes(kappa=0.6, n=126, swap_consistency=0.9, swap_pairs=66)
    assert validation_failures(kappa=0.6, n=125, swap_consistency=None) == ("n < 126",)
    assert validation_failures(kappa=0.59, n=126, swap_consistency=0.95) == ("kappa < 0.6",)
    assert validation_failures(kappa=0.9, n=200, swap_consistency=0.89) == ("swap_consistency < 0.9",)
    assert validation_failures(kappa=0.9, n=200, swap_consistency=None, swap_pairs=0) == ("no swap pairs",)
    assert validation_failures(kappa=None, n=10, swap_consistency=0.5) == (
        "n < 126",
        "kappa undefined",
        "swap_consistency < 0.9",
    )
    assert not validation_passes(kappa=None, n=10, swap_consistency=0.5)


def test_judge_agreement_pairs_the_judge_verdict_with_the_newest_human_label_of_each_item() -> None:
    # The judge says 1 on r04-r12 and 0 on r01-r03; humans agree except on r01 (met) and r12 (not met).
    run = fake_run({JUDGE: ACC_B})
    verdicts: dict[str, LabelValue] = {item: ("not met" if n < 3 else "met") for n, item in enumerate(ITEMS)}
    labels = [label(item, value) for item, value in verdicts.items()]
    labels += [label("r01", "met", at_hours=1), label("r12", "not met", at_hours=2), label("r12", "met", at_hours=1)]

    report = judge_agreement(run, JUDGE, labels)

    assert report.n == 12 and report.pairs[:3] == ((1, 0), (0, 0), (0, 0)) and report.pairs[-1] == (0, 1)
    assert report.observed_agreement == pytest.approx(10 / 12)
    assert (report.tpr, report.tnr) == pytest.approx((8 / 9, 2 / 3))
    assert report.kappa is not None and report.kappa.n == 12
    assert (report.labels_without_sample, report.labels_ungraded, report.samples_without_label, report.unknown_labels) == (
        0,
        0,
        0,
        0,
    )
    assert isinstance(report, Agreement)


def test_judge_agreement_counts_what_it_cannot_pair_and_applies_the_threshold() -> None:
    # r04 ungraded by the judge; r05 labelled unknown; r06 unlabelled; r99 labelled but never asked.
    run = fake_run({JUDGE: {**ACC_B, "r04": (None,)}})
    labels = [label(item, "met") for item in ITEMS if item not in ("r05", "r06")]
    labels += [label("r05", "unknown"), label("r99", "met")]

    report = judge_agreement(run, JUDGE, labels)

    assert report.n == 9 and (report.labels_without_sample, report.labels_ungraded) == (1, 1)
    assert (report.samples_without_label, report.unknown_labels) == (1, 1)
    assert report.kappa is not None and report.kappa.kappa == 0.0  # humans used one category, the judge both
    assert report.tpr == pytest.approx(6 / 9) and report.tnr is None

    # The threshold turns a continuous judge score into a verdict.
    scored = fake_run({JUDGE: {item: (0.1 * n,) for n, item in enumerate(ITEMS)}})  # 0.0 .. 1.1
    low = judge_agreement(scored, JUDGE, [label(item, "met") for item in ITEMS], threshold=0.3)
    high = judge_agreement(scored, JUDGE, [label(item, "met") for item in ITEMS], threshold=0.95)
    assert (low.tpr, high.tpr) == pytest.approx((9 / 12, 2 / 12))

    nothing = judge_agreement(run, "nope", labels)
    assert nothing.n == 0 and nothing.labels_ungraded == 11 and nothing.samples_without_label == 0


def test_judge_not_validated_is_an_exception_core_raises_for_an_unvalidated_judge() -> None:
    with pytest.raises(JudgeNotValidated, match="prompt digest"):
        raise JudgeNotValidated("judge judge:includes has no validation for prompt digest abc")
    assert issubclass(JudgeNotValidated, Exception) and not issubclass(JudgeNotValidated, ValueError)
