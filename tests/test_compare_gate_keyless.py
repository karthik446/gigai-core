"""Two saved runs compared and gated with no API key, no model and no store: files in, verdict out.

The DESIGN 4.9 walk-through: a run saved by ``save_run`` opens with ``open_run`` and feeds ``compare`` and
``gate`` as it is. No provider is touched; the modules under test never import one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gigai_core.comparison import ComparisonRefused, compare
from gigai_core.gate import HeldOutShare, MinLowerBound, NoWorseThan, gate
from gigai_core.run import open_run, save_run
from tests.test_comparison import ITEMS, dataset, fake_run

INCLUDES = "includes"
ALL_RIGHT = {item: (1.0,) for item in ITEMS}
ONE_WRONG = {item: (0.0 if item == "r12" else 1.0,) for item in ITEMS}


def test_two_saved_runs_compare_and_gate_without_a_key_a_model_or_a_store(tmp_path: Path) -> None:
    data = dataset(licences={item: "private" for item in ITEMS[:6]})
    save_run(fake_run({INCLUDES: ALL_RIGHT}, run_id="baseline", data=data), tmp_path / "baseline.jsonl")
    save_run(fake_run({INCLUDES: ONE_WRONG}, run_id="candidate", data=data), tmp_path / "candidate.jsonl")

    baseline = open_run(tmp_path / "baseline.jsonl", dataset=data)
    candidate = open_run(tmp_path / "candidate.jsonl", dataset=data)

    comparison = compare(baseline, candidate, [INCLUDES])
    (includes,) = comparison.metrics
    assert (comparison.run_a, comparison.run_b, comparison.items, comparison.clusters) == ("baseline", "candidate", 12, 10)
    assert (includes.mean_a, includes.mean_b, includes.difference) == pytest.approx((1.0, 11 / 12, -1 / 12))
    assert includes.mcnemar is not None and (includes.mcnemar.b, includes.mcnemar.c) == (1, 0)
    assert includes.test == "mcnemar" and includes.p == 1.0 and includes.ci is not None

    rules = [MinLowerBound(metric=INCLUDES, floor=0.6), NoWorseThan(metric=INCLUDES, margin=0.05), HeldOutShare(minimum=0.5)]
    verdict = gate(candidate, baseline, rules)

    assert verdict.status == "fail" and verdict.exit_code == 1
    assert [v.passed for v in verdict.verdicts] == [True, False, True]
    assert verdict.failed[0].rule == NoWorseThan(metric=INCLUDES, margin=0.05)
    assert verdict.failed[0].reasons == ("includes: difference -0.0833 < -0.0500",)
    assert verdict.failed[0].measured is not None and verdict.failed[0].measured.difference == pytest.approx(-1 / 12)
    assert gate(candidate, baseline, [NoWorseThan(metric=INCLUDES, margin=0.1)]).passed

    # Without a baseline, the single-run rules still gate; the paired rule fails for want of one.
    alone = gate(candidate, None, rules)
    assert alone.status == "fail" and alone.failed[0].reasons == ("no baseline run to pair includes with",)
    assert gate(candidate, None, [MinLowerBound(metric=INCLUDES, floor=0.6), HeldOutShare(minimum=0.5)]).passed

    # A run of another dataset is refused by compare and cannot be gated against.
    other = fake_run({INCLUDES: ALL_RIGHT}, run_id="other", data=dataset(prompt="ask", licences={}))
    with pytest.raises(ComparisonRefused, match="^cannot compare: dataset sha256 differs") as refused:
        compare(other, candidate, [INCLUDES])
    assert refused.value.code == "different_dataset"
    assert gate(candidate, other, rules).exit_code == 2

