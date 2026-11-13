"""The lifted inspect_ai statistics give S-smoke's reference values with numpy blocked.

Fixtures BIN / REAL / Krippendorff and the reference values come from the gigai-2
evals respike S-smoke (§S1.3 table and appendix A.4/A.5, run against scipy, statsmodels and
the `krippendorff` package on 2026-11-09). S records 6 significant digits, so each value is
asserted twice: within 5e-7 of S's printed digits, and within 1e-12 of a closed form computed
here with the standard library (S measured lifted-vs-reference differences <= 1.7e-16).
Ported from gigai 0.3.2 `tests/behaviors/evals/test_stats_lifted.py`; the `grouped` and
`reducer` modules are not vendored in core, so their tests are gone.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import math
import random
import re
import statistics
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist, fmean, pstdev
from types import ModuleType

import pytest

from tests.conftest import REPO, SRC, VENDOR_PREFIX

VENDOR = SRC / "_vendor" / "inspect_ai"
PIN = "9567b431"
NOTICE = "(MIT, Copyright (c) 2024 UK AI Security Institute)"
LIFTED = {
    "std.py": "src/inspect_ai/scorer/_metrics/std.py",
    "krippendorff.py": "src/inspect_ai/scorer/_metrics/krippendorff.py",
    "accuracy.py": "src/inspect_ai/scorer/_metrics/accuracy.py",
}
GIGAI_WRITTEN_HEADERS = {
    "_metric.py": (
        "# gigai-written shim for inspect_ai src/inspect_ai/scorer/_metric.py @ 9567b431; "
        "value_to_float and is_finite_number lifted verbatim "
        "(MIT, Copyright (c) 2024 UK AI Security Institute; see LICENSE)"
    ),
    "__init__.py": (
        "# Vendored inspect_ai statistics @ 9567b431, numpy-free "
        "(MIT, Copyright (c) 2024 UK AI Security Institute; see LICENSE and DIFF.md)"
    ),
}

# S-smoke appendix A.4 fixtures.
BIN = [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1]  # 14 of 20 correct, clusters of 4
REAL = [0.1, 0.4, 0.35, 0.8, 0.9, 0.55, 0.6, 0.2, 0.75, 0.5]
KA_NOM = [["a", "a", "b"], ["b", "b", "b"], ["a", "b", "a"], ["c", "c", "c"], ["a", "a", "a"]]
KA_INT = [[1, 2, 1], [3, 3, 3], [2, 2, 1], [4, 4, 5], [1, 1, 2]]

S_DIGITS = 5e-7  # S prints 6 decimals
CLOSED_FORM = 1e-12

pytestmark = pytest.mark.usefixtures("block_numpy")


@dataclass(frozen=True)
class Lifted:
    """The vendored modules, imported fresh under the numpy block."""

    metric: ModuleType
    std: ModuleType
    krippendorff: ModuleType
    accuracy: ModuleType


@pytest.fixture
def lifted(block_numpy: None) -> Lifted:
    """Import the vendored modules fresh, after `block_numpy` has blocked numpy."""
    assert sys.modules["numpy"] is None

    def load(name: str) -> ModuleType:
        return importlib.import_module(f"{VENDOR_PREFIX}.inspect_ai.{name}")

    return Lifted(metric=load("_metric"), std=load("std"), krippendorff=load("krippendorff"), accuracy=load("accuracy"))


def _bin_scores(m: Lifted) -> list[object]:
    return [
        m.metric.SampleScore(score=m.metric.Score(value="C" if v else "I"), sample_id=i, sample_metadata={"doc": i // 4})
        for i, v in enumerate(BIN)
    ]


def _real_scores(m: Lifted) -> list[object]:
    return [m.metric.SampleScore(score=m.metric.Score(value=v), sample_id=i) for i, v in enumerate(REAL)]


def _sample_sd(xs: list[float]) -> float:
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))


def _student_t_cdf(t: float, df: int) -> float:
    """Student-t CDF in closed form for integer df (Abramowitz & Stegun 26.7.3 / 26.7.4)."""
    theta = math.atan(t / math.sqrt(df))
    c2 = math.cos(theta) ** 2
    if df % 2:
        series, term = (1.0, 1.0) if df > 1 else (0.0, 0.0)
        for j in range(1, (df - 3) // 2 + 1):
            term *= (2 * j) / (2 * j + 1) * c2
            series += term
        return 0.5 + (theta + math.sin(theta) * math.cos(theta) * series) / math.pi
    series, term = 1.0, 1.0
    for j in range(1, (df - 2) // 2 + 1):
        term *= (2 * j - 1) / (2 * j) * c2
        series += term
    return 0.5 + math.sin(theta) * series / 2


def _krippendorff_reference(units: list[list[str]] | list[list[int]], interval: bool) -> Fraction:
    """Krippendorff's alpha from the coincidence matrix, in exact rationals (Krippendorff 2011)."""

    def delta(a: str | int, b: str | int) -> Fraction:
        if interval:
            assert isinstance(a, int) and isinstance(b, int)
            return Fraction((a - b) ** 2)
        return Fraction(int(a != b))

    coincidences: dict[tuple[str | int, str | int], Fraction] = {}
    for unit in units:
        if len(unit) < 2:
            continue
        for i, a in enumerate(unit):
            for j, b in enumerate(unit):
                if i != j:
                    coincidences[(a, b)] = coincidences.get((a, b), Fraction(0)) + Fraction(1, len(unit) - 1)
    marginals: dict[str | int, Fraction] = {}
    for (a, _), count in coincidences.items():
        marginals[a] = marginals.get(a, Fraction(0)) + count
    n = sum(marginals.values(), Fraction(0))
    observed = sum((count * delta(a, b) for (a, b), count in coincidences.items()), Fraction(0))
    expected = sum((marginals[a] * marginals[b] * delta(a, b) for a in marginals for b in marginals), Fraction(0)) / (n - 1)
    return 1 - observed / expected


def test_numpy_is_blocked_and_the_vendored_stats_reimport_under_the_block() -> None:
    assert sys.modules["numpy"] is None
    with pytest.raises(ImportError):
        importlib.import_module("numpy")
    assert not [name for name in sys.modules if name.startswith(VENDOR_PREFIX)]
    importlib.import_module(f"{VENDOR_PREFIX}.inspect_ai.std")
    assert sys.modules["numpy"] is None


def test_accuracy_and_clt_stderr_match_s_and_the_closed_form(lifted: Lifted) -> None:
    bins, reals = _bin_scores(lifted), _real_scores(lifted)

    assert lifted.accuracy.accuracy()(bins) == pytest.approx(0.7, abs=CLOSED_FORM)
    stderr_bin = lifted.std.stderr()(bins)
    stderr_real = lifted.std.stderr()(reals)
    assert abs(stderr_bin - 0.105131) <= S_DIGITS
    assert abs(stderr_real - 0.082006) <= S_DIGITS
    # binary: sample variance (14 * 0.3^2 + 6 * 0.7^2) / 19 = 4.2 / 19
    assert abs(stderr_bin - math.sqrt(4.2 / 19 / 20)) <= CLOSED_FORM
    assert abs(stderr_real - _sample_sd(REAL) / math.sqrt(len(REAL))) <= CLOSED_FORM


def test_std_and_var_match_s_and_the_closed_form(lifted: Lifted) -> None:
    reals = _real_scores(lifted)

    std = lifted.std.std()(reals)
    var = lifted.std.var()(reals)
    assert abs(std - 0.259326) <= S_DIGITS
    assert abs(var - 0.06725) <= CLOSED_FORM  # 0.6725 / 9 * 0.9: S prints it exactly
    assert abs(std - _sample_sd(REAL)) <= CLOSED_FORM


def test_ci_t_is_mean_plus_minus_the_student_t_quantile(lifted: Lifted) -> None:
    interval = lifted.std.ci(level=0.95, method="t")(_real_scores(lifted))

    assert abs(interval["lower"] - 0.3295) <= 5e-5  # S prints 4 decimals
    assert abs(interval["upper"] - 0.7005) <= 5e-5
    mean, se = fmean(REAL), _sample_sd(REAL) / math.sqrt(len(REAL))
    assert abs((interval["lower"] + interval["upper"]) / 2 - mean) <= CLOSED_FORM
    t = (interval["upper"] - mean) / se
    assert abs(_student_t_cdf(t, len(REAL) - 1) - 0.975) <= CLOSED_FORM


def test_ci_wilson_on_bin_matches_s_and_the_closed_form(lifted: Lifted) -> None:
    interval = lifted.std.ci_wilson(level=0.95)(_bin_scores(lifted))

    assert abs(interval["lower"] - 0.481027) <= S_DIGITS
    assert abs(interval["upper"] - 0.854523) <= S_DIGITS
    n, p, z = len(BIN), sum(BIN) / len(BIN), NormalDist().inv_cdf(0.975)
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    assert abs(interval["lower"] - (centre - half)) <= CLOSED_FORM
    assert abs(interval["upper"] - (centre + half)) <= CLOSED_FORM


def test_clustered_stderr_and_clustered_ci_t(lifted: Lifted) -> None:
    bins = _bin_scores(lifted)

    # Miller 2024 (arXiv:2411.00640) appendix A with Inspect's C / (C - 1): cluster deviation
    # sums 0.2, 0.2, 0.2, -0.8, 0.2 -> sqrt(0.8 * 5 / 4) / 20 = 0.05 (S: = statsmodels cluster OLS)
    assert abs(lifted.std.stderr(cluster="doc")(bins) - 0.05) <= CLOSED_FORM
    interval = lifted.std.ci(level=0.95, method="t", cluster="doc")(bins)
    t = (interval["upper"] - 0.7) / 0.05
    assert abs(_student_t_cdf(t, 5 - 1) - 0.975) <= CLOSED_FORM
    with pytest.raises(ValueError):
        lifted.std.stderr(cluster="missing")(bins)


def test_krippendorff_alpha_matches_s_and_the_coincidence_matrix(lifted: Lifted) -> None:
    def samples(units: list[list[str]] | list[list[int]]) -> list[object]:
        return [lifted.metric.SampleScore(score=lifted.metric.Score(value=u), sample_id=i) for i, u in enumerate(units)]

    nominal = lifted.krippendorff.krippendorff_alpha(level="nominal")(samples(KA_NOM))
    interval = lifted.krippendorff.krippendorff_alpha(level="interval")(samples(KA_INT))
    assert abs(nominal - 0.605634) <= S_DIGITS
    assert abs(interval - 0.84) <= CLOSED_FORM
    assert abs(nominal - float(_krippendorff_reference(KA_NOM, interval=False))) <= CLOSED_FORM
    assert abs(interval - float(_krippendorff_reference(KA_INT, interval=True))) <= CLOSED_FORM


def test_bootstrap_stderr_converges_to_the_analytic_value(lifted: Lifted) -> None:
    reals = _real_scores(lifted)
    analytic = pstdev(REAL) / math.sqrt(len(REAL))  # large-B limit of the bootstrap SE of the mean
    assert abs(analytic - 0.077798) <= S_DIGITS

    # S's run_lift.py (appendix A.4) drew 10,000 resamples although its table row says 1000.
    # Both checks below are seed-0 deterministic reproductions, not statistical tolerances: the
    # Monte Carlo SE of the bootstrap SE at B=10000 is ~5.5e-4 (> 4e-4), and over seeds 0..9
    # only 7/10 land within 4e-4 of the analytic value. The binding check is the 5e-7 pin to S's
    # stdlib value 0.077927, which also assumes CPython keeps the `random.choices` algorithm
    # (Python's reproducibility guarantee covers only `random()`).
    random.seed(0)
    at_10000 = lifted.std.bootstrap_stderr(num_samples=10000)(reals)
    assert abs(at_10000 - 0.077798) <= 4e-4
    assert abs(at_10000 - 0.077927) <= S_DIGITS
    # At B=1000 the Monte Carlo SE of the SE is ~0.0778 / sqrt(2000) = 1.7e-3; over seeds 0..9
    # only 1/10 land within 4e-4 and 10/10 within 5e-3 (max 4.03e-3), so 5e-3 (~3 Monte Carlo
    # SEs) is the honest bound.
    random.seed(0)
    at_1000 = lifted.std.bootstrap_stderr(num_samples=1000)(reals)
    assert abs(at_1000 - 0.077798) <= 5e-3
    random.seed(0)
    assert lifted.std.bootstrap_stderr(num_samples=1000)(reals) == at_1000


def test_bootstrap_ci_and_cluster_bootstrap(lifted: Lifted) -> None:
    random.seed(0)
    interval = lifted.std.ci(level=0.95, method="bootstrap", num_samples=10000)(_real_scores(lifted))
    # S: within 5e-3 of scipy's percentile bootstrap; the normal-theory bounds are mean -/+ 1.96 * 0.0778
    assert abs(interval["lower"] - (0.515 - 1.96 * 0.077798)) <= 1e-2
    assert abs(interval["upper"] - (0.515 + 1.96 * 0.077798)) <= 1e-2
    random.seed(0)
    clustered = lifted.std.ci(level=0.95, method="bootstrap", num_samples=2000, cluster="doc")(_bin_scores(lifted))
    assert 0.5 <= clustered["lower"] < 0.7 < clustered["upper"] <= 0.75  # cluster means are 0.5 or 0.75
    # _quantile is numpy's default (linear, Hyndman and Fan type 7)
    assert lifted.std._quantile([1.0, 2.0, 3.0, 4.0], 0.25) == 1.75
    assert lifted.std._quantile([4.0, 1.0, 3.0, 2.0], 0.9) == pytest.approx(3.7)


def test_non_finite_and_degenerate_inputs_raise_where_upstream_gives_nan(lifted: Lifted) -> None:
    """Pins the vendored side of DIFF.md "Behaviour differences from upstream" (upstream numpy returns nan)."""

    def samples(values: list[float]) -> list[object]:
        return [
            lifted.metric.SampleScore(score=lifted.metric.Score(value=v), sample_id=i, sample_metadata={"doc": i // 2})
            for i, v in enumerate(values)
        ]

    std = lifted.std
    nan, inf = float("nan"), float("inf")
    for bad in (nan, inf, -inf):
        scores = samples([0.1, 0.4, bad, 0.8])
        for name, metric in {
            "stderr": std.stderr(),
            "std": std.std(),
            "ci_t": std.ci(method="t"),
            "bootstrap_stderr": std.bootstrap_stderr(num_samples=100),
        }.items():
            # which exception depends on the CPython patch release: AttributeError "numerator" up to 3.13.1, ValueError "inf or nan" later
            with pytest.raises((AttributeError, ValueError), match=r"numerator|inf or nan encountered in data"):
                metric(scores)
                pytest.fail(f"{name} on {bad} did not raise")
        # unchanged from upstream: clustered SE and clustered t interval give nan
        assert math.isnan(std.stderr(cluster="doc")(scores))
        assert all(math.isnan(bound) for bound in std.ci(method="t", cluster="doc")(scores).values())
    assert std.var()(samples([0.1, 0.4, inf, 0.8])) == inf
    assert std.var()(samples([0.1, -inf, 0.4, 0.8])) == -inf
    assert math.isnan(std.var()(samples([0.1, 0.4, nan, 0.8])))  # unchanged

    both_infinities = samples([0.1, inf, -inf, 0.8])
    for metric in (
        std.ci(method="t"),
        std.bootstrap_stderr(num_samples=100),
        std.ci(method="bootstrap", num_samples=100),
        std.ci(method="bootstrap", num_samples=100, cluster="doc"),
        std.stderr(cluster="doc"),
        std.ci(method="t", cluster="doc"),
    ):
        with pytest.raises(ValueError, match=r"-inf \+ inf in fsum"):
            metric(both_infinities)

    with pytest.raises(statistics.StatisticsError, match="pstdev requires at least one data point"):
        std.bootstrap_stderr(num_samples=0)(samples([0.1, 0.5]))


def test_a_nan_score_silently_corrupts_the_bootstrap_ci(lifted: Lifted) -> None:
    """Documents the hazard in DIFF.md: upstream's np.quantile gives nan bounds; the vendored one may not.

    `_quantile` sorts, and `sorted()` on a list containing NaN depends on where the NaN lands, so a
    NaN score can yield a finite, wrong interval. Callers must drop or reject non-finite scores.
    Which seed lands the NaN where depends on the Python version's random sequence, so the test
    asserts the hazard over seeds, never on one seed's numbers.
    """
    nan = float("nan")
    assert lifted.std._quantile([0.2, 0.5, 0.9, nan], 0.025) == 0.2225
    assert math.isnan(lifted.std._quantile([nan, 0.2, 0.5, 0.9], 0.025))

    values = [nan] + [0.05 * i for i in range(19)]
    scores = [lifted.metric.SampleScore(score=lifted.metric.Score(value=v), sample_id=i) for i, v in enumerate(values)]
    intervals: list[dict[str, float]] = []
    for seed in range(20):
        random.seed(seed)
        intervals.append(lifted.std.ci(level=0.95, method="bootstrap", num_samples=1000)(scores))
    assert any(not math.isnan(bound) for interval in intervals for bound in interval.values()), "no seed gave a finite bound"


def diff_sections(diff_md: str) -> dict[str, list[str]]:
    block = diff_md.split("```diff\n", 1)[1].rsplit("```", 1)[0]
    sections: dict[str, list[str]] = {}
    current: list[str] = []
    for line in block.splitlines():
        if line.startswith("--- "):
            current = []
        elif line.startswith("+++ "):
            sections[line[4:].rsplit("/", 1)[1]] = current
        else:
            current.append(line)
    return sections


def reverse_apply(vendored: list[str], diff: list[str]) -> list[str]:
    """Rebuild the upstream file from the vendored file and its `diff -u` (upstream -> vendored)."""
    upstream: list[str] = []
    cursor = 0
    for line in diff:
        hunk = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if hunk:
            start = int(hunk.group(1)) - (0 if line.split("+", 1)[1].split()[0].endswith(",0") else 1)
            upstream.extend(vendored[cursor:start])
            cursor = start
        elif line.startswith(" ") or line == "":
            assert vendored[cursor] == line[1:], (cursor, line)
            upstream.append(line[1:])
            cursor += 1
        elif line.startswith("-"):
            upstream.append(line[1:])
        elif line.startswith("+"):
            assert vendored[cursor] == line[1:], (cursor, line)
            cursor += 1
    return upstream + vendored[cursor:]


def test_diff_md_records_the_vendored_and_upstream_hashes() -> None:
    diff_md = (VENDOR / "DIFF.md").read_text(encoding="utf-8")
    assert "9567b431e7174480e56826b435da8f9f1de73bb8" in diff_md
    rows = re.findall(r"^\| `([^`]+)` \| (.+?) \| (`[0-9a-f]{64}`|-) \| `([0-9a-f]{64})` \|$", diff_md, re.MULTILINE)
    table = {name: (upstream_sha.strip("`"), vendored_sha) for name, _path, upstream_sha, vendored_sha in rows}

    on_disk = {path.name for path in VENDOR.iterdir() if path.is_file() and path.name != "DIFF.md"}
    assert set(table) == on_disk
    for name, (_, vendored_sha) in table.items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == vendored_sha, f"{name} changed without updating DIFF.md"
    assert table["LICENSE"][0] == table["LICENSE"][1]

    sections = diff_sections(diff_md)
    assert set(sections) == set(LIFTED)
    for name, diff in sections.items():
        vendored = (VENDOR / name).read_text(encoding="utf-8").splitlines()
        upstream = "\n".join(reverse_apply(vendored, diff)) + "\n"
        assert hashlib.sha256(upstream.encode("utf-8")).hexdigest() == table[name][0], f"{name}: diff and upstream hash disagree"


def test_every_vendored_python_file_carries_its_header() -> None:
    python_files = {path.name for path in VENDOR.glob("*.py")}
    assert python_files == set(LIFTED) | set(GIGAI_WRITTEN_HEADERS)
    for name, upstream_path in LIFTED.items():
        first = (VENDOR / name).read_text(encoding="utf-8").splitlines()[0]
        assert first == f"# Lifted from inspect_ai {upstream_path} @ {PIN} {NOTICE}"
    for name, header in GIGAI_WRITTEN_HEADERS.items():
        assert (VENDOR / name).read_text(encoding="utf-8").splitlines()[0] == header
    assert "MIT License" in (VENDOR / "LICENSE").read_text(encoding="utf-8")
    assert "Copyright (c) 2024 UK AI Security Institute" in (VENDOR / "LICENSE").read_text(encoding="utf-8")


def test_nothing_under_gigai_core_imports_numpy() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if not isinstance(node, ast.Import | ast.ImportFrom | ast.Call):
                continue
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) in {"__import__", "import_module"}:
                modules = [arg.value for arg in node.args[:1] if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
            offenders += [f"{path.relative_to(REPO)}:{node.lineno} {name}" for name in modules if name.split(".")[0] == "numpy"]
    assert offenders == []


def _python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-I", "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=60, check=False)


def test_fresh_interpreter_imports_std_with_numpy_absent() -> None:
    result = _python(
        f"""
        import sys
        sys.modules["numpy"] = None
        import {VENDOR_PREFIX}.inspect_ai.std
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_an_importable_numpy_is_never_imported(tmp_path: Path) -> None:
    (tmp_path / "numpy").mkdir()
    (tmp_path / "numpy" / "__init__.py").write_text("__version__ = '0.0-fake'\n", encoding="utf-8")
    result = _python(
        f"""
        import importlib.util, random, sys
        sys.path.insert(0, {str(tmp_path)!r})
        assert importlib.util.find_spec("numpy") is not None
        from {VENDOR_PREFIX}.inspect_ai import accuracy, krippendorff, std
        from {VENDOR_PREFIX}.inspect_ai._metric import SampleScore, Score
        scores = [SampleScore(score=Score(value=v), sample_id=i, sample_metadata={{"g": i % 3}}) for i, v in enumerate([0.1, 0.4, 0.8, 0.9, 0.3, 0.6])]
        random.seed(0)
        for m in (std.stderr(), std.stderr(cluster="g"), std.std(), std.var(), std.bootstrap_stderr(num_samples=50),
                  std.ci(), std.ci(method="bootstrap", num_samples=50), std.ci_wilson(), accuracy.accuracy()):
            m(scores)
        krippendorff.krippendorff_alpha(level="interval")([SampleScore(score=Score(value=[1, 2, 1]))] * 2)
        print("numpy" in sys.modules)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
