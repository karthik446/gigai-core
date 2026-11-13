# inspect_ai statistics, vendored numpy-free

- Upstream: https://github.com/UKGovernmentBEIS/inspect_ai at commit `9567b431e7174480e56826b435da8f9f1de73bb8`, files fetched from `https://raw.githubusercontent.com/UKGovernmentBEIS/inspect_ai/9567b431e7174480e56826b435da8f9f1de73bb8/<upstream path>` on 2026-11-10.
- Licence: MIT, "Copyright (c) 2024 UK AI Security Institute"; the upstream `LICENSE` is copied unchanged beside the files. Every lifted file starts with `# Lifted from inspect_ai <upstream path> @ 9567b431 (MIT, Copyright (c) 2024 UK AI Security Institute)`.
- Recipe: gigai-2 evals respike S-smoke §S1.2 and appendix A.7 (numpy-free hunks applied with `patch`, cleanly, no offset or fuzz), plus the edits a flat package needs (all imports point at `._metric`).
- `tests/test_stats_lifted.py` recomputes the vendored sha256 column, rebuilds each upstream file by reverse-applying the diff below and checks it against the upstream sha256 column. Change a vendored file and you must regenerate this file (the diff with `diff -u --label inspect_ai@9567b431/<upstream path> --label src/gigai_core/_vendor/inspect_ai/<file> <upstream> <vendored>`, then both hash columns).

## Files

| vendored file | upstream path | upstream sha256 | vendored sha256 |
|---|---|---|---|
| `std.py` | `src/inspect_ai/scorer/_metrics/std.py` | `e2d605ec523763dd2c3acbf11636f83fe537eda6daf66739e8d4b90dedd0dee3` | `fbdb12971324e117a41e43a8b9e805b3873ead2568dac5f480ca6b383765a7fd` |
| `krippendorff.py` | `src/inspect_ai/scorer/_metrics/krippendorff.py` | `edf785510825ac8934594af804157f42d31c1ce04f2b0da56c5496e54cf7ae9a` | `832feec0e46c5c6421891b04e6d78cdcbebf14af010a1b3bee80ce7caa0d6756` |
| `accuracy.py` | `src/inspect_ai/scorer/_metrics/accuracy.py` | `b9793301c57b16dbd2cc1b7e8de8c80c9f6d5f18654737c9a4724bedab6f257d` | `e9ff2590eeea438aaedea3c975849439993b89e507f41175d5796a8b85305632` |
| `LICENSE` | `LICENSE` | `c593c2afc81388521eaeb03f0d1a4b01bc3e5147ed9cb16c1f129e5be353b6af` | `c593c2afc81388521eaeb03f0d1a4b01bc3e5147ed9cb16c1f129e5be353b6af` |
| `_metric.py` | - (gigai-written shim for `src/inspect_ai/scorer/_metric.py`) | - | `2a321da72c933e40a7af51af61b50792d654ed0bea85dc3df6e978cdfaddaf99` |
| `__init__.py` | - (gigai-written) | - | `723688c43001e176cc8985510ce37404cf931c6e79f0c192995e98ba64897a26` |

## What changed (counted from the diff below: `-` and `+` lines, headers excluded)

| file | header (+) | import edits (-/+) | numpy removal (-/+) | total diff lines |
|---|---|---|---|---|
| `std.py` | 1 | 2 | 74 | 77 |
| `krippendorff.py` | 1 | 2 | 0 | 3 |
| `accuracy.py` | 1 | 2 | 0 | 3 |
| total | 3 | 6 | 74 | 83 |

- Numpy removal (74 diff lines, all in `std.py`): every lazy `import numpy as np` becomes `pass  # numpy import removed`; `np.std`/`np.var` with `ddof=1` become `statistics.stdev`/`statistics.variance`, `np.std` (ddof 0) becomes `statistics.pstdev`, `np.mean` becomes `statistics.fmean`, `np.sqrt` becomes `math.sqrt`, `np.random.choice(..., replace=True)` becomes `random.choices`, `np.quantile` becomes `_quantile` (linear interpolation, numpy's default, Hyndman and Fan 1996 type 7), `np.prod(1 - k / np.arange(...))` becomes `math.prod(...)` over a generator; `std.py` gains `import random`, `import statistics`.
- Import edits (3 source lines): `from .._metric import` becomes `from ._metric import` in std, krippendorff, accuracy. S had 2 edited lines because it kept upstream's package tree.
- Not carried into gigai-core: gigai 0.3.x also vendored `grouped.py` (`grouped`) and `reducer.py` (`mode_score`, `majority_score`, `mean_score`, `median_score`, `at_least`, `pass_at`, `pass_k`, `max_score`, `collect_score`); core's statistics layer does its own grouping and keeps one score per sample, so both files and the `ScoreReducer`/`score_reducer` shim pieces were dropped.
- Retyped for pyright strict in gigai-core (gigai-written shims only, upstream-derived files unchanged): `_metric.py` and `scorers/_shim.py` gain full annotations (`metric()` keeps its passthrough behaviour, with overloads), and the `__init__.py` docstrings no longer name gigai paths; the `sha256` rows below are of the retyped files. The test homes named for the scorer shims (`tests/test_scorers_lifted.py`, `tests/test_judge_validation.py`) arrive with the graders packets.

## Behaviour differences from upstream

The hunks only replace numpy calls, but the CPython `statistics` functions and `sorted()` they rely on behave differently from numpy on non-finite input. Measured on Python 3.13.1 against upstream `std.py` @ 9567b431 run with numpy 2.5.4 (score values `[0.1, 0.4, x, 0.8]`; the reader re-measured on 3.12.8 and 3.13.1 with the same results). `tests/test_stats_lifted.py::test_non_finite_and_degenerate_inputs_raise_where_upstream_gives_nan` and `::test_a_nan_score_silently_corrupts_the_bootstrap_ci` pin the vendored side, so changing it is a deliberate act.

**Callers (V3b, V4 and later) MUST drop or reject non-finite (NaN, `+inf`, `-inf`) score values before calling these metrics.** Most rows below raise, but the first row returns a finite, wrong interval without any error.

| input | metric | upstream (numpy) | vendored |
|---|---|---|---|
| any NaN value | `ci(method="bootstrap")` | `nan` bounds (`np.quantile` propagates NaN) | **silently wrong**: `_quantile` sorts bootstrap means that include NaN, and `sorted()` with NaN depends on where the NaN lands, so either bound can come back finite and wrong (even `lower > upper`). 19 values `0.05 * i` plus one NaN, B=1000, NaN at each of the 20 positions x seeds 0..19: 142/400 runs return at least one finite bound and 15/400 return two (e.g. NaN first, seed 16 on Python 3.13.1: `[0.43175, 0.44269]`; 3.11 and 3.12 give `[0.43175, nan]` for that seed, so the test loops over seeds 0..19 instead of pinning one); the reader saw 9/20 seeds with a finite interval on its placement. `_quantile([0.2, 0.5, 0.9, nan], 0.025) == 0.2225` but `_quantile([nan, 0.2, 0.5, 0.9], 0.025)` is `nan` |
| any NaN, `+inf` or `-inf` value | `stderr()`, `std()`, `ci(method="t")`, `bootstrap_stderr()` | `nan` | raises (from `statistics.stdev`/`pstdev`) `AttributeError: 'float' object has no attribute 'numerator'` up to Python 3.13.1, `ValueError: inf or nan encountered in data` on later patch releases (CI runners); the test accepts either |
| `+inf` or `-inf` value (only one sign) | `var()` | `nan` | returns that `inf` |
| both `+inf` and `-inf` values | `ci(method="t")`, `bootstrap_stderr()`, `ci(method="bootstrap")`, `ci(method="bootstrap", cluster=...)`, `stderr(cluster=...)`, `ci(method="t", cluster=...)` | `nan` | raises `ValueError: -inf + inf in fsum` (`statistics.fmean`) |
| `bootstrap_stderr(num_samples=0)` | | `nan` | raises `statistics.StatisticsError: pstdev requires at least one data point` |

Unchanged (same result in both): NaN in `var()` (`nan`); NaN or one-signed inf in `stderr(cluster=...)` / `ci(method="t", cluster=...)` (`nan`); `accuracy()` and `ci_wilson()`; `ci(method="bootstrap", num_samples=0)` raises `IndexError` in both (different message).

The bootstrap draws from Python's `random` module (`random.choices`) instead of `numpy.random`, so a seeded run gives different, equally distributed resamples and `random.seed` replaces `numpy.random.seed`. Bit-for-bit reproduction of a seeded value relies on CPython keeping the `random.choices` algorithm; Python's reproducibility guarantee covers only `random()`.

## Scorers (`scorers/`)

The `match`, `includes` and `pattern` scorer bodies, their helpers and the model-graded judge template are vendored from the same commit into the `scorers/` subpackage, under the same `LICENSE` (above). Their hash table, change counts, behaviour differences and diff are in `scorers/DIFF.md`, checked by `tests/test_scorers_lifted.py`; this file covers the top-level statistics files only.

## Diff against upstream (`diff -u`, generated, not hand-edited)

```diff
--- inspect_ai@9567b431/src/inspect_ai/scorer/_metrics/std.py
+++ src/gigai_core/_vendor/inspect_ai/std.py
@@ -1,8 +1,11 @@
+# Lifted from inspect_ai src/inspect_ai/scorer/_metrics/std.py @ 9567b431 (MIT, Copyright (c) 2024 UK AI Security Institute)
 import math
+import random
+import statistics
 from logging import getLogger
 from typing import Literal, cast
 
-from .._metric import (
+from ._metric import (
     Metric,
     SampleScore,
     Value,
@@ -35,7 +38,7 @@
     """
 
     def metric(scores: list[SampleScore]) -> float:
-        import numpy as np
+        pass  # numpy import removed
 
         values = [to_float(score.score.value) for score in scores]
         if not values:
@@ -43,13 +46,13 @@
             # numpy empty-slice warnings from the resampling loop), mirroring
             # the insufficient-data guards in stderr()/std()/var().
             return 0.0
-        std = np.std(
+        std = statistics.pstdev(
             [
-                np.mean(np.random.choice(values, len(values), replace=True))
+                statistics.fmean(random.choices(values, k=len(values)))
                 for _ in range(num_samples)
             ]
         )
-        return cast(float, std.item())
+        return std
 
     return metric
 
@@ -89,7 +92,7 @@
     For details, see Appendix A of https://arxiv.org/pdf/2411.00640.
     The version here uses a finite cluster correction (unlike the paper)
     """
-    import numpy as np
+    pass  # numpy import removed
 
     cluster_count = len(partition)
 
@@ -99,19 +102,19 @@
     if cluster_count < 2:
         return 0.0
 
-    cluster_arrays = [np.asarray(group, dtype=float) for group in partition]
-    values = np.concatenate(cluster_arrays)
-    mean = float(np.mean(values))
+    cluster_arrays = [[float(v) for v in group] for group in partition]
+    values = [v for group in cluster_arrays for v in group]
+    mean = statistics.fmean(values)
 
     # sum_i sum_j (s_i - mean)(s_j - mean) over a cluster is the square
     # of that cluster's deviation sum. Computing the identity directly
     # avoids materialising a k-by-k outer product for every cluster.
     clustered_variance = 0.0
     for cluster_data in cluster_arrays:
-        clustered_variance += ((cluster_data - mean).sum()) ** 2
+        clustered_variance += (sum(v - mean for v in cluster_data)) ** 2
 
     # Multiply by C / (C - 1) to unbias the variance estimate
-    standard_error = np.sqrt(
+    standard_error = math.sqrt(
         clustered_variance * cluster_count / (cluster_count - 1)
     ) / len(values)
 
@@ -144,7 +147,7 @@
         return _clustered_stderr(_cluster_partition(scores, cluster, values, "stderr"))
 
     def metric(scores: list[SampleScore]) -> float:
-        import numpy as np
+        pass  # numpy import removed
 
         values = [to_float(score.score.value) for score in scores]
         n = len(values)
@@ -155,10 +158,10 @@
             return 0
 
         # Calculate the sample standard deviation
-        sample_std = np.std(values, ddof=1)
+        sample_std = statistics.stdev(values)
 
         # Calculate the standard error of the mean
-        standard_error = sample_std / np.sqrt(n)
+        standard_error = sample_std / math.sqrt(n)
 
         return cast(float, standard_error)
 
@@ -219,7 +222,7 @@
     tail = (1.0 - level) / 2.0
 
     def metric_fn(scores: list[SampleScore]) -> Value:
-        import numpy as np
+        pass  # numpy import removed
 
         values = [to_float(score.score.value) for score in scores]
 
@@ -240,7 +243,7 @@
             return {"lower": point, "upper": point}
 
         if method == "t":
-            mean = float(np.mean(values))
+            mean = statistics.fmean(values)
             if partition is not None:
                 se = _clustered_stderr(partition)
                 df = max(len(partition) - 1, 1)
@@ -251,8 +254,8 @@
             return {"lower": mean - t * se, "upper": mean + t * se}
         else:
             boot_means = _bootstrap_means(values, partition, num_samples)
-            lower = float(np.quantile(boot_means, tail))
-            upper = float(np.quantile(boot_means, 1.0 - tail))
+            lower = _quantile(boot_means, tail)
+            upper = _quantile(boot_means, 1.0 - tail)
             return {"lower": lower, "upper": upper}
 
     return metric_fn
@@ -393,14 +396,14 @@
 
 def _clt_stderr(values: list[float]) -> float:
     """Central Limit Theorem standard error of the mean of `values`."""
-    import numpy as np
+    pass  # numpy import removed
 
     n = len(values)
     # standard deviation divides by n - ddof, so guard against n < 2
     if (n - 1) < 1:
         return 0.0
-    sample_std = np.std(values, ddof=1)
-    return cast(float, sample_std / np.sqrt(n))
+    sample_std = statistics.stdev(values)
+    return sample_std / math.sqrt(n)
 
 
 def _bootstrap_means(
@@ -414,26 +417,34 @@
     resamples whole clusters with replacement (cluster bootstrap) so
     within-cluster correlation is preserved.
     """
-    import numpy as np
+    pass  # numpy import removed
 
     if partition is None:
-        data = np.asarray(values, dtype=float)
-        n = len(data)
+        n = len(values)
         return [
-            float(np.mean(np.random.choice(data, n, replace=True)))
+            statistics.fmean(random.choices(values, k=n))
             for _ in range(num_samples)
         ]
 
-    cluster_arrays = [np.asarray(group, dtype=float) for group in partition]
+    cluster_arrays = [[float(v) for v in group] for group in partition]
     num_clusters = len(cluster_arrays)
     means: list[float] = []
     for _ in range(num_samples):
-        picks = np.random.choice(num_clusters, num_clusters, replace=True)
-        resampled = np.concatenate([cluster_arrays[i] for i in picks])
-        means.append(float(np.mean(resampled)))
+        picks = random.choices(range(num_clusters), k=num_clusters)
+        resampled = [v for i in picks for v in cluster_arrays[i]]
+        means.append(statistics.fmean(resampled))
     return means
 
 
+def _quantile(values: list[float], q: float) -> float:
+    """Linear-interpolation quantile (numpy default, Hyndman-Fan type 7)."""
+    xs = sorted(values)
+    pos = q * (len(xs) - 1)
+    lo = math.floor(pos)
+    hi = min(lo + 1, len(xs) - 1)
+    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
+
+
 def _t_inv_cdf(p: float, df: int) -> float:
     """Student-t inverse CDF, dependency-free.
 
@@ -552,7 +563,7 @@
     """
 
     def metric(scores: list[SampleScore]) -> float:
-        import numpy as np
+        pass  # numpy import removed
 
         values = [to_float(score.score.value) for score in scores]
         n = len(values)
@@ -563,7 +574,7 @@
             return 0
 
         # Calculate the sample standard deviation
-        sample_std = np.std(values, ddof=1)
+        sample_std = statistics.stdev(values)
 
         return cast(float, sample_std)
 
@@ -588,7 +599,7 @@
     """
 
     def metric(scores: list[SampleScore]) -> float:
-        import numpy as np
+        pass  # numpy import removed
 
         values = [to_float(score.score.value) for score in scores]
         n = len(values)
@@ -597,7 +608,7 @@
         if (n - 1) < 1:
             return 0
 
-        variance = np.var(values, ddof=1)
+        variance = statistics.variance(values)
 
         return cast(float, variance)
 
--- inspect_ai@9567b431/src/inspect_ai/scorer/_metrics/krippendorff.py
+++ src/gigai_core/_vendor/inspect_ai/krippendorff.py
@@ -1,9 +1,10 @@
+# Lifted from inspect_ai src/inspect_ai/scorer/_metrics/krippendorff.py @ 9567b431 (MIT, Copyright (c) 2024 UK AI Security Institute)
 from collections import Counter
 from collections.abc import Callable
 from logging import getLogger
 from typing import Literal
 
-from .._metric import (
+from ._metric import (
     Metric,
     SampleScore,
     ValueToFloat,
--- inspect_ai@9567b431/src/inspect_ai/scorer/_metrics/accuracy.py
+++ src/gigai_core/_vendor/inspect_ai/accuracy.py
@@ -1,6 +1,7 @@
+# Lifted from inspect_ai src/inspect_ai/scorer/_metrics/accuracy.py @ 9567b431 (MIT, Copyright (c) 2024 UK AI Security Institute)
 from logging import getLogger
 
-from .._metric import (
+from ._metric import (
     Metric,
     SampleScore,
     ValueToFloat,
```
