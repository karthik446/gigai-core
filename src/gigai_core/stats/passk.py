"""pass@k and pass^k from ``n`` samples of which ``c`` passed (new in gigai-core; DESIGN 4.7).

Unbiased estimators over samples drawn without replacement: Chen et al. 2021, "Evaluating Large Language
Models Trained on Code", https://arxiv.org/abs/2107.03374, appendix on pass@k (fetched 2026-11-09).
"""

from __future__ import annotations

import math

from ..models import CoreModel
from .intervals import checked_int


class PassK(CoreModel):
    """pass@k (at least one of k passes) and pass^k (all k pass) estimated from ``n`` samples, ``c`` passing."""

    n: int
    c: int
    k: int
    pass_at_k: float
    pass_hat_k: float


def _checked(n: int, c: int, k: int) -> tuple[int, int, int]:
    n = checked_int(n, "n", minimum=1)
    c = checked_int(c, "c", minimum=0)
    k = checked_int(k, "k", minimum=1)
    if c > n:
        raise ValueError(f"c must not exceed n, got c={c}, n={n}")
    if k > n:
        raise ValueError(f"k must not exceed n, got k={k}, n={n}")
    return n, c, k


def pass_at_k(n: int, c: int, k: int) -> float:
    """The probability that at least one of k samples passes, estimated from n samples of which c passed.

    Chen et al. 2021 (https://arxiv.org/abs/2107.03374): 1 - C(n - c, k) / C(n, k), the same quantity as
    1 - prod_{i = n - c + 1..n} (1 - k / i); exact with integer binomials, so it lies in [0, 1] and is
    exactly 1 when fewer than k samples failed.

    Args:
        n: Samples drawn (``n >= 1``).
        c: Samples that passed (``0 <= c <= n``).
        k: Samples that would be drawn (``1 <= k <= n``).

    Returns:
        The estimate in [0, 1].
    """
    n, c, k = _checked(n, c, k)
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """The probability that all of k samples pass, estimated from n samples of which c passed.

    C(c, k) / C(n, k): the share of k-subsets of the n samples that are all passes (Chen et al. 2021,
    https://arxiv.org/abs/2107.03374, read for pass^k the same way as for pass@k); exactly 0 when
    fewer than k samples passed.

    Args:
        n: Samples drawn (``n >= 1``).
        c: Samples that passed (``0 <= c <= n``).
        k: Samples that would be drawn (``1 <= k <= n``).

    Returns:
        The estimate in [0, 1].
    """
    n, c, k = _checked(n, c, k)
    return math.comb(c, k) / math.comb(n, k)


def pass_k(n: int, c: int, k: int) -> PassK:
    """Both estimates, pass@k and pass^k, for one (n, c, k) (https://arxiv.org/abs/2107.03374).

    Args:
        n: Samples drawn (``n >= 1``).
        c: Samples that passed (``0 <= c <= n``).
        k: Samples that would be drawn (``1 <= k <= n``).

    Returns:
        The inputs with both estimates.
    """
    n, c, k = _checked(n, c, k)
    return PassK(n=n, c=c, k=k, pass_at_k=pass_at_k(n, c, k), pass_hat_k=pass_hat_k(n, c, k))
