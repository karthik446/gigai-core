"""Typed statistics for evals: intervals, paired tests, agreement, power and pass@k (DESIGN 4.7).

Every function returns a frozen `CoreModel` whose floats are finite; non-finite input raises
`NonFiniteScoreError` and clustered estimates on fewer than `MIN_CLUSTERS` clusters raise
`TooFewClustersError`. Stdlib only, plus the numpy-free inspect_ai lift in `gigai_core._vendor`.
"""

from .intervals import (
    MAX_BOOTSTRAP_RESAMPLES,
    MIN_CLUSTERS,
    ClusterBootstrap,
    Interval,
    NonFiniteScoreError,
    TooFewClustersError,
    WilsonInterval,
    paired_cluster_bootstrap,
    require_min_clusters,
    wilson_interval,
)
from .kappa import (
    Auc,
    Kappa,
    KrippendorffAlpha,
    auc_hanley_mcneil,
    cohen_kappa,
    krippendorff_alpha,
)
from .passk import PassK, pass_at_k, pass_hat_k, pass_k
from .power import (
    MdeContinuous,
    MdeRow,
    MinimumDetectableDifference,
    NoiseBound,
    VarianceComponents,
    mde_continuous,
    mde_sample_size,
    mde_table,
    minimum_detectable_difference,
    noise_bound,
    variance_components,
)
from .tests import (
    WILCOXON_EXACT_MAX_N,
    BenjaminiHochberg,
    Holm,
    McNemarExact,
    Tost,
    WilcoxonSignedRank,
    benjamini_hochberg,
    holm,
    mcnemar_exact,
    tost,
    wilcoxon_signed_rank,
)

__all__ = [
    "MAX_BOOTSTRAP_RESAMPLES",
    "MIN_CLUSTERS",
    "WILCOXON_EXACT_MAX_N",
    "Auc",
    "BenjaminiHochberg",
    "ClusterBootstrap",
    "Holm",
    "Interval",
    "Kappa",
    "KrippendorffAlpha",
    "McNemarExact",
    "MdeContinuous",
    "MdeRow",
    "MinimumDetectableDifference",
    "NoiseBound",
    "NonFiniteScoreError",
    "PassK",
    "TooFewClustersError",
    "Tost",
    "VarianceComponents",
    "WilcoxonSignedRank",
    "WilsonInterval",
    "auc_hanley_mcneil",
    "benjamini_hochberg",
    "cohen_kappa",
    "holm",
    "krippendorff_alpha",
    "mcnemar_exact",
    "mde_continuous",
    "mde_sample_size",
    "mde_table",
    "minimum_detectable_difference",
    "noise_bound",
    "paired_cluster_bootstrap",
    "pass_at_k",
    "pass_hat_k",
    "pass_k",
    "require_min_clusters",
    "tost",
    "variance_components",
    "wilcoxon_signed_rank",
    "wilson_interval",
]
