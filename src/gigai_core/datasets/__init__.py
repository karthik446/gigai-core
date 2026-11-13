"""The public evaluation sets: ten loaders, six shipped ``ci`` slices, and the release builder.

``load_slice(name)`` reads a shipped slice from the wheel; ``build_release(name, directory)`` downloads a set's pinned
upstream files and builds a slice on the user's machine. Attribution and licence terms are in ``NOTICES.md``.
"""

from __future__ import annotations

from . import (
    contractnli,
    ifeval,
    jsonschemabench,
    judgebench,
    mtbench_human,
    quality,
    ragtruth,
    rewardbench2,
    tinygsm8k,
    tinymmlu,
)
from .loader import (
    LOADERS,
    SEED,
    SETS,
    SHIPPED,
    DatasetDownloadError,
    DatasetLoader,
    ParquetUnavailable,
    SliceCount,
    SliceNotShipped,
    SourceFile,
    Suite,
    build_release,
    load_slice,
)

__all__ = [
    "LOADERS",
    "SEED",
    "SETS",
    "SHIPPED",
    "DatasetDownloadError",
    "DatasetLoader",
    "ParquetUnavailable",
    "SliceCount",
    "SliceNotShipped",
    "SourceFile",
    "Suite",
    "build_release",
    "contractnli",
    "ifeval",
    "jsonschemabench",
    "judgebench",
    "load_slice",
    "mtbench_human",
    "quality",
    "ragtruth",
    "rewardbench2",
    "tinygsm8k",
    "tinymmlu",
]
