"""gigai-core: typed evaluation records, graders, statistics and run files.

The façade (DESIGN 4.9): the names a lesson cell imports with ``from gigai_core import ...``. Everything else is
imported from its module. Importing this package loads pydantic and the standard library only.
"""

from __future__ import annotations

from importlib.metadata import version

from .agreement import judge_agreement
from .budget import CallBudget
from .comparison import compare
from .dataset import Dataset, DatasetItem
from .gate import MinLowerBound, NoWorseThan, gate
from .graders import (
    answer_letter,
    autoevals_judge,
    bool_field,
    constant,
    ifeval,
    includes,
    json_valid,
    label_share,
    labelled,
    last_number,
    llm_judge,
    match,
    model_graded_qa,
    openevals_judge,
    pairwise_judge,
    pattern,
    ranks_first,
    read_labels,
    schema_valid,
    verdict_equals,
    with_wrong_labels,
    write_labels,
)
from .models import CoreModel
from .providers import CallSettings, ModelProvider
from .recorded import RecordedProvider
from .run import Run, open_run, run, save_run
from .stats import cohen_kappa
from .trace import JsonlTrace, read_spans

__version__ = version("gigai-core")

__all__ = [
    "CallBudget",
    "CallSettings",
    "CoreModel",
    "Dataset",
    "DatasetItem",
    "JsonlTrace",
    "MinLowerBound",
    "ModelProvider",
    "NoWorseThan",
    "RecordedProvider",
    "Run",
    "answer_letter",
    "autoevals_judge",
    "bool_field",
    "cohen_kappa",
    "compare",
    "constant",
    "gate",
    "ifeval",
    "includes",
    "json_valid",
    "judge_agreement",
    "label_share",
    "labelled",
    "last_number",
    "llm_judge",
    "match",
    "model_graded_qa",
    "open_run",
    "openevals_judge",
    "pairwise_judge",
    "pattern",
    "ranks_first",
    "read_labels",
    "read_spans",
    "run",
    "save_run",
    "schema_valid",
    "verdict_equals",
    "with_wrong_labels",
    "write_labels",
]
