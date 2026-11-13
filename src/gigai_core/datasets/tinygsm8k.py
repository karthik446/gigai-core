"""tinyGSM8k (Maia Polo et al., 2024): 100 grade-school maths problems that stand in for the GSM8K test set.

Licence MIT: tinyBenchmarks (c) 2024 Felipe Maia Polo, GSM8K (c) 2021 OpenAI; attribution in NOTICES.md. Each item is
one problem; the gold answer is the number after ``####`` in the GSM8K solution, commas dropped.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from random import Random

from ..dataset import DatasetItem
from ..models import CoreModel
from .loader import (
    SetLoader,
    SliceCount,
    SourceFile,
    Suite,
    flat,
    parquet_rows,
    register,
    typed_rows,
)

REVISION = "b870449ed96ee3e308748b550908332450ec4098"
_FILE = "test-00000-of-00001.parquet"
SOURCES = (
    SourceFile(
        name=_FILE,
        url=f"https://huggingface.co/datasets/tinyBenchmarks/tinyGSM8k/resolve/{REVISION}/main/{_FILE}",
        sha256="07c2857ce7f4f4cf123066fede0f52f2aab7458c0c9bae3293c3f656a9eac53c",
    ),
)
COUNTS: Mapping[Suite, SliceCount] = {
    "ci": SliceCount(total=10),
    "release": SliceCount(total=100),
}
_DRAW = "test rows in file order (ids tinygsm8k-000..099), random.Random(20261109).sample"
RECIPES: Mapping[Suite, str] = {
    "ci": f"All 100 {_DRAW} 10 problems.",
    "release": f"All 100 {_DRAW} all 100 problems.",
}
ASK = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "'ANSWER: $NUMBER' (without quotes) where $NUMBER is the answer to the problem."
)
_ANSWER = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)\s*$")


class _Row(CoreModel):
    question: str
    answer: str


def items(raw: Path) -> Sequence[DatasetItem]:
    """Every problem, in file order."""
    out: list[DatasetItem] = []
    for index, row in enumerate(typed_rows(parquet_rows(raw / _FILE), _Row)):
        answer = row.answer.strip()
        match = _ANSWER.search(answer)
        if not match:
            raise ValueError(f"tinygsm8k row {index}: no '####' answer")
        number = match.group(1).replace(",", "")
        out.append(
            DatasetItem(
                id=f"tinygsm8k-{index:03d}",
                prompt=f"{ASK}\n\n{row.question.strip()}",
                expected=number,
                cluster_id=f"tinygsm8k-{index:03d}",
                source_ids={"row": index},
            )
        )
    return out


def select(pool: Sequence[DatasetItem], suite: Suite, rng: Random) -> Sequence[DatasetItem]:
    """A flat random sample of the suite's size."""
    return rng.sample(pool, flat(COUNTS[suite]))


LOADER = register(
    SetLoader(
        name="tinygsm8k",
        spdx="MIT",
        sources=SOURCES,
        counts=COUNTS,
        recipes=RECIPES,
        items=items,
        select=select,
    )
)
