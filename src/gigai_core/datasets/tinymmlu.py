"""tinyMMLU (Maia Polo et al., 2024): 100 four-choice MMLU questions that stand in for the full MMLU test set.

Licence MIT: tinyBenchmarks (c) 2024 Felipe Maia Polo, MMLU (c) 2020 Dan Hendrycks; attribution in NOTICES.md. Each
item is one question with its choices lettered A-D; the gold answer is the correct letter (upstream stores a 0-based
index) and the cluster is the MMLU subject.
"""

from __future__ import annotations

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

REVISION = "cd6a8ee9e1e6f3f18f683535f6c096c386a85da3"
_FILE = "test-00000-of-00001.parquet"
SOURCES = (
    SourceFile(
        name=_FILE,
        url=f"https://huggingface.co/datasets/tinyBenchmarks/tinyMMLU/resolve/{REVISION}/all/{_FILE}",
        sha256="256f81ffdd6e64699df096c02afafba33626bfca08baed484c275e351993f317",
    ),
)
COUNTS: Mapping[Suite, SliceCount] = {
    "ci": SliceCount(total=10),
    "release": SliceCount(total=100),
}
_DRAW = "config all, test rows in file order (ids tinymmlu-000..099), random.Random(20261109).sample"
RECIPES: Mapping[Suite, str] = {
    "ci": f"All 100 {_DRAW} 10 questions.",
    "release": f"All 100 {_DRAW} all 100 questions.",
}
LETTERS = "ABCD"
ASK = (
    "Answer the following multiple choice question. The last line of your response should be of the form "
    "'ANSWER: $LETTER' (without quotes) where LETTER is one of A,B,C,D."
)


class _Row(CoreModel):
    question: str
    choices: Sequence[str]
    answer: int
    subject: str


def items(raw: Path) -> Sequence[DatasetItem]:
    """Every question, in file order."""
    out: list[DatasetItem] = []
    for index, row in enumerate(typed_rows(parquet_rows(raw / _FILE), _Row)):
        choices = "\n".join(f"{LETTERS[i]}) {choice}" for i, choice in enumerate(row.choices))
        out.append(
            DatasetItem(
                id=f"tinymmlu-{index:03d}",
                prompt=f"{ASK}\n\n{row.question.strip()}\n\n{choices}",
                expected=LETTERS[row.answer],
                cluster_id=row.subject,
                source_ids={"row": index, "subject": row.subject},
                tags=(row.subject,),
            )
        )
    return out


def select(pool: Sequence[DatasetItem], suite: Suite, rng: Random) -> Sequence[DatasetItem]:
    """A flat random sample of the suite's size."""
    return rng.sample(pool, flat(COUNTS[suite]))


LOADER = register(
    SetLoader(
        name="tinymmlu",
        spdx="MIT",
        sources=SOURCES,
        counts=COUNTS,
        recipes=RECIPES,
        items=items,
        select=select,
    )
)
