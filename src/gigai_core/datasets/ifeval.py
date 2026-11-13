"""IFEval (Zhou et al., 2023): prompts with formatting instructions that code can verify ("no commas", "3 sections").

Licence Apache-2.0, Google LLC; attribution in NOTICES.md. Each item is one prompt; the gold answer is its
``instruction_id_list`` with the matching ``kwargs``, which the upstream verifiers (``instruction_following_eval``)
check against a response.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from random import Random

from ..dataset import DatasetItem
from .loader import SetLoader, SliceCount, SourceFile, Suite, flat, register

HF_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
_FILE = "ifeval_input_data.jsonl"
SOURCES = (
    SourceFile(
        name=_FILE,
        url=f"https://huggingface.co/datasets/google/IFEval/resolve/{HF_REVISION}/{_FILE}",
        sha256="6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2",
    ),
)
COUNTS: Mapping[Suite, SliceCount] = {
    "ci": SliceCount(total=10),
    "release": SliceCount(total=100),
}
_DRAW = "prompts by key, random.Random(20261109).sample (not the datasets library's shuffle, which needs numpy)."
RECIPES: Mapping[Suite, str] = {
    "ci": f"All 541 {_DRAW} 10 prompts.",
    "release": f"All 541 {_DRAW} 100 prompts.",
}


def items(raw: Path) -> Sequence[DatasetItem]:
    """Every prompt, by ``key``."""
    rows = [json.loads(line) for line in (raw / _FILE).read_text(encoding="utf-8").splitlines()]
    return [
        DatasetItem(
            id=f"ifeval-{row['key']}",
            prompt=row["prompt"],
            expected={
                "instruction_id_list": row["instruction_id_list"],
                "kwargs": row["kwargs"],
            },
            cluster_id=str(row["key"]),
            source_ids={"key": row["key"]},
            tags=tuple(row["instruction_id_list"]),
        )
        for row in sorted(rows, key=lambda row: row["key"])
    ]


def select(pool: Sequence[DatasetItem], suite: Suite, rng: Random) -> Sequence[DatasetItem]:
    """A flat random sample of the suite's size."""
    return rng.sample(pool, flat(COUNTS[suite]))


LOADER = register(
    SetLoader(
        name="ifeval",
        spdx="Apache-2.0",
        sources=SOURCES,
        counts=COUNTS,
        recipes=RECIPES,
        items=items,
        select=select,
    )
)
