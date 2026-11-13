"""JSONSchemaBench (Geng et al., 2025): write one JSON value that validates against a real-world JSON Schema.

Licence MIT (dataset card), the JSONSchemaBench authors; attribution in NOTICES.md. Each item is one schema of the
``Github_medium`` test split. The gold answer is the schema itself: any value it accepts is correct. Items ask for
``output_mode = schema``, so the request carries the schema and a text-only adapter appends it to the prompt (the
prompt does not repeat it).
"""

from __future__ import annotations

import json
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

REVISION = "5bd0f4640badc6f3f02df796421d21cb0ca0b141"
_FILE = "test-00000-of-00001.parquet"
SOURCES = (
    SourceFile(
        name=_FILE,
        url=f"https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench/resolve/{REVISION}/Github_medium/{_FILE}",
        sha256="75faffa85c98edf2db596e3e1d2b5bf580b864389f578599c38c6512daf9854d",
    ),
)
COUNTS: Mapping[Suite, SliceCount] = {
    "ci": SliceCount(total=10),
    "release": SliceCount(total=100),
}
_DRAW = "schemas by unique_id, random.Random(20261109).sample (not the datasets library's shuffle, which needs numpy)."
RECIPES: Mapping[Suite, str] = {
    "ci": f"All 586 Github_medium test {_DRAW} 10 schemas.",
    "release": f"All 586 Github_medium test {_DRAW} 100 schemas.",
}
ASK = (
    "Write one JSON value that a real system could send and that validates against the JSON Schema this request "
    "carries. Fill in realistic values, not placeholders."
)


class _Row(CoreModel):
    unique_id: str
    json_schema: str


def items(raw: Path) -> Sequence[DatasetItem]:
    """Every schema, by ``unique_id``."""
    return [
        DatasetItem(
            id=f"jsonschemabench-{row.unique_id}",
            prompt=ASK,
            expected={"schema": json.loads(row.json_schema)},
            cluster_id=row.unique_id,
            source_ids={
                "unique_id": row.unique_id,
                "subset": "Github_medium",
                "split": "test",
            },
            tags=("output_mode:schema",),
        )
        for row in sorted(typed_rows(parquet_rows(raw / _FILE), _Row), key=lambda row: row.unique_id)
    ]


def select(pool: Sequence[DatasetItem], suite: Suite, rng: Random) -> Sequence[DatasetItem]:
    """A flat random sample of the suite's size."""
    return rng.sample(pool, flat(COUNTS[suite]))


LOADER = register(
    SetLoader(
        name="jsonschemabench",
        spdx="MIT",
        sources=SOURCES,
        counts=COUNTS,
        recipes=RECIPES,
        items=items,
        select=select,
    )
)
