"""JudgeBench (Tan et al., 2024): pick the correct one of two responses to a hard question, both by the same model.

Licence MIT (dataset card), the JudgeBench authors; attribution in NOTICES.md. Each item is one pair of the ``gpt``
split (GPT-4o-2024-05-13 wrote both responses). The gold answer is the pair's label (``A>B`` or ``B>A``), derived
from the source benchmark's verified answer key, not from votes. A and B are swapped by a per-pair seeded coin so
the source's 193/157 label skew does not carry over; ``source_ids`` keeps the source label and the swap (as 0/1).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from operator import itemgetter
from pathlib import Path
from random import Random

from ..dataset import DatasetItem
from .loader import (
    SEED,
    SetLoader,
    SliceCount,
    SourceFile,
    Suite,
    register,
    strata,
    stratified,
)

REVISION = "57dd5e0b9817d07f05ec8f45a91b2ce1e310e308"
_FILE = "gpt-00000-of-00001.jsonl"
SOURCES = (
    SourceFile(
        name=_FILE,
        url=f"https://huggingface.co/datasets/ScalerLab/JudgeBench/resolve/{REVISION}/data/{_FILE}",
        sha256="d5106306b38dead1b2e964ecf5534a3e7b2fc4b8d3b8da327599edfcebd76042",
    ),
)
#: Pairs per source family (the 14 ``mmlu-pro-*`` sources are one family), proportional to the split's 154/98/56/42.
COUNTS: Mapping[Suite, SliceCount] = {
    "ci": SliceCount(
        per_stratum={
            "mmlu-pro": 9,
            "livebench-reasoning": 6,
            "livebench-math": 3,
            "livecodebench": 2,
        }
    ),
    "release": SliceCount(
        per_stratum={
            "mmlu-pro": 132,
            "livebench-reasoning": 84,
            "livebench-math": 48,
            "livecodebench": 36,
        }
    ),
}
_KEPT = ("pair_id", "original_id", "source", "label", "response_model")
_FLIP = {"A>B": "B>A", "B>A": "A>B"}
_DRAW = (
    "All 350 pairs of the gpt split by pair_id; A/B swapped when random.Random(f'20261109:{pair_id}').random() < 0.5; "
    "random.Random(20261109) draws per source family (mmlu-pro, livebench-reasoning, livebench-math, livecodebench)"
)
RECIPES: Mapping[Suite, str] = {
    "ci": f"{_DRAW} 9/6/3/2 pairs.",
    "release": f"{_DRAW} 132/84/48/36 pairs.",
}
ASK = (
    "Below are a question and two responses to it, A and B. Exactly one response reaches the correct answer. "
    'Decide which. Answer with bare JSON: {"verdict": "A>B"} when A is the correct one, {"verdict": "B>A"} when B is.'
)


def items(raw: Path) -> Sequence[DatasetItem]:
    """Every pair of the gpt split, by ``pair_id``, A and B swapped by a per-pair seeded coin."""
    out: list[DatasetItem] = []
    rows = map(json.loads, (raw / _FILE).read_text(encoding="utf-8").splitlines())
    for row in sorted(rows, key=itemgetter("pair_id")):
        swapped = Random(f"{SEED}:{row['pair_id']}").random() < 0.5
        first, second = (row["response_B"], row["response_A"]) if swapped else (row["response_A"], row["response_B"])
        label = _FLIP[row["label"]] if swapped else row["label"]
        family = "mmlu-pro" if row["source"].startswith("mmlu-pro-") else row["source"]
        out.append(
            DatasetItem(
                id=f"judgebench-{row['pair_id']}",
                prompt=f"{ASK}\n\nQUESTION:\n{row['question']}\n\nRESPONSE A:\n{first}\n\nRESPONSE B:\n{second}",
                expected={"label": label},
                cluster_id=f"{row['source']}:{row['original_id']}",
                source_ids={key: row[key] for key in _KEPT} | {"swapped": int(swapped)},
                tags=(family, row["source"], label),
            )
        )
    return out


def select(pool: Sequence[DatasetItem], suite: Suite, rng: Random) -> Sequence[DatasetItem]:
    """Pairs stratified by source family."""
    return stratified(pool, lambda item: item.tags[0], strata(COUNTS[suite]), rng)


LOADER = register(
    SetLoader(
        name="judgebench",
        spdx="MIT",
        sources=SOURCES,
        counts=COUNTS,
        recipes=RECIPES,
        items=items,
        select=select,
    )
)
