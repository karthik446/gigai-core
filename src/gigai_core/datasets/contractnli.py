"""ContractNLI (Koreeda & Manning, 2021): decide 17 fixed hypotheses against one non-disclosure agreement.

Licence CC-BY-4.0, (c) Hitachi America, Ltd.; attribution in NOTICES.md. Each item is one test-split document with
its 17 hypotheses, labelled Entailment, Contradiction or NotMentioned, with evidence spans as character offsets. The
authors publish no inter-annotator agreement, so every item carries the ``audit_pending`` tag until our own audit.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from random import Random

from ..dataset import DatasetItem
from ..values import JsonValue
from .loader import (
    SetLoader,
    SliceCount,
    SourceFile,
    Suite,
    length_band,
    register,
    strata,
    stratified,
)

URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"
SOURCES = (
    SourceFile(
        name="contract-nli.zip",
        url=URL,
        sha256="e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b",
    ),
)
COUNTS: Mapping[Suite, SliceCount] = {
    "ci": SliceCount(per_stratum={"<8k": 3, "8-24k": 5, "24-60k": 2}),
    "release": SliceCount(per_stratum={"<8k": None, "8-24k": None, "24-60k": None}),
}
RECIPES: Mapping[Suite, str] = {
    "ci": "Test split, documents by id, banded by text length (<8k/8-24k/24-60k chars); 3/5/2 per band (the 20/33/7 "
    "base proportions) with random.Random(20261109).sample; one item per document with all 17 hypotheses.",
    "release": "Test split: all 123 documents, grouped by length band (<8k/8-24k/24-60k chars), by id inside a band; "
    "one item per document with all 17 hypotheses.",
}
ASK = (
    "Read the non-disclosure agreement, then decide for every hypothesis whether the agreement entails it, "
    'contradicts it, or does not mention it. Answer with bare JSON: {"labels": {"<hypothesis id>": "<label>"}}, '
    "one entry per hypothesis id, each label Entailment, Contradiction or NotMentioned."
)


def items(raw: Path) -> Sequence[DatasetItem]:
    """Every test-split document, by ``id``, with all 17 hypotheses in the prompt."""
    with zipfile.ZipFile(raw / "contract-nli.zip") as archive:
        data = json.loads(archive.read("contract-nli/test.json"))
    keys = sorted(data["labels"], key=lambda key: int(key.split("-")[1]))
    hypotheses = "\n".join(f"{key}: {data['labels'][key]['hypothesis']}" for key in keys)
    out: list[DatasetItem] = []
    for doc in sorted(data["documents"], key=lambda doc: doc["id"]):
        notes = doc["annotation_sets"][0]["annotations"]
        labels: dict[str, JsonValue] = {key: notes[key]["choice"] for key in keys}
        spans: dict[str, JsonValue] = {
            key: [doc["spans"][i] for i in notes[key]["spans"]] for key in keys if notes[key]["spans"]
        }
        expected: JsonValue = {"labels": labels, "spans": spans}
        out.append(
            DatasetItem(
                id=f"contractnli-test-{doc['id']}",
                prompt=f"{ASK}\n\nAGREEMENT:\n{doc['text']}\n\nHYPOTHESES:\n{hypotheses}",
                expected=expected,
                cluster_id=str(doc["id"]),
                source_ids={
                    "split": "test",
                    "document_id": doc["id"],
                    "file_name": doc["file_name"],
                },
                tags=("audit_pending", length_band(len(doc["text"]))),
            )
        )
    return out


def select(pool: Sequence[DatasetItem], suite: Suite, rng: Random) -> Sequence[DatasetItem]:
    """Documents stratified by their length band."""
    return stratified(pool, lambda item: item.tags[1], strata(COUNTS[suite]), rng)


LOADER = register(
    SetLoader(
        name="contractnli",
        spdx="CC-BY-4.0",
        sources=SOURCES,
        counts=COUNTS,
        recipes=RECIPES,
        items=items,
        select=select,
    )
)
