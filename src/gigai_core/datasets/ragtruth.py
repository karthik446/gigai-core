"""RAGTruth (Niu et al., ACL 2024): find the claims in an LLM's answer or summary that its source does not support.

Licence MIT, (c) 2023 Particle Media; attribution in NOTICES.md. Each item is one human-annotated response with the
task its model was given (source passages included). The gold answer is the response-level verdict plus the
annotated hallucination spans (character offsets into the response); ``implicit_true`` spans are not counted.
"""

from __future__ import annotations

import json
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
    register,
    strata,
    stratified,
)

_RAW = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"
SOURCES = (
    SourceFile(
        name="response.jsonl",
        url=f"{_RAW}/response.jsonl",
        sha256="e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
    ),
    SourceFile(
        name="source_info.jsonl",
        url=f"{_RAW}/source_info.jsonl",
        sha256="0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
    ),
)
_STRATA = ("QA/spans", "QA/none", "Summary/spans", "Summary/none")
COUNTS: Mapping[Suite, SliceCount] = {
    "ci": SliceCount(per_stratum=dict.fromkeys(_STRATA, 5)),
    "release": SliceCount(per_stratum=dict.fromkeys(_STRATA, 100)),
}
#: Responses kept per (source_id, label). RC1's one-per-source release recipe is short: QA has 83 sources with spans.
CAP: Mapping[Suite, int] = {"ci": 1, "release": 2}
_FILTER = 'split == "test", quality == "good", task_type QA or Summary, responses by id; random.Random(20261109)'
_DRAW = "with >= 1 span and the same number without, sampled with the same generator."
RECIPES: Mapping[Suite, str] = {
    "ci": f"{_FILTER} picks 1 response per source_id and label (any model); per task 5 {_DRAW}",
    "release": f"{_FILTER} picks 2 responses per source_id and label (any model); per task 100 {_DRAW}",
}
ASK = (
    "A model was given the task below and wrote the response after it. Find every part of the response that the "
    "task's source material does not support: claims it contradicts and claims it never makes. Answer with bare JSON: "
    '{"supported": true or false, "unsupported_spans": ["<exact quote from the response>", ...]}; supported is true '
    "only when the list is empty."
)


def items(raw: Path) -> Sequence[DatasetItem]:
    """Every annotated response, by numeric ``id``, with its task's source text in the prompt."""
    sources = {row["source_id"]: row for row in map(json.loads, _lines(raw / "source_info.jsonl"))}
    responses = map(json.loads, _lines(raw / "response.jsonl"))
    out: list[DatasetItem] = []
    for row in sorted(responses, key=lambda row: int(row["id"])):
        source = sources[row["source_id"]]
        spans: list[JsonValue] = [{key: span[key] for key in ("start", "end", "text")} for span in row["labels"]]
        spans = [span for span, label in zip(spans, row["labels"], strict=True) if not label["implicit_true"]]
        expected: JsonValue = {"hallucinated": bool(spans), "spans": spans}
        out.append(
            DatasetItem(
                id=f"ragtruth-{row['id']}",
                prompt=f"{ASK}\n\nTASK:\n{source['prompt']}\n\nRESPONSE:\n{row['response']}",
                expected=expected,
                cluster_id=row["source_id"],
                source_ids={
                    "response_id": row["id"],
                    "source_id": row["source_id"],
                    "model": row["model"],
                },
                tags=(
                    source["task_type"],
                    "spans" if spans else "none",
                    row["split"],
                    row["quality"],
                ),
            )
        )
    return out


def select(pool: Sequence[DatasetItem], suite: Suite, rng: Random) -> Sequence[DatasetItem]:
    """Good test responses of QA and Summary, capped per source and label, then stratified by task and label."""
    groups: dict[tuple[str, str], list[DatasetItem]] = {}
    for item in pool:
        if item.tags[0] in ("QA", "Summary") and item.tags[2:] == ("test", "good"):
            groups.setdefault((item.cluster_id, item.tags[1]), []).append(item)
    kept = [pick for group in groups.values() for pick in rng.sample(group, min(CAP[suite], len(group)))]
    return stratified(kept, lambda item: f"{item.tags[0]}/{item.tags[1]}", strata(COUNTS[suite]), rng)


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


LOADER = register(
    SetLoader(
        name="ragtruth",
        spdx="MIT",
        sources=SOURCES,
        counts=COUNTS,
        recipes=RECIPES,
        items=items,
        select=select,
    )
)
