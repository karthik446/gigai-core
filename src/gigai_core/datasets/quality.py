"""QuALITY v1.0.1 (Pang et al., NAACL 2022): four-option questions that need the whole of a long article.

Licence CC-BY-4.0, NYU Machine Learning for Language; attribution in NOTICES.md. Each item is one dev-split question
with its full article (11k-34k characters) in the prompt; the gold answer is the option letter. Articles come from
Project Gutenberg, the Open American National Corpus and other open sources, each with its own ``license`` field.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from random import Random

from ..dataset import DatasetItem
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

URL = "https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev"
_FILE = URL.rsplit("/", 1)[1]
SOURCES = (
    SourceFile(
        name=_FILE,
        url=URL,
        sha256="99852d874994078e4b4112b71ceca4dd35aa3a24ff6d3a35c051be25295b4fef",
    ),
)
COUNTS: Mapping[Suite, SliceCount] = {
    "ci": SliceCount(per_stratum={"8-24k": 1, "24-60k": 1}),
    "release": SliceCount(per_stratum={"8-24k": 24, "24-60k": 36}),
}
_DRAW = "with random.Random(20261109).sample; per article 2 questions drawn from difficult == 1, then 3 from the rest."
RECIPES: Mapping[Suite, str] = {
    "ci": f"Dev split, articles by id, banded by length; 1 article from 8-24k and 1 from 24-60k chars {_DRAW}",
    "release": f"Dev split, articles by id, banded by length; 24 articles from 8-24k and 36 from 24-60k chars {_DRAW}",
}
ASK = "Read the article, then answer the multiple-choice question about it. Reply with ANSWER: <letter> only."
LETTERS = "ABCD"


def items(raw: Path) -> Sequence[DatasetItem]:
    """Every dev-split question, by article then question id, with the whole article in the prompt."""
    rows = [json.loads(line) for line in (raw / _FILE).read_text(encoding="utf-8").splitlines()]
    out: list[DatasetItem] = []
    for row in sorted(rows, key=lambda row: row["set_unique_id"]):
        for question in sorted(row["questions"], key=lambda question: question["question_unique_id"]):
            options = "\n".join(f"{letter}) {option}" for letter, option in zip(LETTERS, question["options"]))
            answer = LETTERS[question["gold_label"] - 1]
            out.append(
                DatasetItem(
                    id=f"quality-{question['question_unique_id']}",
                    prompt=f"{ASK}\n\nARTICLE:\n{row['article']}\n\nQUESTION:\n{question['question']}\n\n{options}",
                    expected=answer,
                    cluster_id=row["article_id"],
                    source_ids={
                        "article_id": row["article_id"],
                        "question_unique_id": question["question_unique_id"],
                    },
                    tags=(
                        length_band(len(row["article"])),
                        "difficult" if question["difficult"] else "easy",
                    ),
                )
            )
    return out


def select(pool: Sequence[DatasetItem], suite: Suite, rng: Random) -> Sequence[DatasetItem]:
    """Articles stratified by length band; per article 2 difficult questions, then 3 of the rest."""
    articles: dict[str, list[DatasetItem]] = {}
    for item in pool:
        articles.setdefault(item.cluster_id, []).append(item)
    ordered = sorted(articles.values(), key=lambda questions: questions[0].cluster_id)
    chosen = stratified(ordered, lambda questions: questions[0].tags[0], strata(COUNTS[suite]), rng)
    out: list[DatasetItem] = []
    for questions in chosen:
        hard = rng.sample([item for item in questions if item.tags[1] == "difficult"], 2)
        out += hard + rng.sample([item for item in questions if item not in hard], 3)
    return out


LOADER = register(
    SetLoader(
        name="quality",
        spdx="CC-BY-4.0",
        sources=SOURCES,
        counts=COUNTS,
        recipes=RECIPES,
        items=items,
        select=select,
    )
)
