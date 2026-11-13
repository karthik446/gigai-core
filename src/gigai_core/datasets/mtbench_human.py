"""MT-Bench human judgments (Zheng et al., 2023): pick the better of two assistants' answers, as expert humans did.

Licence CC BY 4.0, LMSYS; attribution in NOTICES.md. Each item is one (question, turn, model A, model B) that two or
more human judges compared, all picking the same side (no tie), with no Claude model on either side, so a Claude
judge never grades its own family. The gold answer is that unanimous winner, in the source's A/B order.

This is the one multi-turn set: the judging instruction is the item's ``system`` and each conversation turn is one
scripted user turn (the user's question with both assistants' answers to it), so the judge sees the conversation
turn by turn and decides on the last one; ``prompt`` is empty.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from random import Random

from ..dataset import DatasetItem
from ..models import CoreModel
from ..providers import Turn
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

REVISION = "f7d2896d2cc5d80f8b55c2bbc722613555233c25"
_FILE = "human-00000-of-00001-25f4910818759289.parquet"
SOURCES = (
    SourceFile(
        name=_FILE,
        url=f"https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/resolve/{REVISION}/data/{_FILE}",
        sha256="4877bc46a40929f4082c3c79593700fb897b1d6c7f4c473032694a01322f5769",
    ),
)
COUNTS: Mapping[Suite, SliceCount] = {
    "ci": SliceCount(total=20),
    "release": SliceCount(total=293),
}
_FILTER = (
    "human split grouped by (question_id, turn, model_a, model_b): >= 2 judgments, all the same non-tie winner, "
    "no claude model on either side (293 of 2,396 groups); random.Random(20261109).sample"
)
RECIPES: Mapping[Suite, str] = {
    "ci": f"{_FILTER} 20 items.",
    "release": f"{_FILTER} all 293 items.",
}
ASK = (
    "Below is a conversation between a user and two AI assistants, A and B, who each answered on their own. Decide "
    "which assistant answers the last user question better: helpful, relevant, accurate, deep, and detailed enough. "
    'Length and position must not sway you. Answer with bare JSON: {"winner": "A"} or {"winner": "B"}.'
)


class _Message(CoreModel):
    role: str
    content: str


class _Row(CoreModel):
    question_id: int
    turn: int
    model_a: str
    model_b: str
    winner: str
    judge: str
    conversation_a: Sequence[_Message]
    conversation_b: Sequence[_Message]


def items(raw: Path) -> Sequence[DatasetItem]:
    """Every unanimously judged (question, turn, model A, model B) group, by that key."""
    groups: dict[tuple[int, int, str, str], list[_Row]] = {}
    for row in typed_rows(parquet_rows(raw / _FILE), _Row):
        groups.setdefault((row.question_id, row.turn, row.model_a, row.model_b), []).append(row)
    out: list[DatasetItem] = []
    for (question, turn, model_a, model_b), rows in sorted(groups.items()):
        winners = {row.winner for row in rows}
        if len(rows) < 2 or len(winners) > 1 or winners == {"tie"} or "claude" in f"{model_a} {model_b}":
            continue
        if min(len(rows[0].conversation_a), len(rows[0].conversation_b)) < 2 * turn:
            raise ValueError(f"mtbench_human question {question}: a conversation is shorter than turn {turn}")
        winner = {"model_a": "A", "model_b": "B"}[winners.pop()]
        judges = ",".join(sorted(row.judge for row in rows))
        out.append(
            DatasetItem(
                id=f"mtbench_human-{question}-t{turn}-{model_a}-{model_b}",
                prompt="",
                expected={"winner": winner},
                cluster_id=str(question),
                turns=_turns(rows[0], turn),
                system=ASK,
                source_ids={"question_id": question, "turn": turn, "model_a": model_a, "model_b": model_b, "judges": judges},
                tags=(f"turn:{turn}", f"winner:{winner}"),
            )
        )
    return out


def select(pool: Sequence[DatasetItem], suite: Suite, rng: Random) -> Sequence[DatasetItem]:
    """A flat random sample of the suite's size."""
    return rng.sample(pool, flat(COUNTS[suite]))


def _turns(row: _Row, turn: int) -> tuple[Turn, ...]:
    """One scripted user turn per conversation turn up to ``turn``: the question and both assistants' answers."""
    return tuple(
        Turn(
            role="user",
            content=f"USER:\n{row.conversation_a[2 * index].content}\n\nASSISTANT A:\n"
            f"{row.conversation_a[2 * index + 1].content}\n\nASSISTANT B:\n{row.conversation_b[2 * index + 1].content}",
        )
        for index in range(turn)
    )


LOADER = register(
    SetLoader(
        name="mtbench_human",
        spdx="CC-BY-4.0",
        sources=SOURCES,
        counts=COUNTS,
        recipes=RECIPES,
        items=items,
        select=select,
    )
)
