"""Judge agreement against human labels, and the pass rule that makes a judge allowed to gate.

A judge (a model grader) is validated against humans on 0/1 verdicts: Cohen's kappa with its interval, the true
positive and true negative rates. The pass rule is kappa ``>= MIN_KAPPA`` over ``>= MIN_UNITS`` labelled items
and, when measured, a swap consistency ``>= MIN_SWAP``.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import CoreModel
from .records import LabelRecord
from .run import Run
from .stats import Kappa, cohen_kappa

MIN_KAPPA = 0.6
MIN_UNITS = 126
MIN_SWAP = 0.90


class JudgeNotValidated(Exception):
    """A judge was asked to gate with no stored validation for its prompt digest."""


class Agreement(CoreModel):
    """Judge against human over (human, judge) 0/1 pairs.

    ``kappa`` is ``None`` only when both raters used one category (or there are no pairs); one rater using one
    category gives kappa 0. ``tpr`` and ``tnr`` are ``None`` for a rate without cases, ``observed_agreement``
    ``None`` without pairs.
    ``kappa_undefined`` says why kappa is ``None``. The counts after ``pairs`` are what ``judge_agreement`` set
    aside: labels whose item the run never asked, labels whose item the judge left ungraded, judged items
    without a label, and labels of ``unknown`` (which pair with nothing).
    """

    n: int
    kappa: Kappa | None
    observed_agreement: float | None
    tpr: float | None
    tnr: float | None
    pairs: tuple[tuple[int, int], ...] = ()
    kappa_undefined: str | None = None
    labels_without_sample: int = 0
    labels_ungraded: int = 0
    samples_without_label: int = 0
    unknown_labels: int = 0


def agreement(pairs: Sequence[tuple[int, int]]) -> Agreement:
    """Kappa with its interval and the judge's rates over ``(human, judge)`` pairs of 0/1 verdicts.

    Args:
        pairs: One ``(human, judge)`` pair of 0/1 verdicts per item.

    Returns:
        The agreement; ``kappa`` is ``None`` (with ``kappa_undefined`` saying why) when there are no pairs or
        both raters used one category only.
    """
    table = [[0, 0], [0, 0]]
    for human, judge in pairs:
        table[human][judge] += 1
    positives = table[1][0] + table[1][1]
    negatives = table[0][0] + table[0][1]
    kappa: Kappa | None = None
    undefined: str | None = None
    try:
        kappa = cohen_kappa([human for human, _ in pairs], [judge for _, judge in pairs])
    except ValueError as reason:
        undefined = str(reason)
    return Agreement(
        n=len(pairs),
        kappa=kappa,
        observed_agreement=(table[0][0] + table[1][1]) / len(pairs) if pairs else None,
        tpr=table[1][1] / positives if positives else None,
        tnr=table[0][0] / negatives if negatives else None,
        pairs=tuple(pairs),
        kappa_undefined=undefined,
    )


def judge_agreement(run: Run, grader_id: str, labels: Sequence[LabelRecord], *, threshold: float = 0.5) -> Agreement:
    """Pair each sample's judge score (>= ``threshold`` is 1) with the human label of the same item.

    The judge's verdict on an item is its per-item mean over the repeats (``Run.values``), ``1`` when at least
    ``threshold``. The human verdict is the newest label of the item (by ``recorded_at``; a later label in
    ``labels`` wins a tie): ``met`` is 1, ``not met`` 0, ``unknown`` pairs with nothing and is counted. Pairs
    follow the order the run first asked its items.

    Args:
        run: The run the judge graded.
        grader_id: The judge's grader id.
        labels: The human labels; those of items the run did not ask are counted, not paired.
        threshold: The judge score from which a verdict is 1.

    Returns:
        The agreement over the paired items, with the counts of what could not be paired.
    """
    judged = run.values(grader_id)
    asked = dict.fromkeys(sample.item_id for sample in run.samples)
    newest: dict[str, LabelRecord] = {}
    for label in labels:
        held = newest.get(label.item_id)
        if held is None or label.recorded_at >= held.recorded_at:
            newest[label.item_id] = label
    pairs: list[tuple[int, int]] = []
    unknown = 0
    for item in asked:
        label = newest.get(item)
        if label is None or item not in judged:
            continue
        if label.label == "unknown":
            unknown += 1
            continue
        pairs.append((1 if label.label == "met" else 0, 1 if judged[item] >= threshold else 0))
    return agreement(pairs).model_copy(
        update={
            "labels_without_sample": sum(item not in asked for item in newest),
            "labels_ungraded": sum(item in asked and item not in judged for item in newest),
            "samples_without_label": sum(item not in newest for item in judged),
            "unknown_labels": unknown,
        }
    )


def validation_failures(
    *, kappa: float | None, n: int, swap_consistency: float | None, swap_pairs: int | None = None
) -> tuple[str, ...]:
    """Why a judge may not gate; empty when it may.

    The rule: kappa at least ``MIN_KAPPA`` over at least ``MIN_UNITS`` labelled items and, when measured, a swap
    consistency of at least ``MIN_SWAP``.

    Args:
        kappa: The judge's kappa against humans; ``None`` when undefined (both raters used one category only).
        n: The number of labelled items the kappa is over.
        swap_consistency: The share of swapped pairs the judge scored consistently; ``None`` when not measured.
        swap_pairs: How many pairs were swapped; ``0`` means a swap was asked for but no pair could be swapped.

    Returns:
        The reasons, in the order the rule states them.
    """
    reasons: list[str] = []
    if n < MIN_UNITS:
        reasons.append(f"n < {MIN_UNITS}")
    if kappa is None:
        reasons.append("kappa undefined")
    elif kappa < MIN_KAPPA:
        reasons.append(f"kappa < {MIN_KAPPA}")
    if swap_pairs == 0:
        reasons.append("no swap pairs")
    elif swap_consistency is not None and swap_consistency < MIN_SWAP:
        reasons.append(f"swap_consistency < {MIN_SWAP}")
    return tuple(reasons)


def validation_passes(
    *, kappa: float | None, n: int, swap_consistency: float | None, swap_pairs: int | None = None
) -> bool:
    """True when ``validation_failures`` is empty: the judge may gate.

    Args:
        kappa: The judge's kappa against humans; ``None`` when undefined.
        n: The number of labelled items the kappa is over.
        swap_consistency: The share of swapped pairs scored consistently; ``None`` when not measured.
        swap_pairs: How many pairs were swapped; ``0`` means none could be.

    Returns:
        Whether the pass rule is met.
    """
    return not validation_failures(kappa=kappa, n=n, swap_consistency=swap_consistency, swap_pairs=swap_pairs)
