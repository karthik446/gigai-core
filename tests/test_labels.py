"""The labels JSONL helpers and the ``labelled`` grader. The format rules relocated from 0.3.2
``tests/behaviors/evals/test_label.py`` are the ones ``read_labels`` ported: extra keys named and cut short, a missing
key named, the highest round accepted, an empty batch refused, and no error echoing a value. The page, the step check,
the stored-label comparisons and the CLI stay in gigai."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from gigai_core.graders.labels import (
    LABELS,
    ROUND_MAX,
    LabelFileError,
    labelled,
    read_labels,
    write_labels,
)
from gigai_core.graders.protocol import CannotGrade
from gigai_core.providers import Turn
from gigai_core.records import LabelRecord
from gigai_core.sample import Sample
from gigai_core.values import JsonValue

SECRET = "SECRET-TEXT"
NOW = datetime(2026, 1, 31, 9, 30, tzinfo=UTC)
ROW: Mapping[str, JsonValue] = {
    "label_id": "label_1",
    "dataset_id": "capitals",
    "item_id": "item-01",
    "item_version": "1",
    "label": "met",
    "source": "human",
    "recorded_at": "2026-01-31T09:30:00Z",
    "who": "kar",
    "round": 1,
}


def _record(**changes: object) -> LabelRecord:
    return LabelRecord.model_validate_json(json.dumps({**ROW, **changes}, default=datetime.isoformat))


def _file(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def _sample(item_id: str) -> Sample:
    return Sample(
        trace_id=f"trace_{item_id}",
        run_id="run_1",
        item_id=item_id,
        item_version="1",
        cluster_id=item_id,
        repeat_index=0,
        input=(Turn(role="user", content="?"),),
        output_text="!",
        output_parsed=None,
        parse_status="not_parsed",
        outcome="succeeded",
        error=None,
        recorded_at=NOW,
    )


# --- read_labels: the relocated format rules ----------------------------------------------------------------------


def test_read_labels_names_every_extra_key_and_cuts_a_long_one_short(tmp_path: Path) -> None:
    path = _file(tmp_path, json.dumps({**ROW, "zeta": 1, "input": 2, SECRET * 10: 3}))

    with pytest.raises(LabelFileError) as caught:
        read_labels(path)

    message = str(caught.value)
    assert message.startswith(f"{path} line 1: key 'SECRET-TEXTSECRET-TEXTSECRET-TEXTSECRET-...', 'input', 'zeta' is not")
    assert "(keys: label_id, dataset_id, item_id" in message
    assert SECRET * 4 not in message
    assert (caught.value.path, caught.value.line) == (path, 1)
    assert caught.value.reason.startswith("key ")


def test_blank_lines_are_skipped_and_the_records_come_back_in_order(tmp_path: Path) -> None:
    second = {**ROW, "label_id": "label_2", "item_id": "item-02", "label": "not met"}
    plain = _file(tmp_path, json.dumps(ROW), json.dumps(second))
    (tmp_path / "padded").mkdir()
    padded = _file(tmp_path / "padded", "", json.dumps(ROW), "   ", json.dumps(second), "", "")

    assert read_labels(plain) == read_labels(padded) == (_record(), _record(**second))


def test_the_boundaries_of_the_format_are_accepted(tmp_path: Path) -> None:
    edge = {**ROW, "critique": "c" * 200, "recorded_at": "2026-01-31T09:31:00.123456Z", "round": 2, "label": "not met"}
    (label,) = read_labels(_file(tmp_path, json.dumps(edge)))
    assert label.critique == "c" * 200 and label.round == 2 and label.label == "not met"
    assert label.recorded_at == datetime(2026, 1, 31, 9, 31, 0, 123456, tzinfo=UTC)


def test_the_highest_round_is_accepted(tmp_path: Path) -> None:
    assert ROUND_MAX == 1000
    (label,) = read_labels(_file(tmp_path, json.dumps({**ROW, "round": ROUND_MAX})))
    assert label.round == ROUND_MAX


MALFORMED: Mapping[str, tuple[str, str]] = {
    "missing key": (json.dumps({key: value for key, value in ROW.items() if key != "label"}), "line 2: missing key label"),
    "label one letter off": (json.dumps({**ROW, "label": f"mett{SECRET}"}), "line 2: 'label': Input should be 'met', 'not met' or 'unknown'"),
    "label null": (json.dumps({**ROW, "label": None}), "line 2: 'label': Input should be"),
    "round zero": (json.dumps({**ROW, "round": 0}), "line 2: round must be an integer from 1 to 1000"),
    "round one over": (json.dumps({**ROW, "round": 1001}), "line 2: round must be an integer from 1 to 1000"),
    "round float": (json.dumps({**ROW, "round": 1.0}), "line 2: 'round': Input should be a valid integer"),
    "round bool": (json.dumps({**ROW, "round": True}), "line 2: 'round': Input should be a valid integer"),
    "round text": (json.dumps({**ROW, "round": f"1{SECRET}"}), "line 2: 'round': Input should be a valid integer"),
    "recorded_at without zone": (json.dumps({**ROW, "recorded_at": "2026-01-31T09:30:00"}), "line 2: 'recorded_at': "),
    "recorded_at impossible day": (json.dumps({**ROW, "recorded_at": f"2026-02-30T09:30:00Z{SECRET}"}), "line 2: 'recorded_at': "),
    "who a number": (json.dumps({**ROW, "who": 1}), "line 2: 'who': Input should be a valid string"),
    "an array": (json.dumps([SECRET]), "line 2: the line: Input should be an object"),
    "a string": (json.dumps(SECRET), "line 2: the line: Input should be an object"),
    "cut short": (json.dumps({**ROW, "critique": SECRET})[:-5], "line 2: the line: Invalid JSON"),
    "NaN": (json.dumps({**ROW, "round": 1}).replace('"round": 1', '"round": NaN'), "line 2: 'round': Input should be a valid integer"),  # pydantic reads NaN as a float
}  # fmt: skip


@pytest.mark.parametrize("bad, message", MALFORMED.values(), ids=MALFORMED.keys())
def test_a_malformed_line_refuses_the_file_and_never_echoes_a_value(tmp_path: Path, bad: str, message: str) -> None:
    path = _file(tmp_path, json.dumps(ROW), bad)

    with pytest.raises(LabelFileError) as caught:
        read_labels(path)

    assert message in str(caught.value), str(caught.value)
    assert SECRET not in str(caught.value)
    assert caught.value.line == 2


@pytest.mark.parametrize("text", ["", "\n   \n\n"], ids=["empty", "blank lines"])
def test_a_file_with_no_labels_is_refused(tmp_path: Path, text: str) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(LabelFileError, match="line 0: no labels in the file") as caught:
        read_labels(path)
    assert caught.value.line == 0


def test_a_missing_file_is_the_os_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_labels(tmp_path / "nope.jsonl")


def test_an_exact_repeat_of_a_line_is_read_once(tmp_path: Path) -> None:
    second = {**ROW, "label_id": "label_2", "item_id": "item-02"}
    path = _file(tmp_path, json.dumps(ROW), json.dumps(second), json.dumps(ROW), json.dumps({**ROW, "who": "kar"}))

    assert read_labels(path) == (_record(), _record(**second))


@pytest.mark.parametrize(
    "change",
    [{"label": "not met"}, {"critique": SECRET}, {"item_id": f"item-02{SECRET}"}, {"round": 2}],
    ids=["label", "critique", "item_id", "round"],
)
def test_a_label_id_that_comes_back_different_is_refused_naming_both_lines(tmp_path: Path, change: Mapping[str, JsonValue]) -> None:
    path = _file(tmp_path, json.dumps(ROW), "", json.dumps({**ROW, "label_id": "label_2", "item_id": "item-02"}), json.dumps({**ROW, **change}))

    with pytest.raises(LabelFileError) as caught:
        read_labels(path)

    assert str(caught.value) == f"{path} line 4: has the label_id of line 1 with a different label"
    assert SECRET not in str(caught.value) and caught.value.line == 4


# --- write_labels ----------------------------------------------------------------------------------------------------


def test_write_labels_round_trips_through_model_dump_json_and_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    labels = (_record(), _record(label_id="label_2", item_id="item-02", label="unknown", critique="?"))

    write_labels(path, labels)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == [label.model_dump_json() for label in labels]
    assert read_labels(path) == labels
    assert sorted(child.name for child in tmp_path.iterdir()) == ["out.jsonl"]
    with pytest.raises(FileExistsError, match="never overwrites"):
        write_labels(path, labels)
    assert read_labels(path) == labels


def test_write_labels_leaves_nothing_behind_when_the_write_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", failing)
    with pytest.raises(OSError, match="disk full"):
        write_labels(tmp_path / "out.jsonl", (_record(),))
    assert list(tmp_path.iterdir()) == []


# --- labelled ---------------------------------------------------------------------------------------------------------


def test_labelled_grades_met_and_not_met_and_cannot_grade_unknown_or_missing() -> None:
    labels = (
        _record(),
        _record(label_id="label_2", item_id="item-02", label="not met", critique="wrong city"),
        _record(label_id="label_3", item_id="item-03", label="unknown"),
    )
    grader = labelled(labels)
    assert (grader.id, grader.version, grader.kind) == ("human", "1", "human")
    assert labelled(labels, grader_id="kar").id == "kar"
    met = grader.grade(_sample("item-01"), None)
    assert (met.value, met.answer, met.explanation, met.judge) == (1.0, "met", None, None)
    not_met = grader.grade(_sample("item-02"), None)
    assert (not_met.value, not_met.answer, not_met.explanation) == (0.0, "not met", "wrong city")
    with pytest.raises(CannotGrade, match="item 'item-03' is unknown"):
        grader.grade(_sample("item-03"), None)
    with pytest.raises(CannotGrade, match="no label for item 'item-04'"):
        grader.grade(_sample("item-04"), None)
    assert LABELS == ("met", "not met", "unknown")


def test_labelled_takes_the_newest_label_of_an_item_and_a_later_one_on_a_tie() -> None:
    older = _record(label_id="label_1", label="met", recorded_at=NOW - timedelta(hours=1))
    newer = _record(label_id="label_2", label="not met", recorded_at=NOW, round=2)
    assert labelled((newer, older)).grade(_sample("item-01"), None).value == 0.0
    assert labelled((older, newer)).grade(_sample("item-01"), None).value == 0.0
    tie = _record(label_id="label_3", label="unknown", recorded_at=NOW)
    with pytest.raises(CannotGrade, match="unknown"):
        labelled((older, newer, tie)).grade(_sample("item-01"), None)
    assert labelled((tie, newer)).grade(_sample("item-01"), None).value == 0.0
