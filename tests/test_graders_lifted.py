"""The vendored inspect_ai ``match``/``includes``/``pattern`` bodies and the core graders over them.

The hand cases are S-smoke's ``run_scorers.py`` (appendix A.10, summarised at S1.4), run on the vendored bodies
through the shim with numpy blocked; DIFF.md's hash columns and diff are checked the way the statistics' are.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import UTC, datetime

import pytest

from gigai_core.graders import lifted
from gigai_core.graders.protocol import Grader
from gigai_core.providers import Turn
from gigai_core.sample import Sample
from gigai_core.values import JsonValue
from tests.test_stats_lifted import NOTICE, PIN, VENDOR, diff_sections, reverse_apply

SCORERS = VENDOR / "scorers"
LIFTED = {
    "_common.py": "src/inspect_ai/scorer/_common.py",
    "_match.py": "src/inspect_ai/scorer/_match.py",
    "_pattern.py": "src/inspect_ai/scorer/_pattern.py",
    "_target.py": "src/inspect_ai/scorer/_target.py",
    "_unicode.py": "src/inspect_ai/scorer/_unicode.py",
    "text.py": "src/inspect_ai/_util/text.py",
    "judge_template.py": "src/inspect_ai/scorer/_model.py",
}
NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


def _sample(output: str) -> Sample:
    return Sample(
        trace_id="trace_1",
        run_id="run_1",
        item_id="item_1",
        item_version="1",
        cluster_id="item_1",
        repeat_index=0,
        input=(Turn(role="user", content="What is it?"),),
        output_text=output,
        output_parsed=None,
        parse_status="not_parsed",
        outcome="succeeded",
        error=None,
        recorded_at=NOW,
    )


def _value(grader: Grader, output: str, expected: JsonValue) -> float:
    return grader.grade(_sample(output), expected).value


@pytest.mark.usefixtures("block_numpy")
def test_s_smoke_hand_cases_on_the_vendored_bodies() -> None:
    from gigai_core._vendor.inspect_ai._metric import Score as VendorScore
    from gigai_core._vendor.inspect_ai.scorers import includes, match, pattern
    from gigai_core._vendor.inspect_ai.scorers._common import (
        match_str,
        normalize_number,
    )
    from gigai_core._vendor.inspect_ai.scorers._shim import (
        ModelOutput,
        Scorer,
        TaskState,
    )
    from gigai_core._vendor.inspect_ai.scorers._target import Target

    cases = [
        ("match-end numeric", match(location="end", numeric=True), "The answer is 1,234.0", ["1234"], "C"),
        ("match-end numeric wrong", match(location="end", numeric=True), "The answer is 1235", ["1234"], "I"),
        ("match-begin", match(location="begin"), "Paris is the capital", ["paris"], "C"),
        ("match-exact-fail", match(location="exact"), "Paris is the capital", ["Paris"], "I"),
        ("includes", includes(), "I think it is Paris.", ["paris"], "C"),
        ("includes-miss", includes(), "I think it is Rome.", ["paris"], "I"),
        ("pattern ANSWER", pattern(r"ANSWER:\s*(\w+)"), "blah\nANSWER: B", ["B"], "C"),
        ("pattern ANSWER miss", pattern(r"ANSWER:\s*(\w+)"), "blah\nANSWER: C", ["B"], "I"),
        ("pattern gsm8k-style", pattern(r"(?i)ANSWER\s*:\s*\$?(\d[\d,]*(?:\.\d+)?)"), "ANSWER: $1,234", ["1234"], "I"),
        ("empty completion", match(), "", ["x"], "I"),
    ]

    async def run(body: Scorer, text: str, target: list[str]) -> VendorScore:
        return await body(TaskState(ModelOutput(text)), Target(target))

    got = {name: asyncio.run(run(body, text, target)) for name, body, text, target, _ in cases}

    assert {name: score.value for name, score in got.items()} == {name: value for name, *_, value in cases}
    assert got["match-end numeric"].answer == "1234"
    assert got["pattern gsm8k-style"].answer == "1,234"  # the caveat: the group is compared as text
    assert got["empty completion"].reason == "no_response"
    assert all(score.reason is None for name, score in got.items() if name != "empty completion")
    assert normalize_number("1,234.50") == "1234.5"
    assert match_str(" 42.0 ", "42", numeric=True)[1] is True


def test_pattern_compares_text_where_match_normalises_numbers() -> None:
    marker = lifted.pattern(r"ANSWER:\s*(\d+)")
    assert _value(marker, "ANSWER: 1,234", "1234") == 0.0
    assert _value(marker, "ANSWER: 1234", "1234") == 1.0
    assert _value(lifted.match(), "ANSWER: 1,234", "1234") == 1.0
    assert _value(lifted.match(), "So the total is $1,234.00", "1234") == 1.0
    assert _value(lifted.match(), "the colour is Green", "green") == 1.0  # text, at the end
    assert _value(lifted.match(), "green, not blue", "green") == 0.0
    assert _value(lifted.match(), "it is 7", ["6", "7"]) == 1.0  # any target
    assert _value(lifted.includes(), "I think it is Paris.", "paris") == 1.0
    empty = lifted.match().grade(_sample(""), "4")
    assert (empty.value, empty.explanation) == (0.0, "no_response")


def test_lifted_graders_carry_the_pin_as_their_version_and_pass_the_answer_through() -> None:
    assert lifted.LIFTED_VERSION == f"inspect_ai@{PIN}"
    graders: list[Grader] = [lifted.includes(), lifted.match(), lifted.pattern(r"ANSWER:\s*(\w+)")]
    assert [(grader.id, grader.version, grader.kind) for grader in graders] == [
        ("includes", lifted.LIFTED_VERSION, "code"),
        ("match", lifted.LIFTED_VERSION, "code"),
        ("pattern", lifted.LIFTED_VERSION, "code"),
    ]
    score = lifted.match(numeric=True).grade(_sample("The answer is 1,234.0"), 1234)
    assert (score.value, score.answer, score.explanation) == (1.0, "1234", None)
    assert _value(lifted.match(numeric=False), "The answer is 1,234.0", "1234") == 0.0
    assert _value(lifted.match(location="begin"), "Paris is the capital", "paris") == 1.0
    assert _value(lifted.match(location="exact", ignore_case=False), "Paris", "paris") == 0.0
    assert _value(lifted.includes(ignore_case=False), "I think it is Paris.", "paris") == 0.0
    assert _value(lifted.pattern(r"ANSWER:\s*(\w+)", ignore_case=False), "ANSWER: b", "B") == 0.0


def test_scorers_diff_md_records_the_vendored_and_upstream_hashes() -> None:
    diff_md = (SCORERS / "DIFF.md").read_text(encoding="utf-8")
    assert "9567b431e7174480e56826b435da8f9f1de73bb8" in diff_md
    rows = re.findall(r"^\| `([^`]+)` \| (.+?) \| (`[0-9a-f]{64}`|-) \| `([0-9a-f]{64})` \|$", diff_md, re.MULTILINE)
    table = {name: (upstream_sha.strip("`"), vendored_sha) for name, _path, upstream_sha, vendored_sha in rows}

    on_disk = {path.name for path in SCORERS.iterdir() if path.is_file() and path.name != "DIFF.md"}
    assert set(table) == on_disk
    for name, (_, vendored_sha) in table.items():
        assert hashlib.sha256((SCORERS / name).read_bytes()).hexdigest() == vendored_sha, f"{name} changed without updating DIFF.md"

    sections = diff_sections(diff_md)
    assert set(sections) == set(LIFTED)
    for name, diff in sections.items():
        vendored = (SCORERS / name).read_text(encoding="utf-8").splitlines()
        upstream = "\n".join(reverse_apply(vendored, diff)) + "\n"
        assert hashlib.sha256(upstream.encode("utf-8")).hexdigest() == table[name][0], f"{name}: diff and upstream hash disagree"


def test_every_vendored_scorer_file_carries_its_header() -> None:
    assert {path.name for path in SCORERS.glob("*.py")} == set(LIFTED) | {"_shim.py", "__init__.py"}
    for name, upstream_path in LIFTED.items():
        assert (SCORERS / name).read_text(encoding="utf-8").splitlines()[0] == f"# Lifted from inspect_ai {upstream_path} @ {PIN} {NOTICE}"
    for name in ("_shim.py", "__init__.py"):
        first = (SCORERS / name).read_text(encoding="utf-8").splitlines()[0]
        assert f"@ {PIN} (MIT, Copyright (c) 2024 UK AI Security Institute; see ../LICENSE" in first
    assert "MIT License" in (VENDOR / "LICENSE").read_text(encoding="utf-8")
