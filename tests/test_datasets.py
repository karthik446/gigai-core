"""Shipped ci slices, loaders, release builds and the wheel boundary of ``gigai_core.datasets``."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from random import Random
from types import ModuleType

import pytest

from gigai_core.canonical import sha256_file_hex
from gigai_core.dataset import (
    Dataset,
    DatasetDigestMismatch,
    DatasetFileError,
    DatasetItem,
    items_sha256,
)
from gigai_core.datasets import (
    LOADERS,
    SEED,
    SETS,
    SHIPPED,
    DatasetDownloadError,
    SliceNotShipped,
    SourceFile,
    build_release,
    load_slice,
    loader,
    mtbench_human,
    rewardbench2,
    tinygsm8k,
)
from gigai_core.values import JsonValue

from .conftest import SRC

PACKAGE = SRC / "datasets"
DATA = PACKAGE / "data"
NOTICES = (PACKAGE / "NOTICES.md").read_text(encoding="utf-8")
FIXTURES = os.environ.get("GIGAI_V032_FIXTURES")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
GSM8K_FILE = tinygsm8k.LOADER.sources[0].name


def _rows_from(rows: Sequence[Mapping[str, JsonValue]]) -> Callable[[Path], Sequence[Mapping[str, JsonValue]]]:
    def parquet_rows(path: Path) -> Sequence[Mapping[str, JsonValue]]:
        return rows

    return parquet_rows


def _package_root(root: Path) -> Callable[[str], Path]:
    def files(package: str) -> Path:
        return root

    return files


def _parquet(path: Path, columns: Mapping[str, Sequence[JsonValue]]) -> str:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(dict(columns)), path)
    return sha256_file_hex(path)


def _gsm8k_parquet(path: Path, rows: int) -> str:
    return _parquet(
        path,
        {
            "question": [f"What is {i} plus {i}?" for i in range(rows)],
            "answer": [f"{i} + {i} = <<{i}+{i}={2 * i}>>{2 * i}\n#### {2 * i}" for i in range(rows)],
            "extra_column": list(range(rows)),
        },
    )


def _local_source(path: Path, sha256: str | None = None) -> SourceFile:
    return SourceFile(name=path.name, url=path.as_uri(), sha256=sha256 or sha256_file_hex(path))


def test_registry_covers_every_set() -> None:
    assert sorted(LOADERS) == sorted(SETS)
    assert set(SHIPPED) < set(SETS)
    for name, set_loader in LOADERS.items():
        assert set_loader.name == name
        assert set_loader.sources, name
        assert all(HEX64.match(source.sha256) for source in set_loader.sources), name
        assert set(set_loader.counts) == set(set_loader.recipes) == {"ci", "release"}, name


@pytest.mark.parametrize("name", SHIPPED)
def test_shipped_slice_validates_and_matches_digest(name: str) -> None:
    dataset = load_slice(name)
    set_loader = LOADERS[name]
    sidecar = (DATA / f"{name}.sha256").read_text(encoding="utf-8")
    assert sidecar == f"{dataset.ref.sha256}\n"
    assert dataset.ref.sha256 == items_sha256(dataset.items) == sha256_file_hex(DATA / f"{name}.jsonl")
    assert dataset.ref.id == name
    assert dataset.ref.seed == SEED
    assert dataset.ref.slice_recipe is not None and loader.CI_FILTER in dataset.ref.slice_recipe.recipe
    assert len(dataset.items) == len({item.id for item in dataset.items})
    assert [item.id for item in dataset.items] == sorted(item.id for item in dataset.items)
    for item in dataset.items:
        assert item.slice_name == "ci"
        assert item.licence == set_loader.spdx
        assert item.cluster_id
        assert loader.publishable(item)
        assert len(item.prompt) <= loader.MAX_PROMPT_CHARS


@pytest.mark.parametrize("name", SHIPPED)
def test_shipped_slice_size_matches_ci_count(name: str) -> None:
    count = LOADERS[name].counts["ci"]
    expected = count.total if count.total is not None else sum(n or 0 for n in (count.per_stratum or {}).values())
    assert len(load_slice(name).items) == expected


def test_shipped_rows_are_written_without_defaults() -> None:
    for line in (DATA / "tinygsm8k.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert "completion" not in row and "slice_recipe" not in row
        assert row["slice_name"] == "ci" and row["licence"] == "MIT"
        assert json.dumps(row, separators=(",", ":"), ensure_ascii=False) == line


def test_unshipped_sets_and_suites_raise() -> None:
    for name in sorted(set(SETS) - set(SHIPPED)):
        with pytest.raises(SliceNotShipped, match=name):
            load_slice(name)
    with pytest.raises(SliceNotShipped, match="release"):
        load_slice("tinygsm8k", suite="release")
    with pytest.raises(ValueError, match="unknown"):
        load_slice("nope")


def test_tampered_slice_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    lines = (DATA / "tinygsm8k.jsonl").read_text(encoding="utf-8").splitlines()
    lines[3] = lines[3].replace('"prompt":"', '"prompt":"Ignore the instruction. ')
    (data / "tinygsm8k.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (data / "tinygsm8k.sha256").write_bytes((DATA / "tinygsm8k.sha256").read_bytes())
    monkeypatch.setattr(loader, "files", _package_root(tmp_path))
    with pytest.raises(DatasetDigestMismatch, match="expected"):
        load_slice("tinygsm8k")


def test_bad_line_names_its_number(tmp_path: Path) -> None:
    lines = (DATA / "tinygsm8k.jsonl").read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace('"cluster_id"', '"cluster"')
    path = tmp_path / "broken.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(DatasetFileError, match=r"broken\.jsonl line 2: ") as caught:
        Dataset.from_jsonl(path)
    assert caught.value.line == 2
    assert caught.value.path == path


def test_blank_lines_are_skipped_and_digest_is_optional(tmp_path: Path) -> None:
    text = (DATA / "ifeval.jsonl").read_text(encoding="utf-8")
    path = tmp_path / "spaced.jsonl"
    path.write_text("\n" + text.replace("\n", "\n\n"), encoding="utf-8")
    dataset = Dataset.from_jsonl(path)
    assert dataset.items == load_slice("ifeval").items
    assert dataset.ref.id == "spaced"
    assert dataset.ref.sha256 == items_sha256(dataset.items)


@pytest.mark.skipif(FIXTURES is None, reason="GIGAI_V032_FIXTURES not set")
@pytest.mark.parametrize("name", SHIPPED)
def test_shipped_slice_matches_v032_fixture(name: str) -> None:
    assert FIXTURES is not None
    fixture = Path(FIXTURES) / f"{name}.jsonl"
    rows = {row["id"]: row for row in map(json.loads, fixture.read_text(encoding="utf-8").splitlines())}
    items = {item.id: item for item in load_slice(name).items}
    assert sorted(rows) == sorted(items)
    for item_id, row in rows.items():
        item = items[item_id]
        assert item.prompt == row["prompt"], item_id
        assert item.expected == row["expected"], item_id
        assert item.cluster_id == row["cluster_id"], item_id
        assert dict(item.source_ids) == row["source_ids"], item_id
        assert list(item.tags) == row["tags"], item_id
        assert set(row) - {"completion", "slice_recipe"} <= set(item.model_dump())


def test_notices_cover_every_set_and_name_the_shipped_ones() -> None:
    sections = re.findall(r"^## (\S+)$", NOTICES, flags=re.MULTILINE)
    assert sorted(sections) == sorted(SETS)
    for name, set_loader in LOADERS.items():
        body = NOTICES.split(f"\n## {name}\n", 1)[1].split("\n## ", 1)[0]
        assert f"- SPDX: {set_loader.spdx}\n" in body, name
        assert "- Attribution:" in body or "- Licence" in body, name
        where = "`ci` slice" if name in SHIPPED else "download-only"
        assert re.search(rf"^\| {name} \| {re.escape(set_loader.spdx)} \| {where} \|", NOTICES, re.MULTILINE), name
    assert "injection_composite" not in NOTICES


def test_no_src_file_names_the_old_package() -> None:
    package = PACKAGE
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".md", ".jsonl", ".sha256"}:
            assert not re.search(r"gigai[./]", path.read_text(encoding="utf-8")), path


def test_release_build_from_local_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = tmp_path / "upstream" / GSM8K_FILE
    _gsm8k_parquet(upstream, 120)
    monkeypatch.setattr(tinygsm8k.LOADER, "sources", (_local_source(upstream),))
    out = build_release("tinygsm8k", tmp_path / "build")
    assert out == tmp_path / "build" / "tinygsm8k.jsonl"
    assert (tmp_path / "build" / "raw" / "tinygsm8k" / GSM8K_FILE).read_bytes() == upstream.read_bytes()
    digest = out.with_suffix(".sha256").read_text(encoding="utf-8").strip()
    dataset = Dataset.from_jsonl(out, expect_sha256=digest)
    assert len(dataset.items) == 100
    assert [item.id for item in dataset.items] == sorted(item.id for item in dataset.items)
    first = dataset.items[0]
    assert first.slice_name == "release" and first.licence == "MIT"
    assert first.prompt.endswith(f"What is {first.source_ids['row']} plus {first.source_ids['row']}?")
    assert first.expected == str(2 * int(first.source_ids["row"]))
    assert "<<" not in first.prompt
    again = build_release("tinygsm8k", tmp_path / "build")
    assert again.with_suffix(".sha256").read_text(encoding="utf-8").strip() == digest


def test_pinned_sha256_mismatch_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = tmp_path / "upstream" / GSM8K_FILE
    _gsm8k_parquet(upstream, 3)
    monkeypatch.setattr(tinygsm8k.LOADER, "sources", (_local_source(upstream, "0" * 64),))
    with pytest.raises(DatasetDownloadError, match="pinned sha256") as caught:
        build_release("tinygsm8k", tmp_path / "build")
    assert caught.value.code == "sha256_mismatch"
    assert upstream.as_uri() in str(caught.value)


def test_unreachable_source_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "nowhere" / GSM8K_FILE
    source = SourceFile(name=missing.name, url=missing.as_uri(), sha256="0" * 64)
    monkeypatch.setattr(tinygsm8k.LOADER, "sources", (source,))
    with pytest.raises(DatasetDownloadError, match="cannot download") as caught:
        build_release("tinygsm8k", tmp_path / "build")
    assert caught.value.code == "download_failed"
    assert not list((tmp_path / "build").rglob("*.part"))


def test_slice_too_short_for_the_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "raw"
    _gsm8k_parquet(raw / GSM8K_FILE, 5)
    with pytest.raises(ValueError, match="Sample larger than population"):
        loader.build_slice(tinygsm8k.LOADER, raw, "ci")


def test_parquet_rows_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "rows.parquet"
    _parquet(path, {"id": ["a", "b"], "choices": [["x", "y"], ["z"]], "turn": [1, 2]})
    rows = loader.parquet_rows(path)
    assert [dict(row) for row in rows] == [
        {"id": "a", "choices": ["x", "y"], "turn": 1},
        {"id": "b", "choices": ["z"], "turn": 2},
    ]


def test_parquet_unavailable_without_pyarrow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> ModuleType:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(loader, "import_module", missing)
    with pytest.raises(loader.ParquetUnavailable, match="parquet extra"):
        loader.parquet_rows(tmp_path / "missing.parquet")


def test_importing_datasets_loads_no_heavy_modules() -> None:
    code = (
        "import sys, gigai_core.datasets\n"
        "print(sorted(m for m in ('pyarrow', 'click', 'jsonschema', 'sqlite3', 'gigai', 'urllib.request')"
        " if m in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_rewardbench2_ids_are_unique_across_subsets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def rows(path: Path) -> Sequence[Mapping[str, JsonValue]]:
        return [
            {
                "id": str(number),
                "subset": subset,
                "prompt": f"prompt {number}",
                "chosen": [f"{subset}-{number}-good"],
                "rejected": [f"{subset}-{number}-bad-{k}" for k in range(3)],
                "models": [f"m{k}" for k in range(4)],
                "ignored": "column",
            }
            for subset in ("Focus", "Factuality", "Safety")
            for number in (2, 10, 1)
        ]

    monkeypatch.setattr(rewardbench2, "parquet_rows", rows)
    items = rewardbench2.items(tmp_path)
    assert len(items) == 6
    assert len({item.id for item in items}) == 6
    assert [int(str(item.source_ids["id"])) for item in items] == [1, 1, 2, 2, 10, 10]
    for item in items:
        assert item.tags[0] in ("Focus", "Factuality")
        assert set(item.source_ids) == {"id", "subset", "model_A", "model_B", "model_C", "model_D"}
        expected = item.expected
        assert isinstance(expected, dict)
        chosen = str(expected["chosen"])
        assert f"RESPONSE {chosen}:\n{item.tags[0]}-{item.source_ids['id']}-good" in item.prompt
        assert item.tags[1] == f"chosen:{chosen}"


def test_rewardbench2_response_order_is_seeded_per_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row: Mapping[str, JsonValue] = {
        "id": "7",
        "subset": "Focus",
        "prompt": "p",
        "chosen": ["good"],
        "rejected": ["bad-0", "bad-1", "bad-2"],
        "models": ["m0", "m1", "m2", "m3"],
    }
    monkeypatch.setattr(rewardbench2, "parquet_rows", _rows_from([row]))
    first = rewardbench2.items(tmp_path)
    second = rewardbench2.items(tmp_path)
    assert first == second
    assert first[0].expected == {"chosen": "A", "rejected": ["B", "C", "D"]}


def _mtbench_conversation(answer: str) -> list[JsonValue]:
    return [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": f"{answer} one"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": f"{answer} two"},
    ]


def _mtbench_row(turn: int, judge: str, winner: str) -> dict[str, JsonValue]:
    return {
        "question_id": 81,
        "model_a": "alpaca-13b",
        "model_b": "vicuna-13b",
        "winner": winner,
        "judge": judge,
        "conversation_a": _mtbench_conversation("alpha"),
        "conversation_b": _mtbench_conversation("beta"),
        "turn": turn,
    }


def test_mtbench_turns_follow_the_conversation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        _mtbench_row(2, "expert_1", "model_b"),
        _mtbench_row(2, "expert_2", "model_b"),
        _mtbench_row(1, "expert_1", "model_a"),
    ]
    monkeypatch.setattr(mtbench_human, "parquet_rows", _rows_from(rows))
    items = mtbench_human.items(tmp_path)
    assert len(items) == 1
    item = items[0]
    assert item.prompt == "" and item.system is not None and len(item.turns) == 2
    assert all(turn.role == "user" for turn in item.turns)
    assert "first question" in item.turns[0].content and "alpha one" in item.turns[0].content
    assert "beta two" in item.turns[1].content
    assert item.expected == {"winner": "B"}
    assert item.source_ids["judges"] == "expert_1,expert_2"


def test_mtbench_rejects_a_conversation_shorter_than_its_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_mtbench_row(2, "expert_1", "model_b"), _mtbench_row(2, "expert_2", "model_b")]
    rows[1]["conversation_b"] = _mtbench_conversation("beta")[:2]
    rows[0]["conversation_b"] = _mtbench_conversation("beta")[:2]
    monkeypatch.setattr(mtbench_human, "parquet_rows", _rows_from(rows))
    with pytest.raises(ValueError, match=r"question 81: .*turn 2"):
        mtbench_human.items(tmp_path)


def test_build_slice_round_trips_through_write_and_read(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _gsm8k_parquet(raw / GSM8K_FILE, 30)
    dataset = loader.build_slice(tinygsm8k.LOADER, raw, "ci")
    assert len(dataset.items) == 10
    assert dataset.ref.slice_recipe is not None
    assert dataset.ref.slice_recipe.recipe.startswith(tinygsm8k.RECIPES["ci"])
    out = tmp_path / "out" / "tinygsm8k.jsonl"
    digest = loader.write_slice(dataset, out)
    assert out.with_suffix(".sha256").read_text(encoding="utf-8") == f"{digest}\n"
    back = Dataset.from_jsonl(out, expect_sha256=digest)
    assert back.items == dataset.items


def test_build_slice_stamps_the_loader_licence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "raw"
    _gsm8k_parquet(raw / GSM8K_FILE, 120)
    monkeypatch.setattr(tinygsm8k.LOADER, "spdx", "CC-BY-4.0")
    for suite in ("ci", "release"):
        dataset = loader.build_slice(tinygsm8k.LOADER, raw, suite)
        assert {item.licence for item in dataset.items} == {"CC-BY-4.0"}
        assert {item.slice_name for item in dataset.items} == {suite}


def test_loader_bodies_stay_short() -> None:
    for module in sorted(PACKAGE.glob("*.py")):
        if module.name in ("__init__.py", "loader.py"):
            continue
        tree = ast.parse(module.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        assert {"items", "select"} <= {function.name for function in functions}, module.name
        for function in functions:
            assert function.end_lineno is not None
            assert function.end_lineno - function.body[0].lineno + 1 <= 30, f"{module.name}:{function.name}"


def test_shipped_slices_stay_small() -> None:
    for name in SHIPPED:
        assert len((DATA / f"{name}.jsonl").read_bytes().splitlines()) <= 60, name


def test_build_slice_is_deterministic(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _gsm8k_parquet(raw / GSM8K_FILE, 40)
    first = loader.build_slice(tinygsm8k.LOADER, raw, "ci")
    second = loader.build_slice(tinygsm8k.LOADER, raw, "ci")
    assert first == second
    assert first.ref.sha256 == items_sha256(first.items)


def test_build_slice_rejects_overlong_prompts(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _parquet(
        raw / GSM8K_FILE,
        {"question": ["x" * (loader.MAX_PROMPT_CHARS + 1)] + ["q"] * 11, "answer": ["#### 1"] * 12},
    )
    with pytest.raises(ValueError, match=f"over {loader.MAX_PROMPT_CHARS}"):
        loader.build_slice(tinygsm8k.LOADER, raw, "ci")


def test_ci_slice_drops_unpublishable_items(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    questions = [f"q{i}" for i in range(12)]
    questions[0] = "mail me at someone@corp.com"
    questions[1] = "see /Users/someone/file.txt"
    questions[2] = "mail me at someone@example.com"
    _parquet(raw / GSM8K_FILE, {"question": questions, "answer": ["#### 1"] * 12})
    dataset = loader.build_slice(tinygsm8k.LOADER, raw, "ci")
    prompts = [item.prompt for item in dataset.items]
    assert not any("corp.com" in prompt or "/Users/" in prompt for prompt in prompts)
    assert len(dataset.items) == 10


def test_publishable_rules() -> None:
    def item(text: str) -> DatasetItem:
        return DatasetItem(id="x", prompt=text, expected="y", cluster_id="x")

    assert loader.publishable(item("plain text"))
    assert loader.publishable(item("write to admin@example.com"))
    assert not loader.publishable(item("write to admin@company.io"))
    assert not loader.publishable(item("saved under /Users/me/notes"))


def test_stratified_draws_per_stratum() -> None:
    values = [f"{band}-{i}" for band in ("short", "long") for i in range(20)]
    counts = {"short": 3, "long": None}
    picked = loader.stratified(values, lambda value: value.split("-")[0], counts, Random(1))
    assert sum(value.startswith("short") for value in picked) == 3
    assert sum(value.startswith("long") for value in picked) == 20
    with pytest.raises(ValueError, match="short"):
        loader.stratified(values[:2], lambda value: value.split("-")[0], {"short": 3}, Random(1))


def test_length_band_edges() -> None:
    assert loader.length_band(0) == "<8k"
    assert loader.length_band(7_999) == "<8k"
    assert loader.length_band(8_000) == "8-24k"
    assert loader.length_band(24_000) == "24-60k"
