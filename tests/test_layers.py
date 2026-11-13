"""The import graph of `gigai_core` is layered, acyclic and never reaches back into gigai.

DESIGN.md section 3: `models`, `values` and `canonical` import nothing from core; above them
`providers` -> `budget`, `trace`, `dataset`, `stats` -> `sample`, `score` -> `records`, `graders`,
`agent` -> `run` -> `comparison`, `agreement` -> `gate`; `datasets/` imports `dataset` and `recorded`
imports `providers`/`trace` only. Modules that do not exist yet are simply absent; a module that is
not in the table fails the test until the table grows.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from tests.conftest import SRC

PACKAGE = "gigai_core"
VENDOR = SRC / "_vendor"

LAYER = {
    "models": 0,
    "values": 0,
    "canonical": 0,
    "providers": 1,
    "budget": 2,
    "trace": 2,
    "dataset": 2,
    "stats": 2,
    "sample": 3,
    "score": 3,
    "datasets": 3,
    "recorded": 3,
    "records": 4,
    "graders": 4,
    "agent": 4,
    "run": 5,
    "comparison": 6,
    "gate": 7,
    "agreement": 6,
}
# Modules whose imports from core are narrower than "anything below".
ONLY = {
    "stats": {"models", "values"},
    "datasets": {"dataset", "models", "values", "canonical", "providers"},
    "recorded": {"providers", "trace", "models", "values", "canonical"},
}
GIGAI_REFERENCE = re.compile(r"gigai[./]")


def _core_files() -> list[Path]:
    return sorted(path for path in SRC.rglob("*.py") if VENDOR not in path.parents)


def _module_of(path: Path) -> str:
    """The top-level core module a file belongs to (`canonical/encode.py` -> `canonical`); `""` for the root."""
    parts = path.relative_to(SRC).with_suffix("").parts
    if parts == ("__init__",):
        return ""
    return parts[0]


def _imports(path: Path) -> list[tuple[int, str, list[str]]]:
    """Every import of a `gigai_core` module in `path`: (line, dotted module, imported names)."""
    found: list[tuple[int, str, list[str]]] = []
    package = PACKAGE if _module_of(path) == "" else f"{PACKAGE}.{'.'.join(path.relative_to(SRC).parts[:-1])}".rstrip(".")
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name, []) for alias in node.names if alias.name.split(".")[0] == PACKAGE]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            if module.split(".")[0] == PACKAGE:
                found.append((node.lineno, module, [alias.name for alias in node.names]))
    return found


def _top(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else ""


def test_every_core_module_is_in_the_layer_table() -> None:
    modules = {_module_of(path) for path in _core_files()} - {""}
    assert modules <= set(LAYER), f"modules missing from the layer table: {sorted(modules - set(LAYER))}"


def test_no_module_imports_a_layer_at_or_above_itself_and_narrow_modules_stay_narrow() -> None:
    offenders: list[str] = []
    for path in _core_files():
        here = _module_of(path)
        for line, module, _names in _imports(path):
            target = _top(module)
            if target in ("", here, "_vendor"):
                continue
            where = f"{path.relative_to(SRC)}:{line} imports {module}"
            if here == "":
                offenders.append(f"{where} (the package root imports nothing from core)")
            elif LAYER[target] >= LAYER[here]:
                offenders.append(f"{where} (layer {LAYER[target]} >= {LAYER[here]})")
            elif here in ONLY and target not in ONLY[here]:
                offenders.append(f"{where} ({here} may import only {sorted(ONLY[here])})")
    assert offenders == []


def test_the_import_graph_has_no_cycle() -> None:
    graph: dict[str, set[str]] = {}
    for path in _core_files():
        here = _module_of(path)
        for _line, module, _names in _imports(path):
            target = _top(module)
            if target not in ("", here, "_vendor"):
                graph.setdefault(here, set()).add(target)

    def walk(node: str, trail: tuple[str, ...]) -> None:
        if node in trail:
            raise AssertionError(f"import cycle: {' -> '.join(trail[trail.index(node):] + (node,))}")
        for successor in graph.get(node, ()):
            walk(successor, trail + (node,))

    for start in graph:
        walk(start, ())


def test_no_core_file_mentions_gigai_as_an_import_or_a_path() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if GIGAI_REFERENCE.search(line):
                offenders.append(f"{path.relative_to(SRC)}:{number}: {line.strip()}")
    assert offenders == []


def test_no_module_imports_an_underscore_name_from_another_module() -> None:
    offenders: list[str] = []
    for path in _core_files():
        here = _module_of(path)
        for line, module, names in _imports(path):
            if _top(module) == here:
                continue
            if "_vendor" in module.split("."):
                continue
            private_segments = [part for part in module.split(".")[1:] if part.startswith("_") and part != "_vendor"]
            private_names = [name for name in names if name.startswith("_")]
            if private_segments or private_names:
                offenders.append(f"{path.relative_to(SRC)}:{line} imports {module} {private_names or ''}".rstrip())
    assert offenders == []
