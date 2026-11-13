"""The façade (DESIGN 4.9): what ``from gigai_core import ...`` offers, and that importing it stays light.

The lesson snippets themselves (1, 2, 4, 5 and 6) run in ``tests/test_lesson_snippets.py``; the last test here
holds that file to the façade, so a snippet never reaches into a module for a name the façade exports.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import subprocess
import sys
import textwrap

import gigai_core
import gigai_core.graders
from tests.conftest import REPO

DESIGN_4_9 = [
    "run",
    "save_run",
    "open_run",
    "Run",
    "Dataset",
    "DatasetItem",
    "CallSettings",
    "ModelProvider",
    "RecordedProvider",
    "CallBudget",
    "JsonlTrace",
    "read_spans",
    "compare",
    "gate",
    "MinLowerBound",
    "NoWorseThan",
    "cohen_kappa",
    "judge_agreement",
    "CoreModel",
    "read_labels",
    "write_labels",
]
GRADER_CONSTRUCTORS = [
    "answer_letter",
    "autoevals_judge",
    "bool_field",
    "constant",
    "ifeval",
    "includes",
    "json_valid",
    "label_share",
    "labelled",
    "last_number",
    "llm_judge",
    "match",
    "model_graded_qa",
    "openevals_judge",
    "pairwise_judge",
    "pattern",
    "ranks_first",
    "schema_valid",
    "verdict_equals",
    "with_wrong_labels",
]
FACADE = sorted(DESIGN_4_9 + GRADER_CONSTRUCTORS)
HEAVY = ("click", "jsonschema", "pyarrow", "gigai", "sqlite3", "numpy", "yaml")


def _constructors_graders_exports() -> list[str]:
    """Every public callable of ``gigai_core.graders`` whose return annotation is ``Grader`` or ``Judge``."""
    found: list[str] = []
    for name in gigai_core.graders.__all__:
        value = getattr(gigai_core.graders, name)
        if inspect.isfunction(value) and inspect.signature(value).return_annotation in ("Grader", "Judge"):
            found.append(name)
    return found


def test_all_is_the_design_list_plus_every_grader_constructor_sorted() -> None:
    assert gigai_core.__all__ == FACADE
    assert gigai_core.__all__ == sorted(gigai_core.__all__)
    assert _constructors_graders_exports() == GRADER_CONSTRUCTORS


def test_every_facade_name_imports_from_the_root_and_none_is_a_module() -> None:
    for name in FACADE:
        value = getattr(gigai_core, name)
        assert not inspect.ismodule(value), f"gigai_core.{name} is a module, not the exported object"
        module = importlib.import_module(value.__module__)
        assert vars(module)[name] is value, f"gigai_core.{name} is not {value.__module__}.{name}"


def test_graders_all_is_sorted_and_names_no_module() -> None:
    exported = gigai_core.graders.__all__
    assert exported == sorted(exported)
    for name in exported:
        assert not inspect.ismodule(getattr(gigai_core.graders, name)), name


def test_importing_the_facade_in_a_fresh_interpreter_loads_no_heavy_module() -> None:
    code = textwrap.dedent(
        f"""
        import sys
        before = set(sys.modules)
        from gigai_core import {", ".join(FACADE)}
        print(sorted(set({HEAVY!r}) & set(sys.modules)))
        third_party = sorted(
            name.split(".")[0] for name in set(sys.modules) - before
            if name.split(".")[0] not in sys.stdlib_module_names and not name.startswith(("gigai_core", "_"))
        )
        print(sorted(set(third_party)))
        """
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    heavy, third_party = result.stdout.splitlines()
    assert heavy == "[]"
    assert set(ast.literal_eval(third_party)) <= {"pydantic", "pydantic_core", "annotated_types", "typing_extensions", "typing_inspection"}, third_party


def test_the_lesson_snippets_import_every_facade_name_from_the_root() -> None:
    tree = ast.parse((REPO / "tests" / "test_lesson_snippets.py").read_text(encoding="utf-8"))
    from_root: set[str] = set()
    from_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("gigai_core"):
            names = [alias.name for alias in node.names]
            if node.module == "gigai_core":
                from_root.update(names)
            else:
                from_modules += [f"{node.module}.{name}" for name in names if name in FACADE]
    assert from_modules == [], f"snippets reach past the façade for: {from_modules}"
    assert {"run", "save_run", "open_run", "compare", "gate", "judge_agreement", "read_labels", "write_labels", "includes", "llm_judge"} <= from_root
