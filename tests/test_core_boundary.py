"""Importing `gigai_core` pulls in nothing but the standard library and pydantic."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import textwrap

from tests.conftest import SRC

FORBIDDEN = ("click", "jsonschema", "pyarrow", "sqlite3", "gigai")


def test_importing_core_and_canonical_loads_none_of_the_forbidden_modules() -> None:
    code = textwrap.dedent(
        f"""
        import sys
        import gigai_core
        import gigai_core.canonical
        print(sorted(set({FORBIDDEN!r}) & set(sys.modules)))
        print(gigai_core.__version__)
        """
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["[]", importlib.metadata.version("gigai-core")]


def test_package_ships_py_typed_and_all_is_the_facade() -> None:
    import gigai_core
    from tests.test_facade import FACADE

    assert (SRC / "py.typed").is_file()
    assert gigai_core.__all__ == FACADE
