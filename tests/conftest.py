from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "gigai_core"
VENDOR_PREFIX = "gigai_core._vendor"


@pytest.fixture
def block_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run a test with numpy unimportable and the vendored packages re-imported under that block.

    `sys.modules["numpy"] = None` makes any `import numpy` raise ImportError whether or
    not numpy is installed; monkeypatch restores the previous entries after the test.
    """
    monkeypatch.setitem(sys.modules, "numpy", None)
    for name in [name for name in sys.modules if name == VENDOR_PREFIX or name.startswith(VENDOR_PREFIX + ".")]:
        monkeypatch.delitem(sys.modules, name)
