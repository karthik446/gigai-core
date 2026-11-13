"""The built wheel carries the six ci slices and reads them without the repository checkout.

Runs only when ``GIGAI_WHEEL_DIR`` names a directory holding one ``gigai_core-*.whl`` (``uv build --wheel``
writes it). The wheel is unpacked into a temporary directory that is put ahead of the checkout on ``PYTHONPATH``;
a subprocess then loads every shipped slice from the unpacked package.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from gigai_core import datasets

WHEEL_DIR = os.environ.get("GIGAI_WHEEL_DIR")
PROBE = """
import json, sys
import gigai_core.datasets as datasets
from importlib.resources import files
package = files("gigai_core.datasets")
names = sorted(p.name for p in (package / "data").iterdir())
loaded = {name: len(datasets.load_slice(name).items) for name in datasets.SHIPPED}
notices = (package / "NOTICES.md").is_file() and "## tinygsm8k" in (package / "NOTICES.md").read_text("utf-8")
report = {"file": datasets.__file__, "files": names, "loaded": loaded, "notices": notices}
print(json.dumps(report | {"pyarrow": "pyarrow" in sys.modules}))
"""


@pytest.mark.skipif(WHEEL_DIR is None, reason="GIGAI_WHEEL_DIR not set")
def test_wheel_ships_and_reads_the_ci_slices(tmp_path: Path) -> None:
    assert WHEEL_DIR is not None
    wheels = sorted(Path(WHEEL_DIR).glob("gigai_core-*.whl"))
    assert len(wheels) == 1, wheels
    site = tmp_path / "site"
    with zipfile.ZipFile(wheels[0]) as wheel:
        wheel.extractall(site)
    env = {**os.environ, "PYTHONPATH": str(site)}
    probe = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, check=True, env=env)
    result = json.loads(probe.stdout.strip())
    assert Path(result["file"]).is_relative_to(site)
    assert result["pyarrow"] is False
    assert sorted(result["loaded"]) == sorted(datasets.SHIPPED)
    assert result["files"] == sorted(f"{name}.{ext}" for name in datasets.SHIPPED for ext in ("jsonl", "sha256"))
    assert all(count > 0 for count in result["loaded"].values())
    assert result["notices"] is True
