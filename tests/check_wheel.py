"""Check what a built `gigai_core` wheel ships: `python tests/check_wheel.py dist/*.whl`.

Must list `py.typed`, the three vendor LICENSE files, the autoevals templates, the datasets NOTICES and the six shipped
slices with their digests; must not list a schema file, migrations or a `store/` package (those stay in gigai).
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

SHIPPED_SETS = ("tinygsm8k", "tinymmlu", "ifeval", "jsonschemabench", "contractnli", "ragtruth")
REQUIRED = (
    "gigai_core/py.typed",
    "gigai_core/_vendor/inspect_ai/LICENSE",
    "gigai_core/_vendor/autoevals/LICENSE",
    "gigai_core/_vendor/openevals/LICENSE",
    "gigai_core/_vendor/autoevals/templates/battle.yaml",
    "gigai_core/_vendor/autoevals/templates/closed_q_a.yaml",
    "gigai_core/_vendor/autoevals/templates/factuality.yaml",
    "gigai_core/datasets/NOTICES.md",
    *(f"gigai_core/datasets/data/{name}.{suffix}" for name in SHIPPED_SETS for suffix in ("jsonl", "sha256")),
)
FORBIDDEN_PARTS = ("migrations", "store")
FORBIDDEN_SUFFIXES = (".schema.json",)


def check(wheel: Path) -> list[str]:
    """Return the problems found in `wheel`; an empty list means it passes."""
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    problems = [f"missing {name}" for name in REQUIRED if name not in names]
    for name in sorted(names):
        parts = name.split("/")
        if any(part in FORBIDDEN_PARTS for part in parts[:-1]) or name.endswith(FORBIDDEN_SUFFIXES):
            problems.append(f"must not ship {name}")
    if not any(name.endswith(".dist-info/METADATA") for name in names):
        problems.append("no METADATA")
    return problems


def main(argv: list[str]) -> int:
    """Check every wheel named on the command line; exit 1 on the first problem list."""
    if not argv:
        print("usage: check_wheel.py dist/*.whl", file=sys.stderr)
        return 2
    status = 0
    for arg in argv:
        problems = check(Path(arg))
        print(f"{arg}: {'OK' if not problems else 'FAIL'}")
        for problem in problems:
            print(f"  {problem}")
        status |= bool(problems)
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
