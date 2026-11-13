"""Two 0.3.2 ``test_loaders`` behaviours that moved to core with no test (A8 recount): what each NOTICES section
must carry for its licence, and ContractNLI's ``audit_pending`` tag."""

from __future__ import annotations

import re

import pytest

from gigai_core.datasets import SETS, load_slice

from .conftest import SRC

NOTICES = (SRC / "datasets" / "NOTICES.md").read_text(encoding="utf-8")
#: What each licence requires to travel with a copy: MIT its copyright and permission notice, Apache-2.0 a copy or
#: the URL of the licence, CC BY 4.0 the licence URI, ODC-By 1.0 the licence URL. Every set also states the changes.
LICENCE_TERMS: dict[str, tuple[str, ...]] = {
    "MIT": ("Copyright (c) ", "Permission is hereby granted, free of charge", "this permission notice shall be included"),
    "Apache-2.0": ("https://www.apache.org/licenses/LICENSE-2.0",),
    "CC-BY-4.0": ("https://creativecommons.org/licenses/by/4.0/",),
    "ODC-By-1.0": ("https://opendatacommons.org/licenses/by/1-0/",),
}


def _notice_sections() -> dict[str, str]:
    return {section.split()[0]: section for section in re.split(r"^## ", NOTICES, flags=re.MULTILINE)[1:]}


@pytest.mark.parametrize("name", SETS)
def test_every_notice_carries_what_its_licence_requires(name: str) -> None:
    section = _notice_sections()[name]
    (licence,) = re.findall(r"^- SPDX: (\S+)$", section, flags=re.MULTILINE)
    for term in (*LICENCE_TERMS[licence], "Changes made:"):
        assert term in section, (name, term)


def test_contractnli_items_are_audit_pending() -> None:
    items = load_slice("contractnli").items
    assert items
    assert all(item.tags[0] == "audit_pending" for item in items)
