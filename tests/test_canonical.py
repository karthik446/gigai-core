"""Canonical JSON bytes, digests and owned/imported text, relocated from gigai 0.3.2.

The byte and digest half of `tests/behaviors/integrity_canonical/test_canonical.py` and both
ownership walks of `test_canonical_ownership.py`. Ids, front matter and schema versions stay in gigai.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from gigai_core.canonical import (
    DuplicateJsonMemberError,
    ExactBytesRequiredError,
    InvalidCanonicalValueError,
    InvalidJsonError,
    InvalidOwnedTextError,
    canonical_json_bytes,
    canonical_json_digest,
    canonicalize_owned_text,
    digest_imported_bytes,
    digest_owned_text,
    parse_json_bytes,
)
from tests.conftest import SRC

VECTORS = Path(__file__).parent / "fixtures" / "canonical-vectors.json"
CANONICAL_PACKAGE = SRC / "canonical"
CANONICAL_API_NAMES = {
    "canonical_json_bytes",
    "canonical_json_digest",
    "canonicalize_owned_text",
    "digest_imported_bytes",
    "digest_owned_text",
    "parse_json_bytes",
    "sha256_hex",
    "sha256_file_hex",
}


def test_frozen_vectors_match_production_bytes_and_digests() -> None:
    fixture = json.loads(VECTORS.read_text(encoding="utf-8"))
    assert fixture["vectors"], "the frozen vector fixture is empty"
    for vector in fixture["vectors"]:
        assert canonical_json_bytes(vector["input"]) == vector["canonical_utf8"].encode("utf-8")
        assert canonical_json_digest(vector["input"]) == vector["sha256"]


def test_ascii_member_order_matches_restricted_jcs_order() -> None:
    value: JsonValue = {"a": 4, "_name": 3, "A": 2, "$name": 1}
    assert canonical_json_bytes(value) == b'{"$name":1,"A":2,"_name":3,"a":4}'


def test_str_subclasses_are_not_canonical_values() -> None:
    class Name(str):
        pass

    value: JsonValue = Name("plain")  # a StrEnum member is the same case
    with pytest.raises(InvalidCanonicalValueError):
        canonical_json_bytes(value)


def test_unicode_is_preserved_without_normalization() -> None:
    composed = canonical_json_digest({"text": "é"})
    decomposed = canonical_json_digest({"text": "e\u0301"})
    assert composed != decomposed


@pytest.mark.parametrize(
    "value",
    [
        {"amount": 1.5},
        {"amount": 9_007_199_254_740_992},
        {"amount": -9_007_199_254_740_992},
        {"not-ascii-é": "value"},
        {"bad key": "value"},
        {1: "value"},
        {"tuple": (1, 2)},
    ],
)
def test_noncanonical_values_fail_with_stable_type(value: object) -> None:
    with pytest.raises(InvalidCanonicalValueError) as raised:
        canonical_json_bytes(value)  # type: ignore[arg-type]
    assert raised.value.code == "invalid_canonical_value"


def test_cyclic_json_value_fails_with_stable_type() -> None:
    value: list[object] = []
    value.append(value)
    with pytest.raises(InvalidCanonicalValueError, match="cycles") as raised:
        canonical_json_bytes(value)  # type: ignore[arg-type]
    assert raised.value.code == "invalid_canonical_value"


def test_duplicate_members_are_rejected_at_any_depth() -> None:
    with pytest.raises(DuplicateJsonMemberError) as raised:
        parse_json_bytes(b'{"outer":{"a":1,"a":2}}')
    assert raised.value.code == "duplicate_json_member"


@pytest.mark.parametrize("data", [b"\xff", b"{", b'{"amount":NaN}'])
def test_invalid_json_fails_closed(data: bytes) -> None:
    error = InvalidJsonError if data != b'{"amount":NaN}' else InvalidCanonicalValueError
    with pytest.raises(error):
        parse_json_bytes(data)


def test_json_parser_requires_exact_bytes() -> None:
    with pytest.raises(ExactBytesRequiredError) as raised:
        parse_json_bytes('{"a":1}')  # type: ignore[arg-type]
    assert raised.value.code == "exact_bytes_required"


def test_owned_text_has_one_canonical_rendering_and_digest() -> None:
    assert canonicalize_owned_text("line one\r\nline two\n\n") == b"line one\nline two\n"
    assert digest_owned_text("line one\rline two") == digest_imported_bytes(b"line one\nline two\n")


@pytest.mark.parametrize("text", ["\ufeffnot allowed", "not\x00allowed", "\ud800"])
def test_invalid_owned_text_has_stable_error(text: str) -> None:
    with pytest.raises(InvalidOwnedTextError) as raised:
        canonicalize_owned_text(text)
    assert raised.value.code == "invalid_owned_text"


def test_owned_text_requires_text_not_bytes() -> None:
    with pytest.raises(InvalidOwnedTextError):
        canonicalize_owned_text(b"already bytes")  # type: ignore[arg-type]


def test_imported_bytes_are_hashed_without_normalization() -> None:
    crlf = b"line\r\n"
    lf = b"line\n"
    assert digest_imported_bytes(crlf) == f"sha256:{hashlib.sha256(crlf).hexdigest()}"
    assert digest_imported_bytes(lf) == f"sha256:{hashlib.sha256(lf).hexdigest()}"
    assert digest_imported_bytes(crlf) != digest_imported_bytes(lf)


def test_imported_digest_rejects_implicit_text_encoding() -> None:
    with pytest.raises(ExactBytesRequiredError) as raised:
        digest_imported_bytes("line\n")  # type: ignore[arg-type]
    assert raised.value.code == "exact_bytes_required"


def _modules_outside_canonical() -> list[Path]:
    return sorted(path for path in SRC.rglob("*.py") if CANONICAL_PACKAGE not in path.parents)


def test_canonical_package_owns_all_core_sha256_implementation() -> None:
    violations: list[str] = []
    for path in _modules_outside_canonical():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                if any(name == "hashlib" or name.startswith("hashlib.") for name in names):
                    violations.append(f"{path.relative_to(SRC)} imports hashlib")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "sha256":
                violations.append(f"{path.relative_to(SRC)} calls sha256")
    assert violations == []


def test_no_second_core_module_defines_canonical_api() -> None:
    violations: list[str] = []
    for path in _modules_outside_canonical():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in CANONICAL_API_NAMES:
                violations.append(f"{path.relative_to(SRC)} defines {node.name}")
    assert violations == []
