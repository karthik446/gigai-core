"""Canonical JSON bytes and SHA-256 digests (the identity half of gigai's ``canonical.py``)."""

from __future__ import annotations

from gigai_core.canonical.digest import (
    canonical_json_digest,
    canonicalize_owned_text,
    digest_imported_bytes,
    digest_owned_text,
    sha256_file_hex,
    sha256_hex,
)
from gigai_core.canonical.encode import (
    MEMBER_NAME,
    SAFE_INTEGER_MAX,
    CanonicalizationError,
    DuplicateJsonMemberError,
    ExactBytesRequiredError,
    InvalidCanonicalValueError,
    InvalidJsonError,
    InvalidOwnedTextError,
    canonical_json_bytes,
    parse_json_bytes,
)

__all__ = [
    "MEMBER_NAME",
    "SAFE_INTEGER_MAX",
    "CanonicalizationError",
    "DuplicateJsonMemberError",
    "ExactBytesRequiredError",
    "InvalidCanonicalValueError",
    "InvalidJsonError",
    "InvalidOwnedTextError",
    "canonical_json_bytes",
    "canonical_json_digest",
    "canonicalize_owned_text",
    "digest_imported_bytes",
    "digest_owned_text",
    "parse_json_bytes",
    "sha256_file_hex",
    "sha256_hex",
]
