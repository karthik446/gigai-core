"""SHA-256 digests of canonical JSON, owned text and imported bytes.

The API names make byte ownership explicit: :func:`canonicalize_owned_text` normalizes text created by the program,
:func:`digest_imported_bytes` hashes imported user bytes without changing them. Callers must never decode and
re-encode imported content before passing it to ``digest_imported_bytes``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import JsonValue

from gigai_core.canonical.encode import (
    ExactBytesRequiredError,
    InvalidOwnedTextError,
    canonical_json_bytes,
)


def _sha256_digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def sha256_hex(data: bytes) -> str:
    """SHA-256 of ``data`` as bare lowercase hex, for ids and sidecars that carry no ``sha256:`` prefix."""
    return hashlib.sha256(data).hexdigest()


def sha256_file_hex(path: Path) -> str:
    """SHA-256 of the file's bytes as bare hex, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_digest(value: JsonValue) -> str:
    """Digest the canonical logical identity of a JSON value, as ``sha256:<hex>``."""
    return _sha256_digest(canonical_json_bytes(value))


def canonicalize_owned_text(text: str) -> bytes:
    """Normalize owned text to UTF-8, LF, and exactly one final LF.

    Args:
        text: Text the program wrote itself.

    Returns:
        The normalized UTF-8 bytes.

    Raises:
        InvalidOwnedTextError: ``text`` is not a string, contains NUL, begins with a BOM or holds a lone surrogate.
    """
    if type(text) is not str:
        raise InvalidOwnedTextError("GigAI-owned text must be a string")
    if "\x00" in text:
        raise InvalidOwnedTextError("GigAI-owned text cannot contain NUL")
    if text.startswith("\ufeff"):
        raise InvalidOwnedTextError("GigAI-owned text cannot begin with a UTF-8 BOM")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.rstrip("\n") + "\n"
    try:
        return normalized.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise InvalidOwnedTextError("GigAI-owned text contains an invalid Unicode surrogate") from exc


def digest_owned_text(text: str) -> str:
    """Digest text after the owned canonical text transformation, as ``sha256:<hex>``."""
    return _sha256_digest(canonicalize_owned_text(text))


def digest_imported_bytes(data: bytes) -> str:
    """Digest imported user bytes exactly as read, without normalization, as ``sha256:<hex>``.

    Raises:
        ExactBytesRequiredError: ``data`` is not ``bytes``.
    """
    if type(data) is not bytes:
        raise ExactBytesRequiredError("imported content must be bytes; text encoding is not implicit")
    return _sha256_digest(data)
