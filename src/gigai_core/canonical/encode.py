"""Restricted-JCS JSON bytes: render and parse.

Core uses a deliberately restricted RFC 8785 JSON domain. Object member names are ASCII identifiers, so Python
code-point ordering and JCS UTF-16 code-unit ordering are identical for every accepted name. Floats are forbidden,
integers are restricted to the interoperable range, and Unicode string values are preserved without normalization.
"""

from __future__ import annotations

import json
import re
from typing import cast

from pydantic import JsonValue

SAFE_INTEGER_MAX = 9_007_199_254_740_991
MEMBER_NAME = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")


class CanonicalizationError(ValueError):
    """Base class for stable canonical-contract failures."""

    code = "canonicalization_error"


class InvalidCanonicalValueError(CanonicalizationError):
    """A value lies outside the restricted canonical JSON domain."""

    code = "invalid_canonical_value"


class DuplicateJsonMemberError(CanonicalizationError):
    """A JSON object repeats a member name."""

    code = "duplicate_json_member"


class InvalidJsonError(CanonicalizationError):
    """Input bytes are not valid UTF-8 JSON."""

    code = "invalid_json"


class InvalidOwnedTextError(CanonicalizationError):
    """Owned text is not a string, contains NUL, a leading BOM or a lone surrogate."""

    code = "invalid_owned_text"


class ExactBytesRequiredError(CanonicalizationError):
    """Bytes were required but text (or another type) was passed."""

    code = "exact_bytes_required"


def _validate_canonical_value(
    value: JsonValue,
    active_containers: set[int] | None = None,
) -> None:
    if value is None or type(value) is bool:
        return
    if type(value) is str:
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise InvalidCanonicalValueError("string contains an invalid Unicode surrogate") from exc
        return
    if type(value) is int:
        if not -SAFE_INTEGER_MAX <= value <= SAFE_INTEGER_MAX:
            raise InvalidCanonicalValueError("integer is outside the JCS interoperable range")
        return
    if type(value) is float:
        raise InvalidCanonicalValueError("identity-bearing GigAI JSON forbids floating-point numbers")

    if type(value) not in (list, dict):
        raise InvalidCanonicalValueError(f"unsupported canonical JSON value: {type(value).__name__}")

    if active_containers is None:
        active_containers = set()
    identity = id(value)
    if identity in active_containers:
        raise InvalidCanonicalValueError("canonical JSON cannot contain cycles")
    active_containers.add(identity)
    try:
        if isinstance(value, list):
            for item in value:
                _validate_canonical_value(item, active_containers)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if type(key) is not str or not MEMBER_NAME.fullmatch(key):
                    raise InvalidCanonicalValueError(f"invalid canonical member name: {key!r}")
                _validate_canonical_value(item, active_containers)
    finally:
        active_containers.remove(identity)


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Render restricted-JCS UTF-8 bytes with no trailing newline.

    Args:
        value: A JSON value inside the restricted domain (no floats, safe integers, identifier member names).

    Returns:
        The canonical UTF-8 bytes: sorted members, no whitespace, non-ASCII preserved.

    Raises:
        InvalidCanonicalValueError: The value lies outside the restricted domain or contains a cycle.
    """
    _validate_canonical_value(value)
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return rendered.encode("utf-8")


def parse_json_bytes(data: bytes) -> JsonValue:
    """Parse UTF-8 JSON while rejecting duplicate object member names.

    Args:
        data: The exact bytes to parse; ``str`` is refused so no implicit encoding happens.

    Returns:
        The parsed value, already checked against the restricted domain.

    Raises:
        ExactBytesRequiredError: ``data`` is not ``bytes``.
        InvalidJsonError: ``data`` is not valid UTF-8 JSON.
        DuplicateJsonMemberError: An object repeats a member name.
        InvalidCanonicalValueError: The parsed value lies outside the restricted domain.
    """
    if type(data) is not bytes:
        raise ExactBytesRequiredError("JSON input must be exact bytes")

    def unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateJsonMemberError(f"duplicate JSON member name: {key!r}")
            result[key] = value
        return result

    try:
        decoded = data.decode("utf-8", errors="strict")
        value = cast(JsonValue, json.loads(decoded, object_pairs_hook=unique_object))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidJsonError("input is not valid UTF-8 JSON") from exc
    _validate_canonical_value(value)
    return value
