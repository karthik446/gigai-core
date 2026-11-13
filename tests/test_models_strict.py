"""One test per row of DESIGN.md's strict-mode table (section 4.1), against tiny `CoreModel` subclasses.

Each test asserts what the table says. Measured on pydantic 2.14.3 and the lowest-direct 2.10.x
resolution; a row the installed pydantic contradicts is fixed to the measured behaviour and listed
in the A1 report under "Deviations from DESIGN.md".
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, cast

import pytest
from pydantic import AwareDatetime, Field, JsonValue, ValidationError

from gigai_core.models import CoreModel


class Count(CoreModel):
    n: int


class Ratio(CoreModel):
    x: float


class Stamped(CoreModel):
    naive: datetime


class Aware(CoreModel):
    at: AwareDatetime


class Located(CoreModel):
    path: Path


class Row(CoreModel):
    cells: tuple[int, ...]


class Tagged(CoreModel):
    tags: Mapping[str, int]


class Kinded(CoreModel):
    kind: Literal["code", "model", "human"]


class Open(CoreModel):
    value: JsonValue


class A(CoreModel):
    kind: Literal["a"]
    x: int


class B(CoreModel):
    kind: Literal["b"]
    y: int


class Holder(CoreModel):
    item: Annotated[A | B, Field(discriminator="kind")]


class Outer(CoreModel):
    inner: Count


def _error_types(info: pytest.ExceptionInfo[ValidationError]) -> list[str]:
    return [error["type"] for error in info.value.errors()]


def test_int_rejects_string_float_and_bool_in_both_modes() -> None:
    assert Count(n=1).n == 1
    for bad in ("1", 1.0, True):
        with pytest.raises(ValidationError):
            Count(n=bad)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            Count.model_validate({"n": bad})
    assert Count.model_validate_json('{"n": 1}').n == 1
    for text in ('{"n": "1"}', '{"n": 1.0}', '{"n": true}'):
        with pytest.raises(ValidationError):
            Count.model_validate_json(text)


def test_float_accepts_int_rejects_string_and_never_nan_or_inf() -> None:
    assert Ratio(x=1).x == 1.0
    assert Ratio.model_validate_json('{"x": 1}').x == 1.0
    with pytest.raises(ValidationError):
        Ratio(x="1.5")  # type: ignore[arg-type]
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError) as info:
            Ratio(x=bad)
        assert _error_types(info) == ["finite_number"]
    for token in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(ValidationError) as info:
            Ratio.model_validate_json(f'{{"x": {token}}}')
        assert _error_types(info) == ["finite_number"]


def test_datetime_rejects_iso_strings_in_python_mode_and_aware_rejects_naive() -> None:
    now = datetime.now(UTC)
    assert Aware(at=now).at == now
    with pytest.raises(ValidationError):
        Stamped(naive="2026-11-13T10:00:00Z")  # type: ignore[arg-type]
    with pytest.raises(ValidationError) as info:
        Aware(at=datetime(2026, 11, 13, 10, 0, 0))
    assert _error_types(info) == ["timezone_aware"]
    parsed = Aware.model_validate_json('{"at": "2026-11-13T10:00:00Z"}')
    assert parsed.at == datetime(2026, 11, 13, 10, 0, 0, tzinfo=UTC)
    with pytest.raises(ValidationError) as info:
        Aware.model_validate_json('{"at": "2026-11-13T10:00:00"}')
    assert _error_types(info) == ["timezone_aware"]


def test_path_rejects_str_in_python_mode_and_accepts_it_from_json() -> None:
    assert Located(path=Path("a/b")).path == Path("a/b")
    with pytest.raises(ValidationError):
        Located(path="a/b")  # type: ignore[arg-type]
    assert Located.model_validate_json('{"path": "a/b"}').path == Path("a/b")
    assert Located.model_validate_json(Located(path=Path("a/b")).model_dump_json()).path == Path("a/b")


def test_tuple_rejects_a_list_in_python_mode_and_accepts_a_json_array() -> None:
    assert Row(cells=(1, 2)).cells == (1, 2)
    with pytest.raises(ValidationError):
        Row(cells=[1, 2])  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Row.model_validate({"cells": [1, 2]})
    assert Row.model_validate_json('{"cells": [1, 2]}').cells == (1, 2)


def test_mapping_keeps_the_dict_and_makes_the_model_unhashable() -> None:
    tagged = Tagged(tags={"a": 1})
    assert type(tagged.tags) is dict
    assert tagged == Tagged(tags={"a": 1})
    assert tagged != Tagged(tags={"a": 2})
    with pytest.raises(TypeError):
        hash(tagged)
    assert hash(Row(cells=(1, 2))) == hash(Row(cells=(1, 2)))
    assert Tagged.model_validate_json('{"tags": {"a": 1}}').tags == {"a": 1}


def test_literal_accepts_the_plain_string_in_both_modes() -> None:
    assert Kinded(kind="code").kind == "code"
    assert Kinded.model_validate_json('{"kind": "model"}').kind == "model"
    with pytest.raises(ValidationError) as info:
        Kinded(kind="judge")  # type: ignore[arg-type]
    assert _error_types(info) == ["literal_error"]


def test_json_value_accepts_nested_values_and_rejects_a_set() -> None:
    nested: JsonValue = {"a": [1, 2.5, "x", None, True, {"b": []}]}
    assert Open(value=nested).value == nested
    assert Open.model_validate_json(Open(value=nested).model_dump_json()).value == nested
    with pytest.raises(ValidationError):
        Open(value={1, 2})  # type: ignore[arg-type]


def test_discriminated_union_reports_the_three_error_types() -> None:
    assert Holder(item=A(kind="a", x=1)).item == A(kind="a", x=1)
    assert Holder.model_validate({"item": {"kind": "b", "y": 2}}).item == B(kind="b", y=2)
    assert Holder.model_validate_json('{"item": {"kind": "a", "x": 1}}').item == A(kind="a", x=1)
    with pytest.raises(ValidationError) as info:
        Holder.model_validate_json('{"item": {"kind": "c", "x": 1}}')
    assert _error_types(info) == ["union_tag_invalid"]
    with pytest.raises(ValidationError) as info:
        Holder.model_validate_json('{"item": {"x": 1}}')
    assert _error_types(info) == ["union_tag_not_found"]
    with pytest.raises(ValidationError) as info:
        Holder.model_validate_json('{"item": {"kind": "a", "x": 1, "z": 2}}')
    assert _error_types(info) == ["extra_forbidden"]


def test_nested_model_accepts_an_instance_or_a_dict_and_a_json_object() -> None:
    assert Outer(inner=Count(n=1)).inner == Count(n=1)
    assert Outer.model_validate({"inner": {"n": 1}}).inner == Count(n=1)
    assert Outer.model_validate_json('{"inner": {"n": 1}}').inner == Count(n=1)


def test_construction_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        cast(Callable[..., object], Count)(1)


def test_json_schema_field_name_raises_no_warning_and_round_trips() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        class Versioned(CoreModel):
            json_schema: str
            schema_id: str

    record = Versioned(json_schema='{"type": "object"}', schema_id="trace/1")
    assert Versioned.model_validate_json(record.model_dump_json()) == record
    with pytest.warns(UserWarning, match="shadows an attribute"):

        class Shadowing(CoreModel):
            schema: str  # type: ignore[assignment]

    assert Shadowing.model_fields.keys() == {"schema"}


def test_assignment_raises_frozen_instance() -> None:
    record = Count(n=1)
    with pytest.raises(ValidationError) as info:
        record.n = 2  # type: ignore[misc]
    assert _error_types(info) == ["frozen_instance"]
    assert record.model_copy(update={"n": 2}).n == 2
    assert record.n == 1


def test_duplicate_json_key_keeps_the_last_value() -> None:
    assert Count.model_validate_json('{"n": 1, "n": 2}').n == 2
