"""`DatasetItem` and `Dataset`: strict, closed, digested in memory."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gigai_core.canonical import sha256_hex
from gigai_core.dataset import (
    Dataset,
    DatasetItem,
    DatasetRef,
    SliceRecipe,
    items_sha256,
)
from gigai_core.providers import Turn


def _item(id: str = "fr", licence: str | None = None) -> DatasetItem:
    return DatasetItem(id=id, cluster_id=id, prompt=f"Capital of {id}?", expected="Paris", licence=licence)


def test_an_item_with_a_stray_key_is_rejected_in_both_modes() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DatasetItem.model_validate({"id": "fr", "cluster": "fr", "cluster_id": "fr", "expected": "Paris"})
    with pytest.raises(ValidationError, match="cluster"):
        DatasetItem.model_validate_json('{"id": "fr", "cluster": "fr", "expected": "Paris"}')


def test_cluster_id_is_required_and_the_retired_cluster_alias_does_not_stand_in() -> None:
    with pytest.raises(ValidationError, match="cluster_id"):
        DatasetItem.model_validate_json('{"id": "fr", "expected": "Paris"}')
    with pytest.raises(TypeError):
        DatasetItem("fr", "fr", "Paris")  # type: ignore[misc]  # keyword-only construction


def test_user_turns_is_the_prompt_or_the_scripted_turns() -> None:
    single = _item()
    assert single.user_turns() == (Turn(role="user", content="Capital of fr?"),)
    scripted = DatasetItem(
        id="chat",
        cluster_id="chat",
        expected=None,
        turns=(Turn(role="user", content="hi"), Turn(role="user", content="and then?")),
    )
    assert scripted.prompt == "" and scripted.user_turns() == scripted.turns


def test_item_defaults_and_open_fields_round_trip() -> None:
    item = DatasetItem(
        id="q1",
        cluster_id="sums",
        expected={"answer": 4, "weights": [0.5, 0.5]},
        prompt="2+2",
        system="Answer tersely.",
        source_ids={"row": 7, "file": "a.jsonl"},
        tags=("easy",),
        slice_name="ci",
        licence="CC-BY-4.0",
    )
    assert item.version == "1"
    assert DatasetItem.model_validate_json(item.model_dump_json()) == item
    with pytest.raises(ValidationError, match="tuple"):
        DatasetItem(id="q1", cluster_id="sums", expected=None, tags=["easy"])  # type: ignore[arg-type]  # strict tuple


def test_dataset_of_converts_a_list_to_a_tuple_and_digests_the_items_as_json_lines() -> None:
    items = [_item("fr"), _item("de")]
    dataset = Dataset.of("capitals", items)
    assert isinstance(dataset.items, tuple) and dataset.items == tuple(items)
    expected = sha256_hex(b"".join(item.model_dump_json(exclude_defaults=True).encode() + b"\n" for item in items))
    assert dataset.ref == DatasetRef(id="capitals", version="1", sha256=expected)
    assert len(dataset.ref.sha256) == 64 and dataset.ref.sha256 == items_sha256(items)
    assert Dataset.of("capitals", items, version="2").ref.version == "2"


def test_the_digest_follows_the_items_and_their_order() -> None:
    same = Dataset.of("capitals", (_item("fr"), _item("de")))
    assert same.ref.sha256 == Dataset.of("other-name", [_item("fr"), _item("de")]).ref.sha256
    assert same.ref.sha256 != Dataset.of("capitals", (_item("de"), _item("fr"))).ref.sha256
    assert same.ref.sha256 != Dataset.of("capitals", (_item("fr"), _item("de", licence="MIT"))).ref.sha256
    assert items_sha256(()) == sha256_hex(b"")


def test_a_dataset_and_its_reference_round_trip_through_json() -> None:
    dataset = Dataset.of("capitals", (_item("fr"),))
    ref = dataset.ref.model_copy(
        update={"slice_recipe": SliceRecipe(recipe="stratified", count=1, stratification={"short": 1}), "seed": 3}
    )
    dataset = Dataset(ref=ref, items=dataset.items)
    assert Dataset.model_validate_json(dataset.model_dump_json()) == dataset
    assert DatasetRef.model_validate_json(ref.model_dump_json()) == ref
    with pytest.raises(ValidationError):
        DatasetRef(id="capitals", version="1", sha256="x", seed="3")  # type: ignore[arg-type]  # strict int
