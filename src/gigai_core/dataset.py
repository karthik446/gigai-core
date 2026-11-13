"""A dataset item: what a trial is asked and what counts as right."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .canonical import sha256_hex
from .models import CoreModel
from .providers import Turn
from .values import JsonValue


class DatasetDigestMismatch(Exception):
    """The file's items do not hash to the digest the caller expected."""


class SliceRecipe(CoreModel):
    """How a slice was cut from its dataset."""

    recipe: str
    count: int
    filter: str | None = None
    stratification: Mapping[str, int] | None = None


class DatasetRef(CoreModel):
    """The dataset slice a run measured: identity, version and the digest the run verifies."""

    id: str
    version: str
    sha256: str
    slice_recipe: SliceRecipe | None = None
    seed: int | None = None


class DatasetItem(CoreModel):
    """One item.

    Single-turn: ``prompt``. Conversation: ``turns`` (scripted user turns) and ``prompt`` empty. Items sharing
    ``cluster_id`` are not independent; compare and the Wilson interval cluster on it.
    """

    id: str
    cluster_id: str
    expected: JsonValue
    prompt: str = ""
    turns: tuple[Turn, ...] = ()
    system: str | None = None
    source_ids: Mapping[str, str | int] = {}
    tags: tuple[str, ...] = ()
    slice_name: str | None = None
    licence: str | None = None
    version: str = "1"

    def user_turns(self) -> tuple[Turn, ...]:
        """The scripted user turns: ``turns``, or one turn made of ``prompt``."""
        if self.turns:
            return self.turns
        return (Turn(role="user", content=self.prompt),)


def items_sha256(items: Sequence[DatasetItem]) -> str:
    """The digest ``Dataset.of`` records: bare-hex SHA-256 of the items as JSON lines.

    Each line is the item's ``model_dump_json`` with fields left at their default omitted.

    Omitting defaults keeps a saved digest valid when a later core version adds an optional field to ``DatasetItem``.
    It is the digest of the bytes a line-per-item file written that way holds, so a slice written and read back hashes
    the same whether it came from a file or from memory.
    """
    lines = b"".join(item.model_dump_json(exclude_defaults=True).encode("utf-8") + b"\n" for item in items)
    return sha256_hex(lines)


class Dataset(CoreModel):
    """A set of items plus the reference a run records."""

    ref: DatasetRef
    items: tuple[DatasetItem, ...]

    @classmethod
    def of(cls, id: str, items: Sequence[DatasetItem], *, version: str = "1") -> Dataset:
        """Digest the items and build the reference."""
        kept = tuple(items)
        ref = DatasetRef(id=id, version=version, sha256=items_sha256(kept))
        return cls(ref=ref, items=kept)
