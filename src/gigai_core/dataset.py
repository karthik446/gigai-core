"""A dataset item: what a trial is asked and what counts as right."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import ValidationError

from .canonical import sha256_hex
from .models import CoreModel
from .providers import Turn
from .values import JsonValue


class DatasetDigestMismatch(Exception):
    """The file's items do not hash to the digest the caller expected."""


class DatasetFileError(Exception):
    """A slice line this version cannot read: a stray key (the retired ``cluster`` alias), a missing field, bad JSON.

    The message is ``<path> line <n>: <reason>``; ``path`` and ``line`` are also attributes.
    """

    def __init__(self, path: Path, line: int, reason: str) -> None:
        """Name the file, the 1-based line and why it could not be read."""
        super().__init__(f"{path} line {line}: {reason}")
        self.path = path
        self.line = line


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

    @classmethod
    def from_jsonl(cls, path: Path, *, expect_sha256: str | None = None) -> Dataset:
        """Read one slice file, one ``DatasetItem.model_validate_json`` per line.

        The dataset id is the file's stem. Raises ``DatasetFileError`` with the 1-based line number for a line that
        does not validate (a stray key such as the retired ``cluster`` alias is an error, not a silent drop) and
        ``DatasetDigestMismatch`` when ``expect_sha256`` differs from ``items_sha256`` of the items read.
        """
        items: list[DatasetItem] = []
        with path.open("rb") as file:
            for number, raw in enumerate(file, start=1):
                line = raw.rstrip(b"\r\n")
                if not line:
                    continue
                try:
                    items.append(DatasetItem.model_validate_json(line))
                except ValidationError as error:
                    raise DatasetFileError(path, number, str(error)) from error
        digest = items_sha256(items)
        if expect_sha256 is not None and digest != expect_sha256:
            raise DatasetDigestMismatch(f"{path}: items hash to {digest}, expected {expect_sha256}")
        return cls(ref=DatasetRef(id=path.stem, version="1", sha256=digest), items=tuple(items))
