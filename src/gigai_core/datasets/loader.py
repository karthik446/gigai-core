"""Dataset loaders as a Protocol instead of ``ModuleType`` attributes read by name.

Each public set has one loader module (``contractnli``, ``ragtruth``, ``quality``, ``ifeval``, ``jsonschemabench``,
``rewardbench2``, ``judgebench``, ``mtbench_human``, ``tinygsm8k``, ``tinymmlu``) exposing ``LOADER``: the pinned
upstream files, the per-suite counts and recipes, ``items(raw)`` (every upstream item as a ``DatasetItem``) and
``select(pool, suite, rng)`` (the suite's subset, drawn with ``Random(SEED)``).

Six sets ship their ``ci`` slice inside the wheel under ``data/`` (``load_slice``); the other four are download-only
for licence reasons (``NOTICES.md``) and are built on the user's machine with ``build_release``, which fetches the
upstream files once and checks them against their pinned sha256. Parquet files are read with ``pyarrow`` (the
``parquet`` extra), imported only when a build reads one, so importing this package never loads it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib import import_module
from importlib.resources import as_file, files
from pathlib import Path
from random import Random
from typing import Literal, Protocol, TypeVar, cast

from ..canonical import sha256_file_hex
from ..dataset import Dataset, DatasetItem, DatasetRef, SliceRecipe, items_sha256
from ..models import CoreModel
from ..values import JsonValue

Suite = Literal["ci", "release", "upgrade"]
"""A dataset slice size ("how many items"), used only by the loaders. It is not a run field and not a gig concept."""

SEED = 20261109
#: Longest prompt (or sum of scripted user turns) an item may carry, in characters.
MAX_PROMPT_CHARS = 60_000
#: The sets with a loader module, in the order the docs list them.
SETS = (
    "contractnli",
    "ragtruth",
    "quality",
    "ifeval",
    "jsonschemabench",
    "rewardbench2",
    "judgebench",
    "mtbench_human",
    "tinygsm8k",
    "tinymmlu",
)
#: The sets whose ``ci`` slice ships inside the wheel (design decision D13; the others are download-only).
SHIPPED = (
    "tinygsm8k",
    "tinymmlu",
    "ifeval",
    "jsonschemabench",
    "contractnli",
    "ragtruth",
)
#: Appended to every ``ci`` recipe: a shipped slice holds no e-mail address or home-folder path.
CI_FILTER = (
    " Before sampling, items whose text holds an e-mail address outside example.com/example.test or a macOS"
    " home-folder path are dropped."
)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)")
_SAFE_EMAIL_DOMAINS = {"example.test", "example.com"}
_PACKAGE = "gigai_core.datasets"

T = TypeVar("T")
R = TypeVar("R", bound=CoreModel)


class SourceFile(CoreModel):
    """One upstream file: where it comes from and the sha256 it must have."""

    name: str
    url: str
    sha256: str


class SliceCount(CoreModel):
    """How many items a suite takes: a flat count, or a count per stratum (``None`` means all of that stratum)."""

    total: int | None = None
    per_stratum: Mapping[str, int | None] | None = None


class DatasetLoader(Protocol):
    """One public set: pinned sources, per-suite counts and recipes, and the two steps that build a slice."""

    name: str
    spdx: str
    sources: Sequence[SourceFile]
    counts: Mapping[Suite, SliceCount]
    recipes: Mapping[Suite, str]

    def items(self, raw: Path) -> Sequence[DatasetItem]:
        """Every upstream item, read from the files named in ``sources`` under ``raw``."""
        ...

    def select(self, pool: Sequence[DatasetItem], suite: Suite, rng: Random) -> Sequence[DatasetItem]:
        """The ``suite`` subset of ``pool``, drawn with ``rng``."""
        ...


class SetLoader:
    """A ``DatasetLoader`` assembled from a loader module's constants and its ``items`` and ``select`` functions."""

    __slots__ = ("_items", "_select", "counts", "name", "recipes", "sources", "spdx")

    def __init__(
        self,
        *,
        name: str,
        spdx: str,
        sources: Sequence[SourceFile],
        counts: Mapping[Suite, SliceCount],
        recipes: Mapping[Suite, str],
        items: Callable[[Path], Sequence[DatasetItem]],
        select: Callable[[Sequence[DatasetItem], Suite, Random], Sequence[DatasetItem]],
    ) -> None:
        """Keep the module's data and its two functions."""
        self.name = name
        self.spdx = spdx
        self.sources = sources
        self.counts = counts
        self.recipes = recipes
        self._items = items
        self._select = select

    def items(self, raw: Path) -> Sequence[DatasetItem]:
        """Every upstream item, read from the files named in ``sources`` under ``raw``."""
        return self._items(raw)

    def select(self, pool: Sequence[DatasetItem], suite: Suite, rng: Random) -> Sequence[DatasetItem]:
        """The ``suite`` subset of ``pool``, drawn with ``rng``."""
        return self._select(pool, suite, rng)


class DatasetDownloadError(Exception):
    """An upstream file could not be fetched or does not match its pinned sha256.

    ``code`` is ``download_failed`` or ``sha256_mismatch``; the message names the URL and the path to fill by hand.
    """

    def __init__(self, code: Literal["download_failed", "sha256_mismatch"], message: str) -> None:
        """Keep the code beside the message."""
        super().__init__(message)
        self.code = code


class SliceNotShipped(Exception):
    """The slice is not inside the wheel; ``build_release`` downloads the upstream files and builds it."""

    def __init__(self, name: str, suite: Suite) -> None:
        """Name the set and suite asked for and the route to a slice."""
        super().__init__(
            f"{name}: the {suite} slice is not shipped in the wheel (licence terms, see NOTICES.md); build it with "
            f"build_release({name!r}, <directory>), which downloads the upstream files pinned in LOADER.sources"
        )
        self.name = name
        self.suite = suite


class ParquetUnavailable(Exception):
    """``pyarrow`` is not installed: a parquet-backed set needs the ``parquet`` extra (``gigai-core[parquet]``)."""


#: The loaders registered by the ten modules, by set name; filled when the package is imported.
LOADERS: dict[str, DatasetLoader] = {}


def register(loader: DatasetLoader) -> DatasetLoader:
    """Put ``loader`` in ``LOADERS`` under its name and give it back (each loader module calls it once)."""
    if loader.name not in SETS:
        raise ValueError(f"unknown dataset {loader.name!r}; known: {', '.join(SETS)}")
    LOADERS[loader.name] = loader
    return loader


def _loader(name: str) -> DatasetLoader:
    try:
        return LOADERS[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; known: {', '.join(SETS)}") from None


def flat(count: SliceCount) -> int:
    """The flat total of ``count``; a loader whose counts are not stratified reads it."""
    if count.total is None:
        raise ValueError("this suite's count is stratified, not flat")
    return count.total


def strata(count: SliceCount) -> Mapping[str, int | None]:
    """The per-stratum counts of ``count``; a loader whose counts are stratified reads it."""
    if count.per_stratum is None:
        raise ValueError("this suite's count is flat, not stratified")
    return count.per_stratum


def length_band(chars: int) -> str:
    """The prompt-length band of a document of ``chars`` characters (the RC1 stratification bands)."""
    if chars < 8_000:
        return "<8k"
    if chars < 24_000:
        return "8-24k"
    return "24-60k" if chars < MAX_PROMPT_CHARS else ">60k"


def stratified(
    values: Iterable[T],
    key: Callable[[T], str],
    counts: Mapping[str, int | None],
    rng: Random,
) -> list[T]:
    """``counts[k]`` values drawn without replacement from those whose ``key`` is ``k`` (``None``: all of them).

    Groups are drawn in ``counts`` order, keeping ``values`` order inside a group, so callers sort first.
    """
    groups: dict[str, list[T]] = {}
    for value in values:
        groups.setdefault(key(value), []).append(value)
    picked: list[T] = []
    for name, count in counts.items():
        pool = groups.get(name, [])
        if count is not None and count > len(pool):
            raise ValueError(f"recipe draws {count} from {name!r}, which has {len(pool)}")
        picked += pool if count is None else rng.sample(pool, count)
    return picked


def publishable(item: DatasetItem) -> bool:
    """Whether ``item`` may sit in a shipped slice: no ``/Users/`` path, no e-mail outside example domains."""
    text = item.model_dump_json()
    if "/Users/" in text:
        return False
    return all(match.group(1).lower() in _SAFE_EMAIL_DOMAINS for match in _EMAIL.finditer(text))


def typed_rows(rows: Iterable[Mapping[str, JsonValue]], model: type[R]) -> list[R]:
    """Each row validated into ``model`` from its declared fields only (upstream files carry columns no loader reads)."""
    return [model.model_validate({key: row[key] for key in model.model_fields if key in row}) for row in rows]


class _Table(Protocol):
    def to_pylist(self) -> list[dict[str, JsonValue]]: ...


class _ParquetModule(Protocol):
    def read_table(self, source: Path) -> _Table: ...


def parquet_rows(path: Path) -> Sequence[Mapping[str, JsonValue]]:
    """The rows of the parquet file at ``path``; raises ``ParquetUnavailable`` without ``pyarrow``."""
    try:
        module = import_module("pyarrow.parquet")  # here, not at the top: importing pyarrow costs about 1 s cold
    except ModuleNotFoundError as error:
        raise ParquetUnavailable(
            f"{path.name} is a parquet file; install the parquet extra (gigai-core[parquet])"
        ) from error
    return cast(_ParquetModule, module).read_table(path).to_pylist()


def _text_length(item: DatasetItem) -> int:
    return sum(len(turn.content) for turn in item.user_turns())


def _ref(loader: DatasetLoader, suite: Suite, count: int, sha256: str) -> DatasetRef:
    recipe = loader.recipes[suite] + (CI_FILTER if suite == "ci" else "")
    return DatasetRef(
        id=loader.name,
        version="1",
        sha256=sha256,
        slice_recipe=SliceRecipe(recipe=recipe, count=count),
        seed=SEED,
    )


def build_slice(loader: DatasetLoader, raw: Path, suite: Suite) -> Dataset:
    """The ``suite`` slice of ``loader``'s set, drawn from the upstream files in ``raw``, items sorted by ``id``.

    Every item carries the loader's SPDX as ``licence`` and ``suite`` as ``slice_name``; the reference carries the
    recipe (plus ``CI_FILTER`` for ``ci``) and ``SEED``. A prompt over ``MAX_PROMPT_CHARS`` is a loader error.
    """
    pool = list(loader.items(raw))
    for item in pool:
        if _text_length(item) > MAX_PROMPT_CHARS:
            raise ValueError(f"{item.id}: prompt is {_text_length(item)} chars, over {MAX_PROMPT_CHARS}")
    if suite == "ci":
        pool = [item for item in pool if publishable(item)]
    chosen = loader.select(pool, suite, Random(SEED))
    stamped = (item.model_copy(update={"slice_name": suite, "licence": loader.spdx}) for item in chosen)
    items = tuple(sorted(stamped, key=lambda item: item.id))
    return Dataset(ref=_ref(loader, suite, len(items), items_sha256(items)), items=items)


def write_slice(dataset: Dataset, path: Path) -> str:
    """Write ``dataset`` to ``path`` (one ``model_dump_json(exclude_defaults=True)`` per line) and its ``.sha256``.

    The sidecar holds the bare-hex ``items_sha256`` digest, which is also the digest of the file's bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = b"".join(item.model_dump_json(exclude_defaults=True).encode("utf-8") + b"\n" for item in dataset.items)
    path.write_bytes(lines)
    path.with_suffix(".sha256").write_text(f"{dataset.ref.sha256}\n", encoding="utf-8")
    return dataset.ref.sha256


def load_slice(name: str, suite: Suite = "ci") -> Dataset:
    """The shipped ``suite`` slice of ``name``, read from the wheel's ``data/`` and checked against its ``.sha256``.

    Raises ``SliceNotShipped`` for a set or suite that is not in the wheel (only the ``ci`` slice of the
    ``SHIPPED`` sets is), ``DatasetDigestMismatch`` for a tampered file and ``DatasetFileError`` for a bad line.
    """
    loader = _loader(name)
    if suite != "ci" or name not in SHIPPED:
        raise SliceNotShipped(name, suite)
    data = files(_PACKAGE) / "data"
    digest = (data / f"{name}.sha256").read_text(encoding="utf-8").strip()
    with as_file(data / f"{name}.jsonl") as path:
        dataset = Dataset.from_jsonl(path, expect_sha256=digest)
    return Dataset(ref=_ref(loader, suite, len(dataset.items), digest), items=dataset.items)


def _fetch(url: str, target: Path, timeout: float) -> None:
    from urllib.request import urlopen

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.part")
    with urlopen(url, timeout=timeout) as response, partial.open("wb") as handle:
        for block in iter(lambda: response.read(1 << 20), b""):
            handle.write(block)
    partial.replace(target)


def download(loader: DatasetLoader, raw: Path, *, timeout: float = 60.0) -> None:
    """Fetch every absent upstream file of ``loader`` into ``raw`` and check each against its pinned sha256.

    Raises ``DatasetDownloadError`` (``download_failed`` for no network, a timeout or a refused connection;
    ``sha256_mismatch`` for bytes that differ from the pin), naming the URL and the path to fill by hand.
    """
    for source in loader.sources:
        target = raw / source.name
        hint = f"download {source.url} to {target} (sha256 {source.sha256}) and run again"
        if not target.is_file():
            try:
                _fetch(source.url, target, timeout)
            except OSError as error:  # URLError, timeouts and refused connections are all OSError
                message = f"{loader.name}: cannot download {source.name} ({error}); {hint}"
                raise DatasetDownloadError("download_failed", message) from error
        if sha256_file_hex(target) != source.sha256:
            message = f"{loader.name}: {target} does not match the pinned sha256; {hint}"
            raise DatasetDownloadError("sha256_mismatch", message)


def build_release(name: str, directory: Path, *, suite: Suite = "release", timeout: float = 60.0) -> Path:
    """Build ``<directory>/<name>.jsonl`` (plus ``.sha256``) from the upstream files and return its path.

    The upstream files live in ``<directory>/raw/<name>/``; an absent one is downloaded once (``download``).
    """
    loader = _loader(name)
    raw = directory / "raw" / name
    download(loader, raw, timeout=timeout)
    out = directory / f"{name}.jsonl"
    write_slice(build_slice(loader, raw, suite), out)
    return out
