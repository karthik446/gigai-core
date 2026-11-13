"""A downstream consumer type-checked with `mypy --strict`: proves `py.typed` ships and the API is typed."""

from gigai_core.canonical import canonical_json_digest
from gigai_core.models import CoreModel


class Record(CoreModel):
    name: str


digest: str = canonical_json_digest({"name": Record(name="x").name})
