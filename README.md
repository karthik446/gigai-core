# gigai-core

The library half of GigAI: typed evaluation records, a provider seam, a local span trace, graders, paired statistics,
a release gate and one run file format. Every record is a pydantic model (strict, frozen, `extra="forbid"`), every
file is JSON Lines read through `model_validate_json` and written through `model_dump_json`, and the only runtime
dependency is `pydantic`. There is no CLI, no server and no database: `run()` takes a dataset, a provider and graders
and returns a `Run` value. Import name `gigai_core`.

## Install

```text
pip install gigai-core                 # pydantic only
pip install 'gigai-core[schema]'       # + jsonschema, for graders.builtin.schema_valid
pip install 'gigai-core[parquet]'      # + pyarrow, for datasets.build_release on parquet-backed sets
pip install 'gigai-core[inspect]'      # + inspect-ai, agent-client-protocol
```

Python 3.11, 3.12 or 3.13. `schema_valid` without the extra raises
`ModuleNotFoundError: schema_valid needs the schema extra: pip install 'gigai-core[schema]'`; a parquet build without
`pyarrow` raises `datasets.ParquetUnavailable`.

## Quickstart

The examples below run in order as one script and need no API key: the provider answers from a list. A real
provider implements the same four members of `gigai_core.providers.ModelProvider` (`identity`, `capabilities`,
`invoke`, `close`); gigai binds its adapters to it.

```python
import threading
from datetime import UTC, datetime

from gigai_core.providers import Capability, ModelRequest, ModelResult, ProviderIdentity, TokenUsage

NOW = datetime(2026, 1, 1, tzinfo=UTC)


class Scripted:
    """A ModelProvider that answers from a list, in order."""

    identity = ProviderIdentity(provider="scripted", endpoint="memory", adapter="readme")
    capabilities: frozenset[Capability] = frozenset({"text", "system", "messages"})

    def __init__(self, replies: list[str]) -> None:
        self._replies = iter(replies)

    def invoke(self, request: ModelRequest, *, cancel: threading.Event | None = None) -> ModelResult:
        usage = TokenUsage(input_tokens=None, output_tokens=None, total_tokens=None)
        return ModelResult(text=next(self._replies), finish_reason="stop", resolved_model=request.settings.model,
                           usage=usage, cost_status="unavailable", cost_usd=None, latency_ms=0,
                           started_at=NOW, completed_at=NOW)

    def close(self) -> None:
        pass
```

Ask one item, grade the answer with the lifted inspect_ai `includes` scorer, and read the run:

```python
from gigai_core import CallBudget, CallSettings, Dataset, DatasetItem, includes, run
from gigai_core.budget import Price

dataset = Dataset.of("capitals", (DatasetItem(id="fr", cluster_id="fr", prompt="Capital of France?", expected="Paris"),))
price = Price(model="some/model", prompt_usd_per_token=1e-6, completion_usd_per_token=1e-6, fetched_at=NOW, source="readme")
budget = CallBudget(max_usd=0.05, max_calls=20, prices={"some/model": price})
result = run(dataset, Scripted(["Paris, of course."]), [includes()], settings=CallSettings(model="some/model"), budget=budget)
print(result.record.spent_usd, result.values("includes"))  # 0.0 {'fr': 1.0}
```

`budget.plan` runs over every planned request before the first call, so a run that cannot fit its cap never
starts; a money cap needs a price for the model (nothing is guessed).

### The trace

Every model call (and, in 0.2.0, every tool call) is a span. With `trace_dir` the spans of a run go to one JSONL
file, `<trace_dir>/<run_id>.jsonl`; without it they stay in memory on the samples.

```python
from pathlib import Path

settings = CallSettings(model="some/model")
result = run(dataset, Scripted(["Paris"]), [includes()], settings=settings, budget=CallBudget(max_calls=20), trace_dir=Path("traces"))
for span in result.trace("fr"):
    print(span.turn, span.kind, span.started_at)  # 0 trial ..., 0 model_call ...
assert result.record.trace_path == Path("traces") / f"{result.record.run_id}.jsonl"
```

### Paired comparison and the gate

Two runs over the same items are compared item by item (per-cluster bootstrap intervals, exact McNemar for 0/1
metrics, Holm correction across metrics). A gate turns that into a verdict: `MinLowerBound` asks for a floor on the
candidate's interval, `NoWorseThan` for a bounded regression against the baseline.

```python
from gigai_core import MinLowerBound, NoWorseThan, compare, gate, open_run, save_run

items = tuple(DatasetItem(id=f"c{n}", cluster_id=f"c{n}", prompt=f"Capital of country {n}?", expected="Paris") for n in range(20))
twenty = Dataset.of("twenty-capitals", items)
for name, replies in (("gem", ["Paris"] * 20), ("min", ["Paris"] * 18 + ["Lyon"] * 2)):
    saved = run(twenty, Scripted(replies), [includes()], settings=settings, budget=CallBudget())
    save_run(saved, Path(f"{name}.jsonl"), inputs_by_reference=False)

a, b = open_run(Path("gem.jsonl")), open_run(Path("min.jsonl"))
comparison = compare(a, b, ["includes"])
verdict = gate(candidate=b, baseline=a, rules=[MinLowerBound(metric="includes", floor=0.6), NoWorseThan(metric="includes", margin=0.05)])
print(round(comparison.metrics[0].difference, 2), verdict.status, verdict.exit_code)  # -0.1 fail 1
assert not verdict.passed and verdict.failed[0].rule == NoWorseThan(metric="includes", margin=0.05)
```

`open_run` raises `RunFileError` with the file and line number on a bad line; `save_run` is the only write path and
refuses to overwrite.

## Modules

`from gigai_core import ...` is the façade (`gigai_core.__all__`, 41 names): `run`, `save_run`, `open_run`, `Run`,
`Dataset`, `DatasetItem`, `CallSettings`, `ModelProvider`, `RecordedProvider`, `CallBudget`, `JsonlTrace`,
`read_spans`, `compare`, `gate`, `MinLowerBound`, `NoWorseThan`, `cohen_kappa`, `judge_agreement`, `CoreModel`,
`read_labels`, `write_labels` and the 20 grader constructors. Everything else is imported from its module:

- `gigai_core.models`: `CoreModel`, the strict frozen base of every record.
- `gigai_core.values`: the shared literals (`ParseStatus`, `Outcome`, `Direction`).
- `gigai_core.canonical`: canonical JSON bytes and sha256 digests.
- `gigai_core.providers`: the seam (`ModelProvider`, `CallSettings`, `ModelRequest`, `ModelResult`, `ProviderError`).
- `gigai_core.budget`: `CallBudget`, `Price`, `budgeted`, `BudgetExceeded`.
- `gigai_core.trace`: the span records, `JsonlTrace`, `ListTrace`, `read_spans`.
- `gigai_core.dataset`: `Dataset`, `DatasetItem`, `Dataset.of`, `Dataset.from_jsonl`.
- `gigai_core.datasets`: the ten loaders, `load_slice`, `build_release`, the six shipped slices and `NOTICES.md`.
- `gigai_core.sample`, `gigai_core.score`, `gigai_core.records`: `Sample`, `Score`, `ScoreRecord`, `RunRecord`, `Prereg`, `LabelRecord`.
- `gigai_core.recorded`: `RecordedProvider`, replay from stored spans.
- `gigai_core.graders`: the `Grader` Protocol, `Judge`, `CannotGrade` (defined in `graders.protocol`) and every constructor, re-exported; submodules `builtin`, `lifted`, `judge` (`llm_judge`, `pairwise_judge`), `labels`.
- `gigai_core.run`: `run`, `Run`, `save_run`, `open_run`.
- `gigai_core.stats`: intervals, tests, corrections, kappa, alpha, pass@k, power.
- `gigai_core.comparison`, `gigai_core.gate`, `gigai_core.agreement`: `compare`, `gate`, `judge_agreement`.

## Layering

Imports only go downward, and `tests/test_layers.py` fails the build on a violation: `models`, `values` and
`canonical` import nothing from the package; `providers` sits above them; `budget`, `trace`, `dataset` and `stats`
above `providers` (`stats` imports only `models` and `values`); then `sample`, `score`, `records`, `datasets`
(`dataset` only) and `recorded` (`providers` and `trace` only); then `graders`; then `run`; then `comparison`,
`agreement` and `gate`. No module imports `gigai`, and nothing imports `click`, `jsonschema`, `pyarrow`, `numpy`,
`yaml` or `sqlite3` at load time; `jsonschema` and `pyarrow` are imported inside the one function each that needs
them, behind the `schema` and `parquet` extras.

## Typing

The package ships `py.typed` and passes pyright strict (`typeCheckingMode = "strict"`, Python 3.11) with no `Any`
on the public surface; sequence fields are tuples, open maps are `Mapping`, timestamps are aware datetimes, and
models are constructed by keyword.

## Licences and datasets

`Apache-2.0 AND MIT`. The package is Apache-2.0 (`LICENSE`). `gigai_core._vendor` carries scorers and templates
from inspect_ai, autoevals and openevals under their MIT licences, each with its `LICENSE` and a `DIFF.md` of the
changes. The six `ci` dataset slices in the wheel (`contractnli`, `ifeval`, `jsonschemabench`, `ragtruth`,
`tinygsm8k`, `tinymmlu`) and the four download-only sets are listed with their licences and attributions in
`src/gigai_core/datasets/NOTICES.md`, also shipped in the wheel.

## Not in 0.1.0

- The agent loop, tools, environments and the conversation graders (`TurnLimits`, `holds_every_turn`,
  `state_equals`): 0.2.0.
- The sqlite ledger, the CLI, provider adapters and the gig layer: they stay in gigai.
- Windows is not tested.

See `CHANGELOG.md` for what each release holds.
