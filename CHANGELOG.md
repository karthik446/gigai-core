# Changelog

Each release is one `## <version> - <date>` block; `release.yml` slices the block of the tagged version into the
GitHub release notes.

## 0.1.0 - 2026-11-13

First release. The library half of GigAI: the evaluation records, graders, statistics and run files that gigai
0.3.2 kept inside its CLI, moved into a package with one runtime dependency.

### What ships

- `gigai_core.models`: `CoreModel`, the pydantic base of every record (`strict=True`, `frozen=True`,
  `extra="forbid"`, `allow_inf_nan=False`; keyword-only construction). Files are read with `model_validate_json`
  / `TypeAdapter.validate_json` and written with `model_dump_json`; a bad line raises an error that names the file
  and the line (`RunFileError`, `TraceSchemaError`, `LabelFileError`).
- `gigai_core.canonical`: canonical JSON encoding and sha256 digests (`canonical_json_bytes`,
  `canonical_json_digest`, `sha256_hex`, `digest_imported_bytes`, `parse_json_bytes`).
- `gigai_core.providers`: the provider seam. `ModelProvider` (a `Protocol`: `identity`, `capabilities`, `invoke`,
  `close`), `CallSettings`, `ModelRequest`, `ModelResult`, `Turn`, `ToolCall`, `ToolSpec`, `TokenUsage`,
  `ProviderIdentity`, `RetryPolicy`, `request_sha256`, `bounded` (a wall-time cap), and `ProviderError` with
  `AuthenticationRequired`, `UnsupportedRequestField`, `OutputNotJSON`, `CallTimedOut`, `CallCancelled`.
- `gigai_core.budget`: `Budget` (Protocol), `CallBudget` (`max_usd`, `max_calls`, a price table; `plan` refuses a
  run that cannot fit before its first call), `Price`, `Reservation`, `budgeted`, `BudgetExceeded`,
  `CallCapExceeded`.
- `gigai_core.recorded`: `RecordedProvider`, which answers from stored `ModelSpan`s keyed by
  `(model, request_sha256)` and never calls out.
- `gigai_core.trace`: the span trace (`TrialSpan`, `TurnSpan`, `ModelSpan`, `ToolSpan`, `ModelCall`,
  `ModelExchange`, `CallError`), `TraceSink` with `ListTrace` and `JsonlTrace` (one JSONL file per run, schema
  `gigai-core-trace:1`), `read_spans`, `TraceSchemaError`.
- `gigai_core.dataset`: `Dataset`, `DatasetItem`, `DatasetRef`, `SliceRecipe`, `Dataset.of`, `Dataset.from_jsonl`,
  `items_sha256`, `DatasetDigestMismatch`.
- `gigai_core.datasets`: ten loaders (`DatasetLoader`) and `load_slice`; six `ci` slices ship in the wheel
  (`contractnli`, `ifeval`, `jsonschemabench`, `ragtruth`, `tinygsm8k`, `tinymmlu`); `quality`, `rewardbench2`,
  `judgebench` and `mtbench_human` are download-only (`build_release`). Licences and attributions are in
  `src/gigai_core/datasets/NOTICES.md`, shipped in the wheel.
- `gigai_core.sample`, `gigai_core.score`, `gigai_core.records`: `Sample`, `Score`, `ScoreRecord`, `JudgeNote`,
  `RunRecord`, `Prereg`, `LabelRecord`, `LabelValue`, `JudgeValidationRecord`.
- `gigai_core.graders`: one `Grader` Protocol (`id`, `version`, `kind`, `grade(sample, expected) -> Score`), `Judge`
  and `CannotGrade` (for a sample a grader cannot judge), defined in `graders.protocol` and re-exported with every
  constructor from `gigai_core.graders` (40 names). `builtin`: `last_number`, `answer_letter`, `json_valid`,
  `schema_valid`, `label_share`, `bool_field`, `ifeval`, `ranks_first`, `verdict_equals`, `constant`,
  `with_wrong_labels`, `parse_json_text`. `lifted` (the vendored inspect_ai, autoevals and openevals scorers):
  `includes`, `match`, `pattern`, `model_graded_qa`, `autoevals_judge`, `openevals_judge`. `judge`: `llm_judge`,
  `pairwise_judge`, the vendored autoevals templates (`load_template`, `render_template`, `swap_blocks`),
  `JudgeVerdict`. `labels`: `read_labels`, `write_labels`, `labelled`, `LabelFileError`.
- `gigai_core.run`: `run(dataset, provider, graders, *, settings, budget, ...)` returns a typed `Run`
  (`values`, `trace`, `ungraded`, `contested`); `save_run` / `open_run` (one JSONL file, schema `gigai-core-run:1`,
  `RunFileError` with the line number).
- `gigai_core.stats`: Wilson and cluster-bootstrap intervals, exact McNemar, Wilcoxon signed-rank, Holm and
  Benjamini-Hochberg corrections, TOST, Cohen's kappa, Krippendorff's alpha, AUC, minimum detectable difference,
  variance components, pass@k.
- `gigai_core.comparison`: `compare(a, b, metrics)` pairs two runs by item and reports per-metric differences with
  intervals and corrected tests (`Comparison`, `MetricComparison`, `ComparisonRefused`).
- `gigai_core.gate`: `gate(candidate, baseline, rules)` with `MinLowerBound`, `NoWorseThan`, `HeldOutShare`;
  `GateResult` (fields `status`, `verdicts`, `reason`; properties `passed`, `failed`, `exit_code`).
- `gigai_core.agreement`: `judge_agreement(run, grader_id, labels)` and `agreement(pairs)` with kappa, swap
  consistency and the validation thresholds (`MIN_KAPPA`, `MIN_UNITS`, `MIN_SWAP`).
- `gigai_core.__init__`: the façade, 41 names in `__all__`: `run`, `save_run`, `open_run`, `Run`, `Dataset`,
  `DatasetItem`, `CallSettings`, `ModelProvider`, `RecordedProvider`, `CallBudget`, `JsonlTrace`, `read_spans`,
  `compare`, `gate`, `MinLowerBound`, `NoWorseThan`, `cohen_kappa`, `judge_agreement`, `CoreModel`, `read_labels`,
  `write_labels`, and the 20 grader constructors (the eleven `builtin` ones, the six `lifted` ones, `llm_judge`,
  `pairwise_judge`, `labelled`); plus `__version__` from the package metadata.

### Not in 0.1.0

- The agent loop, tools and environments (`TurnLimits`, `holds_every_turn`, `state_equals`): 0.2.0.
- The sqlite ledger, the CLI and the gig layer: they stay in gigai.
- `RunRecord.prices` is always empty in 0.1.0: the budget exposes no price table for `run` to copy.

### Dependencies and platforms

- Required: `pydantic>=2.10,<3`. Nothing else at import time: a plain `pip install gigai-core` loads none of
  `click`, `jsonschema`, `pyarrow`, `numpy`, `yaml`, `sqlite3`, `inspect_ai` or `gigai`.
- Extras: `schema` (`jsonschema>=4.23`, needed by `graders.builtin.schema_valid`), `parquet` (`pyarrow>=18`, needed
  by `datasets.build_release` for parquet-backed sets), `inspect` (`inspect-ai==0.3.288`,
  `agent-client-protocol<1`).
- Python 3.11, 3.12 and 3.13; Linux and macOS.
- Licence `Apache-2.0 AND MIT`: the package is Apache-2.0; the vendored scorers and templates under
  `gigai_core._vendor` (autoevals, inspect_ai, openevals) are MIT, each with its own `LICENSE` and `DIFF.md`.
