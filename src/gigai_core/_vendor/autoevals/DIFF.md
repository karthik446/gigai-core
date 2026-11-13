# autoevals judge templates, vendored unchanged

- Upstream: https://github.com/braintrustdata/autoevals at commit `a5dd17115e0cf0724f27de8f5937ca710db71416` (HEAD of the default branch on 2026-11-11, pinned by `git ls-remote`), files fetched from `https://raw.githubusercontent.com/braintrustdata/autoevals/a5dd17115e0cf0724f27de8f5937ca710db71416/<upstream path>` on 2026-11-11.
- Licence: MIT, "Copyright (c) 2023 BrainTrust Data"; the upstream `LICENSE` is copied unchanged beside the files.
- Every template is copied byte for byte (upstream sha256 = vendored sha256): no header line, so the YAML parses as upstream ships it. `__init__.py` is gigai-written.
- What core uses: the judge grader (`gigai_core.graders`) reads `prompt` and `choice_scores` with a parser for this exact shape (a `prompt: |-` block scalar followed by a `choice_scores:` map of quoted keys to numbers) and renders `{{name}}` / `{{{name}}}` placeholders by substitution. Upstream renders them with chevron (mustache) and asks the model through function calling; core asks for the bare choice and reads the first line of the reply. `battle.yaml` is the position-swap probe of judge validation: upstream never swaps, core asks it twice with the two responses in both orders.
- `tests/test_judge_validation.py::test_diff_md_records_the_vendored_hashes` recomputes the vendored column below. Change a vendored file and you must update it.

## Files

| vendored file | upstream path | upstream sha256 | vendored sha256 |
|---|---|---|---|
| `templates/battle.yaml` | `templates/battle.yaml` | `9bb7ab5f71ccdb548e74ac75b7d81e1d5c5c5315a2c985e47c7e64b2494790e6` | `9bb7ab5f71ccdb548e74ac75b7d81e1d5c5c5315a2c985e47c7e64b2494790e6` |
| `templates/closed_q_a.yaml` | `templates/closed_q_a.yaml` | `0cf9d814982bac6bc0363bf68b41ccf167d10dfbe4e686df57dc975ef43a2238` | `0cf9d814982bac6bc0363bf68b41ccf167d10dfbe4e686df57dc975ef43a2238` |
| `templates/factuality.yaml` | `templates/factuality.yaml` | `9ab5f2c251f821746c7a9aa76672074cf955e5697ad3791b8bba9dc2d8a87efc` | `9ab5f2c251f821746c7a9aa76672074cf955e5697ad3791b8bba9dc2d8a87efc` |
| `templates/humor.yaml` | `templates/humor.yaml` | `0f964efb400c388eea2bf8521b57c381b9d37055bcb0b9c0c214cd8cd479bdc4` | `0f964efb400c388eea2bf8521b57c381b9d37055bcb0b9c0c214cd8cd479bdc4` |
| `templates/possible.yaml` | `templates/possible.yaml` | `ecc767a8a4d17d21413e191716b1088c53fedf6f8073f43093eb7637fecdb258` | `ecc767a8a4d17d21413e191716b1088c53fedf6f8073f43093eb7637fecdb258` |
| `templates/security.yaml` | `templates/security.yaml` | `010810d7b17c1601eb8c9cc298e7fcef27ef0bb8a2429574b3d7d8bd36b3a2e9` | `010810d7b17c1601eb8c9cc298e7fcef27ef0bb8a2429574b3d7d8bd36b3a2e9` |
| `templates/sql.yaml` | `templates/sql.yaml` | `70b26448b9d56ac75417bda7162395b100e794be0651c300e233d57b98b084fb` | `70b26448b9d56ac75417bda7162395b100e794be0651c300e233d57b98b084fb` |
| `templates/summary.yaml` | `templates/summary.yaml` | `1492467f239afe8075609c8a2a4faf2add7dd5c0955bcf87b3910e5ea9d74219` | `1492467f239afe8075609c8a2a4faf2add7dd5c0955bcf87b3910e5ea9d74219` |
| `templates/translation.yaml` | `templates/translation.yaml` | `dc7658fc5f88b87574f1c58900f6c6f06e1963a6f11cfa0ab5dcb3beee58be87` | `dc7658fc5f88b87574f1c58900f6c6f06e1963a6f11cfa0ab5dcb3beee58be87` |
| `LICENSE` | `LICENSE` | `63acc1a22bf0cd417131aef4e16ba63f71c43b3c578969c7255bc54457ae8661` | `63acc1a22bf0cd417131aef4e16ba63f71c43b3c578969c7255bc54457ae8661` |
| `__init__.py` | - (gigai-written) | - | `2c59434787aa6beab3be1f2d97ce93da4e7c6b7f66349e29a9c825be17e40c78` |

## What changed

Nothing in the files. Behaviour differences come from how core asks: upstream parses the model's choice out of a function call and can add chain-of-thought (`use_cot`); core's judge grader sends the rendered prompt as plain text with the allowed choices listed after it and reads the choice from the reply's first non-empty line. A choice the template does not list is "no answer" and scores as its lowest choice.
