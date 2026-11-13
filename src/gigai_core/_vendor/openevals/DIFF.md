# openevals judge prompts, vendored unchanged

- Upstream: https://github.com/langchain-ai/openevals at commit `5753dc75d40c2be75ffd38eb6d63bb966cbcc4e3` (HEAD of the default branch on 2026-11-11, pinned by `git ls-remote`), files fetched from `https://raw.githubusercontent.com/langchain-ai/openevals/5753dc75d40c2be75ffd38eb6d63bb966cbcc4e3/<upstream path>` on 2026-11-11. At this commit the prompts live under `python/openevals/prompts/quality/` (the flat `prompts/hallucination.py` path of older releases is gone).
- Licence: MIT, "Copyright (c) 2025 LangChain, Inc."; the upstream `LICENSE` is copied unchanged beside the files.
- Both prompt modules are copied byte for byte (upstream sha256 = vendored sha256); each is one `str` constant (`HALLUCINATION_PROMPT`, `CORRECTNESS_PROMPT`) with `str.format` placeholders. `__init__.py` files are gigai-written.
- What core uses: `openevals_judge` (`gigai_core.graders.lifted`) renders them with `str.format_map` (`{inputs}`, `{outputs}`, `{reference_outputs}`, and `{context}` for hallucination), appends a request to end the reply with one JSON object `{"verdict": "pass" or "fail", "explanation": "<one sentence>"}`, and reads the last JSON object of the reply as `JudgeVerdict` (`pass` 1.0, `fail` 0.0; anything else `CannotGrade`). Upstream feeds them to a LangChain chat model with a structured-output score; core makes no LangChain call.
- `tests/test_judge_validation.py::test_diff_md_records_the_vendored_hashes` recomputes the vendored column below. Change a vendored file and you must update it.

## Files

| vendored file | upstream path | upstream sha256 | vendored sha256 |
|---|---|---|---|
| `prompts/hallucination.py` | `python/openevals/prompts/quality/hallucination.py` | `a69771e2cbdb0931f08f001756c7387f518195acccbcf30ccbe0b67f460abbf8` | `a69771e2cbdb0931f08f001756c7387f518195acccbcf30ccbe0b67f460abbf8` |
| `prompts/correctness.py` | `python/openevals/prompts/quality/correctness.py` | `e5146d7daa79bbf2e590dbd908b2c512f0d564e144cae3eae40a3bdd54d150be` | `e5146d7daa79bbf2e590dbd908b2c512f0d564e144cae3eae40a3bdd54d150be` |
| `LICENSE` | `LICENSE` | `ff1c7be3c5e3be6ae6ee8956ed7ee7c47c731c60e65540b332c108026713b30f` | `ff1c7be3c5e3be6ae6ee8956ed7ee7c47c731c60e65540b332c108026713b30f` |
| `__init__.py` | - (gigai-written) | - | `6f58dcbe6945475e09e1c3cc20215598b5cf6e4f0666ac602beb6e7bc0d65b31` |
| `prompts/__init__.py` | - (gigai-written) | - | `12fdcb756b40889a2c5ea23cd9225821af7a62a246b0f4d5d1f136844924f146` |

## What changed

Nothing in the files.
