# Vendored autoevals LLM-as-judge templates @ a5dd1711 (MIT, Copyright (c) 2023 BrainTrust Data; see LICENSE and DIFF.md)
"""Braintrust autoevals' nine classifier templates (``templates/*.yaml``), copied unchanged.

Each file is ``prompt`` (a mustache-style text with ``{{input}}``, ``{{output}}``, ``{{expected}}`` and template-specific
fields) and ``choice_scores`` (the judge's allowed answers and the score each maps to). Core's judge grader reads
them with a parser for exactly this shape, so neither PyYAML nor chevron is needed.
"""
