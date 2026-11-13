# Vendored inspect_ai statistics @ 9567b431, numpy-free (MIT, Copyright (c) 2024 UK AI Security Institute; see LICENSE and DIFF.md)
"""Inspect's metric statistics, lifted without inspect_ai, pydantic or numpy.

Modules: std (stderr, ci, ci_wilson, bootstrap_stderr, std, var), krippendorff,
accuracy; _metric is the gigai-written shim they import. The ``grouped`` and
``reducer`` modules of the gigai 0.3.x copy are not carried into core.
"""
