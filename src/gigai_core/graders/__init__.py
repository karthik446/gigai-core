"""Graders: the ``Grader`` protocol, the built-in and lifted constructors, the model judges and the label file.

Every constructor returns a ``Grader`` (a model judge returns a ``Judge``); ``run`` takes a sequence of them.
"""

from __future__ import annotations

from .builtin import (
    ParsedJson,
    answer_letter,
    bool_field,
    constant,
    ifeval,
    json_valid,
    label_share,
    last_number,
    parse_json_text,
    ranks_first,
    schema_valid,
    verdict_equals,
    with_wrong_labels,
)
from .judge import (
    JudgeVerdict,
    Template,
    last_json_object,
    llm_judge,
    load_template,
    pairwise_judge,
    render_template,
    swap_blocks,
    template_choice,
)
from .labels import LABELS, LabelFileError, labelled, read_labels, write_labels
from .lifted import (
    LIFTED_VERSION,
    AutoevalsTemplate,
    MatchLocation,
    OpenevalsPrompt,
    autoevals_judge,
    includes,
    match,
    model_graded_qa,
    openevals_judge,
    pattern,
)
from .protocol import CannotGrade, Grader, Judge

__all__ = [
    "AutoevalsTemplate",
    "CannotGrade",
    "Grader",
    "Judge",
    "JudgeVerdict",
    "LABELS",
    "LIFTED_VERSION",
    "LabelFileError",
    "MatchLocation",
    "OpenevalsPrompt",
    "ParsedJson",
    "Template",
    "answer_letter",
    "autoevals_judge",
    "bool_field",
    "constant",
    "ifeval",
    "includes",
    "json_valid",
    "label_share",
    "labelled",
    "last_json_object",
    "last_number",
    "llm_judge",
    "load_template",
    "match",
    "model_graded_qa",
    "openevals_judge",
    "pairwise_judge",
    "parse_json_text",
    "pattern",
    "ranks_first",
    "read_labels",
    "render_template",
    "schema_valid",
    "swap_blocks",
    "template_choice",
    "verdict_equals",
    "with_wrong_labels",
    "write_labels",
]
