"""The model judges: the vendored templates, the prompt and reply helpers, and every judge constructor driven by a
``ScriptedProvider``. No model is called; the judges' agreement with labels is ``test_agreement.py``."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime

import pytest

from gigai_core.graders import judge, lifted
from gigai_core.graders.protocol import CannotGrade, Judge
from gigai_core.providers import CallSettings, Turn
from gigai_core.sample import Sample
from tests.conftest import SRC
from tests.fakes import ScriptedProvider

VENDOR = SRC / "_vendor"
NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
SETTINGS = CallSettings(model="judge-model")
MARKERS = ("RESPONSE A:", "RESPONSE B:")
PAIR = "Pick the correct one.\n\nQUESTION:\n2 + 2?\n\nRESPONSE A:\n4\n\nRESPONSE B:\n5"


def _sample(output: str, *turns: Turn) -> Sample:
    return Sample(
        trace_id="trace_1",
        run_id="run_1",
        item_id="item_1",
        item_version="1",
        cluster_id="item_1",
        repeat_index=0,
        input=turns or (Turn(role="user", content="What is 2 + 2?"),),
        output_text=output,
        output_parsed=None,
        parse_status="not_parsed",
        outcome="succeeded",
        error=None,
        recorded_at=NOW,
    )


# --- relocated from 0.3.2 tests/behaviors/evals/test_judge_validation.py ----------------------------------------


def test_every_vendored_autoevals_template_parses_and_renders() -> None:
    names = sorted(path.stem for path in (VENDOR / "autoevals" / "templates").glob("*.yaml"))
    assert names == ["battle", "closed_q_a", "factuality", "humor", "possible", "security", "sql", "summary", "translation"]
    for name in names:
        template = judge.load_template(name)
        assert template.choice_scores and all(isinstance(score, float) for score in template.choice_scores.values())
        fields = dict.fromkeys(re.findall(r"\{\{\{?([a-zA-Z_]+)\}?\}\}", template.prompt), "x")
        rendered = judge.render_template(template, **fields)
        assert "{{" not in rendered and rendered.endswith(f"Answer with exactly one of: {', '.join(template.choice_scores)}.")
    battle = judge.load_template("battle")
    assert battle.choice_scores == {"Yes": 1.0, "No": 0.0}
    prompt = judge.render_template(battle, instructions="Q", output="first", expected="second")
    assert "[Response 1]\nfirst\n\n[Instruction 2]\nQ\n[Response 2]\nsecond" in prompt
    assert judge.template_choice(battle, " yes.\nbecause") == "Yes"
    assert judge.template_choice(battle, "Maybe") is None
    with pytest.raises(ValueError, match="no value for expected, output"):
        judge.render_template(battle, instructions="Q")
    with pytest.raises(ValueError, match="no vendored autoevals template"):
        judge.load_template("nope")


def test_openevals_prompts_render_their_placeholders() -> None:
    sample = _sample("A", Turn(role="user", content="Q"))
    text = lifted.openevals_judge("correctness", ScriptedProvider([]), settings=SETTINGS).render(sample, "R")
    assert "<input>\nQ\n</input>" in text and "<reference_outputs>\nR\n</reference_outputs>" in text
    assert text.rstrip().endswith('{"verdict": "pass" or "fail", "explanation": "<one sentence>"}.')
    hallucination = lifted.openevals_judge("hallucination", ScriptedProvider([]), settings=SETTINGS)
    rendered = hallucination.render(sample, {"context": "C", "reference": "R"})
    assert "<context>\nC\n</context>" in rendered and "<reference_outputs>\nR\n</reference_outputs>" in rendered
    with pytest.raises(CannotGrade, match="context"):
        hallucination.render(sample, "R")


def test_swap_blocks_exchanges_the_two_responses() -> None:
    prompt = "ASK\n\nQUESTION:\nq\n\nRESPONSE A:\nfirst\nline\n\nRESPONSE B:\nsecond"
    swapped = judge.swap_blocks(prompt, MARKERS)
    assert swapped == "ASK\n\nQUESTION:\nq\n\nRESPONSE A:\nsecond\n\nRESPONSE B:\nfirst\nline"
    assert judge.swap_blocks(swapped, MARKERS) == prompt
    # a response quoting a marker in its own text does not move the split
    quoting = "ASK\n\nRESPONSE A:\nsee RESPONSE B: below\n\nRESPONSE B:\nreplying to\n\nRESPONSE A:\nabove"
    assert judge.swap_blocks(quoting, MARKERS) == "ASK\n\nRESPONSE A:\nreplying to\n\nRESPONSE A:\nabove\n\nRESPONSE B:\nsee RESPONSE B: below"
    with pytest.raises(ValueError, match="no 'RESPONSE A:'"):
        judge.swap_blocks("nothing here", MARKERS)
    with pytest.raises(ValueError, match="no 'RESPONSE A:'"):
        judge.swap_blocks("RESPONSE A: inline\n\nRESPONSE B:\nb", MARKERS)
    assert judge.last_json_object('sure: {"verdict": "A>B"} done', "verdict") == "A>B"
    assert judge.last_json_object("no json", "verdict") is None


def test_diff_md_records_the_vendored_hashes() -> None:
    for project in ("autoevals", "openevals"):
        folder = VENDOR / project
        table = re.findall(r"^\| `([^`]+)` \| [^|]+ \| [^|]+ \| `([0-9a-f]{64})` \|$", (folder / "DIFF.md").read_text(encoding="utf-8"), re.M)
        assert table, project
        listed = {name for name, _ in table}
        present = {str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file() and path.name != "DIFF.md" and "__pycache__" not in path.parts}
        assert listed == present, project
        for name, digest in table:
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, f"{project}/{name}"


# --- the reply helper ---------------------------------------------------------------------------------------------


def test_last_json_object_takes_the_last_object_and_skips_quoted_ones() -> None:
    reply = 'First I thought {"verdict": "A"} but no.\nFinal: {"verdict": "B", "why": {"nested": true}}'
    assert judge.last_json_object(reply, "verdict") == "B"
    assert judge.last_json_object(reply, "why") == {"nested": True}
    assert judge.last_json_object(reply, "missing") is None
    assert judge.last_json_object('{"a": 1} and then {broken', "a") == 1
    assert judge.last_json_object("[1, 2]", "a") is None


# --- llm_judge ----------------------------------------------------------------------------------------------------


def test_llm_judge_reads_pass_and_fail_and_refuses_anything_else() -> None:
    provider = ScriptedProvider(
        [
            'The answer is right.\n{"verdict": "pass", "explanation": "it says 4"}',
            '{"verdict": "fail"}',
            "I cannot decide.",
            '{"verdict": "maybe"}',
        ]
    )
    graded = judge.llm_judge(provider, rubric="The output gives the right sum.", settings=SETTINGS)
    assert (graded.id, graded.version, graded.kind) == ("llm-judge", "1", "model")
    passed = graded.grade(_sample("4"), "4")
    assert (passed.value, passed.answer, passed.explanation) == (1.0, "pass", "it says 4")
    assert passed.judge is not None
    assert (passed.judge.model, passed.judge.prompt_sha256) == ("judge-model", graded.prompt_sha256)
    assert passed.judge.position_swapped is None and len(passed.judge.calls) == 1
    call = passed.judge.calls[0]
    assert call.text.endswith('"it says 4"}') and call.last_turn.role == "user" and call.call.resolved_model == "judge-model"
    failed = graded.grade(_sample("5"), "4")
    assert (failed.value, failed.answer, failed.explanation) == (0.0, "fail", None)
    with pytest.raises(CannotGrade, match="no JSON object: 'I cannot decide.'"):
        graded.grade(_sample("4"), "4")
    with pytest.raises(CannotGrade, match="not a verdict"):
        graded.grade(_sample("4"), "4")
    prompts = [request.turns[-1].content for request in provider.requests]
    assert all(prompt.startswith("The output gives the right sum.\n") for prompt in prompts)
    assert "[BEGIN INPUT]\nWhat is 2 + 2?\n[END INPUT]" in prompts[0] and "[BEGIN OUTPUT]\n5\n[END OUTPUT]" in prompts[1]
    assert prompts[0] == graded.render(_sample("4"), "4")
    assert re.fullmatch(r"[0-9a-f]{64}", graded.prompt_sha256)


def test_llm_judge_reply_preview_is_cut_at_200_characters() -> None:
    graded = judge.llm_judge(ScriptedProvider(["x" * 300]), rubric="r", settings=SETTINGS)
    with pytest.raises(CannotGrade) as info:
        graded.grade(_sample("4"), None)
    assert "x" * 200 in str(info.value) and "x" * 201 not in str(info.value)


@pytest.mark.parametrize(
    "reply",
    [
        '{"verdict": "pass", "explanation": "ok", "confidence": 0.9}',
        '{"verdict": "pass", "explanation": null}',
        '{"verdict": "PASS"}',
        '{"explanation": "ok"}',
    ],
    ids=["extra key", "null explanation", "upper case", "no verdict"],
)
def test_a_verdict_object_must_be_exactly_verdict_and_an_optional_explanation(reply: str) -> None:
    graded = judge.llm_judge(ScriptedProvider([reply, reply]), rubric="r", settings=SETTINGS)
    with pytest.raises(CannotGrade, match="not a verdict"):
        graded.grade(_sample("4"), None)
    with pytest.raises(CannotGrade, match="not a verdict"):
        judge.read_verdict(reply)
    assert judge.last_json_object(reply, "explanation") in ("ok", None)


# --- pairwise_judge -----------------------------------------------------------------------------------------------


def _pair_sample() -> Sample:
    return _sample("", Turn(role="user", content=PAIR))


def test_pairwise_judge_without_swap_scores_agreement_with_the_label() -> None:
    provider = ScriptedProvider(['{"verdict": "A"}', 'I prefer B.\n{"verdict": "B>A"}', "no side"])
    graded = judge.pairwise_judge(provider, rubric="Which is correct?", settings=SETTINGS, swap=False)
    assert (graded.id, graded.kind) == ("pairwise-judge", "model")
    right = graded.grade(_pair_sample(), {"label": "A>B"})
    assert (right.value, right.answer) == (1.0, "A")
    assert right.judge is not None and right.judge.position_swapped is False and len(right.judge.calls) == 1
    wrong = graded.grade(_pair_sample(), {"label": "A"})
    assert (wrong.value, wrong.answer) == (0.0, "B")
    with pytest.raises(CannotGrade, match="names no side"):
        graded.grade(_pair_sample(), {"label": "A"})
    with pytest.raises(CannotGrade, match="no A/B label"):
        graded.grade(_pair_sample(), {"winner": "A"})
    prompt = provider.requests[0].turns[-1].content
    assert prompt.startswith("Which is correct?\n\n") and "RESPONSE A:\n4\n\nRESPONSE B:\n5" in prompt
    assert prompt.rstrip().endswith('{"verdict": "A" or "B"}.')


def test_pairwise_judge_with_swap_asks_twice_and_a_position_biased_judge_scores_zero() -> None:
    consistent = ScriptedProvider(['{"verdict": "A"}', '{"verdict": "B"}'])
    graded = judge.pairwise_judge(consistent, rubric="Which is correct?", settings=SETTINGS)
    score = graded.grade(_pair_sample(), {"label": "A"})
    assert (score.value, score.answer) == (1.0, "A")
    assert score.judge is not None and score.judge.position_swapped is True and len(score.judge.calls) == 2
    first, second = (request.turns[-1].content for request in consistent.requests)
    assert "RESPONSE A:\n4\n\nRESPONSE B:\n5" in first and "RESPONSE A:\n5\n\nRESPONSE B:\n4" in second
    assert second == graded.render(_pair_sample(), {"label": "A"}, swapped=True)
    assert score.judge.calls[0].text == '{"verdict": "A"}' and score.judge.calls[1].text == '{"verdict": "B"}'

    biased = ScriptedProvider(['{"verdict": "A"}', '{"verdict": "A"}'])
    score = judge.pairwise_judge(biased, rubric="r", settings=SETTINGS).grade(_pair_sample(), {"label": "A"})
    assert score.value == 0.0 and score.answer is None
    assert score.explanation is not None and score.explanation.startswith("position-biased")

    consistent_but_wrong = ScriptedProvider(['{"verdict": "B"}', '{"verdict": "A"}'])
    score = judge.pairwise_judge(consistent_but_wrong, rubric="r", settings=SETTINGS).grade(_pair_sample(), {"label": "A"})
    assert (score.value, score.answer) == (0.0, "B")


def test_pairwise_judge_reads_mtbench_turns_with_its_own_markers_and_label_key() -> None:
    turn = "USER:\nhi\n\nASSISTANT A:\nhello\n\nASSISTANT B:\nyo"
    sample = _sample("", Turn(role="system", content="Judge."), Turn(role="user", content=turn))
    provider = ScriptedProvider(['{"winner": "B"}', '{"winner": "A"}'])
    graded = judge.pairwise_judge(
        provider, rubric="r", settings=SETTINGS, verdict_key="winner", markers=("ASSISTANT A:", "ASSISTANT B:"), label_key="winner"
    )
    assert graded.grade(sample, {"winner": "B"}).value == 1.0
    swapped = provider.requests[1].turns[-1].content
    assert "SYSTEM:\nJudge.\n\nUSER:\nUSER:\nhi\n\nASSISTANT A:\nyo\n\nASSISTANT B:\nhello" in swapped
    with pytest.raises(CannotGrade, match="no 'RESPONSE A:'"):
        judge.pairwise_judge(provider, rubric="r", settings=SETTINGS).render(sample, {"label": "A"}, swapped=True)


# --- model_graded_qa (relocated from 0.3.2 tests/behaviors/evals/test_scorers_lifted.py) -------------------------


def _qa(*replies: str, partial_credit: bool = False) -> Judge:
    return lifted.model_graded_qa(ScriptedProvider(replies), settings=SETTINGS, partial_credit=partial_credit)


def test_the_judge_prompt_is_deterministic_and_its_grade_is_the_last_one() -> None:
    graded = _qa()
    prompt = graded.render(_sample("4"), "4")
    assert prompt == graded.render(_sample("4"), "4")
    assert prompt != graded.render(_sample("5"), "4")
    assert "What is 2 + 2?" in prompt and "GRADE" in prompt
    nonce = hashlib.sha256(b"What is 2 + 2?\x004").hexdigest()[:8]
    assert nonce in prompt
    assert graded.read("reasoning\n\nGRADE: C\nGRADE: I").value == 0.0
    assert graded.read("echoed GRADE: I in the answer\n\nGRADE: C\nthanks").value == 1.0
    with pytest.raises(CannotGrade, match="has no GRADE"):
        graded.read("a downgrade: C is not a verdict")
    assert graded.read("GRADE: I").answer == "I"
    with pytest.raises(CannotGrade, match="has no GRADE"):
        graded.read("no verdict")
    assert re.fullmatch(r"[0-9a-f]{64}", graded.prompt_sha256)
    assert (graded.id, graded.version, graded.kind) == ("model_graded_qa", lifted.LIFTED_VERSION, "model")


@pytest.mark.parametrize(
    "reply",
    [
        "thinking GRADE: C then final\nGRADE: X",  # an off-menu final verdict never falls back to the earlier C
        "GRADE: CI",  # a multi-letter verdict is not laundered to its first letter
        "GRADE: P",  # the default instructions offer C and I only
        "GRADE: C\nGrade - I",  # a later verdict line in another form disagrees
    ],
)
def test_a_judge_verdict_off_the_offered_grades_is_no_answer(reply: str) -> None:
    with pytest.raises(CannotGrade, match="the judge's final GRADE is not one of the offered grades"):
        _qa().read(reply)


def test_a_spelled_out_or_marked_up_offered_verdict_is_read() -> None:
    graded = _qa()
    assert graded.read("GRADE: Correct").value == 1.0
    assert graded.read("GRADE: **I**").value == 0.0
    assert graded.read("Grade - I\nGRADE: C\nGrade - C").value == 1.0


def test_model_graded_qa_grades_through_the_provider_and_offers_partial_credit_on_request() -> None:
    graded = _qa("The sum is right.\n\nGRADE: C", "Wrong.\n\nGRADE: I", "Half.\n\nGRADE: P")
    correct = graded.grade(_sample("4"), "4")
    assert (correct.value, correct.answer, correct.explanation) == (1.0, "C", "The sum is right.\n\nGRADE: C")
    assert correct.judge is not None and correct.judge.prompt_sha256 == graded.prompt_sha256
    assert correct.judge.calls[0].last_turn.content == graded.render(_sample("4"), "4")
    assert graded.grade(_sample("5"), "4").value == 0.0
    with pytest.raises(CannotGrade, match="offered grades"):
        graded.grade(_sample("4"), "4")
    partial = _qa("Half.\n\nGRADE: P", partial_credit=True)
    assert partial.prompt_sha256 != graded.prompt_sha256
    assert '"P" for partially correct answers' in partial.render(_sample("4"), "4")
    assert partial.grade(_sample("4"), "4").value == 0.5


# --- autoevals_judge and openevals_judge --------------------------------------------------------------------------


def test_autoevals_judge_renders_the_template_and_scores_the_named_choice() -> None:
    provider = ScriptedProvider(["(A)\nBecause it matches.", "C", "Maybe"])
    graded = lifted.autoevals_judge("factuality", provider, settings=SETTINGS)
    assert (graded.id, graded.version, graded.kind) == ("autoevals_factuality", lifted.LIFTED_VERSION, "model")
    template = judge.load_template("factuality")
    assert graded.prompt_sha256 == hashlib.sha256(template.prompt.encode()).hexdigest()
    prompt = graded.render(_sample("4"), "4")
    assert prompt == judge.render_template(template, input="What is 2 + 2?", output="4", expected="4")
    first = graded.grade(_sample("4"), "4")
    assert (first.value, first.answer) == (template.choice_scores["A"], "A")
    assert graded.grade(_sample("4"), "4").value == template.choice_scores["C"]
    with pytest.raises(CannotGrade, match="names none of A, B, C, D, E: 'Maybe'"):
        graded.grade(_sample("4"), "4")
    with pytest.raises(CannotGrade, match="no value for instructions"):
        lifted.autoevals_judge("battle", provider, settings=SETTINGS).render(_sample("4"), "4")


def test_openevals_judges_read_a_pass_or_fail_verdict() -> None:
    correctness = lifted.openevals_judge("correctness", ScriptedProvider(['{"verdict": "pass"}']), settings=SETTINGS)
    assert (correctness.id, correctness.version) == ("openevals_correctness", lifted.LIFTED_VERSION)
    score = correctness.grade(_sample("4"), "4")
    assert (score.value, score.answer) == (1.0, "pass")
    assert score.judge is not None and score.judge.prompt_sha256 == correctness.prompt_sha256
    hallucination = lifted.openevals_judge(
        "hallucination", ScriptedProvider(['{"verdict": "fail", "explanation": "made up"}']), settings=SETTINGS
    )
    assert hallucination.prompt_sha256 != correctness.prompt_sha256
    score = hallucination.grade(_sample("Paris is in Spain."), {"context": "Paris is in France."})
    assert (score.value, score.answer, score.explanation) == (0.0, "fail", "made up")
    assert "<reference_outputs>\nnull\n</reference_outputs>" in hallucination.render(_sample("x"), {"context": "c"})


# --- the Protocol -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "built",
    [
        judge.llm_judge(ScriptedProvider([]), rubric="r", settings=SETTINGS),
        judge.pairwise_judge(ScriptedProvider([]), rubric="r", settings=SETTINGS),
        lifted.model_graded_qa(ScriptedProvider([]), settings=SETTINGS),
        lifted.autoevals_judge("summary", ScriptedProvider([]), settings=SETTINGS),
        lifted.openevals_judge("hallucination", ScriptedProvider([]), settings=SETTINGS),
    ],
    ids=["llm_judge", "pairwise_judge", "model_graded_qa", "autoevals_judge", "openevals_judge"],
)
def test_every_judge_constructor_satisfies_the_judge_protocol(built: Judge) -> None:
    # ``Judge`` is not runtime-checkable; the parameter's annotation is the static check (pyright strict), and the
    # attribute reads below are the dynamic one.
    assert built.kind == "model" and built.id and built.version
    assert re.fullmatch(r"[0-9a-f]{64}", built.prompt_sha256)
    assert callable(built.render) and callable(built.read) and callable(built.grade)
