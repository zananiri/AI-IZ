from pathlib import Path

from docslides.legal import evaluation as ev

EVAL_SET = Path(__file__).resolve().parents[2] / "evals" / "legal" / "elections_2026_he.json"


def _q(**kw):
    base = {"id": "X", "group": "A", "question": "q", "gold": "g"}
    return ev.EvalQuestion(**{**base, **kw})


def test_eval_set_loads_and_all_patterns_compile():
    import re

    _, questions = ev.load_eval_set(EVAL_SET)
    assert len(questions) == 16 and {q.group for q in questions} == {"A", "B", "C"}
    for q in questions:
        for pattern in q.key_facts + q.traps + [p for g in q.evidence for p in g]:
            re.compile(pattern)
        assert q.group == "C" or q.evidence


def test_gold_answers_satisfy_their_own_key_facts():
    _, questions = ev.load_eval_set(EVAL_SET)
    for q in questions:
        if q.key_facts:
            assert ev.fact_coverage(q.gold, q.key_facts) == 1.0, q.id
        assert not ev.trap_hits(q.gold, q.traps), q.id


def test_baseline_classification():
    a, c = _q(), _q(group="C")
    assert ev.classify_baseline(a, "abstained", False, []) == "abstained"
    assert ev.classify_baseline(a, "correct", False, []) == "correct_without_context"
    assert ev.classify_baseline(a, "incorrect", True, []) == "hallucination"
    assert ev.classify_baseline(c, "correct", False, []) == "abstained"
    assert ev.classify_baseline(c, "correct", False, ["500 ש\"ח"]) == "hallucination"


def test_after_scoring_retrieval_answer_citation_and_multihop():
    q = _q(group="B", evidence=[["רשות ציבורית"], ["גורם ציבורי"]])
    scores = ev.score_after(q, "correct", False, [], ["...רשות ציבורית...", "...גורם ציבורי..."], ["...רשות ציבורית..."])
    assert scores["retrieval"] == 1.0 and scores["citation"] == 0.5 and scores["answer"] == 1.0
    assert abs(scores["score"] - 2.5 / 3) < 1e-9

    c = _q(group="C", traps=[r"\d+\s*ש\"ח"])
    assert ev.score_after(c, "correct", False, [], [], [])["passed"]
    assert not ev.score_after(c, "correct", False, ["1,000 ש\"ח"], [], [])["passed"]
    assert not ev.score_after(c, "incorrect", True, [], [], [])["passed"]


def test_summary_and_report():
    questions = [_q(id="A1"), _q(id="C1", group="C")]
    before = {"A1": {"classification": "hallucination"}, "C1": {"classification": "abstained"}}
    after = {"A1": {"verdict": "correct", "score": 1.0, "retrieval": 1.0, "answer": 1.0, "citation": 1.0},
             "C1": {"verdict": "correct", "score": 1.0, "passed": True}}
    summary = ev.summarize(questions, before, after)
    assert summary["before"]["score"] == 50 and summary["after"]["score"] == 100
    assert summary["after"]["hallucinations"] == 0
    assert "| A1 | hallucination |" in ev.render_report({"eval_set": "t"}, questions, before, after, summary)


def test_rescore_after_reuses_stored_coverage():
    q = _q(group="A")
    record = {"retrieval": 1.0, "citation": 0.0}
    assert ev.rescore_after(q, record, "correct", False, [])["score"] == 2 / 3
    assert ev.rescore_after(q, record, "abstained", False, [])["score"] == 1 / 3


# --- grader safeguards ---------------------------------------------------------------------


def test_judge_input_uses_gershayim_so_the_judge_cannot_truncate_on_a_quote():
    from docslides.legal import evaluation as ev

    q = ev.EvalQuestion(id="A1", group="A", question='מה הסכום בש"ח?', gold='2.5 מיליון ש"ח', key_facts=["2[.,]5"])
    text = ev._judge_input(q, 'מעל 2.5 מיליון ש"ח')
    assert '"' not in text and "ש״ח" in text


def test_only_answers_holding_every_key_fact_get_the_contradiction_check():
    from docslides.legal import evaluation as ev

    q = ev.EvalQuestion(id="B8", group="B", question="?", gold="!", key_facts=["a", "b"])
    c = ev.EvalQuestion(id="C1", group="C", question="?", gold="!", traps=["x"])
    assert ev.needs_contradiction_check(q, "incorrect", 1.0, [])
    assert ev.needs_contradiction_check(q, "partially_correct", 1.0, [])
    assert not ev.needs_contradiction_check(q, "incorrect", 0.5, [])  # a key fact is missing
    assert not ev.needs_contradiction_check(q, "abstained", 1.0, [])
    assert not ev.needs_contradiction_check(c, "incorrect", None, [])  # unanswerables are judged as before


def test_the_question_shown_to_the_model_uses_gershayim():
    from docslides.legal import pipeline

    assert "יו״ר" in pipeline._question_block('האם סמכויות יו"ר הוועדה חלות?')
    assert '"' in pipeline._question_block('Does "section 5" apply?')  # not Hebrew: untouched
