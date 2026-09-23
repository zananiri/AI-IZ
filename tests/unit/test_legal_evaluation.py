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
