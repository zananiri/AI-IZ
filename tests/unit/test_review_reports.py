"""israeli_legal_eval_v3 is built to the 100-question criteria, and scripts/legal_data/review_reports.py
scores like score.py and keeps system names out of the attorney page."""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "legal_data"))

import review_reports

V3 = ROOT / "legal_txt" / "Evals" / "israeli_legal_eval_v3"


def _load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v3_matches_the_previous_test_criteria():
    gold, questions = _load(V3 / "gold.jsonl"), _load(V3 / "questions.jsonl")
    assert [g["id"] for g in gold] == [q["id"] for q in questions]
    assert Counter(g["category"] for g in gold) == {
        "rule_recall": 22, "rule_conclusion": 18, "mcq_bar": 16, "rule_application": 12, "interpretation": 8,
        "citation_grounding": 7, "issue_spotting": 7, "abstention": 6, "temporal_amendment": 4}
    assert set(Counter(g["answer_letter"] for g in gold if g["scoring"] == "exact_letter").values()) == {4}
    assert Counter(g["answer_label"] for g in gold if g["scoring"] == "label_then_judge") == {"כן": 9, "לא": 9}
    for g, q in zip(gold, questions):
        assert g["gold_answer"] and g["key_points"] and q["instructions"] and q["question"]
        if g["scoring"] == "exact_letter":
            assert g["answer_letter"] in q["options"] and g["gold_answer"].startswith(g["answer_letter"])
        if g["scoring"] == "label_then_judge":
            assert g["gold_answer"].startswith(g["answer_label"]) and g["key_points"][0] == g["answer_label"]
    cases = json.loads((V3 / "cases_gold.json").read_text(encoding="utf-8"))["cases"]
    assert len(cases) == 5
    for c in cases:
        assert (V3 / "cases" / c["file"]).exists()
        assert sum(c["rubric_weights"].values()) == 100 and c["must_not"] and c["red_flags"]


def _run(tmp_path, answers, judged):
    run = tmp_path / "run"
    (run / "questions").mkdir(parents=True)
    (run / "cases").mkdir()
    for name, rows in (("answers_rag.jsonl", answers), ("judged_rag.jsonl", judged)):
        (run / "questions" / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return run


def test_scores_follow_score_py_and_the_attorney_page_names_no_components(tmp_path):
    gold = {g["id"]: g for g in _load(V3 / "gold.jsonl")}
    mcq = next(g for g in gold.values() if g["scoring"] == "exact_letter")
    yes_no = next(g for g in gold.values() if g["scoring"] == "label_then_judge")
    recall = next(g for g in gold.values() if g["category"] == "rule_recall")
    wrong_letter = next(x for x in "אבגד" if x != mcq["answer_letter"])
    answers = [{"id": mcq["id"], "answer": f"{wrong_letter}\nנימוק"},
               {"id": yes_no["id"], "answer": f"{yes_no['answer_label']}, כי ... </start_of_turn>"},
               {"id": recall["id"], "answer": "תשובה", "retrieved": [{"title": recall["citations"][0]["law"],
                                                                      "section": recall["citations"][0]["section"]}]}]
    judged = [{"id": yes_no["id"], "correctness": 1, "grounding": "grounded", "hallucination": False, "key_points_hit": [1],
               "needs_review": False, "note": "The model missed the exception; retrieval was fine."},
              {"id": recall["id"], "correctness": 2, "grounding": "grounded", "hallucination": False, "key_points_hit": [],
               "needs_review": False, "note": "ok"}]
    data = review_reports.collect(V3, _run(tmp_path, answers, judged), {"run_name": "t"})
    by_id = {i["id"]: i for i in data["items"]}
    assert by_id[mcq["id"]]["score"] == 0.0 and by_id[mcq["id"]]["diag"] != "right"
    assert by_id[yes_no["id"]]["score"] == 0.75 and "</start_of_turn>" not in by_id[yes_no["id"]]["answer"]
    assert by_id[recall["id"]]["score"] == 1.0 and by_id[recall["id"]]["govRetrieved"] is True
    assert by_id[recall["id"]]["keyHit"] == [True] * len(recall["key_points"])
    assert data["summary"]["scored"] == 3 and data["summary"]["questions"] == 100

    page = review_reports.attorney_html(data)
    assert "dir=\"rtl\"" in page and "סטטיסטיקה כללית" in page and "תוצאות לפי סוג השאלה" in page
    assert not review_reports.BANNED.search(page.split("<script>", 1)[1].split("const D =", 1)[1])
    review_reports.technical_html(data)  # builds


def test_scrub_removes_component_names():
    text = review_reports.scrub("The model (Gemma via Ollama) retrieved the chunk; המודל טעה")
    assert not review_reports.BANNED.search(text) and "המערכת" in text
