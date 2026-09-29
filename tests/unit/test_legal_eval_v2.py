"""israeli_legal_eval_v2: the build is complete and consistent, and the verify/select/consistency
helpers of scripts/legal_data/eval_v2.py behave (with a fake corpus -- no Chroma needed)."""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "legal_data"))

import eval_v2

V2 = ROOT / "legal_txt" / "Evals" / "israeli_legal_eval_v2"


def _load(name):
    return [json.loads(line) for line in (V2 / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_the_built_set_is_consistent():
    gold, questions = _load("gold.jsonl"), _load("questions.jsonl")
    assert [g["id"] for g in gold] == [q["id"] for q in questions]
    assert len({g["id"] for g in gold}) == len(gold)
    splits = Counter(g["split"] for g in gold)
    assert splits["test"] >= 380 and splits["dev"] >= 190 and splits["robustness"] == 75
    for g in gold:
        assert g["scoring"] in ("judge", "label_then_judge", "exact_letter")
        if g["scoring"] == "label_then_judge":
            assert g["answer_label"] in ("כן", "לא"), g["id"]
        if g["source"] == "v2_new":
            assert g["confidence"] == "medium" and g["verified"] is None
            assert all(c["law"] for c in g["citations"]), g["id"]
        if g["source"] == "v2_paraphrase":
            assert next(x for x in gold if x["id"] == g["paraphrase_group"])["split"] == "test"
    for q in questions:
        assert q["instructions"] and q["question"]


def test_gold_numbers_skip_sections_and_years():
    text = "לפי סעיף 25י, התקרה היא הנמוך מבין 3 חודשים לבין שליש – 18,000 ש\"ח (תיקון 2017)."
    assert eval_v2.gold_numbers(text) == {"3", "18000"}


class _Corpus:
    def __init__(self):
        self.index = {'חוק השכירות והשאילה, התשל"א-1971': {"chunks": 3}}
        self.sections = {'חוק השכירות והשאילה, התשל"א-1971': {"25י": [("laws", "rent:25י#p1")]}}

    def texts(self, refs):
        return "(ב) לא יעלה על הנמוך מבין שליש ... לבין דמי שכירות בעד 3 חודשים"


def _item(**kw):
    base = {"id": "X", "category": "rule_recall", "gold_answer": "הנמוך מבין 3 חודשים לבין שליש.",
            "citations": [{"law_code": "RNT", "law": 'חוק השכירות והשאילה, התשל"א-1971', "section": "25י"}]}
    return {**base, **kw}


def test_verify_finds_the_section_and_flags_numbers_the_statute_does_not_hold():
    ok = eval_v2.verify_item(_item(), _Corpus())
    assert ok["status"] == "VERIFIED" and ok["gold_chunks"] == ["rent:25י#p1"]
    bad = eval_v2.verify_item(_item(gold_answer="הנמוך מבין 4 חודשים לבין שליש."), _Corpus())
    assert bad["status"] == "NUMBERS_MISSING" and "4" in bad["notes"][0]
    computed = eval_v2.verify_item(_item(category="computation", gold_answer="18,000 ש\"ח"), _Corpus())
    assert computed["status"] == "VERIFIED"  # computed values are not looked for in the statute
    missing = eval_v2.verify_item(_item(citations=[{**_item()["citations"][0], "section": "99"}]), _Corpus())
    assert missing["status"] == "SECTION_MISSING"
    assert eval_v2.verify_item(_item(citations=[]), _Corpus())["status"] == "NO_CITATION"


def test_select_keeps_v1_and_verified_new_items_only_when_asked():
    gold = [{"id": "a", "split": "test", "source": "v1"}, {"id": "b", "split": "test", "source": "v2_new", "verified": False},
            {"id": "c", "split": "test", "source": "v2_new", "verified": True}, {"id": "d", "split": "dev", "source": "v1"}]
    qs = [{"id": i} for i in "abcd"]
    assert [g["id"] for g in eval_v2.select(gold, qs, "test", False)[0]] == ["a", "b", "c"]
    kept, kq = eval_v2.select(gold, qs, "test", True)
    assert [g["id"] for g in kept] == ["a", "c"] and [q["id"] for q in kq] == ["a", "c"]


def test_paraphrase_consistency():
    gold = [{"id": "S"}, {"id": "S-P1", "paraphrase_group": "S", "paraphrase_style": "english"},
            {"id": "S-P2", "paraphrase_group": "S", "paraphrase_style": "colloquial"},
            {"id": "T"}, {"id": "T-P1", "paraphrase_group": "T", "paraphrase_style": "english"}]
    rows = [{"id": "S", "score": 1.0}, {"id": "S-P1", "score": 0.5}, {"id": "S-P2", "score": 1.0},
            {"id": "T", "score": 1.0}, {"id": "T-P1", "score": 1.0}]
    out = eval_v2.consistency(gold, rows)
    assert out["groups"] == 2 and out["consistent"] == 1
    assert out["mean_delta_by_style"] == {"english": -0.25, "colloquial": 0.0}
