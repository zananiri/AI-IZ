"""legal/corpus_navigation.py: whole sections, tables of contents, cross-references, context shaping
and doctrine cards -- the 30 Sept retrieval variants, each behind a legal.corpus switch."""

import asyncio
import json
import sys
from pathlib import Path

from docslides.config import get_config
from docslides.legal import corpus_navigation as nav

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "legal_data"))

LAW = 'חוק החוזים (תרופות בשל הפרת חוזה), תשל"א-1970'
REC = "law:remedies"
HEAD = f"{LAW} > סעיף 7 — ביטול"


class FakeChroma:
    """collection.get over a list of (chunk_id, meta, text), with the where filters the module uses."""

    def __init__(self, rows):
        self.rows = rows

    def get(self, where=None, include=None, ids=None):
        conds = where.get("$and", [where]) if where else []
        picked = [r for r in self.rows if all(r[1].get(k) == v for c in conds for k, v in c.items())]
        return {"ids": [r[0] for r in picked], "metadatas": [r[1] for r in picked], "documents": [r[2] for r in picked]}


class FakeCollection:
    def __init__(self, rows):
        self.collection = FakeChroma(rows)


def _meta(section, part=1, count=1, title=LAW, heading="ביטול"):
    return {"record_id": REC, "title": title, "section_number": section, "part_index": part, "part_count": count,
            "breadcrumb": f"{title} > סעיף {section} — {heading}"}


ROWS = [
    (f"{REC}@1970:6", _meta("6", heading="הפרה יסודית"), f"{LAW} > סעיף 6 — הפרה יסודית\n\nהפרה יסודית היא ..."),
    (f"{REC}@1970:7#p1", _meta("7", 1, 2), f"{HEAD} (חלק 1 מתוך 2)\n\n(א) נפגע זכאי לבטל את החוזה אם ההפרה יסודית.\n(ב) היתה ההפרה לא יסודית"),
    (f"{REC}@1970:7#p2", _meta("7", 2, 2), f"{HEAD} (חלק 2 מתוך 2)\n\n(ב) היתה ההפרה לא יסודית, זכאי הנפגע לבטל אחרי ארכה, אלא אם הביטול בלתי צודק; כאמור בסעיף 6."),
]
COLL = FakeCollection(ROWS)


def _hit(chunk_id, text, section="7", score=0.5, category="laws", sources=("q0",), title=LAW):
    return {"id": chunk_id, "category": category, "distance": 0.3, "meta": _meta(section, title=title),
            "text": text, "sources": set(sources), "score": score}


def test_parts_merge_without_repeated_headers_or_overlap():
    merged = nav.merge_parts([ROWS[1][2], ROWS[2][2]])
    assert merged.startswith(HEAD + "\n\n(א)")
    assert "חלק 1 מתוך 2" not in merged and merged.count(HEAD) == 1
    assert merged.count("(ב) היתה ההפרה לא יסודית") == 1  # the overlap between parts appears once
    assert merged.endswith("אלא אם הביטול בלתי צודק; כאמור בסעיף 6.")


def test_a_split_section_is_retrieved_whole_and_its_parts_share_one_slot():
    hits = [_hit(ROWS[2][0], ROWS[2][2], score=0.9), _hit(ROWS[1][0], ROWS[1][2], score=0.4, sources=("k0",))]
    out = nav.expand_sections(hits, collections=lambda _category: COLL, max_tokens=1000)
    assert len(out) == 1
    assert out[0]["whole_section"] and out[0]["chunk_ids"] == [ROWS[1][0], ROWS[2][0]]
    assert "(א) נפגע זכאי" in out[0]["text"] and "בלתי צודק" in out[0]["text"]
    assert out[0]["sources"] == {"q0", "k0"} and out[0]["score"] == 0.9


def test_a_section_over_the_budget_keeps_only_its_retrieved_part():
    hit = _hit(ROWS[2][0], ROWS[2][2])
    assert nav.whole_section(hit, COLL, max_tokens=5) is hit


def test_a_referenced_section_of_the_same_law_is_fetched_once():
    hits = [_hit(ROWS[2][0], ROWS[2][2])]
    out = nav.cross_reference_hits(hits, collections=lambda _category: COLL, limit=3)
    assert [h["meta"]["section_number"] for h in out] == ["6"]
    assert out[0]["sources"] == {"x"}
    # already in the context: not fetched again
    assert nav.cross_reference_hits(hits + out, collections=lambda _category: COLL, limit=3) == []


def test_the_table_of_contents_lists_sections_in_order_with_headings():
    toc = nav.law_toc(COLL, REC)
    assert toc == [("6", "הפרה יסודית"), ("7", "ביטול")]


def test_toc_navigation_adds_the_picked_section_whole_and_ignores_unknown_picks():
    class Picker:
        def __init__(self):
            self.prompt = ""

        async def complete_json(self, messages, *_, **__):
            self.prompt = messages[0].content
            return nav.TocPicks(sections=[nav.TocPick(law=1, section="6"), nav.TocPick(law=1, section="99"),
                                          nav.TocPick(law=4, section="1")])

    picker = Picker()
    hits = [_hit(ROWS[1][0], ROWS[1][2])]
    out = asyncio.run(nav.navigate_toc(picker, "מתי אפשר לבטל?", hits, collections=lambda _c: COLL,
                                       n_laws=2, max_sections=4))
    assert [h["meta"]["section_number"] for h in out] == ["6"] and out[0]["sources"] == {"t"}
    assert "6: הפרה יסודית" in picker.prompt and LAW in picker.prompt


def test_toc_navigation_that_fails_adds_nothing():
    class Broken:
        async def complete_json(self, *_, **__):
            raise RuntimeError("down")

    hits = [_hit(ROWS[1][0], ROWS[1][2])]
    assert asyncio.run(nav.navigate_toc(Broken(), "ש", hits, collections=lambda _c: COLL)) == []


def test_regulations_are_capped_unless_the_plan_names_one():
    hits = [_hit(f"r{i}", "t", section=str(i), category="procedural_rules") for i in range(4)] + [_hit("l1", "t", "1")]
    assert [h["id"] for h in nav.cap_regulations(hits, 2, ["חוק החוזים"])] == ["r0", "r1", "l1"]
    assert len(nav.cap_regulations(hits, 2, ["תקנות סדר הדין האזרחי"])) == 5
    assert len(nav.cap_regulations(hits, None, [])) == 5


def test_context_groups_a_law_together_in_section_order():
    other = "חוק המכר, תשכ\"ח-1968"
    hits = [_hit("a", "t", "12"), {**_hit("b", "t", "3", title=other), "meta": {**_meta("3", title=other), "record_id": "law:sale"}},
            _hit("c", "t", "2"), _hit("d", "t", "12א")]
    assert [h["id"] for h in nav.group_by_law(hits)] == ["c", "a", "d", "b"]


def test_trim_keeps_the_first_hit_and_what_fits():
    from docslides.cleaning.tokens import count_tokens

    hits = [_hit("a", "מילה " * 50), _hit("b", "מילה " * 50), _hit("c", "מילה")]
    budget = count_tokens(hits[0]["text"]) + count_tokens("מילה")
    assert [h["id"] for h in nav.trim_to_budget(hits, budget)] == ["a", "c"]
    assert [h["id"] for h in nav.trim_to_budget(hits, 1)] == ["a"]  # the best always stays


def test_doctrine_cards_match_by_trigger_and_are_labelled_as_notes():
    cards = ({"id": "A", "title": "שימוע", "text": "חובת שימוע", "triggers": ["שימוע", "פיטורים"], "sources": ["פסיקה"]},
             {"id": "B", "title": "דין מקל", "text": "סעיף 5", "triggers": ["לפני התיקון+עבירה"]},
             {"id": "C", "title": "x", "text": "y", "triggers": ["פיטורים"], "status": "rejected"})
    assert [c["id"] for c in nav.match_doctrine_cards("שימוע לפני פיטורים", cards, 2)] == ["A"]
    assert nav.match_doctrine_cards("מה היה לפני התיקון?", cards, 2) == []  # "+": every part must appear
    assert [c["id"] for c in nav.match_doctrine_cards("עבירה שנעברה לפני התיקון", cards, 2)] == ["B"]
    block = nav.render_doctrine_cards([cards[0]])
    assert block.startswith("<doctrine_notes>") and "not statute text" in block and "מקורות: פסיקה" in block


def test_the_shipped_doctrine_cards_are_well_formed_drafts():
    path = Path(__file__).resolve().parents[2] / "legal_txt" / "doctrine_cards.jsonl"
    cards = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(cards) >= 15 and len({c["id"] for c in cards}) == len(cards)
    for card in cards:
        assert card["title"] and card["text"] and card["triggers"] and card["sources"]
        assert card["status"] == "draft_needs_lawyer_review" and card["reviewed_by_lawyer"] is False


def test_whole_sections_are_off_by_default_and_retrieval_is_unchanged():
    corpus_cfg = get_config().legal.corpus
    assert not (corpus_cfg.whole_sections or corpus_cfg.toc_navigation or corpus_cfg.cross_references
                or corpus_cfg.extract_then_answer or corpus_cfg.completeness_check or corpus_cfg.law_grouped_context)
    assert corpus_cfg.regulation_cap is None and corpus_cfg.doctrine_cards_path is None


def test_variants_set_the_corpus_switches(monkeypatch):
    import eval_run

    corpus_cfg = get_config().legal.corpus
    for key in ("whole_sections", "toc_navigation", "cross_references", "extract_then_answer", "completeness_check"):
        monkeypatch.setattr(corpus_cfg, key, getattr(corpus_cfg, key))
    changed = eval_run.apply_variants(["whole_sections", "completeness"])
    assert changed == {"whole_sections": True, "extract_then_answer": True, "completeness_check": True}
    assert corpus_cfg.whole_sections and corpus_cfg.completeness_check


def test_completeness_revises_an_answer_that_misses_an_element():
    import eval_run

    class Model:
        def __init__(self, missing):
            self.missing, self.revise_calls = missing, 0

        async def complete_json(self, *_, **__):
            return eval_run.Completeness(missing=self.missing)

        async def complete_text(self, *_, **__):
            self.revise_calls += 1
            return "הנפגע רשאי לבטל אחרי ארכה, אלא אם הביטול בלתי צודק (סעיף 7(ב))."

    extraction = eval_run.Extraction(elements=["ארכה"], exceptions=["אלא אם הביטול בלתי צודק"])
    text = "הנפגע רשאי לבטל אחרי ארכה (סעיף 7(ב))."
    revised, missing = asyncio.run(eval_run.complete_answer(Model(["בלתי צודק"]), "ש", extraction, text))
    assert "בלתי צודק" in revised and missing == ["בלתי צודק"]

    model = Model([])
    assert asyncio.run(eval_run.complete_answer(model, "ש", extraction, text)) == (text, [])
    assert model.revise_calls == 0
    assert "חריגים וסייגים" in eval_run.render_extraction(extraction)
