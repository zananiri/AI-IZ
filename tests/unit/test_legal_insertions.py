"""Chunking of what an amending law inserts, and of a gazette preamble and
signature block (legal/insertions.py, legal/chunking.py), plus the Hebrew
keyword terms used by legal/keyword.py."""

import pytest

from docslides.config import get_config
from docslides.legal import chunking
from docslides.legal.chunking import SourceMeta, chunk_sections
from docslides.legal.insertions import find_insertion, join_spaced_section_numbers
from docslides.legal.keyword import KeywordIndex, terms
from docslides.legal.structure import parse_sections

LAW = """\
תיקונים עקיפים:
חוק הבחירות לכנסת [נוסח משולב], התשכ"ט-1969 - מס' 79
* התקבל בכנסת ביום ב' באב התשפ"ו (16 ביולי 2026);
הצעת החוק פורסמה בהצעות חוק הכנסת - 1209.
1 ס"ח התשי"א, עמ' 78.
רשומות
⟦מטרה⟧
1. מטרתו של חוק זה לקבוע הוראות מיוחדות.
⟦תיקון חוק הבחירות לכנסת - מס' 79⟧
2. חוק הבחירות לכנסת [נוסח משולב], התשכ"ט-1969, ייקרא כך:
(1) לפני פרק י'4 יבוא:
״פרק י'3ג: הצבעת מפונים
(א) בפרק זה - "מפונה" - מי שעזב את ביתו.
(ב) המידע יימחק לא יאוחר מ־14 ימים.
⟦הרכב ועדות 116 יז 11. הקלפי למפונים⟧
(א) ועדות הקלפי יורכבו מנציגי הסיעות.
(ב) יחולו הוראות סעיף 21א.";
(2) בסעיף 62(ג), במקום "ה־40" יבוא "ה־43";
בנימין נתניהו
ראש הממשלה
"""

META = SourceMeta(law_id="law", law_name='חוק הבחירות (הוראות מיוחדות), התשפ"ו-2026', effective_date_start="2026-07-16",
                  status="current", source_type="statute", source_origin="knesset")


@pytest.fixture(autouse=True)
def word_token_count(monkeypatch):
    monkeypatch.setattr(chunking, "count_tokens", lambda text: len(text.split()))
    monkeypatch.setattr(get_config().legal.ingestion, "chunk_max_tokens", 500)


def test_spaced_section_numbers_are_joined_but_ordinary_text_is_not():
    assert join_spaced_section_numbers("בסעיף 116 יז 12(ג) כהגדרתו בסעיף 116 יז 10;") == \
        "בסעיף 116יז12(ג) כהגדרתו בסעיף 116יז10;"
    assert join_spaced_section_numbers("סעיפים 6 או 7 ו-62 עד 64") == "סעיפים 6 או 7 ו-62 עד 64"


def test_inserted_provisions_are_found_with_a_lost_number_inferred():
    insertion = find_insertion(join_spaced_section_numbers(
        "(1) לפני פרק י'4 יבוא:\n״פרק י'3ג: הצבעת מפונים\n(א) הגדרות.\n(ב) מחיקה.\n"
        "⟦הרכב ועדות 116 יז 11. הקלפי למפונים⟧\n(א) ועדות.".replace("⟦", "").replace("⟧", "")
    ))
    assert [(p.number, p.inferred_number, p.title) for p in insertion.provisions] == [
        ("116יז10", True, None), ("116יז11", False, "הרכב ועדות הקלפי למפונים"),
    ]
    assert insertion.context[-1] == "״פרק י'3ג: הצבעת מפונים"
    assert find_insertion('(2) בסעיף 62(ג), במקום "ה־40" יבוא "ה־43";') is None  # wording, not a provision


def test_amending_chunks_are_one_per_inserted_provision_with_both_numbers():
    by_id = {c.metadata.chunk_id.split(":", 1)[1]: c for c in chunk_sections(parse_sections(LAW), META, "2026-09-23")}

    assert {"2>116יז10", "2>116יז11", "2"} <= set(by_id)
    assert "62(ג)" in by_id["2"].text and "62(ג)" not in by_id["2>116יז11"].text  # the next instruction
    chunk = by_id["2>116יז10"]
    assert chunk.metadata.inserted_section == "116יז10"
    assert chunk.metadata.display_section == "2 › 116יז10"
    assert "מוסיף את סעיף 116יז10 לחוק הבחירות לכנסת" in chunk.text
    assert "לפני פרק י'4 יבוא" in chunk.text and "מ־14 ימים" in chunk.text  # instruction kept as context
    assert "ועדות הקלפי" not in chunk.text  # the next inserted section is its own chunk
    assert [ref["sections"] for ref in chunk.metadata.amends] == [["116יז10"]]
    assert by_id["2>116יז11"].text.count("הרכב ועדות הקלפי למפונים") == 1


def test_preamble_keeps_the_passage_note_and_amendment_list_only():
    chunks = [c for c in chunk_sections(parse_sections(LAW), META, "2026-09-23") if c.metadata.section_number == "preamble"]
    labels = [c.metadata.subsection_number for c in chunks]
    assert labels == ["קבלת החוק", "תיקונים עקיפים"]
    assert "התקבל בכנסת ביום ב' באב" in chunks[0].text and "ס״ח" not in chunks[0].text
    assert "רשומות" not in chunks[0].text + chunks[1].text


def test_signature_block_is_dropped_from_the_last_section():
    last = [c for c in chunk_sections(parse_sections(LAW), META, "2026-09-23") if c.metadata.section_number == "2"][-1]
    assert "נתניהו" not in last.text and "ראש הממשלה" not in last.text


def test_keyword_terms_strip_hebrew_prefixes_and_quotes():
    assert {"היוועדות", "יוועדות"} <= set(terms("בהיוועדות"))
    assert "כב" in terms('כ"ב') and "כב" in terms("כ״ב")
    assert "116יז10" in terms("סעיף 116 יז 10")
    index = KeywordIndex([("a", "דיון בהיוועדות חזותית"), ("b", "מחיקת מידע סטטיסטי")])
    assert index.search("האם אפשר לקיים היוועדות חזותית?", 5)[0][0] == "a"
