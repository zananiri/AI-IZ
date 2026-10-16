"""Reshumot-style extraction problems, reproduced with hand-placed characters
(right-to-left lines, number runs, duplicated page copies) plus the
text-level fixes, chunk dedupe, MMR and the parser's quote/title handling."""

import numpy as np

from docslides.legal import pdf_text
from docslides.legal.chunking import SourceMeta, chunk_sections, normalized_body
from docslides.legal.retrieval import _mmr
from docslides.legal.structure import parse_sections


def _visual_line(visual: str, baseline: float = 100.0, width: float = 5.0):
    """Characters laid out left-to-right exactly as a PDF paints `visual`."""
    return [pdf_text._Char(c, i * width, (i + 1) * width, baseline) for i, c in enumerate(visual)]


def test_rtl_line_keeps_numbers_left_to_right():
    # Painted as ".5 2026 ביולי 16" reversed for display: an RTL line reads right to left.
    logical = "16 ביולי 2026"
    visual = "2026 " + "ביולי"[::-1] + " 16"
    assert pdf_text._order_line(_visual_line(visual)) == logical


def test_section_number_and_decimal_keep_their_punctuation_after_the_run():
    visual = "ןוילמ 2.5 ." + "5"  # "5. ... 2.5 מיליון" painted RTL
    text = pdf_text._order_line(_visual_line(visual))
    assert text.startswith("5.")
    assert "2.5" in text


def test_ltr_lines_are_read_left_to_right():
    assert pdf_text._order_line(_visual_line("Section 12 applies")) == "Section 12 applies"


def test_normalize_line_fixes_reordering_leftovers():
    norm = pdf_text._normalize_line
    assert norm(".5 על אף האמור") == "5. על אף האמור"
    assert norm("(א ) הועסק") == "(א) הועסק"
    assert norm('התשי"א- , 1951 ובשל') == 'התשי"א-1951, ובשל'
    assert norm('במקום "% 25 " יבוא') == 'במקום "25%" יבוא'
    assert norm("(ה) (ה) (1) רשות") == "(ה) (1) רשות"
    assert norm("בסעיף 28 כה3(א)(2)") == "בסעיף 28כה3(א)(2)"
    assert norm("לפי סעיפים 17 ב עד 17 ד") == "לפי סעיפים 17ב עד 17ד"


def test_parser_uses_marked_margin_titles_and_ignores_quoted_amendment_text():
    text = "\n".join([
        "⟦העסקת עובד רשות ציבורית בוועדת בחירות לשם קיום הבחירות⟧",
        "4. (א) הועסק עובד רשות ציבורית.",
        "(ב) עובד יעדכן את הממונה.",
        "⟦תיקון חוק הבחירות לכנסת - מס' 79⟧",
        "6. בחוק הבחירות לכנסת -",
        "(1) בסעיף 7, בסופו יבוא \"או י'3ג\";",
        "(2) אחרי סעיף 24 יבוא:",
        "⟦קיום ישיבות בהיוועדות חזותית⟧",
        "24א. (א) נדרשה הוועדה לקיים ישיבה,",
        "(1) יושב ראש הוועדה הורה על כך;",
        "(2) כל חברי הוועדה הסכימו לכך.\"",
        "(3) בסעיף 62(ג), במקום \"ה־40\" יבוא \"ה־43\".",
        "7. הוראה נוספת.",
    ])
    sections = {s.number: s for s in parse_sections(text)}
    assert list(sections) == ["4", "6", "7"]  # "24א" is quoted text, not a section of this law
    assert sections["4"].title == "העסקת עובד רשות ציבורית בוועדת בחירות לשם קיום הבחירות"
    assert sections["6"].title == "תיקון חוק הבחירות לכנסת - מס' 79"
    assert [s.label for s in sections["6"].subsections] == ["1", "2", "3"]
    assert "קיום ישיבות בהיוועדות חזותית" in sections["6"].text
    assert sections["6"].cross_refs == []  # "סעיף 7" in an amendment is the amended law's


def test_duplicate_chunk_bodies_are_dropped_and_ids_stay_unique():
    body = "רשות ציבורית לא תפטר עובד כאמור בסעיף קטן משום שהועסק על ידי ועדת בחירות לשם קיום הבחירות"
    text = f"1. {body}\n2. {body}\n3. (בוטל)\n4. (בוטל)"
    meta = SourceMeta(law_id="x", law_name="חוק", effective_date_start="2026-07-16", status="current",
                      source_type="statute", source_origin="knesset")
    chunks = chunk_sections(parse_sections(text), meta, "2026-09-23")
    numbers = [c.metadata.section_number for c in chunks]
    assert numbers == ["1", "3", "4"]  # section 2 duplicated section 1; short "(בוטל)" kept
    assert normalized_body("h\n\nנֶחֱזּוּת, עמוקה!") == "נחזותעמוקה"


def test_mmr_skips_a_near_duplicate_for_a_different_relevant_chunk():
    a = np.array([1.0, 0.0, 0.0])
    a_copy = np.array([0.99, 0.141, 0.0]) / np.linalg.norm([0.99, 0.141, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    pool = [("a", "", {}, 0.10, a), ("a2", "", {}, 0.11, a_copy), ("b", "", {}, 0.30, b)]
    picked = [c[0] for c in _mmr(pool, 2, 0.7)]
    assert picked == ["a", "b"]
