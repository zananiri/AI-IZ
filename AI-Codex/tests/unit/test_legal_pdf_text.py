"""Reshumot-style extraction problems, reproduced with hand-placed characters
(right-to-left lines, number runs, duplicated page copies) plus the
text-level fixes, chunk dedupe, MMR and the parser's quote/title handling."""

import numpy as np

from aicodex.legal import pdf_text
from aicodex.legal.chunking import SourceMeta, chunk_sections, normalized_body
from aicodex.legal.retrieval import _mmr
from aicodex.legal.structure import parse_sections


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


def test_parser_reads_spanish_articles_and_ignores_quoted_amendment_text():
    text = "\n".join([
        "TÍTULO I",
        "Disposiciones generales",
        "Artículo 4. Contratación de personal.",
        "1. Se podrá contratar personal temporal.",
        "2. El personal informará a su superior.",
        "Artículo 6. Modificación de la Ley Orgánica 5/1985, del Régimen Electoral General.",
        "La Ley Orgánica 5/1985 queda modificada como sigue:",
        "Uno. Se añade un artículo 24 bis, con la siguiente redacción:",
        "«Artículo 24 bis. Sesiones por videoconferencia.",
        "1. La junta podrá reunirse por videoconferencia,",
        "2. si todos sus miembros lo aceptan.»",
        "Dos. En el artículo 62, se sustituye «cuarenta» por «cuarenta y tres».",
        "Artículo 7. Otra disposición.",
        "Sin perjuicio de lo dispuesto en el artículo 4.",
    ])
    sections = {s.number: s for s in parse_sections(text)}
    assert list(sections) == ["4", "6", "7"]  # "24 bis" is quoted text, not an article of this law
    assert sections["4"].title == "Contratación de personal"
    assert sections["4"].division == "TÍTULO I. Disposiciones generales"
    assert [s.label for s in sections["4"].subsections] == ["1", "2"]
    assert "Sesiones por videoconferencia" in sections["6"].text
    assert sections["6"].cross_refs == []  # "artículo 62" in an amendment is the amended law's
    assert sections["7"].cross_refs == ["4"]


def test_duplicate_chunk_bodies_are_dropped_and_ids_stay_unique():
    body = "La administración no podrá despedir al trabajador por haber sido contratado por la junta electoral durante el proceso"
    text = f"Artículo 1.\n{body}\nArtículo 2.\n{body}\nArtículo 3.\n(Derogado)\nArtículo 4.\n(Derogado)"
    meta = SourceMeta(law_id="x", law_name="Ley", effective_date_start="2026-07-16", status="current",
                      source_type="statute", source_origin="boe")
    chunks = chunk_sections(parse_sections(text), meta, "2026-09-23")
    numbers = [c.metadata.section_number for c in chunks]
    assert numbers == ["1", "3", "4"]  # article 2 duplicated article 1; short "(Derogado)" kept
    assert normalized_body("h\n\nultrafalsificación, ¡profunda!") == "ultrafalsificaciónprofunda"


def test_mmr_skips_a_near_duplicate_for_a_different_relevant_chunk():
    a = np.array([1.0, 0.0, 0.0])
    a_copy = np.array([0.99, 0.141, 0.0]) / np.linalg.norm([0.99, 0.141, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    pool = [("a", "", {}, 0.10, a), ("a2", "", {}, 0.11, a_copy), ("b", "", {}, 0.30, b)]
    picked = [c[0] for c in _mmr(pool, 2, 0.7)]
    assert picked == ["a", "b"]
