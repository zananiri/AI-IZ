from types import SimpleNamespace

from docslides.legal import amendments
from docslides.legal.models import ChunkMetadata
from docslides.legal.prompts import format_evidence
from docslides.legal.retrieval import RetrievedLegalChunk

PREAMBLE = """תיקונים עקיפים:
חוק הבחירות לכנסת [נוסח משולב], התשכ"ט-1969 - מס' 79
חוק הבחירות לכנסת [נוסח משולב], התשכ"ט-1969 - מס' 80
חוק הבחירות (דרכי תעמולה), התשי"ט-1959 - הוראת שעה - מס' 43
חוק המפלגות, התשנ"ב-1992 - מס' 31"""
TOC = amendments.parse_toc(PREAMBLE)
ELECTIONS = 'חוק הבחירות לכנסת [נוסח משולב], התשכ"ט-1969'


def test_law_key_ignores_consolidation_tag_year_and_punctuation():
    assert amendments.law_key(ELECTIONS) == "חוק הבחירות לכנסת"
    assert amendments.law_key('חוק הבחירות לכנסת, התשכ"ט-1969') == "חוק הבחירות לכנסת"
    assert amendments.law_key('חוק הבחירות (דרכי תעמולה), התשי"ט-1959') == "חוק הבחירות דרכי תעמולה"


def test_toc_parsing():
    assert (ELECTIONS, "79", False) in TOC
    assert ('חוק הבחירות (דרכי תעמולה), התשי"ט-1959', "43", True) in TOC


def test_amendment_from_title_resolves_short_name_and_sections():
    ref = amendments.extract_amendment(
        "תיקון חוק הבחירות לכנסת - מס' 80",
        "header > סעיף 7(4)\n\n(4) בסעיף 62(ג), במקום \"ה־40\" יבוא \"ה־43\"",
        TOC,
    )
    assert (ref.target, ref.number, ref.sections, ref.temporary) == (ELECTIONS, "80", ["62"], False)

    temp = amendments.extract_amendment(
        "תיקון חוק התעמולה - הוראת שעה - מס' 43", "h\n\n(2) אחרי סעיף 2א1 יבוא:\n2א2. (א) אדם המפרסם", TOC
    )
    assert temp.target_key == "חוק הבחירות דרכי תעמולה" and temp.temporary
    assert temp.sections == ["2א1", "2א2"]


def test_amendment_from_opening_words_and_non_amending_sections():
    ref = amendments.extract_amendment(None, 'h\n\nבחוק המפלגות, התשנ"ב-1992, בסעיף 28כה3(א)(2), אחרי', TOC)
    assert (ref.target_key, ref.number, ref.sections) == ("חוק המפלגות", "31", ["28כה3"])
    assert amendments.extract_amendment("מטרה", "h\n\nמטרתו של פרק זה לקבוע הוראות", TOC) is None


def _principal(section="62", start="1969-01-01"):
    return SimpleNamespace(law_id="elections", law_name=ELECTIONS, law_key="", section_number=section,
                           effective_date_start=start)


def _index(effective="2026-07-16", sections=("62",)):
    ref = amendments.AmendmentRef(ELECTIONS, "חוק הבחירות לכנסת", "80", list(sections), False)
    amending = {"law_id": "amending-2026", "law_name": "חוק הבחירות לכנסת העשרים ושש", "gazette": "ספר החוקים 3546",
                "effective_date_start": effective}
    return {"חוק הבחירות לכנסת": [(ref, amending)]}


def test_notes_are_dated_and_section_specific():
    [note] = amendments.notes_for(_principal("62(ג)"), _index())
    assert note.touches_section and note.effective == "2026-07-16"
    assert "this section" in note.describe() and "ספר החוקים 3546" in note.describe()

    [law_level] = amendments.notes_for(_principal("5"), _index())
    assert not law_level.touches_section and "sections 62" in law_level.describe()


def test_a_version_starting_after_the_amendment_is_not_flagged():
    assert amendments.notes_for(_principal(start="2026-08-01"), _index()) == []
    assert amendments.notes_for(_principal(start="2026-07-16"), _index()) == []


def test_notes_merge_without_mutating_the_index():
    index = _index()
    ref2 = amendments.AmendmentRef(ELECTIONS, "חוק הבחירות לכנסת", "80", ["63"], False)
    index["חוק הבחירות לכנסת"].append((ref2, index["חוק הבחירות לכנסת"][0][1]))
    [note] = amendments.notes_for(_principal("63"), index)
    assert note.touches_section and note.ref.sections == ["62", "63"]
    assert index["חוק הבחירות לכנסת"][0][0].sections == ["62"]


def test_evidence_carries_amended_by_and_amends():
    meta = ChunkMetadata(
        chunk_id="s62", source_id="s62", section_key="s62", law_id="elections", law_name=ELECTIONS, chapter=None,
        part=None, section_number="62", subsection_number=None, breadcrumb="", effective_date_start="1969-01-01",
        effective_date_end=None, status="current", source_type="statute", source_origin="knesset",
        ingestion_date="2026-09-23", language="he",
        amends=amendments.encode([amendments.AmendmentRef("חוק אחר", "חוק אחר", "3", ["1"], False)]),
    )
    notes = {"s62": amendments.notes_for(meta, _index())}
    text = format_evidence({"s62": [RetrievedLegalChunk("s62", "body", meta, 0.1, "search")]}, notes)
    assert 'amended_by="מס\' 80' in text and "this section" in text
    assert 'amends="חוק אחר — מס\' 3 — סעיפים 1"' in text
