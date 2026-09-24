"""Behaviour needed once several laws share the index: "applied with modifications"
is not an amendment, subsection replacements are insertions, scan artifacts are
repaired, multi-part questions are split, and the pipeline knows when a question
names a law or matches several."""

from docslides.legal import amendments, retrieval
from docslides.legal.insertions import find_insertion
from docslides.legal.pdf_text import _normalize_line
from docslides.legal.prompts import format_evidence

APPLIES = """הוראות חוק המעצרים יחולו על נאשם שהוגש נגדו כתב אישום לפי חוק זה, בשינויים המחויבים ובשינויים אלה:
(5) בסעיף 62(א), במקום "90 ימים" יקראו "150 ימים"."""


def test_applying_a_law_with_modifications_is_not_an_amendment_of_it():
    ref = amendments.extract_amendment("מעצר אחרי הגשת כתב אישום", APPLIES, [])
    assert (ref.target, ref.relation, ref.sections) == ("חוק המעצרים", "reads_as", ["62"])
    assert "not an amendment" in ref.describe()

    numbered = amendments.extract_amendment("תיקון חוק התעמולה - הוראת שעה - מס' 43",
                                            'יקראו את חוק התעמולה, כך: (1) בסעיף 2 ...', [("חוק התעמולה", "43", True)])
    assert numbered.relation == "amends"  # the gazette numbers it as an amendment

    index = amendments.build_index([{"amends": amendments.encode([ref]), "law_id": "oct7"}])
    assert index == {}  # never reported as "amended_by" on the Arrests Law


def test_replacing_a_subsection_is_an_insertion_split_into_its_paragraphs():
    insertion = find_insertion(
        'בחוק החוזים (חלק כללי), התשל"ג-1973, בסעיף 25, במקום\nסעיף קטן (א) יבוא:\n'
        '"(א) (1) אופן הפרשנות יהיה ככל שהסכימו הצדדים.\n(2) חוזה עסקי יפורש לפי לשונו.\n'
        '(5) לעניין סעיף קטן זה - (ב) לא יהיה תוקף להסכמה."\n'
    )
    (provision,) = insertion.provisions
    assert provision.number == "25(א)"
    assert provision.lines[0].startswith("(1) אופן הפרשנות")
    assert insertion.trailing == []


def test_mirrored_brackets_and_split_section_numbers_from_old_scans_are_repaired():
    assert _normalize_line("8 .)א(לענין הכנת הבחירות") == "8.(א)לענין הכנת הבחירות"
    assert _normalize_line("תשי\"ט­ 1959") == "תשי\"ט-1959"
    assert _normalize_line("בסעיף 21(ג) ו-(ד)") == "בסעיף 21(ג) ו-(ד)"  # correct brackets untouched


def test_two_part_questions_are_split_into_their_clauses():
    assert retrieval.question_parts("מתי יו״ר הכנסת יכול למנוע העסקה של עובד, והאם הוא יכול להאציל סמכות זו?") == [
        "מתי יו״ר הכנסת יכול למנוע העסקה של עובד", "האם הוא יכול להאציל סמכות זו"]
    assert retrieval.question_parts("מה קובע סעיף 25?") == []


def test_a_law_the_question_names_is_recognized(monkeypatch):
    names = {"c": 'חוק החוזים (חלק כללי) (תיקון מס\' 3), התשפ"ו-2026',
             "o": 'חוק העמדה לדין בשל אירועי טבח 7 באוקטובר 2023, התשפ"ו-2026'}
    monkeypatch.setattr(retrieval, "_derived_indexes", lambda: {"law_names": names})
    assert retrieval._named_law_ids("מה קובע סעיף 25(ב1) לחוק החוזים?") == {"c"}
    assert retrieval._named_law_ids("מה קובע סעיף 25?") == set()


def test_evidence_says_when_an_amended_law_is_not_in_the_index():
    from docslides.legal.models import ChunkMetadata
    from docslides.legal.retrieval import RetrievedLegalChunk

    ref = amendments.AmendmentRef('חוק החוזים (חלק כללי), התשל"ג-1973', "חוק החוזים חלק כללי", "3", ["25"], False)
    meta = ChunkMetadata(
        chunk_id="c:1", source_id="c:1", section_key="c:1", law_id="c", law_name="חוק", chapter=None, part=None,
        section_number="1", subsection_number=None, breadcrumb="חוק", effective_date_start="2026-01-05",
        effective_date_end=None, status="current", source_type="statute", source_origin="knesset",
        ingestion_date="2026-09-24", language="he", amends=amendments.encode([ref]),
    )
    text = format_evidence({"c:1": [RetrievedLegalChunk("c:1", "חוק > סעיף 1\n\nטקסט", meta, 0.3, "search")]},
                           {}, indexed_law_keys={"חוק העמדה לדין"})
    assert "its text is not in the index" in text


def test_inner_lists_of_the_same_marker_kind_stay_inside_their_subsection():
    from docslides.legal.structure import parse_sections

    text = ("1. מטרת החוק." + chr(10) + "25. (א) דיון יתקיים באופן שיבטיח את כל אלה:\n(1) ראשון;\n(2) (א) לפני הדיון תתקיים שיחה.\n"
            "(ב) במהלך הדיון תתאפשר שיחה.\n(3) שלישי;\n(7) פרוטוקול לא יאוחר מ־24 שעות;\n"
            "(ב) דיון יתקיים באולם שאושר.\n(ג) השר יקבע הוראות.")
    (section,) = [s for s in parse_sections(text) if s.number == "25"]
    assert [s.label for s in section.subsections] == ["א", "ב", "ג"]
    assert "24 שעות" in section.subsections[0].text and "במהלך הדיון" in section.subsections[0].text
