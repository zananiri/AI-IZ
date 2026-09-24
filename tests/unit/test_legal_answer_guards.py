"""The checks between the draft and the answer: removing sentences whose
citations failed, a garbled source_id, words in the wrong script, sections the
index doesn't hold, and what the eval judge is shown."""

from docslides.legal.citations import expand_citations, parse_citations, remove_cited_sentences
from docslides.legal.evaluation import with_citations
from docslides.legal.models import ChunkMetadata
from docslides.legal.retrieval import _holds
from docslides.legal.script_check import allowed_words, foreign_words, replace_words


def _cite(n: int, source: str = "law-a@2026-05-11:3") -> str:
    return f"[[CITE: claim_id=C{n} | source_id={source} | relation=supports]]"


def test_removing_a_citation_takes_its_sentence_and_the_tokens_attached_to_it():
    text = f"First. {_cite(1)} Second, unsupported. {_cite(2)}{_cite(3)}\nThird. {_cite(4)} tail"
    cleaned, removed = remove_cited_sentences(text, {2})
    assert removed == {1, 2}  # C3 sits right after C2: the same sentence
    assert cleaned == f"First. {_cite(1)}\nThird. {_cite(4)} tail"
    assert [c.claim_id for c in parse_citations(cleaned)] == ["C1", "C4"]


def test_a_source_id_with_a_garbled_version_date_resolves_to_the_one_match():
    meta = ChunkMetadata(
        chunk_id="x", source_id="law-a@2026-05-11:3", section_key="x", law_id="law-a", law_name="חוק", chapter=None,
        part=None, section_number="3", subsection_number=None, breadcrumb="", effective_date_start="2026-05-11",
        effective_date_end=None, status="current", source_type="statute", source_origin="knesset",
        ingestion_date="2026-09-24", language="he",
    )
    evidence = {"law-a@2026-05-11:3": meta, "law-a@2026-05-11:4": meta}
    assert parse_citations(expand_citations(_cite(1, "law-a@2026-11:3"), evidence))[0].source_id == "law-a@2026-05-11:3"
    assert parse_citations(expand_citations(_cite(1, "law-b@2026-11:3"), evidence))[0].source_id == "law-b@2026-11:3"


def test_foreign_words_flags_mixed_and_wrong_script_words_but_not_evidence_terms():
    allowed = allowed_words(["הודעה באמצעות WhatsApp"])
    answer = f"השר מ報導 ב октяבר, הugo של חוזה simultaneously דרך WhatsApp בשנת 2026. {_cite(1)}"
    assert foreign_words(answer, "he", allowed) == ["מ報導", "октяבר", "הugo", "simultaneously"]
    assert foreign_words("Under חוק הבחירות the minister 報導", "en", allowed_words(["חוק הבחירות לכנסת"])) == ["報導"]


def test_replace_words_swaps_whole_words_outside_citation_tokens_only():
    text = f"ugo מ報導 בזמן. {_cite(1, 'ugo@1:1')}"
    assert replace_words(text, {"מ報導": "מדווח", "ugo": ""}) == f" מדווח בזמן. {_cite(1, 'ugo@1:1')}"


def test_a_subsection_only_referred_to_by_an_amendment_is_not_held():
    inserted = {"section_number": "1", "inserted_section": "25(א)(5)"}
    refers = "הסכמה הנוגדת את הוראות סעיף קטן (ב1) בטלה."
    assert not _holds("25", "ב1", refers, inserted)
    assert _holds("25", "א", refers, inserted)
    assert _holds("25", "ב1", "25. (א) ...\n(ב1) נוסח הסעיף", {"section_number": "25"})  # the whole section
    assert not _holds("25", "ב1", "(א) ...", {"section_number": "25", "subsection_number": "א"})
    amending = {"section_number": "7", "subsection_number": "4"}
    assert _holds("62", "ג", 'בסעיף 62(ג), במקום "ה־40" יבוא "ה־43"', amending)
    assert _holds("116יז10", "ד", "", {"section_number": "6", "inserted_section": "116יז10(ד)"})


def test_the_judge_sees_which_provisions_an_answer_cites():
    assert with_citations("Jerusalem.", []) == "Jerusalem."
    shown = with_citations("Jerusalem.", ["חוק העמדה לדין, section 4"])
    assert shown.endswith("Provisions cited in the answer:\n- חוק העמדה לדין, section 4")
