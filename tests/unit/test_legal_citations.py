from docslides.legal.citations import (
    expand_citations,
    format_citation,
    parse_citations,
    render_with_footnotes,
    sentence_before,
)
from docslides.legal.models import ChunkMetadata
from docslides.legal.validation import (
    AUTO_NOTE_MARK,
    check_draft_citations,
    clean_memorandum,
    record_contrary_search_notes,
    validate_memorandum,
)
from docslides.llm.schemas import ResearchMemorandum

LAW = 'חוק החוזים (חלק כללי), תשל"ג-1973'  # contains an ASCII double quote on purpose


def cite(claim="C1", source="law@1973:14", relation="supports", source_type="statute"):
    return format_citation(
        claim_id=claim, source_id=source, law=LAW, section="14", effective="current",
        source_type=source_type, relation=relation,
    )


def meta(source_id="law@1973:14", section="14", source_type="statute"):
    return ChunkMetadata(
        chunk_id=source_id, source_id=source_id, section_key=source_id, law_id="law", law_name=LAW, chapter=None,
        part=None, section_number=section, subsection_number=None, breadcrumb="", effective_date_start="1973-06-01",
        effective_date_end=None, status="current", source_type=source_type, source_origin="knesset",
        ingestion_date="2026-09-23", language="he",
    )


def memo(**overrides):
    base = {
        "issues": [{"issue_id": "I1", "question": "q", "legal_domain": "contracts"}],
        "facts_relied_on": [{"fact_id": "F1", "text": "mistake"}],
        "governing_law": [{"claim_id": "C1", "text": "A mistaken party may rescind.", "issue_id": "I1", "fact_ids": ["F1"]}],
        "supporting_authority": [
            {"claim_id": "C1", "source_id": "law@1973:14", "law": LAW, "section": "14", "effective": "current",
             "source_type": "statute"}
        ],
        "contrary_authority": [],
        "contrary_search_performed": True,
        "unresolved_questions": ["C1: searched for contrary authority, none found in the evidence"],
    }
    base.update(overrides)
    return ResearchMemorandum.model_validate(base)


def test_parse_handles_quotes_inside_law_names():
    text = f"A party may rescind. {cite()} Done."
    [citation] = parse_citations(text)
    assert citation.law == LAW
    assert citation.source_id == "law@1973:14"
    assert citation.relation == "supports"


def test_render_numbers_footnotes_by_source():
    text = f"One. {cite()} Two. {cite('C2')} Three. {cite('C3', 'law@1973:3')}"
    display, _, numbers = render_with_footnotes(text)
    assert numbers == [1, 1, 2]
    assert "[[CITE" not in display and "[1]" in display and "[2]" in display


def test_sentence_before_token():
    text = f"Intro sentence. The mistaken party may rescind. {cite()} Next."
    [citation] = parse_citations(text)
    assert sentence_before(text, citation.start) == "The mistaken party may rescind."


def test_valid_memorandum_passes_gate():
    assert validate_memorandum(memo(), {"law@1973:14": meta()}) == []


def test_gate_catches_invented_source_wrong_type_and_missing_contrary_search():
    bad = memo(
        supporting_authority=[
            {"claim_id": "C1", "source_id": "invented:1", "law": LAW, "section": "1", "effective": "current",
             "source_type": "statute"},
        ],
        contrary_search_performed=False,
        unresolved_questions=[],
    )
    errors = validate_memorandum(bad, {"law@1973:14": meta()})
    assert any("contrary_search_performed" in e for e in errors)
    assert any("never invent a source_id" in e for e in errors)
    assert any("no contrary_authority" in e for e in errors)

    wrong_type = memo(
        supporting_authority=[
            {"claim_id": "C1", "source_id": "law@1973:14", "law": LAW, "section": "14", "effective": "current",
             "source_type": "ruling"},
        ]
    )
    assert any("carried through unchanged" in e for e in validate_memorandum(wrong_type, {"law@1973:14": meta()}))


def test_unsupported_claim_must_be_listed_as_unresolved():
    errors = validate_memorandum(memo(supporting_authority=[], unresolved_questions=[]), {})
    assert any("C1 has no supporting_authority" in e for e in errors)


def test_draft_citation_structural_checks():
    evidence = {"law@1973:14": meta()}
    good = f"A mistaken party may rescind. {cite()}"
    assert check_draft_citations(good, memo(), evidence) == {}

    bad = f"New claim. {cite('C7')} Wrong relation. {cite(relation='contrary')} Bad type. {cite(source_type='ruling')}"
    problems = check_draft_citations(bad, memo(), evidence)
    assert "not established in Pass A" in problems[0][0]
    assert "contrary authority" in problems[1][0]
    assert "differs from the source" in problems[2][0]


def test_expand_short_form_fills_from_metadata_and_keeps_conflicting_values():
    evidence = {"law@1973:14": meta()}
    short = "May rescind. [[CITE: claim_id=C1 | source_id=law@1973:14 | relation=supports]]"
    [citation] = parse_citations(expand_citations(short, evidence))
    assert (citation.law, citation.section, citation.effective, citation.source_type) == (LAW, "14", "current", "statute")

    wrong = "May rescind. [[CITE: claim_id=C1 | source_id=law@1973:14 | source_type=ruling | relation=supports]]"
    assert parse_citations(expand_citations(wrong, evidence))[0].source_type == "ruling"  # left for the check to flag
    unknown = "X. [[CITE: claim_id=C1 | source_id=nope:1 | relation=supports]]"
    assert parse_citations(expand_citations(unknown, evidence))[0].law == ""


def test_bare_claim_id_is_not_an_explanation_and_junk_entries_are_dropped():
    gamed = memo(supporting_authority=[], unresolved_questions=["C1", "],"])
    errors = validate_memorandum(clean_memorandum(gamed), {})
    assert any("C1 has no supporting_authority" in e for e in errors)
    assert clean_memorandum(memo(authority_conflicts=["],", " real conflict "])).authority_conflicts == ["real conflict"]


def test_missing_contrary_note_is_auto_recorded_only_when_the_search_was_affirmed():
    bare = memo(unresolved_questions=[])
    fixed, added = record_contrary_search_notes(bare)
    assert len(added) == 1 and AUTO_NOTE_MARK in added[0]
    assert validate_memorandum(fixed, {"law@1973:14": meta()}) == []

    not_searched, added = record_contrary_search_notes(memo(unresolved_questions=[], contrary_search_performed=False))
    assert added == [] and validate_memorandum(not_searched, {"law@1973:14": meta()})

    unsupported, added = record_contrary_search_notes(memo(unresolved_questions=[], supporting_authority=[]))
    assert added == []  # an unsupported claim still has to be listed as unresolved by the model
    assert any("no supporting_authority" in e for e in validate_memorandum(unsupported, {}))


def test_a_citation_on_a_fragment_or_a_lead_in_is_flagged():
    evidence = {"law@1973:14": meta()}
    for fragment in ("ובנוסף,", "שונו מספרים בהתאם לחוקים הבאים:"):
        problems = check_draft_citations(f"{fragment} {cite()}", memo(), evidence)
        assert "doesn't state the claim" in problems[0][-1]
    two_tokens = f"A mistaken party may rescind. {cite()}{cite()}"  # the second shares the first's sentence
    assert check_draft_citations(two_tokens, memo(), evidence) == {}


def test_an_authority_conflict_that_denies_any_conflict_is_dropped():
    cleaned = clean_memorandum(memo(authority_conflicts=["אין סתירה בין סעיף 3 לסעיף 5", "No conflict found.",
                                                         "סעיף 3 סותר את סעיף 5"]))
    assert cleaned.authority_conflicts == ["סעיף 3 סותר את סעיף 5"]
