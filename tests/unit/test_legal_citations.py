import pytest

from docslides.legal.citations import (
    CitationLockError,
    expand_citations,
    format_citation,
    lock,
    lock_problems,
    parse_citations,
    render_with_footnotes,
    sentence_before,
    unlock,
)
from docslides.legal.models import ChunkMetadata
from docslides.legal.resources import MemorySnapshot, TierStatus, evaluate_tier, suggestion
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


def test_lock_round_trip_and_tamper_detection():
    text = f"First claim. {cite()} Second claim. {cite('C2', 'law@1973:3', 'contrary')}"
    locked, citation_lock = lock(text)
    assert "[[CITE:1]]" in locked and "[[CITE:2]]" in locked and "claim_id" not in locked
    assert unlock(locked, citation_lock) == text
    assert unlock(locked.replace("[[CITE:2]]", "[[ CITE : 2 ]]"), citation_lock) == text  # spacing tolerated

    assert lock_problems(locked.replace("[[CITE:2]]", ""), citation_lock)
    swapped = locked.replace("[[CITE:1]]", "@@").replace("[[CITE:2]]", "[[CITE:1]]").replace("@@", "[[CITE:2]]")
    with pytest.raises(CitationLockError):
        unlock(swapped, citation_lock)
    with pytest.raises(CitationLockError):
        unlock(locked + " [[CITE: claim_id=\"C9\"]]", citation_lock)


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


def test_tier_suggestion_logic():
    snapshot = MemorySnapshot(ram_available_gb=6, ram_total_gb=16, vram_free_gb=4, vram_total_gb=8)

    class Tier:
        def __init__(self, backend, min_gb):
            self.min_memory_gb = min_gb
            self.llm = type("LLM", (), {"backend": backend})()

    assert evaluate_tier(Tier("ollama", 16), snapshot, loaded=False) == (10, False)
    assert evaluate_tier(Tier("ollama", 8), snapshot, loaded=False) == (10, True)
    assert evaluate_tier(Tier("vllm", 8), snapshot, loaded=False) == (4, False)
    assert evaluate_tier(Tier("ollama", 16), snapshot, loaded=True)[1] is True

    heavy = TierStatus("heavy", "Heavy", "24b", 16, 10, False, False)
    light = TierStatus("light", "Light", "12b", 8, 10, False, True)
    result = suggestion("heavy", [heavy, light])
    assert result["suggest"] == "light"
    assert result["message"].startswith("Insufficient RAM detected for the Heavy DictaLM model.")
    assert suggestion("light", [heavy, light]) == {"suggest": None, "message": None}


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
