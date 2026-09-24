"""End-to-end Legal pipeline runs against scripted fake models and a fake
retrieval result: exercises routing, the Pass A gate, citation
verification, the citation lock, polish rejection/retry and the audit log
without any model server."""

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from docslides.config import get_config
from docslides.legal import pipeline
from docslides.legal.citations import format_citation
from docslides.legal.models import ChunkMetadata
from docslides.legal.retrieval import RetrievalResult, RetrievedLegalChunk
from docslides.llm import schemas

LAW = 'חוק החוזים (חלק כללי), תשל"ג-1973'
SOURCE = "contracts@1973-06-01:14"
CHUNK_TEXT = f"{LAW} > סעיף 14 — טעות\n\n(א) מי שהתקשר בחוזה עקב טעות רשאי לבטל את החוזה."
CITE = format_citation(
    claim_id="C1", source_id=SOURCE, law=LAW, section="14", effective="current",
    source_type="statute", relation="supports",
)

GOOD_MEMO = {
    "issues": [{"issue_id": "I1", "question": "rescission for mistake", "legal_domain": "contracts"}],
    "facts_relied_on": [{"fact_id": "F1", "text": "the user contracted by mistake"}],
    "governing_law": [{"claim_id": "C1", "text": "A party who contracted by mistake may rescind.", "issue_id": "I1",
                       "fact_ids": ["F1"], "source_ids": [SOURCE]}],
    "contrary_authority": [],
    "contrary_search_performed": True,
    "unresolved_questions": ["C1: searched the evidence for contrary authority; none found"],
}


class FakeClient:
    """Pops scripted responses per call site; records every call."""

    def __init__(self, model, json_responses=None, text_responses=None):
        self.model = model
        self.json_responses = {k: list(v) for k, v in (json_responses or {}).items()}
        self.text_responses = {k: list(v) for k, v in (text_responses or {}).items()}
        self.calls = []

    async def complete_json(self, messages, call_site, schema, sampling=None, **_):
        self.calls.append((call_site.name, messages))
        queue = self.json_responses.get(call_site.name)
        if not queue:
            raise AssertionError(f"unexpected JSON call to {call_site.name}")
        return schema.model_validate(queue.pop(0))

    async def complete_text(self, messages, call_site, sampling=None, **_):
        self.calls.append((call_site.name, messages))
        queue = self.text_responses.get(call_site.name)
        if not queue:
            raise AssertionError(f"unexpected text call to {call_site.name}")
        return queue.pop(0)

    def names(self):
        return [name for name, _ in self.calls]


def _retrieval() -> RetrievalResult:
    meta = ChunkMetadata(
        chunk_id=SOURCE, source_id=SOURCE, section_key=SOURCE, law_id="contracts", law_name=LAW,
        chapter="פרק ב'", part=None, section_number="14", subsection_number=None, breadcrumb=f"{LAW} > סעיף 14",
        effective_date_start="1973-06-01", effective_date_end=None, status="current", source_type="statute",
        source_origin="knesset", ingestion_date="2026-09-23", language="he",
    )
    chunk = RetrievedLegalChunk(SOURCE, CHUNK_TEXT, meta, 0.2, "search")
    return RetrievalResult(chunks=[chunk], low_relevance=False, best_distance=0.2, bundle_verification="signed")


@pytest.fixture
def wire(monkeypatch, tmp_path):
    monkeypatch.setattr(get_config().legal, "audit_dir", str(tmp_path))
    queries = []

    def fake_retrieve(query):
        queries.append(query)
        return _retrieval()

    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(pipeline, "amendment_index", dict)  # never touch the real vector DB

    def install(qwen, dicta):
        monkeypatch.setattr(pipeline, "get_legal_orchestrator_client", lambda: qwen)
        monkeypatch.setattr(pipeline, "get_legal_dicta_client", lambda tier: dicta)
        return queries

    return install


async def _noop_status(message):
    pass


def _run(query, tier="heavy"):
    return asyncio.run(pipeline.run_legal_turn(query, tier, "job1", _noop_status))


def _audit(result):
    return json.loads(Path(result.audit_path).read_text(encoding="utf-8").splitlines()[-1])


def test_hebrew_turn_normalizes_revises_memo_and_rejects_tag_breaking_polish(wire):
    draft_he = f"הצד הטועה רשאי לבטל את החוזה. {CITE}"
    qwen = FakeClient(
        "qwen",
        json_responses={
            "legal_research_memo": [{**GOOD_MEMO, "contrary_search_performed": False}, GOOD_MEMO],
            "legal_draft": [{"answer_draft": draft_he, "escalation_flag": False}],
            "legal_citation_verification": [{"verdict": "entailed", "explanation": "סעיף 14(א)"}],
            "legal_equivalence_check": [{"equivalent": True, "discrepancies": []}],
        },
    )
    dicta = FakeClient(
        "dicta-24b",
        text_responses={
            "legal_query_normalization": ["האם ניתן לבטל חוזה שנכרת עקב טעות?"],
            "legal_hebrew_polish": [
                "הצד שטעה רשאי לבטל את החוזה.",  # dropped the citation tag -> rejected
                "הצד שטעה רשאי לבטל את החוזה. [[CITE:1]]",
            ],
        },
    )
    queries = wire(qwen, dicta)

    result = _run("אפשר לבטל חוזה שחתמתי בטעות?")

    assert result.reply_language == "he"
    assert queries == ["האם ניתן לבטל חוזה שנכרת עקב טעות?"]  # retrieval used Dicta's normalized form
    assert "legal_language_id" not in qwen.names()  # Hebrew detected by script, no model call
    assert qwen.names().count("legal_research_memo") == 2  # gate sent the first memo back
    assert result.polish_status == "accepted"
    assert result.output["answer_draft"] == f"הצד שטעה רשאי לבטל את החוזה. {CITE}"
    assert result.footnotes[0]["verified"] and result.footnotes[0]["section"] == "14"
    assert not result.output["escalation_flag"]
    # Dicta only ever received the Hebrew instructions, and the re-polish request is Hebrew too.
    system_prompts = [messages[0].content for _, messages in dicta.calls]
    assert all(p.startswith("אתה ") for p in system_prompts)
    assert dicta.calls[-1][1][1].content.startswith("הליטוש הקודם נדחה")

    entry = _audit(result)
    assert entry["dicta_tier"] == "heavy" and entry["dicta_used"] is True
    assert len(entry["memorandum_attempts"]) == 2 and entry["memorandum_attempts"][0]["errors"]
    assert entry["polish_attempts"][0]["lock_problems"]
    assert entry["final_integrity"]["tokens_match_lock"] is True


def test_non_equivalent_polish_falls_back_to_verified_draft(wire):
    draft_he = f"בית המשפט רשאי לבטל את החוזה. {CITE}"
    discrepancy = {"location": "משפט 1", "pre_polish": "רשאי", "post_polish": "חייב", "issue": "שינוי מאפשרות לחובה"}
    qwen = FakeClient(
        "qwen",
        json_responses={
            "legal_research_memo": [GOOD_MEMO],
            "legal_draft": [{"answer_draft": draft_he, "escalation_flag": False}],
            "legal_citation_verification": [{"verdict": "entailed", "explanation": "ok"}],
            "legal_equivalence_check": [{"equivalent": False, "discrepancies": [discrepancy]}] * 2,
        },
    )
    dicta = FakeClient(
        "dicta-24b",
        text_responses={
            "legal_query_normalization": ["האם בית המשפט יכול לבטל חוזה?"],
            "legal_hebrew_polish": ["בית המשפט חייב לבטל את החוזה. [[CITE:1]]"] * 2,
        },
    )
    wire(qwen, dicta)

    result = _run("האם בית המשפט יכול לבטל חוזה?")

    assert result.polish_status == "rejected_used_draft"
    assert result.output["answer_draft"] == draft_he
    assert result.notes


def test_english_turn_skips_dicta_and_escalates_unverifiable_citation(wire):
    invented = CITE.replace(SOURCE, "invented:99")
    qwen = FakeClient(
        "qwen",
        json_responses={
            "legal_language_id": [{"language": "en"}],
            "legal_research_memo": [GOOD_MEMO],
            "legal_draft": [
                {"answer_draft": f"A mistaken party may rescind. {invented}", "escalation_flag": False},
                {"answer_draft": f"A mistaken party may rescind. {invented}", "escalation_flag": False},
            ],
        },
    )
    dicta = FakeClient("dicta-24b")  # any call raises
    wire(qwen, dicta)

    result = _run("Can I cancel a contract I signed by mistake?")

    assert result.reply_language == "en"
    assert dicta.calls == [] and not result.dicta_used
    assert qwen.names().count("legal_draft") == 2  # one redraft after the failed verification
    assert result.output["escalation_flag"]
    assert any("could not be verified" in r for r in result.escalation_reasons)
    assert result.footnotes[0]["verified"] is False


def test_memo_failing_gate_after_revisions_escalates_without_drafting(wire):
    bad = {**GOOD_MEMO, "contrary_search_performed": False}
    qwen = FakeClient(
        "qwen",
        json_responses={"legal_language_id": [{"language": "fr"}], "legal_research_memo": [bad, bad, bad]},
    )
    wire(qwen, FakeClient("dicta"))

    result = _run("Puis-je annuler un contrat signé par erreur ?")

    assert "legal_draft" not in qwen.names()
    assert result.output["escalation_flag"]
    assert result.display_answer.startswith("Je n'ai pas pu")


def test_memo_schema_limits_each_claim_to_retrieved_sources():
    turn_schema = schemas.grounded_memorandum_schema([SOURCE])
    wire_schema = json.dumps(turn_schema.model_json_schema(), ensure_ascii=False)
    assert SOURCE in wire_schema and '"minItems": 1' in wire_schema
    invented = {**GOOD_MEMO, "governing_law": [{**GOOD_MEMO["governing_law"][0], "source_ids": ["made-up:1"]}]}
    with pytest.raises(ValidationError):
        turn_schema.model_validate(invented)


def test_a_claim_naming_no_source_gets_the_evidence_it_quotes(wire):
    quoting = {**GOOD_MEMO, "governing_law": [{**GOOD_MEMO["governing_law"][0], "source_ids": [],
                                              "text": "מי שהתקשר בחוזה עקב טעות רשאי לבטל את החוזה."}]}
    qwen = FakeClient(
        "qwen",
        json_responses={
            "legal_language_id": [{"language": "en"}],
            "legal_research_memo": [quoting],
            "legal_draft": [{"answer_draft": f"A mistaken party may rescind. {CITE}", "escalation_flag": False}],
            "legal_citation_verification": [{"verdict": "entailed", "explanation": "ok"}],
        },
    )
    wire(qwen, FakeClient("dicta"))

    result = _run("Can I cancel a contract I signed by mistake?")

    support = result.output["research_memorandum"]["supporting_authority"]
    assert [(a["source_id"], a["attached_by"], a["section"], a["law"]) for a in support] == [
        (SOURCE, "pipeline", "14", LAW)
    ]
    assert qwen.names().count("legal_research_memo") == 1 and not result.output["escalation_flag"]


def test_short_form_citation_is_expanded_and_a_citationless_draft_escalates(wire):
    short = f"[[CITE: claim_id=C1 | source_id={SOURCE} | relation=supports]]"
    qwen = FakeClient(
        "qwen",
        json_responses={
            "legal_language_id": [{"language": "en"}, {"language": "en"}],
            "legal_research_memo": [GOOD_MEMO, GOOD_MEMO],
            "legal_draft": [
                {"answer_draft": f"A mistaken party may rescind. {short}", "escalation_flag": False},
                {"answer_draft": "A mistaken party may rescind.", "escalation_flag": False},
                {"answer_draft": "A mistaken party may rescind.", "escalation_flag": False},
            ],
            "legal_citation_verification": [{"verdict": "entailed", "explanation": "ok"}],
        },
    )
    wire(qwen, FakeClient("dicta"))

    result = _run("Can I cancel a contract I signed by mistake?")
    assert result.output["answer_draft"] == f"A mistaken party may rescind. {CITE}"
    assert result.footnotes[0]["verified"] and not result.output["escalation_flag"]

    uncited = _run("Can I cancel a contract I signed by mistake?")
    assert uncited.output["escalation_flag"]
    assert any("no [[CITE]] tokens" in r for r in uncited.escalation_reasons)


def test_memo_revisions_start_from_the_best_attempt_not_the_last(wire):
    # One problem the gate can't auto-fix: a claim linked to a fact that doesn't exist.
    nearly = {**GOOD_MEMO, "governing_law": [{**GOOD_MEMO["governing_law"][0], "fact_ids": ["F9"]}]}
    worse = {**nearly, "governing_law": [{**nearly["governing_law"][0], "source_ids": []}],
             "authority_conflicts": ["C1 lists no source_ids and is not listed in unresolved_questions"]}
    qwen = FakeClient(
        "qwen",
        json_responses={
            "legal_language_id": [{"language": "en"}],
            "legal_research_memo": [nearly, worse, GOOD_MEMO],
            "legal_draft": [{"answer_draft": f"A mistaken party may rescind. {CITE}", "escalation_flag": False}],
            "legal_citation_verification": [{"verdict": "entailed", "explanation": "ok"}],
        },
    )
    wire(qwen, FakeClient("dicta"))

    result = _run("Can I cancel a contract I signed by mistake?")

    memo_calls = [messages for name, messages in qwen.calls if name == "legal_research_memo"]
    revised_from = memo_calls[2][2].content  # the assistant turn the third attempt was asked to fix
    assert f'"source_ids":["{SOURCE}"]' in revised_from.replace(" ", "")  # attempt 1, not the worse attempt 2
    assert not result.output["escalation_flag"]
    assert result.output["research_memorandum"]["authority_conflicts"] == []


def test_failed_gate_keeps_the_attempt_with_fewest_problems(wire):
    nearly = {**GOOD_MEMO, "governing_law": [{**GOOD_MEMO["governing_law"][0], "fact_ids": ["F9"]}]}
    worse = {**nearly, "governing_law": [{**nearly["governing_law"][0], "source_ids": []}], "unresolved_questions": []}
    qwen = FakeClient(
        "qwen",
        json_responses={"legal_language_id": [{"language": "en"}], "legal_research_memo": [nearly, worse, worse]},
    )
    wire(qwen, FakeClient("dicta"))

    result = _run("Can I cancel a contract I signed by mistake?")

    assert result.output["research_memorandum"]["supporting_authority"]  # attempt 1 kept
    assert len([r for r in result.escalation_reasons if "has no supporting_authority" in r]) == 0
