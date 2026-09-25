"""End-to-end Legal pipeline runs against scripted fake models and a fake
retrieval result: exercises routing, the Pass A gate, citation
verification and the audit log without any model server."""

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
            if call_site.name == "legal_analysis":  # Pass 0 is optional: no notes unless scripted
                return ""
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

    def install(qwen):
        monkeypatch.setattr(pipeline, "get_legal_orchestrator_client", lambda: qwen)
        return queries

    return install


async def _noop_status(message):
    pass


def _run(query):
    return asyncio.run(pipeline.run_legal_turn(query, "job1", _noop_status))


def _audit(result):
    return json.loads(Path(result.audit_path).read_text(encoding="utf-8").splitlines()[-1])


def test_hebrew_turn_revises_memo_and_answers_with_the_verified_draft(wire):
    draft_he = f"הצד הטועה רשאי לבטל את החוזה. {CITE}"
    qwen = FakeClient(
        "qwen",
        json_responses={
            "legal_research_memo": [{**GOOD_MEMO, "contrary_search_performed": False}, GOOD_MEMO],
            "legal_draft": [{"answer_draft": draft_he, "escalation_flag": False}],
            "legal_citation_verification": [{"verdict": "entailed", "explanation": "סעיף 14(א)"}],
        },
    )
    queries = wire(qwen)

    result = _run("אפשר לבטל חוזה שחתמתי בטעות?")

    assert result.reply_language == "he"
    assert queries == ["אפשר לבטל חוזה שחתמתי בטעות?"]  # retrieval searches the question as asked
    assert "legal_language_id" not in qwen.names()  # Hebrew detected by script, no model call
    assert qwen.names().count("legal_research_memo") == 2  # gate sent the first memo back
    assert result.output["answer_draft"] == draft_he
    assert result.footnotes[0]["verified"] and result.footnotes[0]["section"] == "14"
    assert not result.output["escalation_flag"]

    entry = _audit(result)
    assert entry["orchestrator_model"] == "qwen"
    assert len(entry["memorandum_attempts"]) == 2 and entry["memorandum_attempts"][0]["errors"]
    assert entry["final_integrity"]["structural_problems"] == {}


def test_english_turn_withholds_a_draft_whose_only_citation_fails(wire):
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
    wire(qwen)

    result = _run("Can I cancel a contract I signed by mistake?")

    assert result.reply_language == "en"
    assert qwen.names().count("legal_draft") == 2  # one redraft after the failed verification
    assert result.display_answer.startswith("The drafted answer could not be verified")
    assert result.output["escalation_flag"] and result.footnotes == []
    assert any("no answer was given" in r and "invented:99" in r for r in result.escalation_reasons)
    entry = _audit(result)
    assert "rescind" in entry["withheld_draft"] and entry["removed_sentences"][0]["source_id"] == "invented:99"


def test_memo_failing_gate_after_revisions_escalates_without_drafting(wire):
    bad = {**GOOD_MEMO, "contrary_search_performed": False}
    qwen = FakeClient(
        "qwen",
        json_responses={"legal_language_id": [{"language": "fr"}], "legal_research_memo": [bad, bad, bad]},
    )
    wire(qwen)

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
    wire(qwen)

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
    wire(qwen)

    result = _run("Can I cancel a contract I signed by mistake?")
    assert result.output["answer_draft"] == f"A mistaken party may rescind. {CITE}"
    assert result.footnotes[0]["verified"] and not result.output["escalation_flag"]

    uncited = _run("Can I cancel a contract I signed by mistake?")
    assert uncited.output["escalation_flag"]
    assert any("no [[CITE]] tokens" in r for r in uncited.escalation_reasons)
    assert uncited.display_answer.startswith("The drafted answer could not be verified")  # nothing verified to give


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
    wire(qwen)

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
    wire(qwen)

    result = _run("Can I cancel a contract I signed by mistake?")

    assert result.output["research_memorandum"]["supporting_authority"]  # attempt 1 kept
    assert len([r for r in result.escalation_reasons if "has no supporting_authority" in r]) == 0


# --- answer guards: unverified sentences, wrong-script words, several laws, missing sections ----------

TWO_CLAIM_MEMO = {
    **GOOD_MEMO,
    "governing_law": [*GOOD_MEMO["governing_law"],
                      {"claim_id": "C2", "text": "Rescission is possible for ten years.", "issue_id": "I1",
                       "fact_ids": [], "source_ids": [SOURCE]}],
    "unresolved_questions": [*GOOD_MEMO["unresolved_questions"], "C2: searched for contrary authority; none found"],
}
CITE_C2 = CITE.replace('claim_id="C1"', 'claim_id="C2"')

LAW2 = 'חוק העמדה לדין בשל אירועי טבח 7 באוקטובר 2023, התשפ"ו-2026'
SOURCE2 = "oct7@2026-05-11:25"


def _two_law_retrieval(**extra) -> RetrievalResult:
    result = _retrieval()
    meta = ChunkMetadata(
        chunk_id=SOURCE2, source_id=SOURCE2, section_key=SOURCE2, law_id="oct7", law_name=LAW2, chapter=None,
        part=None, section_number="25", subsection_number=None, breadcrumb=f"{LAW2} > סעיף 25",
        effective_date_start="2026-05-11", effective_date_end=None, status="current", source_type="statute",
        source_origin="knesset", ingestion_date="2026-09-23", language="he",
    )
    result.chunks.append(RetrievedLegalChunk(SOURCE2, f"{LAW2} > סעיף 25\n\nדיון בהיוועדות חזותית.", meta, 0.3,
                                             "section_lookup"))
    for key, value in extra.items():
        setattr(result, key, value)
    return result


def test_a_sentence_whose_source_does_not_state_it_is_removed(wire):
    draft = f"הצד הטועה רשאי לבטל את החוזה. {CITE} הביטול אפשרי במשך עשר שנים. {CITE_C2}"
    qwen = FakeClient(
        "qwen",
        json_responses={
            "legal_research_memo": [TWO_CLAIM_MEMO],
            "legal_draft": [{"answer_draft": draft, "escalation_flag": False}] * 2,
            "legal_citation_verification": [
                {"verdict": "entailed", "explanation": "14(א)"}, {"verdict": "not_entailed", "explanation": "no term"},
            ] * 2,
        },
    )
    wire(qwen)

    result = _run("אפשר לבטל חוזה שחתמתי בטעות?")

    assert result.output["answer_draft"] == f"הצד הטועה רשאי לבטל את החוזה. {CITE}"
    assert [n["verified"] for n in result.footnotes] == [True]
    assert any("Removed 1 statement" in r for r in result.escalation_reasons)
    assert _audit(result)["removed_sentences"][0]["sentence"] == "הביטול אפשרי במשך עשר שנים."


def test_words_in_another_script_are_repaired_word_by_word(wire):
    draft = f"הצד הטועה מ報導 רשאי לבטל את החוזה simultaneously. {CITE}"
    qwen = FakeClient(
        "qwen",
        json_responses={
            "legal_research_memo": [GOOD_MEMO],
            "legal_draft": [{"answer_draft": draft, "escalation_flag": False}],
            "legal_citation_verification": [{"verdict": "entailed", "explanation": "14(א)"}],
            # The second replacement is itself foreign, so it isn't applied.
            "legal_script_repair": [{"repairs": [{"word": "מ報導", "replacement": "מדווח"},
                                                 {"word": "simultaneously", "replacement": "at once"}]}],
        },
    )
    wire(qwen)

    result = _run("אפשר לבטל חוזה שחתמתי בטעות?")

    assert result.output["answer_draft"] == f"הצד הטועה מדווח רשאי לבטל את החוזה simultaneously. {CITE}"
    assert any("another script remain" in r and "simultaneously" in r for r in result.escalation_reasons)
    assert _audit(result)["script_check"]["flagged"] == ["מ報導", "simultaneously"]


def test_draft_fields_echoed_into_the_answer_text_are_dropped(wire):
    draft = (f"הצד הטועה רשאי לבטל את החוזה. {CITE}\n\n escalated_flag: true\n escalation_reason: נדרש תאריך.\n"
             " coverage_gaps: אין מידע.")
    qwen = FakeClient(
        "qwen",
        json_responses={
            "legal_research_memo": [GOOD_MEMO],
            "legal_draft": [{"answer_draft": draft, "escalation_flag": False}],
            "legal_citation_verification": [{"verdict": "entailed", "explanation": "14(א)"}],
        },
    )
    wire(qwen)

    result = _run("אפשר לבטל חוזה שחתמתי בטעות?")

    assert result.output["answer_draft"] == f"הצד הטועה רשאי לבטל את החוזה. {CITE}"
    assert "legal_script_repair" not in qwen.names()  # nothing left to repair
    assert len(_audit(result)["dropped_field_lines"]) == 3


def test_an_answer_leaving_out_a_law_in_play_says_so_up_front(wire, monkeypatch):
    qwen = FakeClient(
        "qwen",
        json_responses={
            # Asked (twice) to add a claim for the second law; it never does.
            "legal_research_memo": [GOOD_MEMO] * 3,
            "legal_draft": [{"answer_draft": f"הצד הטועה רשאי לבטל את החוזה. {CITE}", "escalation_flag": False}],
            "legal_citation_verification": [{"verdict": "entailed", "explanation": "14(א)"}],
        },
    )
    wire(qwen)
    monkeypatch.setattr(pipeline, "retrieve", lambda q: _two_law_retrieval(laws_in_play=[LAW, LAW2]))

    result = _run("מה קובע סעיף 25?")

    memo_calls = [messages for name, messages in qwen.calls if name == "legal_research_memo"]
    assert len(memo_calls) == 3 and f"no claim cites {LAW2}" in memo_calls[1][-1].content
    assert qwen.names().count("legal_draft") == 1  # the memorandum has nothing on LAW2 to cite
    assert result.output["answer_draft"].startswith("השאלה אינה חד־משמעית")
    assert f"אינה עוסקת ב{LAW2}" in result.output["answer_draft"].split("\n\n")[0]
    assert any("leaves out" in r and LAW2 in r for r in result.escalation_reasons)


def test_a_draft_citing_one_law_of_two_the_memo_covers_is_revised(wire, monkeypatch):
    memo = {**TWO_CLAIM_MEMO, "governing_law": [TWO_CLAIM_MEMO["governing_law"][0],
                                                {**TWO_CLAIM_MEMO["governing_law"][1], "source_ids": [SOURCE2]}]}
    cite2 = CITE_C2.replace(SOURCE, SOURCE2).replace(LAW, LAW2).replace('section="14"', 'section="25"')
    both = f"השאלה אינה חד־משמעית. בחוק החוזים: ניתן לבטל. {CITE} בחוק העמדה לדין: דיון בהיוועדות. {cite2}"
    qwen = FakeClient(
        "qwen",
        json_responses={
            "legal_research_memo": [memo],
            "legal_draft": [{"answer_draft": f"ניתן לבטל. {CITE}", "escalation_flag": False},
                            {"answer_draft": both, "escalation_flag": False}],
            "legal_citation_verification": [{"verdict": "entailed", "explanation": "ok"}] * 3,
        },
    )
    wire(qwen)
    monkeypatch.setattr(pipeline, "retrieve", lambda q: _two_law_retrieval(laws_in_play=[LAW, LAW2]))

    result = _run("מה קובע סעיף 25?")

    revision = [messages for name, messages in qwen.calls if name == "legal_draft"][1][-1].content
    assert f"the memorandum has claims for {LAW2}" in revision
    assert result.output["answer_draft"] == both and not result.output["escalation_flag"]


def test_a_named_section_missing_from_the_index_is_said_up_front(wire, monkeypatch):
    qwen = FakeClient(
        "qwen",
        json_responses={
            "legal_research_memo": [GOOD_MEMO],
            "legal_draft": [{"answer_draft": f"הצד הטועה רשאי לבטל את החוזה. {CITE}", "escalation_flag": False}],
            "legal_citation_verification": [{"verdict": "entailed", "explanation": "14(א)"}],
        },
    )
    wire(qwen)
    missing = _retrieval()
    missing.missing_sections = ["25(ב1)"]
    monkeypatch.setattr(pipeline, "retrieve", lambda q: missing)

    result = _run("מה קובע סעיף 25(ב1) לחוק החוזים?")

    memo_prompt = next(messages for name, messages in qwen.calls if name == "legal_research_memo")[1].content
    assert "does not hold the text of section 25(ב1)" in memo_prompt
    assert result.output["answer_draft"].startswith("נוסח סעיף 25(ב1) אינו נמצא במאגר")
    assert any("not in the index: 25(ב1)" in r for r in result.escalation_reasons)


# --- Pass 0 notes, JSON-safe quotes, a draft that can't be generated or ran away, a section in several laws ---


def test_analysis_notes_reach_the_memo_and_the_draft(wire):
    qwen = FakeClient(
        "qwen",
        json_responses={
            "legal_research_memo": [GOOD_MEMO],
            "legal_draft": [{"answer_draft": f"הצד הטועה רשאי לבטל את החוזה. {CITE}", "escalation_flag": False}],
            "legal_citation_verification": [{"verdict": "entailed", "explanation": "14(א)"}],
        },
        text_responses={"legal_analysis": ['Direct answer: yes -- "רשאי לבטל" (14(א)).']},
    )
    wire(qwen)

    result = _run("אפשר לבטל חוזה שחתמתי בטעות?")

    assert qwen.names()[:2] == ["legal_analysis", "legal_research_memo"]
    for name in ("legal_research_memo", "legal_draft"):
        prompt = next(messages for n, messages in qwen.calls if n == name)[1].content
        assert "Analysis notes" in prompt and "״רשאי לבטל״" in prompt  # its quotes can't end a JSON string
    assert result.analysis_notes.startswith("Direct answer: yes")
    assert _audit(result)["analysis_notes"] == result.analysis_notes


def test_the_drafter_sees_the_memo_with_json_safe_quotes(wire):
    claim = 'במקום "7 ימים" יקראו "30 ימים".'
    quoted = {**GOOD_MEMO, "governing_law": [{**GOOD_MEMO["governing_law"][0], "text": claim}]}
    qwen = FakeClient(
        "qwen",
        json_responses={
            "legal_research_memo": [quoted],
            "legal_draft": [{"answer_draft": f"הצד הטועה רשאי לבטל את החוזה. {CITE}", "escalation_flag": False}],
            "legal_citation_verification": [{"verdict": "entailed", "explanation": "14(א)"}],
        },
    )
    wire(qwen)

    result = _run("אפשר לבטל חוזה שחתמתי בטעות?")

    draft_prompt = next(messages for n, messages in qwen.calls if n == "legal_draft")[1].content
    assert "״30 ימים״" in draft_prompt and '\\"30' not in draft_prompt
    assert result.output["research_memorandum"]["governing_law"][0]["text"] == claim  # stored as written


def test_a_draft_that_cannot_be_generated_escalates_instead_of_failing_the_turn(wire):
    qwen = FakeClient(
        "qwen",
        json_responses={"legal_language_id": [{"language": "en"}], "legal_research_memo": [GOOD_MEMO]},
    )  # no legal_draft response: every draft call fails
    wire(qwen)

    result = _run("Can I cancel a contract I signed by mistake?")

    assert result.display_answer.startswith("The drafted answer could not be verified")
    assert any("No well-formed draft" in r for r in result.escalation_reasons)
    assert _audit(result)["draft_attempts"][0]["draft"] is None


def test_a_draft_cut_off_in_a_loop_keeps_each_sentence_once_up_to_its_last_citation():
    short = f"[[CITE: claim_id=C1 | source_id={SOURCE} | relation=supports]]"
    raw = '{"answer_draft": "ניתן לבטל את החוזה. ' + short + " ניתן לבטל את החוזה. " + short + " ניתן לב"
    draft = pipeline._salvage_draft(raw)
    assert draft.answer_draft == f"ניתן לבטל את החוזה. {short}"
    assert draft.escalation_flag and "length limit" in draft.escalation_reason
    assert pipeline._salvage_draft('{"answer_draft": "nothing cited yet') is None
    assert pipeline._salvage_draft("not json at all") is None


def test_a_named_section_several_laws_have_is_flagged_up_front(wire, monkeypatch):
    memo = {**TWO_CLAIM_MEMO, "governing_law": [TWO_CLAIM_MEMO["governing_law"][0],
                                                {**TWO_CLAIM_MEMO["governing_law"][1], "source_ids": [SOURCE2]}]}
    cite2 = CITE_C2.replace(SOURCE, SOURCE2).replace(LAW, LAW2).replace('section="14"', 'section="25"')
    both = f"בחוק החוזים ניתן לבטל חוזה שנכרת בטעות. {CITE} בחוק העמדה לדין הדיון מתקיים בהיוועדות. {cite2}"
    qwen = FakeClient(
        "qwen",
        json_responses={
            "legal_research_memo": [memo],
            "legal_draft": [{"answer_draft": both, "escalation_flag": False}],
            "legal_citation_verification": [{"verdict": "entailed", "explanation": "ok"}] * 2,
        },
    )
    wire(qwen)
    monkeypatch.setattr(pipeline, "retrieve", lambda q: _two_law_retrieval(
        laws_in_play=[LAW, LAW2], ambiguous_sections={"25": [LAW, LAW2]}))

    result = _run("מה קובע סעיף 25?")

    memo_prompt = next(messages for n, messages in qwen.calls if n == "legal_research_memo")[1].content
    assert "section 25, which the question names, exists in several laws" in memo_prompt
    opening = result.output["answer_draft"].split("\n\n")[0]
    assert opening.startswith("השאלה אינה חד־משמעית: סעיף 25 מופיע ביותר מחוק אחד")
    assert opening.endswith("להלן מה שקובע כל אחד מהם.")
    assert result.output["answer_draft"].endswith(cite2)
    assert not any("leaves out" in r for r in result.escalation_reasons)
