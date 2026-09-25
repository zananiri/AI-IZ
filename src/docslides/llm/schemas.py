"""JSON schemas exchanged with the LLM.

Every content-generation call is constrained to one of these schemas via
guided decoding (xgrammar/outlines through vLLM's `guided_json`). We never
parse free-text model output with regex -- if it isn't valid against one of
these models, the call is retried with an error-correction prompt.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, create_model

LayoutType = Literal["title_bullets", "two_column", "section_header", "image_caption", "quote"]


class SlidePlan(BaseModel):
    title: str
    intent: str = Field(description="What this slide should communicate, one short phrase")
    target_bullet_count: int = Field(ge=0, le=8)
    layout_type: LayoutType


class OutlineResult(BaseModel):
    """Stage 1 output: the full slide-by-slide plan for a document."""

    slides: list[SlidePlan]


class SlideContent(BaseModel):
    """Stage 2 output: fully generated content for a single slide."""

    title: str
    bullets: list[str]
    notes: str = ""
    layout_type: LayoutType


class GlossaryTerm(BaseModel):
    source_term: str
    target_term: str
    notes: str = ""


class GlossaryExtraction(BaseModel):
    """Extracted per-chunk during translation to keep terminology consistent."""

    terms: list[GlossaryTerm]


class TranslatedChunk(BaseModel):
    translated_text: str
    new_terms: list[GlossaryTerm] = Field(default_factory=list)


class ChunkSummary(BaseModel):
    """Used to compress chunks into the outline stage's context budget for
    long documents -- see slides/outline.py."""

    summary: str
    key_points: list[str] = Field(default_factory=list)


class ChatIntent(BaseModel):
    """Classifies whether a chat turn with an attached document wants a full
    slide deck generated -- the only request type that needs the dedicated
    pipeline (translate -> outline -> fill -> PPTX assembly) rather than a
    normal chat response with the document's text as context. See
    api/routes_chat.py."""

    wants_slides: bool = Field(
        description="True only if the user explicitly asked for a PowerPoint/slide deck/presentation"
    )
    target_lang: str | None = Field(
        default=None,
        description="ISO 639-1 code for the language the user asked the output in, if any (e.g. 'fr', 'es')",
    )


LegalSourceType = Literal["statute", "regulation", "ruling", "uploaded_document"]


class LegalIssue(BaseModel):
    issue_id: str = Field(description="'I1', 'I2', ...")
    question: str
    legal_domain: str


class LegalFact(BaseModel):
    fact_id: str = Field(description="'F1', 'F2', ...")
    text: str
    source: Literal["user_input"] = "user_input"


class GoverningLawClaim(BaseModel):
    claim_id: str = Field(description="'C1', 'C2', ... -- the only claim IDs the draft may cite")
    text: str = Field(description="The legal proposition")
    issue_id: str
    fact_ids: list[str] = Field(
        default_factory=list, description="fact_ids from facts_relied_on this proposition applies to, if any"
    )


class SupportingAuthority(BaseModel):
    claim_id: str
    source_id: str = Field(description="Exactly as given in the retrieved evidence metadata")
    law: str
    section: str
    effective: str
    source_type: LegalSourceType
    # "pipeline" when legal/validation.ground_memorandum attached the source the claim quotes
    attached_by: Literal["model", "pipeline"] = "model"


class ContraryAuthority(SupportingAuthority):
    note: str = Field(description="How this source qualifies, limits or contradicts the claim")


class ResearchMemorandum(BaseModel):
    """Legal tab Pass A (Qwen): the claim -> evidence graph that the draft is
    only allowed to cite from. Checked by legal/validation.py's gate before
    Pass B may run."""

    issues: list[LegalIssue]
    facts_relied_on: list[LegalFact] = Field(default_factory=list)
    governing_law: list[GoverningLawClaim]
    supporting_authority: list[SupportingAuthority] = Field(default_factory=list)
    contrary_authority: list[ContraryAuthority] = Field(default_factory=list)
    contrary_search_performed: bool
    unresolved_questions: list[str] = Field(default_factory=list)
    temporal_issues: list[str] = Field(default_factory=list)
    authority_conflicts: list[str] = Field(default_factory=list)


_SOURCE_IDS_DESCRIPTION = (
    "source_id of every evidence item that states this proposition, exactly as in the evidence -- at least one"
)


class GroundedClaim(BaseModel):
    claim_id: str = Field(description="'C1', 'C2', ... -- the only claim IDs the draft may cite")
    text: str = Field(description="The legal proposition")
    issue_id: str
    fact_ids: list[str] = Field(
        default_factory=list, description="fact_ids from facts_relied_on this proposition applies to, if any"
    )
    source_ids: list[str] = Field(
        default_factory=list, description=_SOURCE_IDS_DESCRIPTION, json_schema_extra={"minItems": 1}
    )


class GroundedContrary(BaseModel):
    claim_id: str
    source_id: str
    note: str = Field(description="How this source qualifies, limits or contradicts the claim")


class GroundedMemorandum(BaseModel):
    """Pass A as the model writes it. Each claim names its own sources, and the
    model never copies law / section / effective / source_type -- that is
    what small models got wrong or skipped, leaving supporting_authority empty
    and failing the gate. legal/validation.ground_memorandum turns this into
    the ResearchMemorandum everything downstream uses."""

    issues: list[LegalIssue]
    facts_relied_on: list[LegalFact] = Field(default_factory=list)
    governing_law: list[GroundedClaim]
    contrary_authority: list[GroundedContrary] = Field(default_factory=list)
    contrary_search_performed: bool
    unresolved_questions: list[str] = Field(default_factory=list)
    temporal_issues: list[str] = Field(default_factory=list)
    authority_conflicts: list[str] = Field(default_factory=list)


def grounded_memorandum_schema(source_ids: list[str]) -> type[GroundedMemorandum]:
    """GroundedMemorandum with every source_id limited to this turn's retrieved
    sources, and each claim required to name at least one: constrained
    decoding can then neither invent a source nor leave a claim unsourced.
    (minItems is in the JSON schema only, so a backend that doesn't enforce
    it still parses -- ground_memorandum handles that case.)"""
    if not source_ids:
        return GroundedMemorandum
    source_id = Literal[tuple(source_ids)]
    claim = create_model(
        "GroundedClaim",
        __base__=GroundedClaim,
        source_ids=(list[source_id], Field(default_factory=list, description=_SOURCE_IDS_DESCRIPTION,
                                           json_schema_extra={"minItems": 1})),
    )
    contrary = create_model("GroundedContrary", __base__=GroundedContrary, source_id=(source_id, ...))
    return create_model(
        "GroundedMemorandum",
        __base__=GroundedMemorandum,
        governing_law=(list[claim], ...),
        contrary_authority=(list[contrary], Field(default_factory=list)),
    )


class LegalDraft(BaseModel):
    """Legal tab Pass B (Qwen): prose in the reply language with inline
    [[CITE: ...]] tokens referencing Pass A claim IDs only."""

    answer_draft: str
    escalation_flag: bool
    escalation_reason: str | None = None
    coverage_gaps: str | None = None


class EntailmentVerdict(BaseModel):
    """Legal tab citation verification: does the cited evidence actually
    establish the stated relation (supports / contrary) to the claim?

    Verdict first: the explanation-first order tried on 25 Sept 2026 made the
    8B verifier reject correct sentences far more often (11% -> 26%)."""

    verdict: Literal["entailed", "partially_entailed", "not_entailed"]
    explanation: str


class WordRepair(BaseModel):
    word: str = Field(description="The flagged word, exactly as listed")
    replacement: str = Field(description="What it should read in the answer language; empty to drop it")


class ScriptRepair(BaseModel):
    """Legal tab: the answer-language word for each word written in another
    script (legal/script_check.py)."""

    repairs: list[WordRepair]


class EvalJudgement(BaseModel):
    """scripts/eval_legal.py: grades one answer against its gold answer.
    explanation first: the judge compares the facts before it gives a verdict
    (see EntailmentVerdict)."""

    explanation: str = Field(description="Compare the answer's essential facts with the gold answer's, briefly")
    verdict: Literal["correct", "partially_correct", "incorrect", "abstained"]
    fabricated_specifics: bool = Field(
        description="True if the answer states a specific number, date, amount or rule that is not in the gold "
        "answer. The laws and sections it cites don't count."
    )


class EvalContradiction(BaseModel):
    """scripts/eval_legal.py: the narrow second question asked before an answer
    holding every key fact may be graded down. The quote comes before the verdict."""

    conflict: str = Field(description="The conflicting statement, quoted, or empty if none")
    contradicts_gold: bool = Field(
        description="True only if the answer states something that conflicts with a fact in the gold answer"
    )


class ReplyLanguage(BaseModel):
    language: str = Field(description="ISO 639-1 code of the language the question is written in")


SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "outline_result": OutlineResult,
    "slide_content": SlideContent,
    "glossary_extraction": GlossaryExtraction,
    "translated_chunk": TranslatedChunk,
    "chunk_summary": ChunkSummary,
    "chat_intent": ChatIntent,
    "research_memorandum": ResearchMemorandum,
    "legal_draft": LegalDraft,
    "entailment_verdict": EntailmentVerdict,
    "reply_language": ReplyLanguage,
    "eval_judgement": EvalJudgement,
}
