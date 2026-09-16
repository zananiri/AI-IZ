"""JSON schemas exchanged with the LLM.

Every content-generation call is constrained to one of these schemas via
guided decoding (xgrammar/outlines through vLLM's `guided_json`). We never
parse free-text model output with regex -- if it isn't valid against one of
these models, the call is retried with an error-correction prompt.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

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


class LegalQueryPlan(BaseModel):
    """Stage 1 (orchestrator) output for the Legal tab: reformulates the
    user's question into a precise Hebrew legal-research query for the
    Hebrew-analyst model -- see legal/pipeline.py."""

    hebrew_query: str = Field(
        description="The user's legal question, translated and reformulated into clear, precise Hebrew, "
        "self-contained and ready to hand to an Israeli-law legal analysis model"
    )
    topic_summary: str = Field(description="One short phrase (in English) naming the legal topic/area, for status display")


class HebrewLegalFindings(BaseModel):
    """Stage 2 (Hebrew analyst / DictaLM) output for the Legal tab: the
    analysis itself, with citations and relevant laws kept as separate
    fields (rather than embedded in prose) so they can be shown in their own
    panel and carried through verification unchanged."""

    analysis_hebrew: str = Field(description="The full legal analysis and answer, in Hebrew")
    citations: list[str] = Field(default_factory=list, description="Case citations / legal sources relied on, in Hebrew")
    relevant_laws: list[str] = Field(default_factory=list, description="Relevant statutes/laws/sections relied on, in Hebrew")


class LegalFinalAnswer(BaseModel):
    """Stage 3 (orchestrator, verification) output for the Legal tab: the
    answer translated into the user's own language and checked against the
    Hebrew findings; citations/laws are carried through as-is (in Hebrew)."""

    answer: str = Field(description="The final answer to the user, in the user's own language")
    citations: list[str] = Field(default_factory=list)
    relevant_laws: list[str] = Field(default_factory=list)


SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "outline_result": OutlineResult,
    "slide_content": SlideContent,
    "glossary_extraction": GlossaryExtraction,
    "translated_chunk": TranslatedChunk,
    "chunk_summary": ChunkSummary,
    "chat_intent": ChatIntent,
    "legal_query_plan": LegalQueryPlan,
    "hebrew_legal_findings": HebrewLegalFindings,
    "legal_final_answer": LegalFinalAnswer,
}
