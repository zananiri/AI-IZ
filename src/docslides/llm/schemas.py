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


SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "outline_result": OutlineResult,
    "slide_content": SlideContent,
    "glossary_extraction": GlossaryExtraction,
    "translated_chunk": TranslatedChunk,
    "chunk_summary": ChunkSummary,
}
