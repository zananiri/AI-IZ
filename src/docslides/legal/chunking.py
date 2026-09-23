"""Structure-first chunking for Israeli legal sources (ingestion spec 0.1).

One chunk per section (סעיף). A section over the soft token budget
(config.legal.ingestion.chunk_max_tokens) is split at its top-level
subsections; a subsection -- or an unsubdivided section -- that is still
over budget is split into sibling parts at line/sentence boundaries. Parts
share one `source_id` (the citation ID) and get distinct `chunk_id`s. A
single sentence longer than the budget stays whole: the budget is a soft
target, structural and sentence boundaries always win, and nothing is ever
truncated.

Every chunk's text starts with a breadcrumb (law > part > chapter > section)
so the embedding carries the provision's context, not just its words.

IDs: `version_id` = "<law_id>@<effective_date_start>", so several versions
of the same law can sit side by side in the index; `section_key` =
"<version_id>:<section>"; `source_id` adds "(<subsection>)" when a section
was split at subsections; `chunk_id` adds "#p<n>" for multi-part provisions.

Cross-references only ever point at this law's own sections: in an amending
section, "סעיף N" is the amended law's section N and links nothing here.
Hebrew chunks are stored with ״ in place of the ASCII double quote
(`normalize_hebrew_quotes`).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from docslides.cleaning.tokens import count_tokens
from docslides.config import get_config
from docslides.legal import amendments
from docslides.legal.models import ChunkMetadata, LegalChunk, SourceOrigin, SourceType, Status
from docslides.legal.structure import Section, extract_cross_references
from docslides.logging_setup import get_logger

logger = get_logger(__name__)

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.;!?])\s+")
_SHORT_INTRO_TOKENS = 40
_GERSHAYIM = "״"  # U+05F4 HEBREW PUNCTUATION GERSHAYIM


def normalize_hebrew_quotes(text: str) -> str:
    """`text` with ״ (gershayim) in place of every ASCII double quote.

    Hebrew legal text uses the ASCII quote both inside abbreviations
    (התשכ"ט, כ"ב, ס"ח) and around quoted wording. A model that copies one
    into a JSON string without escaping it ends the string there, and
    grammar-constrained decoding then closes the object -- silently cutting
    a memo claim or an answer mid-word. ״ needs no escaping and is the
    proper Hebrew character (prompts.format_evidence already writes it in
    attribute values)."""
    return text.replace('"', _GERSHAYIM)


@dataclass
class SourceMeta:
    """Document-level metadata every chunk inherits (sidecar .meta.json for
    official sources; defaults for uploads -- see legal/sources.py)."""

    law_id: str
    law_name: str
    effective_date_start: str
    status: Status
    source_type: SourceType
    source_origin: SourceOrigin
    effective_date_end: str | None = None
    language: str = "he"
    gazette: str | None = None

    @property
    def version_id(self) -> str:
        return f"{self.law_id}@{self.effective_date_start}"


@dataclass
class _Piece:
    subsection: str | None
    text: str


def _section_label(language: str, number: str) -> str:
    if number == "preamble":
        return "מבוא" if language == "he" else "Preamble"
    return f"{'סעיף' if language == 'he' else 'Section'} {number}"


def _breadcrumb(meta: SourceMeta, section: Section, subsection: str | None) -> str:
    provision = _section_label(meta.language, section.number)
    if subsection:
        provision += f"({subsection})"
    if section.title:
        provision += f" — {section.title}"
    path = [meta.law_name, section.division, section.chapter, section.subchapter, provision]
    return " > ".join(p for p in path if p)


def _units(text: str) -> list[str]:
    units: list[str] = []
    for line in text.splitlines():
        units.extend(s.strip() for s in _SENTENCE_BOUNDARY_RE.split(line) if s.strip())
    return units


def _pack(text: str, budget: int) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in _units(text):
        unit_tokens = count_tokens(unit)
        if current and current_tokens + unit_tokens > budget:
            parts.append("\n".join(current))
            current, current_tokens = [], 0
        current.append(unit)
        current_tokens += unit_tokens
    if current:
        parts.append("\n".join(current))
    return parts


def _pieces(section: Section, budget: int, header_tokens: int) -> list[_Piece]:
    if count_tokens(section.text) + header_tokens <= budget or not section.subsections:
        return [_Piece(None, section.text)]
    pieces: list[_Piece] = []
    intro_is_context = count_tokens(section.intro) <= _SHORT_INTRO_TOKENS if section.intro else False
    if section.intro and not intro_is_context:
        pieces.append(_Piece(None, section.intro))
    for sub in section.subsections:
        text = f"{section.intro}\n{sub.text}" if intro_is_context else sub.text
        pieces.append(_Piece(sub.label, text))
    return pieces


def _part_suffix(language: str, index: int, count: int) -> str:
    return f" (חלק {index} מתוך {count})" if language == "he" else f" (part {index} of {count})"


def chunk_sections(sections: list[Section], meta: SourceMeta, ingestion_date: str) -> list[LegalChunk]:
    budget = get_config().legal.ingestion.chunk_max_tokens
    known_sections = {s.number for s in sections}
    toc = amendments.parse_toc(next((s.text for s in sections if s.number == "preamble"), ""))
    key = amendments.law_key(meta.law_name)
    chunks: list[LegalChunk] = []

    for section in sections:
        section_key = f"{meta.version_id}:{section.number}"
        header_tokens = count_tokens(_breadcrumb(meta, section, None))

        for piece in _pieces(section, budget, header_tokens):
            breadcrumb = _breadcrumb(meta, section, piece.subsection)
            source_id = f"{section_key}({piece.subsection})" if piece.subsection else section_key
            # Same rule parse_sections applies to Section.cross_refs: an amending section's
            # "סעיף 2" is section 2 of the law it amends, not of this one.
            amending = section.number != "preamble" and (
                (section.title or "").startswith("תיקון")
                or amendments.extract_amendment(section.title, piece.text, toc) is not None
            )
            refs = [] if amending else [
                f"{meta.version_id}:{n}"
                for n in extract_cross_references(piece.text, section.number)
                if n in known_sections
            ]
            if count_tokens(piece.text) + count_tokens(breadcrumb) <= budget:
                bodies = [piece.text]
            else:
                bodies = _pack(piece.text, max(budget - count_tokens(breadcrumb), 1))

            for index, body in enumerate(bodies, start=1):
                multipart = len(bodies) > 1
                amended = amendments.extract_amendment(section.title, body, toc) if section.number != "preamble" else None
                header = breadcrumb + (_part_suffix(meta.language, index, len(bodies)) if multipart else "")
                text = f"{header}\n\n{body}"
                stored_breadcrumb = breadcrumb
                if meta.language == "he":
                    text, stored_breadcrumb = normalize_hebrew_quotes(text), normalize_hebrew_quotes(breadcrumb)
                chunks.append(
                    LegalChunk(
                        text=text,
                        metadata=ChunkMetadata(
                            chunk_id=f"{source_id}#p{index}" if multipart else source_id,
                            source_id=source_id,
                            section_key=section_key,
                            law_id=meta.law_id,
                            law_name=meta.law_name,
                            chapter=section.chapter,
                            part=section.subchapter or section.division,
                            section_number=section.number,
                            subsection_number=piece.subsection,
                            breadcrumb=stored_breadcrumb,
                            effective_date_start=meta.effective_date_start,
                            effective_date_end=meta.effective_date_end,
                            status=meta.status,
                            source_type=meta.source_type,
                            source_origin=meta.source_origin,
                            ingestion_date=ingestion_date,
                            language=meta.language,
                            part_index=index,
                            part_count=len(bodies),
                            cross_references=refs,
                            gazette=meta.gazette,
                            law_key=key,
                            amends=amendments.encode([amended]) if amended else [],
                        ),
                    )
                )
    return _dedupe(chunks)


_MIN_DEDUPE_WORDS = 12  # short bodies ("(בוטל)") legitimately repeat


def normalized_body(text: str) -> str:
    """A chunk's body without its breadcrumb header, with points, punctuation
    and spacing removed -- what "the same provision" means for dedupe (here and
    at query time in legal/retrieval.py)."""
    body = text.split("\n\n", 1)[-1]
    body = re.sub("[\u0591-\u05C7]", "", body)
    return re.sub(r"[\W_]+", "", body)


def _dedupe(chunks: list[LegalChunk]) -> list[LegalChunk]:
    """Drops chunks whose normalized body repeats an earlier one (a PDF that
    printed a page twice), and makes chunk ids unique should the parser ever
    emit the same provision number twice -- Chroma rejects duplicate ids."""
    seen_bodies: set[str] = set()
    seen_ids: dict[str, int] = {}
    kept: list[LegalChunk] = []
    for chunk in chunks:
        body = normalized_body(chunk.text)
        if len(chunk.text.split("\n\n", 1)[-1].split()) >= _MIN_DEDUPE_WORDS:
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            if digest in seen_bodies:
                logger.warning("legal_duplicate_chunk_dropped", chunk_id=chunk.metadata.chunk_id)
                continue
            seen_bodies.add(digest)
        chunk_id = chunk.metadata.chunk_id
        if chunk_id in seen_ids:
            seen_ids[chunk_id] += 1
            suffix = f"~{seen_ids[chunk_id]}"
            logger.warning("legal_duplicate_chunk_id", chunk_id=chunk_id)
            chunk.metadata.chunk_id += suffix
            chunk.metadata.source_id += suffix
        else:
            seen_ids[chunk_id] = 1
        kept.append(chunk)
    return kept
