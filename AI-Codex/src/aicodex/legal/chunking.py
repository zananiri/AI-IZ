"""Structure-first chunking for Spanish legal sources (ingestion spec 0.1).

One chunk per article (artículo) or disposition. An article over the soft token budget
(config.legal.ingestion.chunk_max_tokens) is split at its top-level apartados; an apartado -- or an
undivided article -- that is still over budget is split into sibling parts at line/sentence
boundaries. Parts share one `source_id` (the citation ID) and get distinct `chunk_id`s. A single
sentence longer than the budget stays whole: the budget is a soft target, structural and sentence
boundaries always win, and nothing is ever truncated.

Every chunk's text starts with a breadcrumb (law > Título > Capítulo > Artículo 14.2 — title) so
the embedding carries the provision's context, not just its words.

IDs: `version_id` = "<law_id>@<effective_date_start>", so several versions of the same law can sit
side by side in the index; `section_key` = "<version_id>:<article>"; `source_id` adds
"(<apartado>)" when an article was split at apartados; `chunk_id` adds "#p<n>" for multi-part
provisions.

Cross-references only ever point at this law's own articles: in an amending provision, "artículo N"
is the amended law's article N and links nothing here. ASCII double quotes are stored as « » / ”
(`normalize_json_quotes`), so a model copying evidence into a JSON string never ends it early.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import dataclass

from aicodex.cleaning.tokens import count_tokens
from aicodex.config import get_config
from aicodex.legal import amendments
from aicodex.legal.amendments import AmendmentRef
from aicodex.legal.insertions import Insertion, find_insertion, join_spaced_section_numbers
from aicodex.legal.models import ChunkMetadata, LegalChunk, SourceOrigin, SourceType, Status
from aicodex.legal.structure import (
    Section,
    _amends_other_law,
    _split_subsections,
    extract_cross_references,
    is_article_number,
)
from aicodex.logging_setup import get_logger

logger = get_logger(__name__)

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.;!?])\s+")
_SHORT_INTRO_TOKENS = 60  # an opening line this short is repeated as context in each subsection's chunk


def normalize_json_quotes(text: str) -> str:
    """`text` without ASCII double quotes: a quoted span becomes «...», a stray quote ”.

    A model that copies an ASCII quote from the evidence into a JSON string without escaping it
    ends the string there, and grammar-constrained decoding then closes the object -- silently
    cutting a memo claim or an answer mid-word. « » and ” need no escaping and are what Spanish
    legal text uses anyway."""
    if '"' not in text:
        return text
    text = re.sub(r'"([^"\n]{1,400})"', "«\\1»", text)
    return text.replace('"', "”")


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
    language: str = "es"
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
        return "Preámbulo" if language == "es" else "Preamble"
    if is_article_number(number):
        return f"{'Artículo' if language == 'es' else 'Article'} {number}"
    return number  # "Disposición adicional primera", "Anexo I"


def _breadcrumb(meta: SourceMeta, section: Section, subsection: str | None) -> str:
    provision = _section_label(meta.language, section.number)
    if subsection and section.number == "preamble":
        provision += f" — {subsection}"
    elif subsection:
        provision += f".{subsection}" if subsection.isdigit() else f", letra {subsection})"
    if section.title:
        provision += f" — {section.title}"
    path = [meta.law_name, section.division, section.chapter, section.subchapter, provision]
    return " > ".join(p for p in path if p)


def _units(text: str) -> list[str]:
    units: list[str] = []
    for line in text.splitlines():
        units.extend(s.strip() for s in _SENTENCE_BOUNDARY_RE.split(line) if s.strip())
    return units


def _pack(text: str, budget: int, overlap: int = 0) -> list[str]:
    """Sentence units packed into parts of at most `budget` tokens. With `overlap`, each
    part after the first starts with the previous part's last units, up to that many tokens."""
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in _units(text):
        unit_tokens = count_tokens(unit)
        if current and current_tokens + unit_tokens > budget:
            parts.append("\n".join(current))
            carried: list[str] = []
            carried_tokens = 0
            for previous in reversed(current):
                tokens = count_tokens(previous)
                if carried_tokens + tokens > overlap or carried_tokens + tokens + unit_tokens > budget:
                    break
                carried.insert(0, previous)
                carried_tokens += tokens
            current, current_tokens = carried, carried_tokens
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
    return f" (parte {index} de {count})" if language == "es" else f" (part {index} of {count})"


# The promulgation and signatures after a law's last provision ("Por tanto, Mando a todos los
# españoles...", "FELIPE R.", "El Presidente del Gobierno, PEDRO SÁNCHEZ PÉREZ-CASTEJÓN"), which
# otherwise end up in its last chunk.
_SIGNATORY_RE = re.compile(r"^(?:[A-ZÁÉÍÓÚÑ ]{3,40} R\.|El Presidente del Gobierno|La Presidenta del Gobierno|"
                           r"Por tanto,|Mando a todos los|Dado en .{2,40}, a)")
_SENTENCE_END = (".", ";", ":")
_NUMBERED_ITEM_RE = re.compile(r"^(?:Uno|Dos|Tres|Cuatro|Cinco|Seis|Siete|Ocho|Nueve|Diez|\d{1,3})\.\s")  # "Dos. Se modifica ..."


def _strip_signature_block(text: str) -> str:
    lines = text.splitlines()
    for j, line in enumerate(lines):
        if j > 0 and _SIGNATORY_RE.match(line.strip()):
            return "\n".join(lines[:j]).rstrip()
    return text


def _strip_last_section_signature(section: Section) -> None:
    section.text = _strip_signature_block(section.text)
    section.intro = _strip_signature_block(section.intro)
    if section.subsections:
        section.subsections[-1].text = _strip_signature_block(section.subsections[-1].text)


def _inserts_label(language: str, inserted: str, target: str) -> str:
    if language == "es":
        return f"da nueva redacción al artículo {inserted}" + (f" de {target}" if target else "")
    return f"inserts article {inserted}" + (f" into {target}" if target else "")


def _provision_units(lines: list[str], room: int) -> list[tuple[str | None, str]]:
    """An inserted provision as (subsection label, text) units: whole if it
    fits `room`, else split at its top-level subsections (a short opening
    line stays with each) -- the same rule chunk_sections applies to a law's
    own sections."""
    text = "\n".join(lines)
    if count_tokens(text) <= room:
        return [(None, text)]
    intro, subsections = _split_subsections(lines)
    if not subsections:
        return [(None, text)]
    units: list[tuple[str | None, str]] = []
    intro_is_context = bool(intro) and count_tokens(intro) <= _SHORT_INTRO_TOKENS
    if intro and not intro_is_context:
        units.append((None, intro))
    for sub in subsections:
        units.append((sub.label, f"{intro}\n{sub.text}" if intro_is_context else sub.text))
    return units


def chunk_sections(
    sections: list[Section], meta: SourceMeta, ingestion_date: str, budget: int | None = None, overlap_tokens: int = 0
) -> list[LegalChunk]:
    """`budget` defaults to legal.ingestion.chunk_max_tokens; `overlap_tokens` repeats the end of
    one part of a split provision at the start of the next (the corpus index uses it)."""
    budget = budget or get_config().legal.ingestion.chunk_max_tokens
    known_sections = {s.number for s in sections}
    toc = amendments.parse_toc(next((s.text for s in sections if s.number == "preamble"), ""))
    key = amendments.law_key(meta.law_name)
    if sections and sections[-1].number != "preamble":
        _strip_last_section_signature(sections[-1])
    chunks: list[LegalChunk] = []

    def emit(section: Section, subsection: str | None, source_id: str, breadcrumb: str, body: str,
             refs: list[str], amended: AmendmentRef | None, inserted: str = "") -> None:
        if count_tokens(body) + count_tokens(breadcrumb) <= budget:
            bodies = [body]
        else:
            bodies = _pack(body, max(budget - count_tokens(breadcrumb), 1), overlap_tokens)
        for index, part in enumerate(bodies, start=1):
            multipart = len(bodies) > 1
            ref = amended
            if ref is not None and not inserted:  # what this part itself touches
                ref = amendments.extract_amendment(section.title, part, toc) or ref
            header = breadcrumb + (_part_suffix(meta.language, index, len(bodies)) if multipart else "")
            text = f"{header}\n\n{part}"
            text, stored_breadcrumb = normalize_json_quotes(text), normalize_json_quotes(breadcrumb)
            chunks.append(
                LegalChunk(
                    text=text,
                    metadata=ChunkMetadata(
                        chunk_id=f"{source_id}#p{index}" if multipart else source_id,
                        source_id=source_id,
                        section_key=f"{meta.version_id}:{section.number}",
                        law_id=meta.law_id,
                        law_name=meta.law_name,
                        chapter=section.chapter,
                        part=section.subchapter or section.division,
                        section_number=section.number,
                        subsection_number=subsection,
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
                        amends=amendments.encode([ref]) if ref else [],
                        inserted_section=inserted,
                    ),
                )
            )

    def emit_insertion(section: Section, subsection: str | None, source_id: str, breadcrumb: str,
                       insertion: Insertion, amended: AmendmentRef | None) -> None:
        """One chunk per inserted provision (or per subsection of a long one), labelled
        with both numbers and carrying the amending instruction as context."""
        context = "\n".join(insertion.context)
        target = amended.target if amended else ""
        for provision in insertion.provisions:
            number = provision.number or "?"
            title = f" — {provision.title}" if provision.title else ""
            provision_breadcrumb = f"{breadcrumb} > {_inserts_label(meta.language, number, target)}{title}"
            room = budget - count_tokens(provision_breadcrumb) - count_tokens(context)
            ref = AmendmentRef(
                target, amended.target_key if amended else "", amended.number if amended else "",
                [number] if provision.number else [], amended.temporary if amended else False,
            )
            for sub, unit in _provision_units(provision.lines, max(room, 1)):
                inserted = f"{number}({sub})" if sub else number
                unit_breadcrumb = f"{breadcrumb} > {_inserts_label(meta.language, inserted, target)}{title}"
                emit(section, subsection, f"{source_id}>{inserted}", unit_breadcrumb,
                     f"{context}\n{unit}", [], ref, inserted)

    for section in sections:
        section_key = f"{meta.version_id}:{section.number}"
        header_tokens = count_tokens(_breadcrumb(meta, section, None))
        for piece in _pieces(section, budget, header_tokens):
            breadcrumb = _breadcrumb(meta, section, piece.subsection)
            source_id = f"{section_key}({piece.subsection})" if piece.subsection else section_key
            piece_text = join_spaced_section_numbers(piece.text)
            if section.number == "preamble":
                emit(section, piece.subsection, source_id, breadcrumb, piece_text, [], None)
                continue
            amended = amendments.extract_amendment(section.title, piece_text, toc)
            # Same rule parse_sections applies to Section.cross_refs: an amending provision's
            # "artículo 2" is article 2 of the law it amends, not of this one.
            amending = _amends_other_law(section) or amended is not None
            refs = [] if amending else [
                f"{meta.version_id}:{n}"
                for n in extract_cross_references(piece_text, section.number)
                if n in known_sections
            ]
            insertion = find_insertion(piece_text) if amending else None
            if insertion is None:
                emit(section, piece.subsection, source_id, breadcrumb, piece_text, refs, amended)
                continue
            lead = list(itertools.takewhile(lambda line: not _NUMBERED_ITEM_RE.match(line), insertion.context))
            while insertion is not None:
                emit_insertion(section, piece.subsection, source_id, breadcrumb, insertion, amended)
                if not insertion.trailing:
                    break
                # Instructions after the quoted block: their own insertion, or a plain chunk.
                rest = "\n".join(lead + insertion.trailing)
                insertion = find_insertion(rest)
                if insertion is None:
                    emit(section, piece.subsection, source_id, breadcrumb, rest, [],
                         amendments.extract_amendment(section.title, rest, toc) or amended)
    return _dedupe(chunks)


_MIN_DEDUPE_WORDS = 12  # short bodies ("(Derogado)", "(Sin contenido)") legitimately repeat


def normalized_body(text: str) -> str:
    """A chunk's body without its breadcrumb header, with punctuation
    and spacing removed -- what "the same provision" means for dedupe (here and
    at query time in legal/retrieval.py)."""
    body = text.split("\n\n", 1)[-1]
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
