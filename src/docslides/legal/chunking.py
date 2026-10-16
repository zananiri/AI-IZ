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
import itertools
import re
from dataclasses import dataclass

from docslides.cleaning.tokens import count_tokens
from docslides.config import get_config
from docslides.legal import amendments
from docslides.legal.amendments import AmendmentRef
from docslides.legal.insertions import Insertion, find_insertion, join_spaced_section_numbers
from docslides.legal.models import ChunkMetadata, LegalChunk, SourceOrigin, SourceType, Status
from docslides.legal.structure import Section, _split_subsections, extract_cross_references
from docslides.logging_setup import get_logger

logger = get_logger(__name__)

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.;!?])\s+")
_SHORT_INTRO_TOKENS = 60  # an opening line this short is repeated as context in each subsection's chunk
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
    if subsection and section.number == "preamble":
        provision += f" — {subsection}"
    elif subsection:
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
    return f" (חלק {index} מתוך {count})" if language == "he" else f" (part {index} of {count})"


_ENACTED_RE = re.compile(r"^\*?\s*(?:התקבל בכנסת|Passed by the Knesset)")
_FOOTNOTE_REF_RE = re.compile(r"^\d{1,3}\s+ס[\"״]ח\s")
_GAZETTE_HEADER_LINES = {"רשומות", "ספר החוקים", "הערות שוליים:", "עמוד"}
_SIGNATORY_RE = re.compile(r"ראש הממשלה|נשיא המדינה|יושב ראש הכנסת")
_SENTENCE_END = (".", ";", ":", '."', ".״", '";', "״;")
_ENACTED_LABEL = "קבלת החוק"
_TOC_LABEL = "תיקונים עקיפים"
_NUMBERED_ITEM_RE = re.compile(r"^\(\d{1,3}\)")  # "(2) בסעיף 62(ג) ..."


def _preamble_pieces(section: Section) -> list[_Piece] | None:
    """A gazette preamble reduced to what answers questions: when the Knesset
    passed the law (the '* התקבל בכנסת ...' note) and the list of laws it
    amends. Page headers, the title fragment and bare 'ס"ח' footnote
    references are dropped. None if neither part is found (keep it whole)."""
    lines = [line.strip() for line in section.text.splitlines() if line.strip()]
    enacted: list[str] = []
    toc: list[str] = []
    collecting = False
    for line in lines:
        is_toc = bool(amendments.parse_toc(line))
        if _ENACTED_RE.match(line):
            collecting = True
        elif collecting and (is_toc or _FOOTNOTE_REF_RE.match(line) or line in _GAZETTE_HEADER_LINES):
            collecting = False
        if collecting:
            enacted.append(line)
        elif is_toc:
            toc.append(line)
    pieces = []
    if enacted:
        pieces.append(_Piece(_ENACTED_LABEL, "\n".join(enacted)))
    if toc:
        pieces.append(_Piece(_TOC_LABEL, "תיקונים עקיפים:\n" + "\n".join(toc)))
    return pieces or None


def _strip_signature_block(text: str) -> str:
    """Drops the signatories printed after a law's last section ('בנימין
    נתניהו / ראש הממשלה ...'), which otherwise end up in its last chunk."""
    lines = text.splitlines()
    if not any(_SIGNATORY_RE.search(line) and len(line.split()) <= 6 for line in lines[-6:]):
        return text
    cut = len(lines)
    for j in range(len(lines) - 1, -1, -1):
        stripped = lines[j].strip()
        if not stripped:
            continue
        if len(stripped.split()) > 6 or stripped.endswith(_SENTENCE_END):
            break
        cut = j
    return "\n".join(lines[:cut])


def _strip_last_section_signature(section: Section) -> None:
    section.text = _strip_signature_block(section.text)
    section.intro = _strip_signature_block(section.intro)
    if section.subsections:
        section.subsections[-1].text = _strip_signature_block(section.subsections[-1].text)


def _inserts_label(language: str, inserted: str, target: str) -> str:
    if language == "he":
        return f"מוסיף את סעיף {inserted}" + (f" ל{target}" if target else "")
    return f"inserts section {inserted}" + (f" into {target}" if target else "")


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
    hebrew = meta.language == "he"
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
            stored_breadcrumb = breadcrumb
            if hebrew:
                text, stored_breadcrumb = normalize_hebrew_quotes(text), normalize_hebrew_quotes(breadcrumb)
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
        pieces = _preamble_pieces(section) if section.number == "preamble" and hebrew else None

        for piece in pieces or _pieces(section, budget, header_tokens):
            breadcrumb = _breadcrumb(meta, section, piece.subsection)
            source_id = f"{section_key}({piece.subsection})" if piece.subsection else section_key
            piece_text = join_spaced_section_numbers(piece.text) if hebrew else piece.text
            if section.number == "preamble":
                emit(section, piece.subsection, source_id, breadcrumb, piece_text, [], None)
                continue
            amended = amendments.extract_amendment(section.title, piece_text, toc)
            # Same rule parse_sections applies to Section.cross_refs: an amending section's
            # "סעיף 2" is section 2 of the law it amends, not of this one.
            amending = (section.title or "").startswith("תיקון") or amended is not None
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
