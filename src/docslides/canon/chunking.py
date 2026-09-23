"""Structure-aware chunking for canon/civil-law text: one chunk per atomic
legal provision (a canon's `§` paragraph, or a civil-law article) rather
than the token-budget paragraph packing in cleaning/chunking.py, which is
built for slide-deck ingestion and would cut a canon apart mid-provision.

Every chunk is prefixed with a synthetic breadcrumb header (code + canon/
article number + book/title/chapter, or law name/date for civil law) so
short provisions still carry their hierarchical context into the embedding
-- this is what makes citation-style queries ("what does canon law say
about X") retrieve precisely instead of matching on generic legal prose.

`ProvisionRecord`s are produced by canon/parsing.py (which does the actual
HTML/PDF scraping, gated behind the optional `canon` dependency group); this
module has no such dependency and is unit-tested directly against
hand-built records.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from docslides.cleaning.tokens import count_tokens
from docslides.config import get_config

CodeName = Literal["cic", "cceo", "vcs_law"]

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """A dependency-free sentence split for the rare oversized-provision
    fallback below -- deliberately NOT cleaning/segmentation.py's spaCy/
    Stanza-backed splitter, since that would pull the heavy `lang` extra
    (spaCy for en/it/etc., Stanza for ar/he) into the `canon` extra's
    footprint just for this corner case. Good enough here: this only picks
    chunk-boundary points for embeddings, not translation-quality output."""
    return [s.strip() for s in _SENTENCE_BOUNDARY_RE.split(text) if s.strip()]

_CODE_LABEL: dict[CodeName, str] = {"cic": "CIC", "cceo": "CCEO", "vcs_law": ""}
_UNIT_LABEL: dict[CodeName, str] = {"cic": "Can.", "cceo": "Can.", "vcs_law": "Art."}


@dataclass
class ProvisionRecord:
    """One atomic legal provision as extracted from source, before
    token-budget sub-splitting."""

    code: CodeName
    number: str  # e.g. "1055" (canon) or "12" (article)
    paragraph: str | None  # e.g. "1" for "Can. 1055 §1"; None if undivided
    breadcrumb: str  # e.g. "Book IV: Sanctifying Function > Title VII: Marriage"
    text: str
    source_url: str
    language: str  # "en" | "la" | "it"


@dataclass
class Chunk:
    id: str
    text: str  # breadcrumb header + provision text -- what actually gets embedded
    code: CodeName
    number: str
    paragraph: str | None
    breadcrumb: str
    source_url: str
    language: str
    token_count: int


def citation_label(code: CodeName, number: str, paragraph: str | None = None) -> str:
    """The short citation form shown in the UI's citations panel and
    embedded as each chunk's header, e.g. 'CIC Can. 1055 §1', 'Art. 12'."""
    label = _CODE_LABEL[code]
    unit = _UNIT_LABEL[code]
    number_part = f"{unit} {number}" + (f" §{paragraph}" if paragraph else "")
    return f"{label} {number_part}".strip()


def _header(record: ProvisionRecord) -> str:
    prefix = citation_label(record.code, record.number, record.paragraph)
    return f"{prefix} — {record.breadcrumb}" if record.breadcrumb else prefix


def _chunk_id(record: ProvisionRecord, part: int | None = None) -> str:
    base = f"{record.code}:{record.number}"
    if record.paragraph:
        base += f":{record.paragraph}"
    if part is not None:
        base += f":{part}"
    return base


def chunk_provision(record: ProvisionRecord) -> list[Chunk]:
    """One provision -> one chunk, unless it exceeds the configured token
    budget (rare -- a handful of long canons), in which case it's split at
    sentence boundaries with the same header repeated on each part so
    retrieval context is never lost."""
    max_tokens = get_config().cleaning.max_chunk_tokens
    header = _header(record)
    full_text = f"{header}\n\n{record.text}"
    full_tokens = count_tokens(full_text)

    if full_tokens <= max_tokens:
        return [
            Chunk(
                id=_chunk_id(record),
                text=full_text,
                code=record.code,
                number=record.number,
                paragraph=record.paragraph,
                breadcrumb=record.breadcrumb,
                source_url=record.source_url,
                language=record.language,
                token_count=full_tokens,
            )
        ]

    header_tokens = count_tokens(header)
    sentences = _split_sentences(record.text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = header_tokens
    part = 1

    def flush() -> None:
        nonlocal current, current_tokens, part
        if not current:
            return
        text = f"{header} (part {part})\n\n{' '.join(current)}"
        chunks.append(
            Chunk(
                id=_chunk_id(record, part),
                text=text,
                code=record.code,
                number=record.number,
                paragraph=record.paragraph,
                breadcrumb=record.breadcrumb,
                source_url=record.source_url,
                language=record.language,
                token_count=count_tokens(text),
            )
        )
        part += 1
        current = []
        current_tokens = header_tokens

    for sentence in sentences:
        sent_tokens = count_tokens(sentence)
        if current_tokens + sent_tokens > max_tokens and current:
            flush()
        current.append(sentence)
        current_tokens += sent_tokens
    flush()

    return chunks


def chunk_provisions(records: list[ProvisionRecord]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for record in records:
        chunks.extend(chunk_provision(record))
    return chunks
