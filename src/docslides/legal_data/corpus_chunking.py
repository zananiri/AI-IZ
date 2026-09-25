"""Chunks for the corpus index (scripts/legal_data/vectorize.py).

Laws and regulations: a record's sections become legal/structure.Section objects and go through
legal/chunking.chunk_sections -- the same breadcrumbs ("law > פרק > סעיף N(א) — title"),
subsection splitting and token budget as the Legal tab's own index -- with `overlap` tokens
repeated between the parts of a split provision. Text under a heading outside any numbered
section (schedules, forms) and records with no sections at all are packed by paragraph under a
"law > heading" header.

Judgments: groups of paragraphs up to the budget, each chunk prefixed with
"court | case number | case name | date", consecutive groups overlapping by `overlap` tokens.

Every chunk carries three texts: `text` (displayed, stored as the Chroma document), `embed_text`
(niqqud stripped, quotes unified -- what is embedded) and `lexical_text` (also final letters
folded -- for a keyword index), and metadata of scalars only, since Chroma allows nothing else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from docslides.cleaning.tokens import count_tokens
from docslides.legal.chunking import SourceMeta, chunk_sections
from docslides.legal.structure import Section, Subsection
from docslides.legal_data.hebrew import normalize_for_embedding, normalize_for_index

_SENTENCE_RE = re.compile(r"(?<=[.;!?:])\s+")


@dataclass
class CorpusChunk:
    chunk_id: str
    text: str
    embed_text: str
    lexical_text: str
    metadata: dict


def _ymd(value: str | None) -> int | None:
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", value or "")
    return int("".join(match.groups())) if match else None


def base_metadata(record: dict, rhash: str) -> dict:
    meta = {
        "record_id": record["id"],
        "record_hash": rhash,
        "category": record["category"],
        "authority_level": record["authority_level"],
        "status": record.get("status") or "unknown",
        "source": record["source"],
        "license": record["license"],
        "title": record["title"][:500],
    }
    optional = {
        "effective_ymd": _ymd(record.get("effective_date")),
        "law_id": record.get("law_id"),
        "case_number": record.get("case_number"),
        "court": record.get("court"),
        "decision_ymd": _ymd(record.get("decision_date")),
        "doc_type": record.get("doc_type"),
        "text_quality": record.get("quality", {}).get("extraction"),
        "encoding": record.get("quality", {}).get("encoding"),
    }
    meta.update({k: v for k, v in optional.items() if v not in (None, "")})
    return meta


def _split_long(text: str, budget: int) -> list[str]:
    """A paragraph longer than the budget, as sentence-packed pieces (hard-cut as a last resort)."""
    pieces, current, tokens = [], [], 0
    for sentence in _SENTENCE_RE.split(text):
        if not sentence.strip():
            continue
        size = count_tokens(sentence)
        if size > budget:  # one enormous "sentence" (a table row, a list without stops)
            if current:
                pieces.append(" ".join(current))
                current, tokens = [], 0
            step = max(budget * 2, 200)  # ~2 characters per token for Hebrew, conservatively
            pieces += [sentence[i : i + step] for i in range(0, len(sentence), step)]
            continue
        if current and tokens + size > budget:
            pieces.append(" ".join(current))
            current, tokens = [], 0
        current.append(sentence)
        tokens += size
    if current:
        pieces.append(" ".join(current))
    return pieces


def pack_paragraphs(paragraphs: list[str], budget: int, overlap: int) -> list[str]:
    """Paragraphs packed into groups of at most `budget` tokens; each group after the first starts
    with the previous group's last paragraphs, up to `overlap` tokens."""
    units: list[tuple[str, int]] = []
    for paragraph in paragraphs:
        size = count_tokens(paragraph)
        pieces = [paragraph] if size <= budget else _split_long(paragraph, budget)
        units += [(piece, size if len(pieces) == 1 else count_tokens(piece)) for piece in pieces]
    groups: list[str] = []
    current: list[tuple[str, int]] = []
    tokens = 0
    for unit, size in units:
        if current and tokens + size > budget:
            groups.append("\n".join(u for u, _ in current))
            carried: list[tuple[str, int]] = []
            carried_tokens = 0
            for previous, previous_size in reversed(current):
                if carried_tokens + previous_size > overlap or carried_tokens + previous_size + size > budget:
                    break
                carried.insert(0, (previous, previous_size))
                carried_tokens += previous_size
            current, tokens = carried, carried_tokens
        current.append((unit, size))
        tokens += size
    if current:
        groups.append("\n".join(u for u, _ in current))
    return groups


def _finish(chunk_id: str, text: str, metadata: dict, fold_finals: bool) -> CorpusChunk:
    return CorpusChunk(chunk_id, text, normalize_for_embedding(text, fold_finals), normalize_for_index(text), metadata)


def _packed(record: dict, meta: dict, header: str, body: str, key: str, budget: int, overlap: int,
            fold_finals: bool, extra: dict | None = None) -> list[CorpusChunk]:
    paragraphs = [p.strip() for p in body.splitlines() if p.strip()]
    groups = pack_paragraphs(paragraphs, max(budget - count_tokens(header), 50), overlap)
    return [
        _finish(f"{record['id']}:{key}#p{i}", f"{header}\n\n{group}",
                {**meta, **(extra or {}), "breadcrumb": header[:500], "part_index": i, "part_count": len(groups)},
                fold_finals)
        for i, group in enumerate(groups, 1)
    ]


def legislation_chunks(record: dict, rhash: str, budget: int, overlap: int, fold_finals: bool) -> list[CorpusChunk]:
    meta = base_metadata(record, rhash)
    records = record.get("sections") or []
    numbered = [s for s in records if s["kind"] in ("section", "preamble")]
    chunks: list[CorpusChunk] = []
    if numbered:
        sections = [
            Section(number=s["number"], title=s.get("title"), text=s["text"], division=s.get("division"),
                    chapter=s.get("chapter"), subchapter=s.get("subchapter"), intro=s.get("intro") or "",
                    subsections=[Subsection(label=sub["label"], text=sub["text"]) for sub in s.get("subsections", [])])
            for s in numbered
        ]
        source = SourceMeta(
            law_id=record["id"], law_name=record["title"], effective_date_start=record.get("effective_date") or "",
            status="repealed" if record.get("status") == "repealed" else "current",
            source_type="statute" if record["category"] == "laws" else "regulation",
            source_origin="knesset" if record["category"] == "laws" else "reshumot",
            language=record.get("language") or "he",
        )
        for chunk in chunk_sections(sections, source, (record.get("retrieved_at") or "")[:10], budget, overlap):
            m = chunk.metadata
            extra = {"section_number": m.section_number, "breadcrumb": m.breadcrumb[:500],
                     "part_index": m.part_index, "part_count": m.part_count}
            if m.subsection_number:
                extra["subsection"] = m.subsection_number
            if m.inserted_section:
                extra["inserted_section"] = m.inserted_section
            chunks.append(_finish(m.chunk_id, chunk.text, {**meta, **extra}, fold_finals))
    for n, block in enumerate(s for s in records if s["kind"] == "schedule"):
        header = f"{record['title']} > {block.get('title') or block['number']}"
        chunks += _packed(record, meta, header, block["text"], f"schedule{n}", budget, overlap, fold_finals,
                          {"section_number": f"schedule:{block.get('title') or block['number']}"[:200]})
    if not records:
        chunks += _packed(record, meta, record["title"], record["text"], "text", budget, overlap, fold_finals)
    return chunks


def judgment_chunks(record: dict, rhash: str, budget: int, overlap: int, fold_finals: bool) -> list[CorpusChunk]:
    meta = base_metadata(record, rhash)
    header = " | ".join(str(p) for p in (record.get("court"), record.get("case_number"), record.get("case_name"),
                                           record.get("decision_date")) if p)
    return _packed(record, meta, header or record["title"], record["text"], "judgment", budget, overlap, fold_finals)


def chunk_record(record: dict, rhash: str, budget: int, overlap: int, fold_finals: bool) -> list[CorpusChunk]:
    if record["category"] == "supreme_court":
        chunks = judgment_chunks(record, rhash, budget, overlap, fold_finals)
    else:
        chunks = legislation_chunks(record, rhash, budget, overlap, fold_finals)
    # A page can repeat a section number (e.g. a second numbering in transitional provisions):
    # Chroma needs unique ids.
    seen: dict[str, int] = {}
    for chunk in chunks:
        seen[chunk.chunk_id] = seen.get(chunk.chunk_id, 0) + 1
        if seen[chunk.chunk_id] > 1:
            chunk.chunk_id = f"{chunk.chunk_id}~{seen[chunk.chunk_id]}"
    return chunks
