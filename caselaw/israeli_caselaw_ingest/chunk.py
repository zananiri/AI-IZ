"""Stage 4: structure-aware chunking of documents.parquet into chunks/year=YYYY/part-NNNNN.parquet.

The body (after the header) is cut into segments first: a numbered paragraph ("12. "), a
Hebrew-letter item ("ב. "), a section heading (רקע, דיון והכרעה, סוף דבר ...), or a block after a
blank line; unmarked lines continue the segment above. Segments are split into sentences
(". ? ! :" followed by a space or newline), measured with the embedding model's tokenizer, and
packed into chunks of min_tokens..max_tokens: a chunk ends at a paragraph boundary once it holds
min_tokens, or before the sentence that would take it past max_tokens, and the next chunk repeats
up to overlap_tokens of whole sentences from its end. The operative section ("סוף דבר", "אשר על
כן", "התוצאה היא") always starts a new chunk, without overlap, so holding chunks hold only the
holding; a heading does too once the current chunk holds heading_break_min_tokens. A single sentence longer than max_tokens is the only thing cut inside a
sentence, at word boundaries.

Each document also gets a header chunk (index 0): the parsed header, or one built from metadata."""

from __future__ import annotations

import datetime as dt
import math
import re
from collections import Counter
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq

from .citations import extract_citations
from .config import config_hash, paths_for
from .state import atomic_write_json, atomic_write_parquet, log, mark_done, read_json, stage_done
from .tokens import get_tokenizer

CHUNK_SCHEMA = pa.schema([
    ("chunk_id", pa.string()), ("doc_id", pa.string()), ("case_citation", pa.string()),
    ("case_name", pa.string()), ("doc_type", pa.string()), ("proceeding_type", pa.string()),
    ("division", pa.string()), ("decision_date", pa.date32()), ("judges", pa.list_(pa.string())),
    ("parties", pa.list_(pa.string())), ("lang", pa.string()), ("section", pa.string()),
    ("is_holding", pa.bool_()), ("chunk_index", pa.int32()), ("n_chunks", pa.int32()),
    ("char_start", pa.int64()), ("char_end", pa.int64()), ("token_count", pa.int32()),
    ("context_prefix", pa.string()), ("text", pa.large_string()), ("citations", pa.list_(pa.string())),
    ("source_url", pa.string()), ("source_dataset", pa.string()), ("ingested_at", pa.timestamp("s", tz="UTC")),
])

NUMBERED_RE = re.compile(r"^\s*\d{1,3}\.\s")
LETTER_RE = re.compile(r"^\s*(?:[א-ת]\.|\(?[א-ת]{1,2}\))\s")
_ENUM_ONLY_RE = re.compile(r"^\s*(?:\d{1,3}|[א-ת]{1,2})\.$")
_SENT_END_RE = re.compile(r"(?<=[.?!:])(?:\s+)")
_WORD_RE = re.compile(r"\S+")


@dataclass
class Unit:
    start: int
    end: int
    para_start: bool = False
    heading: bool = False
    holding: bool = False
    holding_start: bool = False
    tokens: int = 0


def _strip_marker(line: str) -> str:
    return re.sub(r"^\s*(?:\d{1,3}\.|[א-ת]\.)\s*", "", line).strip().rstrip(":").strip()


def is_heading(line: str, headings: set[str]) -> bool:
    core = _strip_marker(line)
    return 0 < len(core) <= 40 and core in headings


def holding_start_offset(lines: list[tuple[int, str]], headings: set[str], markers: list[str]) -> int | None:
    """Where the operative section starts: the last "סוף דבר"-style heading, else the last line that
    opens with a holding marker."""
    marker_re = re.compile(r"^(?:" + "|".join(re.escape(m) for m in markers) + r")(?:[\s,:]|$)")
    heading_hits, line_hits = [], []
    for pos, line in lines:
        core = _strip_marker(line)
        if not core:
            continue
        if marker_re.match(core):
            (heading_hits if is_heading(line, set(markers)) else line_hits).append(pos)
    if heading_hits:
        return heading_hits[-1]
    return line_hits[-1] if line_hits else None


def segments(text: str, start: int, headings: set[str], markers: list[str]) -> list[Unit]:
    """Segments of text[start:] as units (one per segment, not yet split into sentences)."""
    lines, pos = [], start
    for line in text[start:].split("\n"):
        lines.append((pos, line))
        pos += len(line) + 1
    hold_at = holding_start_offset(lines, headings, markers)
    segs: list[Unit] = []
    current: Unit | None = None
    for pos, line in lines:
        if not line.strip():
            current = None
            continue
        end = pos + len(line)
        heading = is_heading(line, headings)
        starts = current is None or heading or bool(NUMBERED_RE.match(line) or LETTER_RE.match(line)) \
            or (current is not None and current.heading) or (hold_at is not None and pos == hold_at)
        if starts:
            current = Unit(start=pos, end=end, para_start=True, heading=heading)
            segs.append(current)
        else:
            current.end = end
    for seg in segs:
        seg.holding = hold_at is not None and seg.start >= hold_at
        seg.holding_start = hold_at is not None and seg.start == hold_at
    return segs


def sentences(text: str, seg: Unit) -> list[Unit]:
    """A segment split at sentence ends; an enumerator alone ("1.") never ends a sentence."""
    piece = text[seg.start:seg.end]
    cuts, last = [], 0
    for m in _SENT_END_RE.finditer(piece):
        if _ENUM_ONLY_RE.match(piece[last:m.start()]):
            continue
        cuts.append((last, m.start()))
        last = m.end()
    if last < len(piece):
        cuts.append((last, len(piece)))
    units = []
    for i, (a, b) in enumerate(c for c in cuts if piece[c[0]:c[1]].strip()):
        units.append(Unit(start=seg.start + a, end=seg.start + b, para_start=seg.para_start and i == 0,
                          heading=seg.heading, holding=seg.holding, holding_start=seg.holding_start and i == 0))
    return units


def split_long(text: str, unit: Unit, max_tokens: int, tokenizer) -> list[Unit]:
    """A sentence over max_tokens, cut at word boundaries into pieces under it."""
    words = [(m.start(), m.end()) for m in _WORD_RE.finditer(text, unit.start, unit.end)]
    pieces = max(2, math.ceil(unit.tokens / (max_tokens * 0.9)))
    per = max(1, math.ceil(len(words) / pieces))
    out = []
    for i in range(0, len(words), per):
        group = words[i:i + per]
        out.append(Unit(start=group[0][0], end=group[-1][1], para_start=unit.para_start and i == 0,
                        heading=unit.heading, holding=unit.holding, holding_start=unit.holding_start and i == 0))
    counts = tokenizer.count([text[u.start:u.end] for u in out])
    for u, n in zip(out, counts):
        u.tokens = n
    if any(u.tokens > max_tokens for u in out) and per > 1:
        return [v for u in out for v in (split_long(text, u, max_tokens, tokenizer) if u.tokens > max_tokens else [u])]
    return out


def body_units(text: str, start: int, cfg: dict, tokenizer) -> list[Unit]:
    ccfg = cfg["chunk"]
    headings = set(ccfg["headings"]) | set(ccfg["holding_markers"])
    units = [u for seg in segments(text, start, headings, ccfg["holding_markers"]) for u in sentences(text, seg)]
    for u, n in zip(units, tokenizer.count([text[u.start:u.end] for u in units])):
        u.tokens = n
    out = []
    for u in units:
        out.extend(split_long(text, u, ccfg["max_tokens"], tokenizer) if u.tokens > ccfg["max_tokens"] else [u])
    return out


def pack(units: list[Unit], min_tokens: int, max_tokens: int, overlap_tokens: int,
         heading_break_min_tokens: int = 100) -> list[list[Unit]]:
    """Group units into chunks (see the module docstring). A heading starts a new chunk once the
    current one holds heading_break_min_tokens (a heading right after a few lines joins them)."""
    chunks: list[list[Unit]] = []
    cur: list[Unit] = []
    cur_tok = 0
    fresh = 0  # units in cur that are not overlap from the previous chunk

    def flush(overlap: bool) -> None:
        nonlocal cur, cur_tok, fresh
        if fresh:
            chunks.append(cur)
        carry: list[Unit] = []
        if overlap and fresh:
            total = 0
            for u in reversed(cur[1:]):  # never the whole chunk
                if total + u.tokens > overlap_tokens:
                    break
                carry.insert(0, u)
                total += u.tokens
        cur, cur_tok, fresh = carry, sum(u.tokens for u in carry), 0

    for u in units:
        fresh_tok = sum(v.tokens for v in cur[len(cur) - fresh:]) if fresh else 0
        if u.holding_start or (u.heading and u.para_start and fresh_tok >= heading_break_min_tokens):
            if fresh:
                flush(overlap=False)
            else:
                cur, cur_tok = [], 0
        elif fresh and cur_tok + u.tokens > max_tokens:
            flush(overlap=True)
        elif fresh and u.para_start and cur_tok >= min_tokens:
            flush(overlap=True)
        if cur and cur_tok + u.tokens > max_tokens:
            cur, cur_tok = [], 0  # the overlap and this unit don't fit together: drop the overlap
        cur.append(u)
        cur_tok += u.tokens
        fresh += 1
    if fresh:
        chunks.append(cur)
    return chunks


def context_prefix(doc: dict) -> str:
    date = doc["decision_date"].isoformat() if doc.get("decision_date") else None
    judges = ", ".join(doc.get("judge_last_names") or [])
    parts = [doc.get("case_citation"), doc.get("doc_type"), date, judges]
    return "[" + " | ".join(p for p in parts if p) + "]"


def metadata_header(doc: dict) -> str:
    lines = [doc.get("court") or "בית המשפט העליון", doc.get("case_citation"), doc.get("case_name")]
    if doc.get("judges"):
        lines.append("בפני: " + ", ".join(doc["judges"]))
    if doc.get("parties"):
        lines.append("הצדדים: " + " נגד ".join(doc["parties"][:2]) if len(doc["parties"]) == 2
                     else "הצדדים: " + "; ".join(doc["parties"]))
    lines += [doc.get("doc_type"), doc["decision_date"].isoformat() if doc.get("decision_date") else None]
    return "\n".join(line for line in lines if line)


def chunk_document(doc: dict, cfg: dict, tokenizer, now: dt.datetime) -> list[dict]:
    ccfg = cfg["chunk"]
    text = doc["text"] or ""
    prefix = context_prefix(doc)
    body_start = doc["body_start"] if doc.get("header_parsed") else 0
    header_text = doc["header_text"] if doc.get("header_parsed") and doc.get("header_text") else metadata_header(doc)
    header_span = (0, body_start) if doc.get("header_parsed") else (0, 0)
    groups = pack(body_units(text, body_start, cfg, tokenizer), ccfg["min_tokens"], ccfg["max_tokens"],
                  ccfg["overlap_tokens"], ccfg.get("heading_break_min_tokens", 100))
    spans = [header_span] + [(g[0].start, g[-1].end) for g in groups]
    texts = [header_text] + [text[a:b] for a, b in spans[1:]]
    holding = [False] + [all(u.holding for u in g) for g in groups]
    counts = tokenizer.count(texts)
    base = {k: doc.get(k) for k in ("doc_id", "case_citation", "case_name", "doc_type", "proceeding_type",
                                     "division", "decision_date", "judges", "parties", "lang",
                                     "source_url", "source_dataset")}
    rows = []
    for i, (chunk_text, (a, b), is_holding, n) in enumerate(zip(texts, spans, holding, counts)):
        rows.append({**base, "chunk_id": f"{doc['document_hash']}:{i}",
                     "section": "header" if i == 0 else ("holding" if is_holding else "body"),
                     "is_holding": is_holding, "chunk_index": i, "n_chunks": len(texts),
                     "char_start": a, "char_end": b, "token_count": n, "context_prefix": prefix,
                     "text": chunk_text, "citations": extract_citations(chunk_text), "ingested_at": now})
    return rows


def run_chunk(cfg: dict, force: bool = False) -> dict:
    paths = paths_for(cfg)
    clean_done = stage_done(paths.state, "clean")
    done = stage_done(paths.state, "chunk")
    if done and not force and clean_done and done.get("clean") == clean_done.get("config_hash") \
            and done.get("chunk_settings") == config_hash([cfg["chunk"], cfg["embed"]["model"]]):
        log(f"chunk: done already ({done['chunks']:,} chunks); --force to redo")
        return done
    if not clean_done or not paths.documents.exists():
        raise SystemExit("run the clean stage first")
    tokenizer = get_tokenizer(cfg)
    settings_hash = config_hash({"chunk": cfg["chunk"], "model": cfg["embed"]["model"], "tokenizer": tokenizer.name,
                                 "clean": clean_done.get("config_hash")})
    out = paths.chunks
    done_dir = out / "_done"
    marker = read_json(out / "_settings.json")
    if force or (marker and marker.get("config_hash") != settings_hash):
        for p in list(out.rglob("*.parquet")) + list(done_dir.glob("*.json")):
            p.unlink()
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "_settings.json", {"config_hash": settings_hash, "tokenizer": tokenizer.name})
    log(f"chunk: tokenizer {tokenizer.name}")

    pf = pq.ParquetFile(paths.documents)
    total_docs = pf.metadata.num_rows
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    batch_size = cfg["chunk"]["doc_batch_size"]
    for i, batch in enumerate(pf.iter_batches(batch_size=batch_size)):
        name = f"part-{i:05d}"
        if (done_dir / f"{name}.json").exists():
            continue
        by_year: dict[int, list[dict]] = {}
        docs = batch.to_pylist()
        del batch
        for doc in docs:
            for row in chunk_document(doc, cfg, tokenizer, now):
                by_year.setdefault(doc["year"], []).append(row)
        for year, rows in by_year.items():
            atomic_write_parquet(out / f"year={year}" / f"{name}.parquet", pa.Table.from_pylist(rows, schema=CHUNK_SCHEMA))
        n = sum(len(r) for r in by_year.values())
        atomic_write_json(done_dir / f"{name}.json", {"documents": len(docs), "chunks": n,
                                                      "years": {str(y): len(r) for y, r in by_year.items()}})
        log(f"chunk: batch {i} ({min((i + 1) * batch_size, total_docs):,}/{total_docs:,} documents): {n:,} chunks")

    totals = Counter()
    for p in done_dir.glob("part-*.json"):
        s = read_json(p)
        totals["documents"] += s["documents"]
        totals["chunks"] += s["chunks"]
    mark_done(paths.state, "chunk", config_hash=settings_hash, chunks=totals["chunks"], documents=totals["documents"],
              tokenizer=tokenizer.name, clean=clean_done.get("config_hash"), chunk_settings=config_hash([cfg["chunk"], cfg["embed"]["model"]]))
    log(f"chunk: {totals['chunks']:,} chunks from {totals['documents']:,} documents -> {out}")
    return dict(totals)


def chunk_files(chunks_dir) -> list:
    """Every chunk file, in the one fixed order the embed and BM25 stages rely on."""
    return sorted(chunks_dir.glob("year=*/part-*.parquet"), key=lambda p: (p.parent.name, p.name))
