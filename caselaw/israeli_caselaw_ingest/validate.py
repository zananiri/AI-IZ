"""Stage 6: validation report (reports/validation_report.md) and sanity queries.

Streams the chunk files column by column: document and chunk counts, token-length histogram,
share of mojibake repaired and of headers parsed, top cited precedents, chunks per year, the
decision-date range. Fails (after writing the report) when a chunk is dated on or after
max_decision_date or a chunk_id repeats."""

from __future__ import annotations

import datetime as dt
from collections import Counter

import pyarrow.parquet as pq

from .chunk import chunk_files
from .config import paths_for
from .state import atomic_write_text, log, stage_done

BINS = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 1000]


def _bin(n: int) -> str:
    for lo, hi in zip(BINS, BINS[1:]):
        if n < hi:
            return f"{lo}-{hi - 1}"
    return f">={BINS[-1]}"


def _hist(counter: Counter) -> list[str]:
    order = [f"{lo}-{hi - 1}" for lo, hi in zip(BINS, BINS[1:])] + [f">={BINS[-1]}"]
    peak = max(counter.values(), default=1)
    return [f"| {b} | {counter.get(b, 0):,} | {'█' * round(30 * counter.get(b, 0) / peak)} |" for b in order]


def _search_section(cfg: dict) -> list[str]:
    from .search import Searcher

    paths = paths_for(cfg)
    k = cfg["validate"]["top_k"]
    searcher = Searcher(cfg)
    have_bm25 = paths.bm25.exists() and any(paths.bm25.glob("shard-*/done.json"))
    have_dense = stage_done(paths.state, "embed") is not None and paths.lancedb.exists()
    lines = ["## Sanity queries", ""]
    if not have_bm25 and not have_dense:
        return lines + ["No BM25 or vector index built yet.", ""]
    for query in cfg["validate"]["queries"]:
        lines += [f"### {query}", ""]
        rankings = []
        if have_bm25:
            rankings.append(("BM25", searcher.bm25(query, k)))
        if have_dense:
            try:
                rankings.append(("dense", searcher.dense(query, k)))
                if have_bm25:
                    rankings.append(("hybrid (RRF)", searcher.hybrid(query, k)))
            except Exception as exc:  # noqa: BLE001 -- model not available offline, etc.
                lines.append(f"dense search unavailable: {type(exc).__name__}: {exc}")
        rows = searcher.rows([cid for _, hits in rankings for cid, _ in hits])
        for label, hits in rankings:
            lines += [f"**{label}**", "", "| # | score | citation | type | date | section | source_url | excerpt |",
                      "|--:|--:|---|---|---|---|---|---|"]
            for rank, (cid, score) in enumerate(hits, 1):
                r = rows.get(cid, {})
                excerpt = " ".join((r.get("text") or "").split())[:140].replace("|", "/")
                lines.append(f"| {rank} | {score:.3f} | {r.get('case_citation') or ''} | {r.get('doc_type') or ''} | "
                             f"{r.get('decision_date') or ''} | {r.get('section') or ''} | {r.get('source_url') or ''} | {excerpt} |")
            if not hits:
                lines.append("| - | | no hits | | | | | |")
            lines.append("")
    return lines


def run_validate(cfg: dict) -> dict:
    paths = paths_for(cfg)
    if not stage_done(paths.state, "chunk"):
        raise SystemExit("run the chunk stage first")
    cutoff = dt.date.fromisoformat(str(cfg["filter"]["max_decision_date"]))
    files = chunk_files(paths.chunks)
    tokens_body, tokens_header = Counter(), Counter()
    per_year, sections, cited = Counter(), Counter(), Counter()
    ids: set[str] = set()
    duplicates = 0
    min_date = max_date = None
    late = 0
    chunks = 0
    docs = set()
    for f in files:
        t = pq.read_table(f, columns=["chunk_id", "doc_id", "decision_date", "section", "token_count", "citations"])
        chunks += t.num_rows
        for cid, did, date, section, n, cites in zip(*(t.column(c).to_pylist() for c in t.column_names)):
            if cid in ids:
                duplicates += 1
            ids.add(cid)
            docs.add(did)
            (tokens_header if section == "header" else tokens_body)[_bin(n)] += 1
            sections[section] += 1
            per_year[date.year] += 1
            if min_date is None or date < min_date:
                min_date = date
            if max_date is None or date > max_date:
                max_date = date
            if date >= cutoff:
                late += 1
            if section != "header":
                cited.update(cites)
    del ids

    doc_stats = Counter()
    repair = Counter()
    doc_rows = 0
    if paths.documents.exists():
        dt_ = pq.read_table(paths.documents, columns=["encoding_repair", "header_parsed", "lang", "date_source"])
        doc_rows = dt_.num_rows
        repair.update(dt_.column("encoding_repair").to_pylist())
        doc_stats["header_parsed"] = sum(dt_.column("header_parsed").to_pylist())
        doc_stats.update(f"lang:{x}" for x in dt_.column("lang").to_pylist())
        doc_stats.update(f"date:{x}" for x in dt_.column("date_source").to_pylist())
    filt = stage_done(paths.state, "filter") or {}
    repaired = sum(v for k, v in repair.items() if k not in ("none", "unrepaired"))
    embed_state = stage_done(paths.state, "embed")
    chunk_state = stage_done(paths.state, "chunk") or {}

    lines = [
        "# Validation report", "",
        f"Run directory: `{paths.work}`" + (f" (sample of {cfg['sample_n']:,} documents)" if cfg.get("sample_n") else ""), "",
        "## Counts", "",
        f"- dataset rows read: {filt.get('read', 0):,}; documents kept by the filter: {filt.get('kept', 0):,}",
        f"- documents.parquet rows: {doc_rows:,}; documents with chunks: {len(docs):,}",
        f"- chunks: {chunks:,} ({sections.get('header', 0):,} header, {sections.get('body', 0):,} body, "
        f"{sections.get('holding', 0):,} holding)",
        f"- tokenizer used for chunk sizes: {chunk_state.get('tokenizer', '?')}",
        f"- mojibake repaired: {repaired:,} documents ({repaired / max(doc_rows, 1):.2%}); still garbled: "
        f"{repair.get('unrepaired', 0):,}; by method: {dict(repair)}",
        f"- header parsed: {doc_stats['header_parsed']:,} documents ({doc_stats['header_parsed'] / max(doc_rows, 1):.1%})",
        "- language: " + ", ".join(f"{k[5:]} {v:,}" for k, v in doc_stats.items() if k.startswith("lang:")),
        "- decision date from: " + ", ".join(f"{k[5:]} {v:,}" for k, v in doc_stats.items() if k.startswith("date:")),
        f"- vector store: {embed_state['store']} ({embed_state['rows']:,} rows, {embed_state['model']})" if embed_state
        else "- vector store: not built (embed not run)",
        "", "## Checks", "",
        f"- decision dates: {min_date} .. {max_date}; chunks dated on/after {cutoff}: **{late}** "
        + ("✅" if late == 0 else "❌"),
        f"- duplicate chunk_id: **{duplicates}** " + ("✅" if duplicates == 0 else "❌"),
        "", "## Token length (body and holding chunks)", "", "| tokens | chunks | |", "|---|--:|---|", *_hist(tokens_body),
        "", "## Token length (header chunks)", "", "| tokens | chunks | |", "|---|--:|---|", *_hist(tokens_header),
        "", "## Top cited precedents", "", "| citation | chunks citing |", "|---|--:|",
        *[f"| {c} | {n:,} |" for c, n in cited.most_common(30)],
        "", "## Chunks per year", "", "| year | chunks |", "|---|--:|",
        *[f"| {y} | {n:,} |" for y, n in sorted(per_year.items())], "",
    ]
    lines += _search_section(cfg)
    report = "\n".join(lines)
    atomic_write_text(paths.reports / "validation_report.md", report)
    log(f"validate: report -> {paths.reports / 'validation_report.md'}")
    if late or duplicates:
        raise SystemExit(f"validation failed: {late} chunks on/after {cutoff}, {duplicates} duplicate chunk ids")
    return {"chunks": chunks, "documents": len(docs), "min_date": str(min_date), "max_date": str(max_date)}


def peek(cfg: dict, n: int = 5, seed: int = 0) -> str:
    """n random chunks (body/holding preferred), as Markdown."""
    import random

    paths = paths_for(cfg)
    files = chunk_files(paths.chunks)
    rng = random.Random(seed)
    sizes = [pq.ParquetFile(f).metadata.num_rows for f in files]
    total = sum(sizes)
    picks = sorted(rng.sample(range(total), min(n, total)))
    out, offset, fi = [], 0, 0
    for p in picks:
        while p >= offset + sizes[fi]:
            offset += sizes[fi]
            fi += 1
        row = pq.read_table(files[fi]).slice(p - offset, 1).to_pylist()[0]
        out += [f"### {row['chunk_id']}  ({row['section']}, chunk {row['chunk_index']}/{row['n_chunks'] - 1}, "
                f"{row['token_count']} tokens)", "",
                f"- prefix: {row['context_prefix']}", f"- citations: {', '.join(row['citations']) or '-'}",
                f"- source_url: {row['source_url']}", "", "```text", row["text"], "```", ""]
    return "\n".join(out)
