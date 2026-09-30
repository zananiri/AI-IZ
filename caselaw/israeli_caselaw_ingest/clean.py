"""Stage 3: clean each filtered part into documents_parts/, then merge them into documents.parquet
(one row per kept document: full cleaned text + metadata, so chunks can be regenerated without
cleaning again).

Per document: mojibake repair, NFC, directional marks removed, space runs collapsed (paragraph
breaks kept), boilerplate lines removed, header parsed (court, panel, parties, body start; the
dataset metadata fills what the header doesn't give), language, and the official download URL
rebuilt for citation -- never fetched."""

from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import pyarrow as pa
import pyarrow.parquet as pq

from .config import config_hash, paths_for
from .encoding import hebrew_count, normalize, repair_text
from .header import clean_judge, parse_header
from .state import atomic_write_json, atomic_write_parquet, log, mark_done, read_json, stage_done

DOCUMENT_SCHEMA = pa.schema([
    ("doc_id", pa.string()), ("document_hash", pa.string()), ("case_id", pa.string()),
    ("case_citation", pa.string()), ("case_name", pa.string()), ("doc_type", pa.string()),
    ("proceeding_type", pa.string()), ("division", pa.string()), ("court", pa.string()),
    ("decision_date", pa.date32()), ("date_source", pa.string()), ("year", pa.int16()),
    ("judges", pa.list_(pa.string())), ("judge_last_names", pa.list_(pa.string())),
    ("parties", pa.list_(pa.string())), ("lawyers", pa.list_(pa.string())),
    ("lang", pa.string()), ("header_parsed", pa.bool_()), ("header_court", pa.string()),
    ("header_judges", pa.list_(pa.string())), ("header_parties", pa.list_(pa.string())),
    ("header_text", pa.large_string()), ("body_start", pa.int64()),
    ("encoding_repair", pa.string()), ("text", pa.large_string()), ("text_len", pa.int64()),
    ("source_url", pa.string()), ("source_dataset", pa.string()),
])

_LATIN_RE = re.compile(r"[A-Za-z]")
_CITATION_IN_TITLE_RE = re.compile(r"[א-ת]{1,4}[\"״][א-ת]{1,2}\s*\d{1,6}/\d{2,4}")


def detect_lang(text: str) -> str:
    sample = text[:5000]
    he, en = hebrew_count(sample), len(_LATIN_RE.findall(sample))
    if he == 0 and en == 0:
        return "unknown"
    return "he" if he >= en else "en"


def remove_boilerplate(text: str, patterns: list[re.Pattern]) -> str:
    kept = [line for line in text.split("\n") if not any(p.search(line) for p in patterns)]
    return "\n".join(kept)


def source_url(template: str, path: str | None, file_name: str | None) -> str | None:
    """The official download URL, rebuilt from the dataset's Path / FileName (reference only)."""
    if not path and not file_name:
        return None
    p = (path or "").replace("/", "\\").strip("\\")
    if p and not p.lower().startswith("hebrewverdicts"):
        p = "HebrewVerdicts\\" + p
    return template.format(path=quote(p, safe="\\._-"), file_name=quote(file_name or "", safe="._-"))


def last_names(judges: list[str], judge_last: list[str]) -> list[str]:
    if judge_last:
        return [j.strip() for j in judge_last if j.strip()]
    out = []
    for j in judges:
        words = clean_judge(j).split()
        if words:
            out.append(words[-1])
    return out


def clean_record(rec: dict, cfg: dict, patterns: list[re.Pattern]) -> dict:
    ccfg = cfg["clean"]
    text, method = repair_text(rec["text"], ccfg["mojibake_min_chars"], ccfg["mojibake_min_ratio"], ccfg["try_ftfy"])
    text = normalize(remove_boilerplate(normalize(text), patterns))
    header = parse_header(text, ccfg["max_header_chars"])
    judges = header.judges if header.parsed and header.judges else rec["judges"]
    parties = header.parties if header.parsed and header.parties else rec["parties"]
    citation = rec["citation"]
    if not citation and rec.get("title"):
        m = _CITATION_IN_TITLE_RE.search(rec["title"])
        citation = m.group(0) if m else None
    date: dt.date = rec["decision_date"]
    return {
        "doc_id": rec["doc_id"], "document_hash": rec["document_hash"], "case_id": rec["case_id"],
        "case_citation": citation, "case_name": rec["case_name"] or rec.get("title"),
        "doc_type": rec["doc_type"], "proceeding_type": rec["proceeding_type"], "division": rec["division"],
        "court": rec["court"] or header.court,
        "decision_date": date, "date_source": rec["date_source"], "year": date.year,
        "judges": judges, "judge_last_names": last_names(rec["judges"] or judges, rec["judge_last"]),
        "parties": parties, "lawyers": rec["lawyers"],
        "lang": detect_lang(text),
        "header_parsed": header.parsed, "header_court": header.court,
        "header_judges": header.judges, "header_parties": header.parties,
        "header_text": header.text, "body_start": header.body_start,
        "encoding_repair": method, "text": text, "text_len": len(text),
        "source_url": source_url(cfg["source"]["source_url_template"], rec["path"], rec["file_name"]),
        "source_dataset": cfg["source"]["dataset_name"],
    }


def run_clean(cfg: dict, force: bool = False) -> dict:
    paths = paths_for(cfg)
    if not stage_done(paths.state, "filter"):
        raise SystemExit("run the filter stage first")
    settings_hash = config_hash({"clean": cfg["clean"], "filter": stage_done(paths.state, "filter").get("config_hash"),
                                 "url": cfg["source"]["source_url_template"]})
    done = stage_done(paths.state, "clean")
    if done and not force and done.get("config_hash") == settings_hash and paths.documents.exists():
        log(f"clean: done already ({done['documents']:,} documents); --force to redo")
        return done
    patterns = [re.compile(p) for p in cfg["clean"]["boilerplate_patterns"]]
    out = paths.documents_parts
    out.mkdir(parents=True, exist_ok=True)
    marker = read_json(out / "_settings.json")
    if force or (marker and marker.get("config_hash") != settings_hash):
        for p in out.glob("part-*"):
            p.unlink()
    atomic_write_json(out / "_settings.json", {"config_hash": settings_hash})

    parts = sorted(paths.filtered.glob("part-*.parquet"))
    for n, src in enumerate(parts):
        dest = out / src.name
        stats_path = out / (src.stem + ".json")
        if stats_path.exists() and dest.exists():
            continue
        records = pq.read_table(src).to_pylist()
        cleaned = [clean_record(r, cfg, patterns) for r in records]
        stats = {"documents": len(cleaned),
                 "encoding_repair": Counter(c["encoding_repair"] for c in cleaned),
                 "header_parsed": sum(c["header_parsed"] for c in cleaned),
                 "lang": Counter(c["lang"] for c in cleaned)}
        atomic_write_parquet(dest, pa.Table.from_pylist(cleaned, schema=DOCUMENT_SCHEMA))
        atomic_write_json(stats_path, stats)
        log(f"clean: {src.name} ({n + 1}/{len(parts)}): {len(cleaned):,} documents, "
            f"{stats['header_parsed']:,} headers parsed, repairs {dict(stats['encoding_repair'])}")

    total = merge_documents(out, paths.documents)
    agg = {"documents": 0, "header_parsed": 0, "encoding_repair": Counter(), "lang": Counter()}
    for p in sorted(out.glob("part-*.json")):
        s = read_json(p)
        agg["documents"] += s["documents"]
        agg["header_parsed"] += s["header_parsed"]
        agg["encoding_repair"].update(s["encoding_repair"])
        agg["lang"].update(s["lang"])
    atomic_write_json(paths.reports / "clean_stats.json", agg)
    mark_done(paths.state, "clean", config_hash=settings_hash, documents=total)
    if not cfg["clean"].get("keep_parts", False):
        for p in out.glob("part-*.parquet"):
            p.unlink()
    log(f"clean: {total:,} documents -> {paths.documents}")
    return {"documents": total}


def merge_documents(parts_dir: Path, dest: Path) -> int:
    """Stream the parts into one parquet file (one row group per part); atomic."""
    parts = sorted(parts_dir.glob("part-*.parquet"))
    if not parts and dest.exists():
        return pq.ParquetFile(dest).metadata.num_rows  # parts already removed after a finished merge
    tmp = dest.with_name(dest.name + ".tmp")
    total = 0
    with pq.ParquetWriter(tmp, DOCUMENT_SCHEMA, compression="zstd") as writer:
        for p in parts:
            table = pq.read_table(p)
            total += table.num_rows
            writer.write_table(table, row_group_size=2000)
    tmp.replace(dest)
    return total
