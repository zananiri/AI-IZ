"""Stage 2: filter the raw dataset, batch by batch, into filtered/part-NNNNN.parquet.

Never loads the file whole: pyarrow iter_batches over the needed columns only. Batch i is written
to part i with its counts beside it (part-NNNNN.json, written last = the batch is done), so a run
that stopped resumes at the first missing batch; the duplicate-hash set is rebuilt from the parts
already written. A sample run stops once sample_n documents are kept."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import schema
from .config import config_hash, paths_for
from .dates import resolve_decision_date
from .download import dataset_path
from .encoding import dedupe_key
from .state import atomic_write_json, atomic_write_parquet, atomic_write_text, log, mark_done, read_json, stage_done

FILTERED_SCHEMA = pa.schema([
    ("doc_id", pa.string()), ("case_id", pa.string()), ("citation", pa.string()),
    ("document_hash", pa.string()), ("case_name", pa.string()), ("title", pa.string()),
    ("doc_type", pa.string()), ("technical", pa.bool_()),
    ("decision_date", pa.date32()), ("date_source", pa.string()), ("case_date", pa.string()),
    ("judges", pa.list_(pa.string())), ("judge_last", pa.list_(pa.string())),
    ("parties", pa.list_(pa.string())), ("lawyers", pa.list_(pa.string())),
    ("court", pa.string()), ("division", pa.string()), ("proceeding_type", pa.string()),
    ("path", pa.string()), ("file_name", pa.string()),
    ("text", pa.large_string()), ("text_len", pa.int64()), ("text_hash", pa.string()),
])

_TYPE_NORM_RE = re.compile(r"[\s\-־–_]+")


def norm_type(value: str | None) -> str:
    if not value:
        return ""
    return _TYPE_NORM_RE.sub(" ", value.replace("״", '"').replace("”", '"').replace("“", '"')).strip()


def classify(row: dict, cols: dict[str, list[str]], fcfg: dict, keep: set[str], decision: str,
             cutoff: dt.date, min_valid: dt.date) -> tuple[str | None, dict]:
    """(drop reason or None, the normalised record)."""
    text = schema.first_value(row, cols["text"])
    text = text if isinstance(text, str) else (None if text is None else str(text))
    doc_type = schema.as_str(schema.first_value(row, cols["type"]))
    technical = schema.as_bool(schema.first_value(row, cols["technical"]))
    date_fields = {name: schema.first_value(row, cols[name])
                   for name in ("meta_verdict_dt", "VerdictsDt", "VerdictDt", "case_dt", "year")}
    rec = {
        "doc_id": schema.as_str(schema.first_value(row, cols["doc_id"])),
        "case_id": schema.as_str(schema.first_value(row, cols["case_id"])),
        "citation": schema.as_str(schema.first_value(row, cols["citation"])),
        "document_hash": schema.as_str(schema.first_value(row, cols["document_hash"])),
        "case_name": schema.as_str(schema.first_value(row, cols["case_name"])),
        "title": schema.as_str(schema.first_value(row, cols["title"])),
        "doc_type": doc_type, "technical": technical,
        "case_date": schema.as_str(date_fields["case_dt"]),
        "judges": schema.as_list(schema.first_value(row, cols["judges"])),
        "judge_last": schema.as_list(schema.first_value(row, cols["judge_last"])),
        "parties": schema.as_list(schema.first_value(row, cols["parties"])),
        "lawyers": schema.as_list(schema.first_value(row, cols["lawyers"])),
        "court": schema.as_str(schema.first_value(row, cols["court"])),
        "division": schema.as_str(schema.first_value(row, cols["division"])),
        "proceeding_type": schema.as_str(schema.first_value(row, cols["proceeding_type"])),
        "path": schema.as_str(schema.first_value(row, cols["path"])),
        "file_name": schema.as_str(schema.first_value(row, cols["file_name"])),
        "text": text, "text_len": len(text or ""),
        "decision_date": None, "date_source": None, "date_notes": [], "text_hash": None,
    }
    if not text or not text.strip():
        return "empty_text", rec
    t = norm_type(doc_type)
    if t == norm_type(decision) and fcfg.get("keep_decisions", True):
        if fcfg.get("decision_require_non_technical", True) and technical is not False:
            return "decision_technical_or_unknown", rec
        if len(text.strip()) < fcfg.get("decision_min_chars", 1500):
            return "decision_too_short", rec
    elif t not in keep:
        return "type_not_kept", rec
    date, source, notes = resolve_decision_date(date_fields, min_valid)
    rec.update(decision_date=date, date_source=source, date_notes=notes)
    if date is None:
        return "no_valid_date", rec
    if date >= cutoff:
        return "on_or_after_cutoff", rec
    return None, rec


def _part_name(i: int) -> str:
    return f"part-{i:05d}"


def _done_parts(out_dir: Path) -> list[int]:
    done = sorted(int(p.stem.split("-")[1]) for p in out_dir.glob("part-*.json"))
    contiguous = []
    for expected, got in enumerate(done):
        if got != expected:
            break
        contiguous.append(got)
    return contiguous


def run_filter(cfg: dict, force: bool = False) -> dict:
    paths = paths_for(cfg)
    fcfg = cfg["filter"]
    settings_hash = config_hash({"filter": fcfg, "sample_n": cfg.get("sample_n"), "batch": cfg["read"]["batch_size"]})
    done = stage_done(paths.state, "filter")
    if done and not force and done.get("config_hash") == settings_hash:
        log(f"filter: done already ({done['kept']:,} kept); --force to redo")
        return done
    src = dataset_path(cfg)
    if not src.exists():
        raise SystemExit(f"{src} not found: run the download stage first")
    cols = schema.check_schema(src)
    paths.reports.mkdir(parents=True, exist_ok=True)
    atomic_write_text(paths.reports / "schema.md", schema.schema_report(src, cols))

    out = paths.filtered
    out.mkdir(parents=True, exist_ok=True)
    marker = read_json(out / "_settings.json")
    if force or (marker and marker.get("config_hash") != settings_hash):
        for p in list(out.glob("part-*")):
            p.unlink()
    atomic_write_json(out / "_settings.json", {"config_hash": settings_hash})

    completed = _done_parts(out)
    seen: set[str] = set()
    kept_total = 0
    for i in completed:
        stats = read_json(out / f"{_part_name(i)}.json")
        kept_total += stats["kept"]
        if fcfg.get("dedupe", True) and (out / f"{_part_name(i)}.parquet").exists():
            seen.update(pq.read_table(out / f"{_part_name(i)}.parquet", columns=["text_hash"]).column(0).to_pylist())
    if completed:
        log(f"filter: resuming after {len(completed)} batches ({kept_total:,} kept so far)")

    keep = {norm_type(t) for t in fcfg["keep_types"]}
    cutoff = dt.date.fromisoformat(str(fcfg["max_decision_date"]))
    min_valid = dt.date.fromisoformat(str(fcfg["min_valid_date"]))
    sample_n = cfg.get("sample_n")
    needed = sorted({c for v in cols.values() for c in v})
    pf = pq.ParquetFile(src)
    total_rows = pf.metadata.num_rows
    fixes_path = paths.reports / "date_fixes.jsonl"
    max_fixes = fcfg.get("max_logged_date_fixes", 200)
    logged_fixes = sum(1 for _ in fixes_path.open(encoding="utf-8")) if completed and fixes_path.exists() else 0
    if not completed:
        fixes_path.unlink(missing_ok=True)

    for i, batch in enumerate(pf.iter_batches(batch_size=cfg["read"]["batch_size"], columns=needed)):
        if sample_n and kept_total >= sample_n:
            break
        if i in completed:
            continue
        rows = batch.to_pylist()
        del batch
        stats = {"read": 0, "kept": 0, "dropped": Counter(), "type_in": Counter(), "type_kept": Counter(),
                 "year_kept": Counter(), "proceeding_kept": Counter(), "date_source": Counter(), "date_fixed": 0}
        kept_rows = []
        fixes = []
        for row in rows:
            stats["read"] += 1
            reason, rec = classify(row, cols, fcfg, keep, fcfg["decision_type"], cutoff, min_valid)
            stats["type_in"][rec["doc_type"] or "(none)"] += 1
            if rec["date_notes"]:
                stats["date_fixed"] += 1
                if logged_fixes + len(fixes) < max_fixes:
                    fixes.append({"doc_id": rec["doc_id"], "citation": rec["citation"], "resolved": str(rec["decision_date"]),
                                  "source": rec["date_source"], "notes": rec["date_notes"]})
            if reason is None and fcfg.get("dedupe", True):
                h = hashlib.sha1(dedupe_key(rec["text"]).encode("utf-8")).hexdigest()
                rec["text_hash"] = h
                if h in seen:
                    reason = "duplicate_text"
                else:
                    seen.add(h)
            if reason:
                stats["dropped"][reason] += 1
                continue
            if rec["text_hash"] is None:
                rec["text_hash"] = hashlib.sha1(dedupe_key(rec["text"]).encode("utf-8")).hexdigest()
            if not rec["document_hash"]:
                rec["document_hash"] = rec["text_hash"]
            if not rec["doc_id"]:
                rec["doc_id"] = rec["document_hash"]
            stats["kept"] += 1
            stats["type_kept"][rec["doc_type"] or "(none)"] += 1
            stats["year_kept"][str(rec["decision_date"].year)] += 1
            stats["proceeding_kept"][rec["proceeding_type"] or "(none)"] += 1
            stats["date_source"][rec["date_source"]] += 1
            kept_rows.append(rec)
            if sample_n and kept_total + stats["kept"] >= sample_n:
                break
        table = pa.Table.from_pylist([{k: r[k] for k in FILTERED_SCHEMA.names} for r in kept_rows], schema=FILTERED_SCHEMA)
        atomic_write_parquet(out / f"{_part_name(i)}.parquet", table)
        if fixes:
            with fixes_path.open("a", encoding="utf-8") as fh:
                for fix in fixes:
                    fh.write(json.dumps(fix, ensure_ascii=False) + "\n")
            logged_fixes += len(fixes)
        atomic_write_json(out / f"{_part_name(i)}.json", stats)
        kept_total += stats["kept"]
        log(f"filter: batch {i} ({(i + 1) * cfg['read']['batch_size']:,}/{total_rows:,} rows read): "
            f"kept {stats['kept']:,} (total {kept_total:,})")

    report = write_filter_report(cfg, out)
    mark_done(paths.state, "filter", config_hash=settings_hash, kept=report["kept"], read=report["read"])
    return report


def aggregate(out: Path) -> dict:
    total = {"read": 0, "kept": 0, "date_fixed": 0}
    counters = {k: Counter() for k in ("dropped", "type_in", "type_kept", "year_kept", "proceeding_kept", "date_source")}
    for p in sorted(out.glob("part-*.json")):
        stats = read_json(p)
        for k in total:
            total[k] += stats[k]
        for k in counters:
            counters[k].update(stats[k])
    return {**total, **counters}


def _table(counter: Counter, header: tuple[str, str], sort_by_key: bool = False, limit: int | None = None) -> list[str]:
    items = sorted(counter.items()) if sort_by_key else counter.most_common(limit)
    return [f"| {header[0]} | {header[1]} |", "|---|--:|"] + [f"| {k} | {v:,} |" for k, v in items]


def write_filter_report(cfg: dict, out: Path) -> dict:
    paths = paths_for(cfg)
    agg = aggregate(out)
    fcfg = cfg["filter"]
    type_rows = ["| Type | read | kept | dropped |", "|---|--:|--:|--:|"]
    for t, n in agg["type_in"].most_common():
        k = agg["type_kept"].get(t, 0)
        type_rows.append(f"| {t} | {n:,} | {k:,} | {n - k:,} |")
    lines = [
        "# Filter report", "",
        f"Source: `{cfg['source']['repo_id']}` / `{cfg['source']['filename']}`"
        + (f" -- **sample run: first {cfg['sample_n']:,} kept documents**" if cfg.get("sample_n") else ""), "",
        f"- rows read: **{agg['read']:,}**",
        f"- kept: **{agg['kept']:,}** ({agg['kept'] / max(agg['read'], 1):.1%})",
        f"- dropped: **{agg['read'] - agg['kept']:,}**",
        f"- rows whose first date field was bogus or unparseable and fell back: {agg['date_fixed']:,} "
        "(examples in `date_fixes.jsonl`)", "",
        "Rules: decision date < " + str(fcfg["max_decision_date"]) + "; Type in " + ", ".join(fcfg["keep_types"])
        + f"; {fcfg['decision_type']} only if non-technical and >= {fcfg['decision_min_chars']:,} chars; "
        "no empty text; exact duplicates (normalised text hash) dropped.", "",
        "## Drop reasons", "", *_table(agg["dropped"], ("reason", "rows")), "",
        "## Per Type", "", *type_rows, "",
        "## Kept per year", "", *_table(agg["year_kept"], ("year", "kept"), sort_by_key=True), "",
        "## Kept per proceeding type (meta_inyan_nm)", "", *_table(agg["proceeding_kept"], ("proceeding", "kept"), limit=60), "",
        "## Decision date taken from", "", *_table(agg["date_source"], ("field", "kept")), "",
    ]
    atomic_write_text(paths.reports / "filter_report.md", "\n".join(lines))
    log(f"filter: {agg['kept']:,} of {agg['read']:,} rows kept -> {paths.reports / 'filter_report.md'}")
    return {"read": agg["read"], "kept": agg["kept"]}
