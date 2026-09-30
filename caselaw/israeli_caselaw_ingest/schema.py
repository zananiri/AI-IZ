"""The dataset's columns, checked against the actual file.

Each logical field lists the columns that may hold it, best first; a row's value is the first
non-empty one among the columns the file actually has. The resolved mapping is written to
reports/schema.md so the column list can be verified before anything relies on it."""

from __future__ import annotations

import ast
import datetime as dt
from pathlib import Path

import pyarrow.parquet as pq

FIELDS: dict[str, list[str]] = {
    "doc_id": ["Id", "id"],
    "case_id": ["CaseId", "case_id"],
    "citation": ["CaseDesc", "meta_case_nbr"],
    "document_hash": ["document_hash"],
    "text": ["text"],
    "title": ["html_title"],
    "case_name": ["CaseName", "meta_case_nm"],
    "type": ["Type", "meta_verdict_ty", "TypeCode"],
    "technical": ["Technical", "meta_is_technical"],
    # Date fields are kept apart: resolution order and time-zone handling differ per field.
    "meta_verdict_dt": ["meta_verdict_dt"],
    "VerdictsDt": ["VerdictsDt"],
    "VerdictDt": ["VerdictDt"],
    "case_dt": ["meta_case_dt"],
    "year": ["Year"],
    "judges": ["meta_judge"],
    "judge_last": ["meta_judge_nm_last"],
    "parties": ["meta_side_nm"],
    "lawyers": ["meta_lawyer_nm"],
    "court": ["meta_court_nm"],
    "division": ["meta_mador_nm"],
    "proceeding_type": ["meta_inyan_nm"],
    "path": ["Path"],
    "file_name": ["FileName", "file_name"],
}

REQUIRED = ("text", "type")
DATE_FIELDS = ("meta_verdict_dt", "VerdictsDt", "VerdictDt")


def resolve_columns(names: list[str]) -> dict[str, list[str]]:
    """Logical field -> the candidate columns the file has (case-sensitive match first, then
    case-insensitive)."""
    lower = {n.lower(): n for n in names}
    resolved = {}
    for field, candidates in FIELDS.items():
        found = []
        for c in candidates:
            if c in names:
                found.append(c)
            elif c.lower() in lower and lower[c.lower()] not in found:
                found.append(lower[c.lower()])
        resolved[field] = found
    return resolved


def check_schema(parquet_path: Path) -> dict[str, list[str]]:
    names = pq.ParquetFile(parquet_path).schema_arrow.names
    resolved = resolve_columns(names)
    missing = [f for f in REQUIRED if not resolved[f]]
    if not any(resolved[f] for f in DATE_FIELDS) and not resolved["year"]:
        missing.append("a decision date (meta_verdict_dt / VerdictsDt / VerdictDt / Year)")
    if missing:
        raise SystemExit(f"{parquet_path} lacks {', '.join(missing)}; its columns are: {names}")
    return resolved


def schema_report(parquet_path: Path, resolved: dict[str, list[str]]) -> str:
    pf = pq.ParquetFile(parquet_path)
    lines = [f"# Dataset schema: {parquet_path.name}", "",
             f"{pf.metadata.num_rows:,} rows, {pf.metadata.num_row_groups} row groups.", "",
             "| column | type |", "|---|---|"]
    lines += [f"| `{f.name}` | {f.type} |" for f in pf.schema_arrow]
    lines += ["", "## Logical fields", "", "| field | columns used (first non-empty wins) |", "|---|---|"]
    lines += [f"| {k} | {', '.join(f'`{c}`' for c in v) or '**missing**'} |" for k, v in resolved.items()]
    used = {c for v in resolved.values() for c in v}
    unused = [n for n in pf.schema_arrow.names if n not in used]
    if unused:
        lines += ["", "Columns not used: " + ", ".join(f"`{n}`" for n in unused)]
    return "\n".join(lines) + "\n"


def first_value(row: dict, columns: list[str]):
    for c in columns:
        v = row.get(c)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        if isinstance(v, (list, tuple)) and not v:
            continue
        return v
    return None


def as_str(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    s = str(v).strip()
    return s or None


def as_list(v) -> list[str]:
    """A list column, or its string form ("['a', 'b']", "a, b", "a; b")."""
    if v is None:
        return []
    if hasattr(v, "tolist") and not isinstance(v, str):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [s for s in (str(x).strip() for x in v if x is not None) if s]
    s = str(v).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (ValueError, SyntaxError):
            s = s[1:-1]
    sep = ";" if ";" in s else ","
    return [p.strip().strip("'\"") for p in s.split(sep) if p.strip().strip("'\"")]


def as_bool(v) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "t", "y"):
        return True
    if s in ("false", "0", "no", "f", "n"):
        return False
    return None
