"""Drop-in ingestion for the Legal tab: every law PDF placed in
config.legal.ingestion.legal_txt_dir (./legal_txt by default) is parsed,
chunked, embedded and signed into the index in one run -- as an official
source, so it's staged and auto-approved with reviewed_by left empty.
Superseding of older versions happens in that approval (legal/staging.py).
Run with scripts/ingest_legal_txt.py.

Metadata: if a PDF has no `<file>.meta.json` sidecar, one is derived from
the text and written next to the PDF for you to check:
  * law_name -- the first title-like line ("חוק ...", "פקודת ...", "תקנות ...",
    "צו ..."), preferring one that carries a year;
  * effective_date_start -- the gazette's publication date from its header
    ("16 ביולי 2026"); else January 1st of the title's year; else the file's
    modification date;
  * gazette -- "ספר החוקים 3546" / "קובץ התקנות N" from the header;
  * source_type/source_origin -- regulation/reshumot for תקנות or צו,
    statute/knesset otherwise.
These are guesses, and the sidecar lists them under "_derived". Edit the
sidecar and re-run: the changed metadata is detected, the guessed version is
retracted, and the corrected one replaces it.

One consequence to know about: the title of an updated consolidated text
usually still shows the enactment year, so it derives the SAME version as
the original and replaces it rather than superseding it. To keep both as
versions, set the new file's effective_date_start in its sidecar.

State (which file produced which law version, by content hash) lives in
staging_dir/_legal_txt_state.json, so unchanged files are skipped.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from docslides.config import get_config
from docslides.legal import bundle, staging
from docslides.legal.sources import (
    extract_text,
    file_sha256,
    is_source_file,
    official_source,
    sidecar_path,
)

_STATE_FILE = "_legal_txt_state.json"
# The bulk corpus (scripts/legal_data/) lives in these subfolders: thousands of downloaded files that
# are vectorized into their own store, never signed into this curated index.
CORPUS_SUBDIRS = frozenset({"laws", "procedural_rules", "supreme_court", "metadata", "_manifests", "_sample"})
_TITLE_RE = re.compile(r"^(?:חוק|פקודת|פקודה|תקנות|צו)\s")
_YEAR_RE = re.compile(r"(?:[-–]\s*|\b)((?:19|20)\d\d)\b")
_HEBREW_MONTHS = [
    "ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני", "יולי", "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר",
    "", "", "מרס",  # alternate spelling of March, index 14 -> month 3
]
_GAZETTE_DATE_RE = re.compile(
    r"\b(\d{1,2})\b(?:\s+\d{3,5})?\s+ב(" + "|".join(m for m in _HEBREW_MONTHS if m) + r")\s+((?:19|20)\d\d)\b"
)
# "ספר החוקים" / "קובץ התקנות" + issue number, allowing the header's other column in between.
_GAZETTE_RE = re.compile(r"(ספר החוקים|קובץ התקנות|ילקוט הפרסומים)[^\n]*?\n?[^\n]*?\b(\d{3,5})\b(?!\s*ב?(?:19|20)\d\d)")
_DERIVED_NOTE = (
    "Auto-derived by scripts/ingest_legal_txt.py -- check the fields listed in _derived "
    "(especially effective_date_start), edit them, and re-run the script to re-index."
)

Action = Literal["indexed", "unchanged", "failed", "dry_run", "retracted", "missing"]


@dataclass
class FileResult:
    path: Path
    action: Action
    message: str = ""
    law_name: str = ""
    batch_id: str | None = None
    chunk_count: int = 0
    derived_fields: list[str] = field(default_factory=list)
    superseded: list[str] = field(default_factory=list)


def _state_path() -> Path:
    return Path(get_config().legal.ingestion.staging_dir) / _STATE_FILE


def _load_state() -> dict[str, dict]:
    path = _state_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_state(state: dict[str, dict]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def _gazette_date(head: str) -> str | None:
    """Publication date from a gazette header ("16 ביולי 2026"; the header's
    columns can put the issue number between day and month)."""
    match = _GAZETTE_DATE_RE.search(head)
    if not match:
        return None
    day, month, year = int(match.group(1)), _HEBREW_MONTHS.index(match.group(2)) % 12 + 1, int(match.group(3))
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def derive_sidecar(path: Path, text: str) -> dict:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    head = "\n".join(lines[:60])
    # A title wrapped onto a second line that starts with its Hebrew year: join them.
    joined: list[str] = []
    for line in lines[:60]:
        if joined and _TITLE_RE.match(joined[-1]) and joined[-1].endswith(",") and line.startswith("התש"):
            joined[-1] = f"{joined[-1]} {line}"
        else:
            joined.append(line)
    candidates = [
        re.sub(r"[\s*]+\d{0,4}$", "", line)  # a table-of-contents page number / footnote star
        for line in joined
        if _TITLE_RE.match(line) and len(line) <= 200
    ]
    with_year = [line for line in candidates if _YEAR_RE.search(line)]
    title = (with_year or candidates or lines[:1] or [path.stem])[0]

    derived = ["law_name", "effective_date_start", "source_type", "source_origin"]
    effective = _gazette_date(head)
    if effective is None:
        year = _YEAR_RE.search(title)
        effective = (
            f"{year.group(1)}-01-01" if year else datetime.fromtimestamp(path.stat().st_mtime).date().isoformat()
        )
    gazette_match = _GAZETTE_RE.search(head)
    is_regulation = title.startswith(("תקנות", "צו"))
    hebrew = len(re.findall(r"[א-ת]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    sidecar = {
        "law_name": title,
        "effective_date_start": effective,
        "effective_date_end": None,
        "status": "current",
        "source_type": "regulation" if is_regulation else "statute",
        "source_origin": "reshumot" if is_regulation else "knesset",
        "language": "he" if hebrew >= latin else "en",
    }
    if gazette_match:
        sidecar["gazette"] = f"{gazette_match.group(1)} {gazette_match.group(2)}"
        derived.append("gazette")
    return {**sidecar, "_derived": derived, "_note": _DERIVED_NOTE}


def _digest(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _process(path: Path, state: dict[str, dict], dry_run: bool) -> FileResult:
    key = str(path.resolve())
    sidecar_file = sidecar_path(path)
    text: str | None = None
    if sidecar_file.exists():
        sidecar = json.loads(sidecar_file.read_text(encoding="utf-8"))
    else:
        text = extract_text(path)
        sidecar = derive_sidecar(path, text)
        if not dry_run:
            sidecar_file.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")

    pdf_sha, meta_sha = file_sha256(path), _digest(sidecar)
    previous = state.get(key)
    if previous and previous["pdf_sha256"] == pdf_sha and previous["meta_sha256"] == meta_sha and not dry_run:
        return FileResult(path, "unchanged", law_name=sidecar.get("law_name", ""))

    source = official_source(path, sidecar, text=text)
    derived = list(sidecar.get("_derived", []))
    if dry_run:
        chunks = staging.build_chunks(source)
        return FileResult(
            path, "dry_run", law_name=source.meta.law_name, chunk_count=len(chunks), derived_fields=derived,
            message=f"version {source.meta.version_id}, {len({c.metadata.section_key for c in chunks})} sections",
        )

    version = (source.meta.law_id, source.meta.effective_date_start)
    warnings = []
    if previous and (previous["law_id"], previous["effective_date_start"]) != version:
        staging.retract_version(previous["law_id"], previous["effective_date_start"])
        warnings.append(f"replaced previous version {previous['law_id']}@{previous['effective_date_start']}")
    for other_key, other in state.items():
        if other_key != key and (other["law_id"], other["effective_date_start"]) == version:
            warnings.append(
                f"same law and effective date as {Path(other_key).name} -- this file replaces its text; "
                "set effective_date_start in the sidecar to keep both as separate versions"
            )

    staging.discard_pending(source.sha256, reason="superseded by a legal_txt re-run")
    batch = staging.approve(staging.stage(source).batch_id, reviewer=None)
    state[key] = {
        "pdf_sha256": pdf_sha,
        "meta_sha256": meta_sha,
        "law_id": source.meta.law_id,
        "effective_date_start": source.meta.effective_date_start,
        "batch_id": batch.batch_id,
    }
    return FileResult(
        path, "indexed", message="; ".join(warnings), law_name=batch.law_name, batch_id=batch.batch_id,
        chunk_count=len(batch.chunks), derived_fields=derived, superseded=batch.superseded,
    )


def run(dry_run: bool = False, prune: bool = False) -> list[FileResult]:
    """Indexes new/changed files in legal_txt_dir. With `prune`, versions
    whose file was deleted from the folder are retracted from the index;
    without it they're only reported."""
    if not dry_run:
        bundle.require_key()
    folder = Path(get_config().legal.ingestion.legal_txt_dir)
    state = _load_state()
    results: list[FileResult] = []

    def in_corpus_subdir(path: Path) -> bool:
        return path.relative_to(folder).parts[0] in CORPUS_SUBDIRS

    for path in sorted(p for p in folder.rglob("*") if is_source_file(p) and not in_corpus_subdir(p)):
        try:
            results.append(_process(path, state, dry_run))
        except Exception as exc:  # noqa: BLE001 -- one bad file mustn't block the rest
            results.append(FileResult(path, "failed", message=str(exc)))
        if not dry_run:
            _save_state(state)  # after every file, so an interrupted run resumes cleanly

    for key in [k for k in state if not Path(k).exists()]:
        entry = state[key]
        if prune and not dry_run:
            changes = staging.retract_version(entry["law_id"], entry["effective_date_start"])
            del state[key]
            _save_state(state)
            results.append(FileResult(Path(key), "retracted", superseded=changes))
        else:
            results.append(FileResult(Path(key), "missing", message="file deleted; still indexed (use --prune)"))
    return results
