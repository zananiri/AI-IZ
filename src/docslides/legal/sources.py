"""Loading legal source files for staging (legal/staging.py).

Official sources (statutes, regulations, rulings) are local files -- .txt,
.md, .docx or text-layer .pdf -- each with a `<file>.meta.json` sidecar
naming the law and its version, e.g.:

    {
      "law_id": "contracts-general-1973",
      "law_name": "חוק החוזים (חלק כללי), תשל\"ג-1973",
      "effective_date_start": "1973-06-01",
      "effective_date_end": null,
      "status": "current",
      "source_type": "statute",
      "source_origin": "knesset",
      "language": "he"
    }

Nothing is fetched from the web here: get official texts from the source
sites yourself, within their terms of use, and drop them in
config.legal.ingestion.sources_dir.

Uploads (the uploads/ folder: memos, firm materials, client documents)
don't need a sidecar. Whatever a sidecar says, they are always tagged
source_type "uploaded_document" / source_origin "manual_upload".

Scanned PDF pages are refused rather than skipped: a chunk set with pages
silently missing would look complete to the reviewer. OCR them first.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel

from docslides.legal.chunking import SourceMeta
from docslides.legal.models import SourceOrigin, SourceType, Status

SUPPORTED_SUFFIXES = {".txt", ".md", ".docx", ".pdf"}
_OFFICIAL_ORIGINS = {"knesset", "reshumot", "court_gov_il"}


class _SidecarMeta(BaseModel):
    law_id: str | None = None
    law_name: str
    effective_date_start: date
    effective_date_end: date | None = None
    status: Status = "current"
    source_type: SourceType
    source_origin: SourceOrigin
    language: str = "he"
    gazette: str | None = None


class _UploadSidecarMeta(BaseModel):
    law_id: str | None = None
    law_name: str | None = None
    effective_date_start: date | None = None
    effective_date_end: date | None = None
    status: Status = "current"
    language: str = "he"


@dataclass
class LoadedSource:
    path: Path
    sha256: str
    text: str
    meta: SourceMeta


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def is_source_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and not path.name.endswith(".meta.json")


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".docx":
        import docx

        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    if suffix == ".pdf":
        # Rebuilt from character positions (legal/pdf_text.py), not ingestion/parser.py:
        # MinerU is a slow layout model built for slide decks, its scan heuristic calls
        # a law's short final page "scanned", and plain text-layer order mangles
        # right-to-left lines, duplicated pages and margin notes in Reshumot PDFs.
        from docslides.legal.pdf_text import extract_pdf_text

        text, scanned = extract_pdf_text(path)
        if scanned:
            raise ValueError(
                f"{path.name}: pages {scanned} have no text layer (scanned). OCR the file and "
                "stage the text version instead -- staging it now would silently drop those pages."
            )
        if looks_visual_order(text):
            raise ValueError(
                f"{path.name}: the PDF's Hebrew text layer is stored in visual (reversed) order, so "
                "extracted words come out backwards. Export the law as text/DOCX (or get it from a "
                "source that publishes logical-order text) and use that instead."
            )
        return text
    raise ValueError(f"Unsupported legal source type: {path.name}")


_FINAL_LETTERS = "ךםןףץ"


def looks_visual_order(text: str) -> bool:
    """Hebrew final letters (ך ם ן ף ץ) only occur word-finally. When a PDF
    stores text in visual order, extraction reverses each word and they pile
    up at word starts instead -- a reliable tell on any real legal text."""
    words = re.findall(r"[א-ת]{2,}", text)
    starts = sum(w[0] in _FINAL_LETTERS for w in words)
    ends = sum(w[-1] in _FINAL_LETTERS for w in words)
    return starts >= 20 and starts > 2 * ends


def _slug(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def load_official_source(path: Path, meta_path: Path | None = None) -> LoadedSource:
    meta_path = meta_path or sidecar_path(path)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{path.name}: official sources need a metadata sidecar at {meta_path.name} "
            "(law_name, effective_date_start, status, source_type, source_origin) -- see legal/sources.py"
        )
    return official_source(path, json.loads(meta_path.read_text(encoding="utf-8")))


def official_source(path: Path, sidecar_data: dict, text: str | None = None) -> LoadedSource:
    """An official source from already-loaded sidecar values (and, optionally,
    already-extracted text -- PDF extraction can be slow)."""
    sidecar = _SidecarMeta.model_validate(sidecar_data)
    if sidecar.source_type == "uploaded_document" or sidecar.source_origin not in _OFFICIAL_ORIGINS:
        raise ValueError(f"{path.name}: official sources can't be tagged as uploads -- put the file in uploads/")
    meta = SourceMeta(
        law_id=sidecar.law_id or f"law-{_slug(sidecar.law_name)}",
        law_name=sidecar.law_name,
        effective_date_start=sidecar.effective_date_start.isoformat(),
        effective_date_end=sidecar.effective_date_end.isoformat() if sidecar.effective_date_end else None,
        status=sidecar.status,
        source_type=sidecar.source_type,
        source_origin=sidecar.source_origin,
        language=sidecar.language,
        gazette=sidecar.gazette,
    )
    return LoadedSource(
        path=path, sha256=file_sha256(path), text=text if text is not None else extract_text(path), meta=meta
    )


def load_upload(path: Path) -> LoadedSource:
    meta_path = sidecar_path(path)
    sidecar = (
        _UploadSidecarMeta.model_validate(json.loads(meta_path.read_text(encoding="utf-8")))
        if meta_path.exists()
        else _UploadSidecarMeta()
    )
    modified = datetime.fromtimestamp(path.stat().st_mtime).date()
    meta = SourceMeta(
        law_id=sidecar.law_id or f"upload-{_slug(path.name)}",
        law_name=sidecar.law_name or path.stem,
        effective_date_start=(sidecar.effective_date_start or modified).isoformat(),
        effective_date_end=sidecar.effective_date_end.isoformat() if sidecar.effective_date_end else None,
        status=sidecar.status,
        source_type="uploaded_document",
        source_origin="manual_upload",
        language=sidecar.language,
    )
    return LoadedSource(path=path, sha256=file_sha256(path), text=extract_text(path), meta=meta)
