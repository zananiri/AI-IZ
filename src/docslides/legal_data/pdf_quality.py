"""PDF text for the corpus, with a quality verdict. Extraction is legal/pdf_text.extract_pdf_text
(right-to-left reconstruction, scanned-page detection); encoding faults go through
legal_data/hebrew.repair_text. `extraction="low"` marks text that shouldn't be trusted as the
document's full content: scanned pages, too few characters per page, or a failed extraction."""

from __future__ import annotations

from pathlib import Path

from docslides.legal_data.hebrew import hebrew_ratio, repair_text
from docslides.legal_data.records import Quality

_MIN_CHARS_PER_PAGE = 200


def extract_pdf_with_quality(path: Path) -> tuple[str, Quality]:
    import pymupdf

    from docslides.legal.pdf_text import extract_pdf_text

    try:
        text, scanned = extract_pdf_text(path)
        with pymupdf.open(path) as document:
            pages = document.page_count
    except Exception as exc:  # noqa: BLE001 -- a broken PDF is reported, not fatal
        return "", Quality(extraction="low", notes=[f"text extraction failed: {type(exc).__name__}: {exc}"])
    text, encoding, notes = repair_text(text)
    extraction = "ok"
    if scanned:
        extraction = "low"
        notes.append(f"pages without a text layer (scanned): {scanned}")
    per_page = len(text.strip()) / max(pages, 1)
    if per_page < _MIN_CHARS_PER_PAGE:
        extraction = "low"
        notes.append(f"only {per_page:.0f} characters per page")
    if len(text) > 200 and hebrew_ratio(text) < 0.3:
        notes.append("little Hebrew text")
    return text, Quality(encoding=encoding, extraction=extraction, notes=notes)
