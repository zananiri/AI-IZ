"""Document parsing with per-page scan-vs-native classification.

MinerU (magic-pdf) is the primary parser for PDF/DOCX/PPTX/XLSX/images: it
classifies each page individually and only scanned pages are routed to OCR.
If MinerU is not installed, or the input type isn't one MinerU handles, we
fall back to PyMuPDF text-layer extraction with a per-page heuristic
(extractable chars per page area) to decide which pages still need OCR.

Nothing here assumes a single language for the whole document -- language
detection happens per page/segment in `language_detect.py`, after parsing.
"""

from __future__ import annotations

import io
from pathlib import Path

from docslides.config import get_config
from docslides.ingestion.models import Page, PageImage, PageKind, ParsedDocument
from docslides.logging_setup import get_logger

logger = get_logger(__name__)

try:
    import pymupdf as fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

try:
    # magic-pdf is MinerU's package name on PyPI.
    from magic_pdf.pipe.UNIPipe import UNIPipe  # type: ignore
    from magic_pdf.pipe.OCRPipe import OCRPipe  # type: ignore

    MINERU_AVAILABLE = True
except ImportError:  # pragma: no cover
    MINERU_AVAILABLE = False

MINERU_SUPPORTED_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".tiff"}
RENDER_DPI = 200


def parse_document(file_path: str | Path) -> ParsedDocument:
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if MINERU_AVAILABLE and suffix in MINERU_SUPPORTED_SUFFIXES:
        try:
            return _parse_with_mineru(file_path)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: fall back on any parser failure
            logger.warning("mineru_parse_failed_falling_back", path=str(file_path), error=str(exc))

    if suffix == ".pdf":
        return _parse_pdf_with_pymupdf(file_path)

    raise ValueError(
        f"No parser available for '{file_path.name}': MinerU is "
        f"{'available' if MINERU_AVAILABLE else 'not installed'} and PyMuPDF fallback only "
        "supports PDF. Install magic-pdf for DOCX/PPTX/XLSX/image support."
    )


def _parse_with_mineru(file_path: Path) -> ParsedDocument:
    """Adapter over MinerU's per-page pipeline.

    MinerU's `UNIPipe` classifies each page as text-based or OCR-needed as
    part of its layout analysis (`pdf_type` / per-page `need_ocr` markers in
    its middle-JSON output). We request its analysis, then translate that
    per-page verdict into our own `Page` model rather than letting MinerU run
    its own (non-multilingual-aware) OCR step -- our OCR routing in
    `ocr/router.py` handles language-specific engine selection instead.

    NOTE: MinerU's Python API has changed across releases; verify this
    adapter against the installed `magic-pdf` version's UNIPipe signature
    before relying on it in production (see README "Model setup" section).
    """
    pdf_bytes = file_path.read_bytes()
    pipe = UNIPipe(pdf_bytes, {"_pdf_type": "", "model_list": []}, image_writer=None)
    pipe.pipe_classify()
    pipe.pipe_analyze()
    middle_json = pipe.pipe_mk_uni_format(img_parent_path="", drop_mode="none")

    pages: list[Page] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf") if fitz else None

    for i, page_info in enumerate(middle_json.get("pdf_info", [])):
        needs_ocr = bool(page_info.get("need_ocr", False))
        native_text = "\n".join(
            block.get("text", "")
            for block in page_info.get("preproc_blocks", [])
            if "text" in block
        )
        image = None
        if needs_ocr and doc is not None:
            pix = doc[i].get_pixmap(dpi=RENDER_DPI)
            image = PageImage(png_bytes=pix.tobytes("png"), dpi=RENDER_DPI)

        pages.append(
            Page(
                index=i,
                kind=PageKind.SCANNED if needs_ocr else PageKind.NATIVE_TEXT,
                native_text=native_text,
                image=image,
                source_engine="mineru",
            )
        )

    logger.info("mineru_parse_complete", path=str(file_path), pages=len(pages))
    return ParsedDocument(source_path=str(file_path), pages=pages)


def _parse_pdf_with_pymupdf(file_path: Path) -> ParsedDocument:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not installed; cannot parse PDF without MinerU.")

    cfg = get_config().scan_detection
    doc = fitz.open(file_path)
    pages: list[Page] = []

    for i, pdf_page in enumerate(doc):
        text = pdf_page.get_text("text")
        rect = pdf_page.rect
        area = max(rect.width * rect.height, 1.0)
        chars_per_area = len(text.strip()) / area

        needs_ocr = chars_per_area < cfg.min_chars_per_page_area
        image = None
        if needs_ocr:
            pix = pdf_page.get_pixmap(dpi=RENDER_DPI)
            image = PageImage(png_bytes=pix.tobytes("png"), dpi=RENDER_DPI)

        pages.append(
            Page(
                index=i,
                kind=PageKind.SCANNED if needs_ocr else PageKind.NATIVE_TEXT,
                native_text=text if not needs_ocr else "",
                image=image,
                source_engine="pymupdf",
            )
        )

    logger.info("pymupdf_parse_complete", path=str(file_path), pages=len(pages))
    return ParsedDocument(source_path=str(file_path), pages=pages)
