"""Document parsing with per-page scan-vs-native classification.

MinerU (magic-pdf) is the primary parser for PDF/DOCX/PPTX/XLSX/images. It
does the layout-aware native-text extraction (paragraph/table/formula
reconstruction), but we never trust its own OCR step -- when no `lang` is
given it silently defaults to a Chinese-biased model
(magic_pdf's `models_config.yml` "ch_lite" entry), and empirically it always
attempts to OCR-fill any character-less region internally regardless of the
`ocr=` flag passed to `doc_analyze` (that flag only tags per-page metadata,
it does not disable the internal gap-filling pass). So instead: we run
MinerU in its native/"TXT" mode, then independently reclassify every page
ourselves with the same chars-per-area heuristic the PyMuPDF fallback below
uses, computed straight from MinerU's own reported page size. Any page that
heuristic calls scanned has its MinerU-produced text DISCARDED (it may be
wrong-language OCR garbage) and gets rendered to an image instead, for our
own multilingual OCR routing in `ocr/router.py` to handle.

If MinerU is not installed, or the input type isn't one MinerU handles, we
fall back to PyMuPDF text-layer extraction with that same per-page heuristic
to decide which pages still need OCR.

Nothing here assumes a single language for the whole document -- language
detection happens per page/segment in `language_detect.py`, after parsing.
"""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
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
    # magic-pdf is MinerU's package name on PyPI. Current (>=1.x) API:
    # read_api builds a Dataset from a file, doc_analyze runs layout/formula/
    # OCR model inference over it, and the InferenceResult's pipe_txt_mode
    # produces the actual per-page text/layout ("PipeResult"). The old
    # magic_pdf.pipe.UNIPipe/OCRPipe classes this adapter used to target were
    # removed entirely somewhere before 1.3.x -- see git history for that
    # version if you need to compare against an even older magic-pdf release.
    from magic_pdf.config.enums import SupportedPdfParseMethod  # noqa: F401
    from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter  # noqa: F401
    from magic_pdf.data.read_api import read_local_images, read_local_office, read_local_pdfs  # noqa: F401
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze  # noqa: F401

    MINERU_AVAILABLE = True
except ImportError:  # pragma: no cover
    MINERU_AVAILABLE = False

MINERU_SUPPORTED_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".tiff"}
RENDER_DPI = 200


def parse_document(file_path: str | Path) -> ParsedDocument:
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    mineru_error: Exception | None = None

    if MINERU_AVAILABLE and suffix in MINERU_SUPPORTED_SUFFIXES:
        try:
            return _parse_with_mineru(file_path)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: fall back on any parser failure
            mineru_error = exc
            logger.warning("mineru_parse_failed_falling_back", path=str(file_path), error=str(exc))

    if suffix == ".pdf":
        return _parse_pdf_with_pymupdf(file_path)

    if mineru_error is not None:
        raise ValueError(
            f"No parser available for '{file_path.name}': MinerU is installed but failed to "
            f"parse this file, and PyMuPDF fallback only supports PDF. MinerU error: {mineru_error}"
        ) from mineru_error

    raise ValueError(
        f"No parser available for '{file_path.name}': MinerU is "
        f"{'available' if MINERU_AVAILABLE else 'not installed'} and PyMuPDF fallback only "
        "supports PDF. Install magic-pdf for DOCX/PPTX/XLSX/image support."
    )


def _load_mineru_dataset(file_path: Path):
    """Build MinerU's Dataset object for the given input type.

    DOCX/PPTX go through MinerU's own LibreOffice-based converter
    (`read_local_office`). XLSX isn't in that function's own suffix filter,
    but its underlying conversion primitive is format-agnostic (any type
    LibreOffice's `--convert-to pdf` handles), so we call it directly. Both
    need `soffice` (LibreOffice) on PATH -- see scripts/setup.* .
    """
    suffix = file_path.suffix.lower()
    if suffix in {".docx", ".pptx"}:
        return read_local_office(str(file_path))[0]
    if suffix in {".png", ".jpg", ".jpeg", ".tiff"}:
        return read_local_images(str(file_path), suffixes=[suffix])[0]
    if suffix == ".xlsx":
        return _load_xlsx_dataset(file_path)
    return read_local_pdfs(str(file_path))[0]


def _load_xlsx_dataset(file_path: Path):
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.utils.office_to_pdf import ConvertToPdfError, convert_file_to_pdf

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            convert_file_to_pdf(str(file_path), tmp_dir)
        except ConvertToPdfError as exc:
            raise RuntimeError(f"LibreOffice conversion failed for '{file_path.name}': {exc}") from exc
        pdf_path = Path(tmp_dir) / f"{file_path.stem}.pdf"
        return PymuDocDataset(FileBasedDataReader("").read(str(pdf_path)))


def _render_scanned_pages(doc, page_count: int, source_engine: str) -> list[Page]:
    """Image-only input: MinerU itself reports no native-text method is even
    possible for it, so there's nothing for a layout/native-text pass to
    find -- skip straight to rendering for our own OCR pipeline."""
    pages: list[Page] = []
    for i in range(page_count):
        image = None
        if doc is not None:
            pix = doc[i].get_pixmap(dpi=RENDER_DPI)
            image = PageImage(png_bytes=pix.tobytes("png"), dpi=RENDER_DPI)
        pages.append(Page(index=i, kind=PageKind.SCANNED, native_text="", image=image, source_engine=source_engine))
    return pages


def _parse_with_mineru(file_path: Path) -> ParsedDocument:
    ds = _load_mineru_dataset(file_path)

    if SupportedPdfParseMethod.TXT not in ds.supported_methods():
        doc = fitz.open(stream=ds.data_bits(), filetype="pdf") if fitz else None
        pages = _render_scanned_pages(doc, len(ds), source_engine="mineru")
        logger.info("mineru_parse_complete", path=str(file_path), pages=len(pages), image_only=True)
        return ParsedDocument(source_path=str(file_path), pages=pages)

    with tempfile.TemporaryDirectory() as tmp_dir:
        image_writer = FileBasedDataWriter(tmp_dir)
        # Always TXT mode / ocr=False: see module docstring for why we never
        # rely on MinerU's own OCR step, even for pages it ends up filling
        # in internally anyway -- we reclassify and discard below instead.
        infer_result = ds.apply(doc_analyze, ocr=False, lang=None)
        pipe_result = infer_result.pipe_txt_mode(image_writer, debug_mode=False)
        middle = json.loads(pipe_result.get_middle_json())
        content_items = pipe_result.get_content_list(image_dir_or_bucket_prefix="images")

    text_by_page: dict[int, list[str]] = defaultdict(list)
    for item in content_items:
        text = item.get("text")
        if text:
            text_by_page[item.get("page_idx", 0)].append(text)

    cfg = get_config().scan_detection
    doc = fitz.open(stream=ds.data_bits(), filetype="pdf") if fitz else None

    pages: list[Page] = []
    for i, page_info in enumerate(middle.get("pdf_info", [])):
        width, height = page_info.get("page_size", [0, 0])
        area = max(width * height, 1.0)
        native_text = "\n".join(text_by_page.get(i, []))
        chars_per_area = len(native_text.strip()) / area

        needs_ocr = chars_per_area < cfg.min_chars_per_page_area
        image = None
        if needs_ocr:
            native_text = ""  # discard MinerU's own possibly-wrong-language OCR fill-in
            if doc is not None:
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
