from pathlib import Path

from docslides.ingestion.models import PageKind
from docslides.ingestion.parser import _parse_pdf_with_pymupdf

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def test_scanned_arabic_pdf_is_flagged_scanned():
    parsed = _parse_pdf_with_pymupdf(FIXTURES_DIR / "scanned_arabic.pdf")
    assert parsed.page_count == 1
    page = parsed.pages[0]
    assert page.kind == PageKind.SCANNED
    assert page.image is not None
    assert page.native_text == ""
