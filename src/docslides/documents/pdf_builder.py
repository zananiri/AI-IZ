"""Render a `PdfSpec` to a PDF with PyMuPDF's Story (HTML/CSS layout, already
a core dependency): headings, paragraphs, bullet lists and tables flow across
A4 pages. MuPDF's HTML engine does the bidi and shaping, so Hebrew and Arabic
read right to left with joined letters, and its bundled Noto fallback fonts
cover the scripts the base font lacks -- nothing extra to install, offline."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pymupdf

from docslides.config import get_config
from docslides.llm.schemas import PdfSpec
from docslides.logging_setup import get_logger

logger = get_logger(__name__)

_MARGIN = 54  # 0.75 inch
_CSS = """
body { font-family: sans-serif; font-size: 11pt; line-height: 1.4; color: #1f2937; }
h1 { font-size: 22pt; color: #c2410c; margin-bottom: 2pt; }
.subtitle { font-size: 13pt; color: #6b7280; margin-top: 0; }
h2 { font-size: 15pt; color: #c2410c; margin-top: 14pt; border-bottom: 1px solid #f97316; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; }
th { background-color: #f97316; color: #ffffff; font-weight: bold; }
th, td { border: 1px solid #d1d5db; padding: 3pt 5pt; }
.bullet { margin: 2pt 12pt; }
"""


def _html(spec: PdfSpec, rtl: bool) -> str:
    """MuPDF lays text out right to left under dir="rtl", but not list markers or table columns:
    for RTL, bullets are paragraphs with the marker after the text (MuPDF keeps a leading neutral
    on the left, even after an RLM) and each table row is written last column first."""
    direction = "rtl" if rtl else "ltr"
    parts = [f'<div dir="{direction}"><h1>{escape(spec.title)}</h1>']
    if spec.subtitle:
        parts.append(f'<p class="subtitle">{escape(spec.subtitle)}</p>')
    for section in spec.sections:
        parts.append(f"<h2>{escape(section.heading)}</h2>")
        parts.extend(f"<p>{escape(p)}</p>" for p in section.paragraphs if p.strip())
        if section.bullets and rtl:
            parts.extend(f'<p class="bullet">{escape(b)} \u2022</p>' for b in section.bullets)
        elif section.bullets:
            items = "".join(f"<li>{escape(b)}</li>" for b in section.bullets)
            parts.append(f"<ul>{items}</ul>")
        if section.table and section.table.columns:
            width = len(section.table.columns)

            def ordered(cells: list[str]) -> list[str]:
                cells = (cells + [""] * width)[:width]
                return cells[::-1] if rtl else cells

            header = "".join(f"<th>{escape(c)}</th>" for c in ordered(section.table.columns))
            body = "".join(
                "<tr>" + "".join(f"<td>{escape(c)}</td>" for c in ordered(row)) + "</tr>"
                for row in section.table.rows
            )
            parts.append(f"<table><tr>{header}</tr>{body}</table>")
    parts.append("</div>")
    return "".join(parts)


def build_pdf(spec: PdfSpec, output_path: str | Path, lang: str = "en") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rtl = get_config().languages.is_rtl(lang)

    story = pymupdf.Story(html=_html(spec, rtl), user_css=_CSS)
    page_rect = pymupdf.paper_rect("a4")
    content_rect = page_rect + (_MARGIN, _MARGIN, -_MARGIN, -_MARGIN)
    writer = pymupdf.DocumentWriter(str(output_path))
    pages, more = 0, True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(content_rect)
        story.draw(device)
        writer.end_page()
        pages += 1
    writer.close()

    with pymupdf.open(str(output_path)) as doc:
        doc.set_metadata({"title": spec.title, "creator": "Clara - Offline Large Language Model"})
        doc.saveIncr()
    logger.info("pdf_built", output=str(output_path), pages=pages)
    return output_path
