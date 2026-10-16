"""Generate a file the user asked for in the chat: an Excel workbook, a
PowerPoint deck or a PDF document.

Gemma 4 writes the content as a guided-JSON spec (llm/schemas.py:
`SpreadsheetSpec`, `PresentationSpec`, `PdfSpec`) -- from the request alone,
or from an attached document's text -- and a deterministic builder renders it:
openpyxl (documents/xlsx_builder.py), python-pptx through the slide pipeline's
own builder (slides/pptx_builder.py, so the configured .potx template and RTL
handling apply), or PyMuPDF (documents/pdf_builder.py). The model never writes
file bytes or code, so every file opens cleanly and nothing it generates is
executed.

The content is written directly in the requested language: each spec is one
structured reply, which the chunk-by-chunk TranslateGemma path can't translate
without breaking its structure.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from docslides.config import get_config
from docslides.documents.pdf_builder import build_pdf
from docslides.documents.xlsx_builder import build_xlsx
from docslides.llm.client import ChatMessage, LLMCallSite, LLMClient, SamplingParams
from docslides.llm.prompts import LANGUAGE_NAMES
from docslides.llm.schemas import PdfSpec, PresentationSpec, SlideContent, SpreadsheetSpec
from docslides.slides.pptx_builder import build_pptx

FORMAT_LABELS = {"xlsx": "Excel workbook", "pptx": "PowerPoint presentation", "pdf": "PDF document"}

_SPEC_BY_FORMAT: dict[str, type[BaseModel]] = {
    "xlsx": SpreadsheetSpec,
    "pptx": PresentationSpec,
    "pdf": PdfSpec,
}

_GUIDANCE = {
    "xlsx": (
        "Design an Excel workbook. Give each sheet a short tab name, a header row of column names, "
        "and data rows with one value per column. Write numbers as JSON numbers (no currency "
        "symbols or units in the cell -- put the unit in the column name). Where totals, averages "
        "or derived values help, use Excel formulas as strings starting with '=' (e.g. "
        "'=SUM(B2:B6)', '=B2*C2'); the header is row 1, so the first data row is row 2. Use "
        "realistic, internally consistent values; never leave the sheet empty."
    ),
    "pptx": (
        "Design a PowerPoint presentation of 5 to 12 slides. Each slide has a short title, 3 to 6 "
        "concise bullets (no trailing periods, no more than ~15 words each), and speaker notes "
        "that expand on the bullets in two or three sentences. Use layout_type 'section_header' "
        "for a slide that opens a new part, 'two_column' for comparisons, otherwise "
        "'title_bullets'. Do not add a title slide -- it is created from the title and subtitle."
    ),
    "pdf": (
        "Write a well-structured document. Give it a title and an optional subtitle, then sections "
        "each with a heading and full paragraphs; use bullets for lists and a table (column names "
        "plus rows of cell text) where tabular data helps. Write complete, polished prose."
    ),
}

_MAX_TOKENS = {"xlsx": 6000, "pptx": 5000, "pdf": 6000}


def _filename_stem(title: str) -> str:
    stem = re.sub(r"[^\w\- ]+", "", title, flags=re.UNICODE).strip().replace(" ", "_")
    return stem[:60] or "document"


async def generate_spec(
    client: LLMClient,
    fmt: str,
    request: str,
    lang: str,
    identity_prompt: str,
    document_text: str | None = None,
) -> BaseModel:
    language = LANGUAGE_NAMES.get(lang, lang)
    system = (
        f"{identity_prompt}\n\nThe user asked you to create a {FORMAT_LABELS[fmt]}. "
        f"{_GUIDANCE[fmt]} Write all of its text in {language}. Return only JSON matching the "
        "schema; the application turns it into the file."
    )
    if document_text:
        system += (
            "\n\nBase the content on the attached document; its extracted text follows.\n"
            f"--- DOCUMENT START ---\n{document_text}\n--- DOCUMENT END ---"
        )
    messages = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=request),
    ]
    return await client.complete_json(
        messages,
        LLMCallSite("document_generation"),
        schema=_SPEC_BY_FORMAT[fmt],
        sampling=SamplingParams(temperature=0.4, max_tokens=_MAX_TOKENS[fmt]),
        enable_thinking=False,
    )


def build_document(spec: BaseModel, fmt: str, lang: str, job_id: str) -> Path:
    """Render `spec` to `paths.output_dir/<title>_<job_id>.<fmt>`."""
    filename = f"{_filename_stem(spec.title)}_{job_id}.{fmt}"
    output_path = Path(get_config().paths.output_dir) / filename
    if isinstance(spec, SpreadsheetSpec):
        return build_xlsx(spec, output_path, lang)
    if isinstance(spec, PdfSpec):
        return build_pdf(spec, output_path, lang)
    if isinstance(spec, PresentationSpec):
        title_slide = SlideContent(
            title=spec.title,
            bullets=[spec.subtitle] if spec.subtitle else [],
            layout_type="section_header",
        )
        return build_pptx([(slide, lang) for slide in (title_slide, *spec.slides)], output_path)
    raise ValueError(f"unsupported document spec: {type(spec).__name__}")
