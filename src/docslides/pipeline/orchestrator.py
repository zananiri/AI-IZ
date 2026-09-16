"""End-to-end pipeline: ingestion -> OCR -> cleaning -> translation ->
outline -> slide fill -> PPTX assembly, emitting "status" events at every
stage for the chat UI's persistent status line.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from docslides.api.events import event_bus
from docslides.cleaning.chunking import chunk_document
from docslides.cleaning.text_cleaning import clean_pages
from docslides.config import get_config
from docslides.ingestion.language_detect import detect_language
from docslides.ingestion.models import PageKind, ParsedDocument
from docslides.ingestion.parser import parse_document
from docslides.llm.client import get_client
from docslides.llm.schemas import SlideContent
from docslides.logging_setup import get_logger
from docslides.ocr.router import recognize_page
from docslides.slides.fill import fill_all_slides
from docslides.slides.outline import generate_outline
from docslides.slides.fill import fill_slide
from docslides.slides.pptx_builder import build_pptx
from docslides.translation.glossary import Glossary
from docslides.translation.translator import translate_chunk

logger = get_logger(__name__)


async def _ocr_and_extract_pages(job_id: str, parsed: ParsedDocument) -> tuple[list[str], list[str]]:
    page_texts: list[str] = []
    page_languages: list[str] = []
    total = parsed.page_count

    for page in parsed.pages:
        await event_bus.publish_status(
            job_id, f"Detecting scanned pages (page {page.index + 1}/{total})"
        )

        if page.kind == PageKind.NATIVE_TEXT:
            text = page.native_text
            lang = detect_language(text) or "en"
        else:
            # Language must be known before OCR routing picks an engine; run
            # a quick pass on any available native fragments first, else
            # default to the document-level majority language once seen, or
            # "en" as a last resort so routing never crashes on an unknown page.
            lang = detect_language(page.native_text) or (page_languages[-1] if page_languages else "en")

            await event_bus.publish_status(
                job_id, f"OCR: page {page.index + 1}/{total} ({lang}) -- routing to engine"
            )
            assert page.image is not None
            result = await recognize_page(page.image.png_bytes, lang, page_index=page.index)
            text = result.text
            escalation_note = " (GPU fallback escalation)" if result.escalated else ""
            await event_bus.publish_status(
                job_id,
                f"OCR: {lang} page {page.index + 1}/{total} -- {result.engine}{escalation_note}",
            )

        page_texts.append(text)
        page_languages.append(lang)

    return page_texts, page_languages


async def extract_document_text(job_id: str, file_path: str | Path) -> tuple[str, str]:
    """Parse + OCR + clean a document into a single text blob, for chat turns
    that just need the content as context (translate/rewrite/ask questions)
    rather than the full slide-generation pipeline below. Reuses the same
    parsing/OCR/cleaning stages `run_pipeline` uses, minus chunking,
    translation, and slide assembly."""
    cfg = get_config()
    await event_bus.publish_status(job_id, "Parsing document")
    parsed = await asyncio.to_thread(parse_document, file_path)
    page_texts, page_languages = await _ocr_and_extract_pages(job_id, parsed)
    cleaned_pages = clean_pages(page_texts, page_languages, cfg.cleaning.header_footer_repetition_threshold)
    dominant_lang = max(set(page_languages), key=page_languages.count) if page_languages else "en"
    return "\n\n".join(cleaned_pages), dominant_lang


async def run_pipeline(job_id: str, file_path: str | Path, target_lang: str) -> Path:
    cfg = get_config()
    client = get_client()
    document_id = Path(file_path).stem

    try:
        await event_bus.publish_status(job_id, "Parsing document")
        parsed = await asyncio.to_thread(parse_document, file_path)

        page_texts, page_languages = await _ocr_and_extract_pages(job_id, parsed)

        await event_bus.publish_status(job_id, "Cleaning extracted text")
        cleaned_pages = clean_pages(
            page_texts, page_languages, cfg.cleaning.header_footer_repetition_threshold
        )

        await event_bus.publish_status(job_id, "Chunking document for translation")
        chunks = chunk_document(cleaned_pages, page_languages)

        translated_chunks = []
        glossary = Glossary.load(document_id)
        for i, chunk in enumerate(chunks):
            result = await translate_chunk(client, chunk, target_lang, glossary)
            glossary.save()
            translated_chunks.append(result)
            await event_bus.publish_status(
                job_id,
                f"Translating chunk {i + 1}/{len(chunks)} -- glossary: {len(glossary.terms)} terms",
            )

        await event_bus.publish_status(job_id, "Planning slide outline")
        outline = await generate_outline(client, translated_chunks, target_lang)

        document_context = "\n\n".join(c.translated_text for c in translated_chunks)
        slide_contents: list[SlideContent] = []
        for i, plan in enumerate(outline.slides):
            content = await fill_slide(client, plan, document_context, target_lang)
            slide_contents.append(content)
            await event_bus.publish_status(job_id, f"Generating slide {i + 1}/{len(outline.slides)}")

        await event_bus.publish_status(job_id, "Assembling PPTX")
        output_path = Path(cfg.paths.output_dir) / f"{document_id}_{target_lang}.pptx"
        slides_with_lang = [(content, target_lang) for content in slide_contents]
        build_pptx(slides_with_lang, output_path)

        await event_bus.publish_done(job_id, output_path=str(output_path))
        return output_path

    except Exception as exc:  # noqa: BLE001 -- surfaced to the UI, then re-raised
        logger.exception("pipeline_failed", job_id=job_id, error=str(exc))
        await event_bus.publish_error(job_id, str(exc))
        raise
