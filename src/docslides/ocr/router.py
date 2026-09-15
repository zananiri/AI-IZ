"""Language-aware OCR routing with confidence-based GPU escalation.

For each page/segment: run the configured primary engine for that language's
script first (CPU for Latin scripts, or the VLM engine directly for
Arabic/Hebrew where the config designates it primary). If mean confidence is
below `ocr.confidence_threshold`, escalate to the GPU-based VLM-OCR fallback
via the GPU arbiter (never held resident alongside a vLLM generation batch),
and keep whichever result has higher confidence.

Every page/segment logs which engine ultimately produced its text, for
downstream debugging and QA.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from docslides.config import get_config
from docslides.ocr.engines import ENGINE_REGISTRY, OCRResult
from docslides.ocr.gpu_arbiter import GPUOCRSession
from docslides.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class RoutedOCRResult(OCRResult):
    escalated: bool = False
    page_index: int | None = None
    language: str = ""


async def _run_engine(engine_name: str, png_bytes: bytes, lang: str) -> OCRResult:
    engine = ENGINE_REGISTRY[engine_name]
    if engine.gpu_resident:
        async with GPUOCRSession():
            return await asyncio.to_thread(engine.recognize, png_bytes, lang)
    return await asyncio.to_thread(engine.recognize, png_bytes, lang)


async def recognize_page(
    png_bytes: bytes, lang: str, page_index: int | None = None
) -> RoutedOCRResult:
    cfg = get_config().ocr
    script_cfg = cfg.script_for_language(lang)

    primary_result = await _run_engine(script_cfg.primary, png_bytes, lang)
    best = primary_result
    escalated = False

    if primary_result.mean_confidence < cfg.confidence_threshold:
        logger.info(
            "ocr_low_confidence_trying_cpu_fallback",
            page=page_index,
            lang=lang,
            primary=script_cfg.primary,
            confidence=primary_result.mean_confidence,
        )
        cpu_fallback_result = await _run_engine(script_cfg.fallback_cpu, png_bytes, lang)
        if cpu_fallback_result.mean_confidence > best.mean_confidence:
            best = cpu_fallback_result

        if best.mean_confidence < cfg.confidence_threshold:
            logger.info(
                "ocr_escalating_to_gpu_fallback",
                page=page_index,
                lang=lang,
                fallback_gpu=script_cfg.fallback_gpu,
                confidence=best.mean_confidence,
            )
            gpu_result = await _run_engine(script_cfg.fallback_gpu, png_bytes, lang)
            escalated = True
            if gpu_result.mean_confidence > best.mean_confidence:
                best = gpu_result

    routed = RoutedOCRResult(
        text=best.text,
        mean_confidence=best.mean_confidence,
        engine=best.engine,
        escalated=escalated,
        page_index=page_index,
        language=lang,
    )
    logger.info(
        "ocr_page_complete",
        page=page_index,
        lang=lang,
        engine=routed.engine,
        confidence=routed.mean_confidence,
        escalated=escalated,
    )
    return routed
