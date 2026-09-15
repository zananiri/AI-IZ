"""Stage 2: per-slide content generation.

Qwen3-32B, thinking DISABLED (config: llm.thinking_defaults.slide_fill),
guided JSON decoding against `SlideContent`. Each call gets the slide's plan
from Stage 1 plus the (possibly summarized, same budget logic as the outline
stage) document content as grounding context, so bullets stay faithful to the
source rather than invented.
"""

from __future__ import annotations

import asyncio

from docslides.llm.client import ChatMessage, LLMCallSite, QwenClient
from docslides.llm.prompts import slide_fill_system_prompt
from docslides.llm.schemas import SlideContent, SlidePlan
from docslides.logging_setup import get_logger

logger = get_logger(__name__)


async def fill_slide(
    client: QwenClient,
    slide_plan: SlidePlan,
    document_context: str,
    target_lang: str,
) -> SlideContent:
    system_prompt = slide_fill_system_prompt(target_lang)
    user_prompt = (
        f"Slide plan:\n"
        f"- title: {slide_plan.title}\n"
        f"- intent: {slide_plan.intent}\n"
        f"- target_bullet_count: {slide_plan.target_bullet_count}\n"
        f"- layout_type: {slide_plan.layout_type}\n\n"
        f"Source document content:\n{document_context}"
    )

    result = await client.complete_json(
        messages=[
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ],
        call_site=LLMCallSite("slide_fill"),
        schema=SlideContent,
    )
    assert isinstance(result, SlideContent)
    return result


async def fill_all_slides(
    client: QwenClient,
    slide_plans: list[SlidePlan],
    document_context: str,
    target_lang: str,
    max_concurrency: int = 4,
) -> list[SlideContent]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _bounded(i: int, plan: SlidePlan) -> SlideContent:
        async with semaphore:
            logger.info("generating_slide", slide_index=i, total=len(slide_plans), title=plan.title)
            return await fill_slide(client, plan, document_context, target_lang)

    return await asyncio.gather(*(_bounded(i, p) for i, p in enumerate(slide_plans)))
