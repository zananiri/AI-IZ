"""Stage 1: slide-by-slide outline generation.

Qwen3-32B, thinking ENABLED (config: llm.thinking_defaults.outline_generation),
reads the cleaned/translated document (chunk summaries + structure) and
produces the outline as guided JSON against `OutlineResult`.

For documents whose full translated text would not fit the outline prompt
within `llm.max_model_len`, each chunk is first compressed to a short summary
(thinking disabled, deterministic) so the outline call always sees the whole
document's structure rather than being truncated mid-document.
"""

from __future__ import annotations

import asyncio

from docslides.cleaning.tokens import count_tokens
from docslides.config import get_config
from docslides.llm.client import ChatMessage, LLMCallSite, QwenClient
from docslides.llm.prompts import chunk_summary_system_prompt, outline_system_prompt
from docslides.llm.schemas import ChunkSummary, OutlineResult
from docslides.translation.translator import TranslatedResult
from docslides.logging_setup import get_logger

logger = get_logger(__name__)

# Reserve headroom for the system prompt, schema grammar, and the model's own
# output tokens within max_model_len.
OUTLINE_PROMPT_TOKEN_BUDGET_FRACTION = 0.55


async def _summarize_chunk(client: QwenClient, chunk: TranslatedResult) -> str:
    result = await client.complete_json(
        messages=[
            ChatMessage(role="system", content=chunk_summary_system_prompt()),
            ChatMessage(role="user", content=chunk.translated_text),
        ],
        call_site=LLMCallSite("chunk_summary"),
        schema=ChunkSummary,
    )
    assert isinstance(result, ChunkSummary)
    points = "; ".join(result.key_points)
    return f"{result.summary} (Key points: {points})" if points else result.summary


async def _build_document_summary(client: QwenClient, translated_chunks: list[TranslatedResult]) -> str:
    full_text = "\n\n".join(
        f"[Chunk {c.chunk_index}]\n{c.translated_text}" for c in translated_chunks
    )
    budget = int(get_config().llm.max_model_len * OUTLINE_PROMPT_TOKEN_BUDGET_FRACTION)

    if count_tokens(full_text) <= budget:
        return full_text

    logger.info(
        "document_exceeds_outline_budget_summarizing_chunks",
        num_chunks=len(translated_chunks),
        budget_tokens=budget,
    )
    summaries = await asyncio.gather(*(_summarize_chunk(client, c) for c in translated_chunks))
    return "\n\n".join(f"[Chunk {c.chunk_index} summary]\n{s}" for c, s in zip(translated_chunks, summaries))


async def generate_outline(
    client: QwenClient,
    translated_chunks: list[TranslatedResult],
    target_lang: str,
) -> OutlineResult:
    document_text = await _build_document_summary(client, translated_chunks)
    system_prompt = outline_system_prompt(target_lang, len(translated_chunks))

    result = await client.complete_json(
        messages=[
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=document_text),
        ],
        call_site=LLMCallSite("outline_generation"),
        schema=OutlineResult,
    )
    assert isinstance(result, OutlineResult)
    logger.info("outline_generated", num_slides=len(result.slides), target_lang=target_lang)
    return result
