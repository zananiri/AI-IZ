"""Chunk-level translation via Qwen3-32B itself (no separate NMT model).

Per chunk: mask non-translatable spans -> translate with the accumulated
glossary injected as context -> restore masked spans -> merge any newly
identified terms back into the running glossary for subsequent chunks.
"""

from __future__ import annotations

from dataclasses import dataclass

from docslides.cleaning.chunking import Chunk
from docslides.cleaning.masking import mask_non_translatable_spans, restore_masks
from docslides.llm.client import ChatMessage, LLMCallSite, QwenClient
from docslides.llm.prompts import glossary_extraction_note, translation_system_prompt
from docslides.llm.schemas import TranslatedChunk
from docslides.logging_setup import get_logger
from docslides.translation.glossary import Glossary

logger = get_logger(__name__)


@dataclass
class TranslatedResult:
    chunk_index: int
    source_text: str
    translated_text: str


async def translate_chunk(
    client: QwenClient,
    chunk: Chunk,
    target_lang: str,
    glossary: Glossary,
) -> TranslatedResult:
    masked = mask_non_translatable_spans(chunk.text, chunk.language)

    system_prompt = translation_system_prompt(chunk.language, target_lang, glossary.terms)
    user_prompt = f"{masked.text}\n\n{glossary_extraction_note()}"

    result = await client.complete_json(
        messages=[
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ],
        call_site=LLMCallSite("translation"),
        schema=TranslatedChunk,
    )
    assert isinstance(result, TranslatedChunk)

    restored = restore_masks(result.translated_text, masked.spans)
    glossary.add_terms(result.new_terms)

    logger.info(
        "chunk_translated",
        chunk_index=chunk.index,
        source_lang=chunk.language,
        target_lang=target_lang,
        new_glossary_terms=len(result.new_terms),
    )
    return TranslatedResult(chunk_index=chunk.index, source_text=chunk.text, translated_text=restored)


async def translate_document(
    client: QwenClient,
    chunks: list[Chunk],
    target_lang: str,
    document_id: str,
) -> list[TranslatedResult]:
    glossary = Glossary.load(document_id)
    results: list[TranslatedResult] = []
    for chunk in chunks:
        result = await translate_chunk(client, chunk, target_lang, glossary)
        results.append(result)
        glossary.save()
    return results
