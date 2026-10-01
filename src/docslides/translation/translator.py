"""Chunk-level translation: TranslateGemma translates, Gemma 4 keeps the terminology consistent.

Per document: Gemma 4 (the general `llm:` model) reads the document once and builds a glossary of
terms that must be rendered the same way throughout (`prepare_glossary`).

Per chunk (`translate_chunk`): TranslateGemma (config translation.translator) translates the text
with its own prompt. Any glossary term that occurs in the source but whose translation doesn't
appear in the output goes back to Gemma 4, which fixes just those terms. Text is not masked for
TranslateGemma: it isn't trained to carry placeholder tokens through, and it renders numbers and
names faithfully on its own.

With no translator configured (the vLLM default), Gemma 4 translates each chunk itself: masked
non-translatable spans, the glossary in the prompt, and new terms returned alongside.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from docslides.cleaning.chunking import Chunk
from docslides.cleaning.masking import (
    mask_non_translatable_spans,
    restore_masks,
    unwrap_trivial_math,
)
from docslides.cleaning.tokens import count_tokens
from docslides.config import get_config
from docslides.llm.client import ChatMessage, LLMCallSite, LLMClient, SamplingParams
from docslides.llm.prompts import (
    glossary_build_system_prompt,
    glossary_extraction_note,
    glossary_fix_system_prompt,
    translategemma_prompt,
    translation_system_prompt,
)
from docslides.llm.schemas import (
    GlossaryExtraction,
    GlossaryTerm,
    TranslatedChunk,
    TranslationRevision,
)
from docslides.logging_setup import get_logger
from docslides.translation.glossary import Glossary

logger = get_logger(__name__)

# Terms Gemma 4 lists up front; the glossary still caps at translation.glossary_max_terms.
_GLOSSARY_BUILD_TERMS = 40
# Share of the general model's context the document may take when building the glossary.
_GLOSSARY_BUILD_CONTEXT_FRACTION = 0.5
# A glossary fix shorter than this share of the draft was cut off or dropped text: keep the draft.
_MIN_REVISION_LENGTH_RATIO = 0.8


@dataclass
class TranslatedResult:
    chunk_index: int
    source_text: str
    translated_text: str


def _fit_to_tokens(text: str, max_tokens: int) -> str:
    tokens = count_tokens(text)
    if tokens <= max_tokens:
        return text
    return text[: max(1, int(len(text) * max_tokens / tokens * 0.95))]


async def prepare_glossary(
    client: LLMClient, document_id: str, text: str, source_lang: str, target_lang: str
) -> Glossary:
    """The document's glossary for this target language: loaded if an earlier run saved one,
    otherwise built by Gemma 4 from the document's text."""
    glossary = Glossary.load(f"{document_id}_{target_lang}")
    if glossary.terms:
        return glossary
    budget = int(get_config().llm.max_model_len * _GLOSSARY_BUILD_CONTEXT_FRACTION)
    try:
        result = await client.complete_json(
            messages=[
                ChatMessage(
                    role="system",
                    content=glossary_build_system_prompt(source_lang, target_lang, _GLOSSARY_BUILD_TERMS),
                ),
                ChatMessage(role="user", content=_fit_to_tokens(unwrap_trivial_math(text), budget)),
            ],
            call_site=LLMCallSite("translation"),
            schema=GlossaryExtraction,
        )
    except Exception as exc:  # noqa: BLE001 -- a glossary helps consistency; translation works without one
        logger.warning("glossary_build_failed", document_id=document_id, error=str(exc))
        return glossary
    assert isinstance(result, GlossaryExtraction)
    glossary.add_terms(result.terms)
    logger.info("glossary_built", document_id=document_id, target_lang=target_lang, terms=len(glossary.terms))
    return glossary


def _occurs(term: str, text: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE) is not None


def _rendered(target_term: str, text: str) -> bool:
    """Whether `target_term` appears in `text`, allowing for inflection: each word of four letters
    or more only needs its stem (all but its last three letters, at least four) to appear --
    "Heiliges Land" counts as present in "im Heiligen Land"."""
    lowered = text.lower()
    for word in re.findall(r"\w+", target_term.lower()):
        stem = word if len(word) < 4 else word[: max(4, len(word) - 3)]
        if stem not in lowered:
            return False
    return True


def _missed_terms(source_text: str, translated_text: str, glossary: Glossary) -> list[GlossaryTerm]:
    return [
        term
        for term in glossary.terms
        if term.target_term.strip()
        and _occurs(term.source_term.strip(), source_text)
        and not _rendered(term.target_term, translated_text)
    ]


async def _fix_glossary_terms(
    client: LLMClient, chunk: Chunk, target_lang: str, draft: str, missed: list[GlossaryTerm]
) -> str:
    try:
        result = await client.complete_json(
            messages=[
                ChatMessage(role="system", content=glossary_fix_system_prompt(chunk.language, target_lang, missed)),
                ChatMessage(role="user", content=f"Source:\n{chunk.text}\n\nTranslation:\n{draft}"),
            ],
            call_site=LLMCallSite("translation"),
            schema=TranslationRevision,
            sampling=SamplingParams(temperature=0.1, top_p=0.9, max_tokens=max(1024, len(draft) // 2)),
        )
    except Exception as exc:  # noqa: BLE001 -- the unfixed translation is still a translation
        logger.warning("glossary_fix_failed", chunk_index=chunk.index, error=str(exc))
        return draft
    assert isinstance(result, TranslationRevision)
    revised = result.translated_text.strip()
    if len(revised) < _MIN_REVISION_LENGTH_RATIO * len(draft):
        logger.warning("glossary_fix_dropped_text", chunk_index=chunk.index)
        return draft
    return revised


async def _translate_with_translator(
    client: LLMClient, translator: LLMClient, chunk: Chunk, target_lang: str, glossary: Glossary
) -> str:
    source = unwrap_trivial_math(chunk.text)
    draft = await translator.complete_text(
        messages=[ChatMessage(role="user", content=translategemma_prompt(chunk.language, target_lang, source))],
        call_site=LLMCallSite("translation"),
        sampling=SamplingParams(temperature=0.1, top_p=0.9, max_tokens=max(1024, len(source) // 2)),
    )
    missed = _missed_terms(source, draft, glossary)
    if missed:
        logger.info("glossary_terms_missed", chunk_index=chunk.index, terms=[t.source_term for t in missed])
        return await _fix_glossary_terms(client, chunk, target_lang, draft, missed)
    return draft


async def _translate_alone(client: LLMClient, chunk: Chunk, target_lang: str, glossary: Glossary) -> str:
    masked = mask_non_translatable_spans(chunk.text, chunk.language)
    result = await client.complete_json(
        messages=[
            ChatMessage(role="system", content=translation_system_prompt(chunk.language, target_lang, glossary.terms)),
            ChatMessage(role="user", content=f"{masked.text}\n\n{glossary_extraction_note()}"),
        ],
        call_site=LLMCallSite("translation"),
        schema=TranslatedChunk,
    )
    assert isinstance(result, TranslatedChunk)
    glossary.add_terms(result.new_terms)
    return restore_masks(result.translated_text, masked.spans)


async def translate_chunk(
    client: LLMClient,
    chunk: Chunk,
    target_lang: str,
    glossary: Glossary,
    translator: LLMClient | None = None,
) -> TranslatedResult:
    """`client` is Gemma 4 (glossary work, or the whole translation when `translator` is None);
    `translator` is TranslateGemma (llm/client.get_translator_client)."""
    if translator is not None:
        translated = await _translate_with_translator(client, translator, chunk, target_lang, glossary)
    else:
        translated = await _translate_alone(client, chunk, target_lang, glossary)
    logger.info(
        "chunk_translated",
        chunk_index=chunk.index,
        source_lang=chunk.language,
        target_lang=target_lang,
        translator=translator.model if translator is not None else client.model,
        glossary_terms=len(glossary.terms),
    )
    return TranslatedResult(chunk_index=chunk.index, source_text=chunk.text, translated_text=translated)
