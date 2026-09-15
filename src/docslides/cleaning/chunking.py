"""Chunk cleaned document text at paragraph/section boundaries within a safe
token budget for Qwen3-32B's context window -- never splitting mid-sentence.

Strategy: split into paragraphs, sentence-segment each paragraph with the
correct per-language model, then greedily pack sentences into chunks up to
`max_chunk_tokens`. A single paragraph longer than the budget is chunked at
sentence boundaries within it rather than dropped or truncated.
"""

from __future__ import annotations

from dataclasses import dataclass

from docslides.cleaning.segmentation import segment_sentences
from docslides.cleaning.tokens import count_tokens
from docslides.config import get_config


@dataclass
class Chunk:
    index: int
    text: str
    language: str
    token_count: int


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_document(page_texts: list[str], page_languages: list[str]) -> list[Chunk]:
    """`page_texts`/`page_languages` are the cleaned, per-page outputs, same
    language granularity as detected during ingestion. Chunks never span a
    language boundary between pages, since each chunk is translated as a unit
    with a single source language."""
    max_tokens = get_config().cleaning.max_chunk_tokens
    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    current_lang: str | None = None

    def flush() -> None:
        nonlocal current_sentences, current_tokens, current_lang
        if current_sentences:
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text="\n".join(current_sentences),
                    language=current_lang or "en",
                    token_count=current_tokens,
                )
            )
        current_sentences = []
        current_tokens = 0

    for page_text, lang in zip(page_texts, page_languages):
        lang = lang or "en"
        if lang != current_lang:
            flush()
            current_lang = lang

        for paragraph in _split_paragraphs(page_text):
            sentences = segment_sentences(paragraph, lang)
            for sentence in sentences:
                sent_tokens = count_tokens(sentence)
                if current_tokens + sent_tokens > max_tokens and current_sentences:
                    flush()
                    current_lang = lang
                current_sentences.append(sentence)
                current_tokens += sent_tokens
            # Prefer to break chunks at paragraph boundaries when we're
            # already reasonably full, rather than packing right up to budget.
            if current_tokens > max_tokens * 0.85:
                flush()
                current_lang = lang

    flush()
    return chunks
