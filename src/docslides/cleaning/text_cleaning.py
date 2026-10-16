"""Document cleaning for large multi-page documents, ahead of translation.

Steps, applied in order:
  1. Unicode NFC normalization; strip control chars and invalid UTF-8 fragments.
  2. Cross-page repeated-line detection (headers/footers/page numbers).
  3. De-hyphenation of line-wrapped words.
  4. Whitespace normalization.

Runs on already-extracted/OCR'd page text; language-agnostic except where
noted (e.g. de-hyphenation assumes a "-" continuation convention shared by
Latin-script languages -- see the guard below).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")
_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")

LATIN_HYPHENATING_LANGS = {"en", "fr", "es", "it", "de"}


def normalize_unicode_and_strip_control(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    # Drop lone surrogates / invalid fragments that can appear from bad decodes.
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return _CONTROL_CHAR_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_LINE_RE.sub("\n\n", text)
    return text.strip()


def dehyphenate(text: str, lang: str) -> str:
    """Join `word-\nword` -> `wordword` for line-wrapped Latin-script text.

    Skipped for non-Latin scripts (ar/he) where "-" is not a line-wrap
    convention and joining could corrupt legitimate hyphenated content.
    """
    if lang not in LATIN_HYPHENATING_LANGS:
        return text
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def detect_repeated_headers_footers(pages_text: list[str], repetition_threshold: float) -> set[str]:
    """Identify lines that repeat across enough pages to be a running
    header/footer/page-number artifact rather than real content."""
    if len(pages_text) < 3:
        return set()

    line_page_counts: Counter[str] = Counter()
    for page_text in pages_text:
        # Only consider the first/last two lines of each page -- headers and
        # footers live at page edges, not in the body.
        lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
        edge_lines = set(lines[:2] + lines[-2:])
        for line in edge_lines:
            line_page_counts[line] += 1

    min_pages = max(3, int(len(pages_text) * repetition_threshold))
    return {
        line
        for line, count in line_page_counts.items()
        if count >= min_pages or _PAGE_NUMBER_LINE_RE.match(line)
    }


def strip_headers_footers(text: str, repeated_lines: set[str]) -> str:
    kept = [
        line
        for line in text.splitlines()
        if line.strip() not in repeated_lines and not _PAGE_NUMBER_LINE_RE.match(line.strip())
    ]
    return "\n".join(kept)


def clean_pages(pages_text: list[str], languages: list[str], repetition_threshold: float) -> list[str]:
    """Full cleaning pipeline over a document's pages. `languages` gives the
    detected language per page (same length as `pages_text`)."""
    normalized = [normalize_unicode_and_strip_control(t) for t in pages_text]
    repeated_lines = detect_repeated_headers_footers(normalized, repetition_threshold)

    cleaned = []
    for text, lang in zip(normalized, languages):
        text = strip_headers_footers(text, repeated_lines)
        text = dehyphenate(text, lang)
        text = normalize_whitespace(text)
        cleaned.append(text)
    return cleaned
