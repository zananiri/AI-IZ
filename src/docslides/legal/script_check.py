"""Words in the wrong script in a Legal answer.

A small multilingual model drafting in Hebrew sometimes drops in a word from
another language mid-sentence: 'מ報導' for 'מדווח', 'октяבר' for 'אוקטובר',
'simultaneously'. The meaning around it is usually right, so the answer is
repaired word by word rather than redrafted: a redraft can change what was
already verified, a word swap can't.

A word is flagged when it mixes letters of two scripts, or when its script is
not the answer language's and it doesn't appear in the evidence or the
question (so a law's English name quoted from the evidence stays).
"""

from __future__ import annotations

import re
import unicodedata

from docslides.legal.citations import outside_citations, strip_citations

_NATIVE_SCRIPT = {"he": "HEBREW", "ar": "ARABIC", "fa": "ARABIC", "ru": "CYRILLIC", "uk": "CYRILLIC",
                  "am": "ETHIOPIC", "el": "GREEK"}
_WORD_RE = re.compile(r"\w+")


def _script(char: str) -> str:
    # "HEBREW LETTER ALEF" -> "HEBREW", "CJK UNIFIED IDEOGRAPH-5831" -> "CJK"
    return unicodedata.name(char, "UNKNOWN").split(" ")[0]


def _scripts(word: str) -> set[str]:
    return {_script(c) for c in word if c.isalpha()}


def allowed_words(texts: list[str]) -> set[str]:
    """Every word of the evidence and the question, lowercased."""
    return {w.lower() for text in texts for w in _WORD_RE.findall(text)}


def foreign_words(text: str, reply_language: str, allowed: set[str]) -> list[str]:
    """Words of `text` (citation tokens excluded) in the wrong script, in order, once each."""
    native = _NATIVE_SCRIPT.get(reply_language, "LATIN")
    found: list[str] = []
    for word in _WORD_RE.findall(strip_citations(text)):
        scripts = _scripts(word)
        if not scripts or word.lower() in allowed or word in found:
            continue
        if len(scripts) > 1 or scripts != {native}:
            found.append(word)
    return found


def replace_words(text: str, replacements: dict[str, str]) -> str:
    """Swaps whole words outside citation tokens; a word replaced by "" is dropped with a space around it."""
    if not replacements:
        return text
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(w) for w in sorted(replacements, key=len, reverse=True))
                         + r")(?!\w)")

    def swap(prose: str) -> str:
        prose = pattern.sub(lambda m: replacements[m.group(1)], prose)
        return re.sub(r"[ \t]{2,}", " ", prose)

    return outside_citations(text, swap)


def sentences_with(text: str, words: list[str]) -> list[str]:
    """The sentences of `text` that contain any of `words`, for the repair prompt."""
    prose = strip_citations(text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?:;])\s+|\n+", prose) if s.strip()]
    return [s for s in sentences if any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", s) for w in words)]
