"""Spanish text hygiene for the corpus.

repair_text() detects and, when it can do so safely, repairs mojibake -- UTF-8 bytes read as
cp1252/Latin-1 ("artÃ­culo" for "artículo", "aÃ±o" for "año"), the classic fault of older
Spanish documents and of some autonomous-community bulletins -- flagging rather than guessing
when it can't.

normalize_for_embedding() unifies quote marks (« » “ ” -> ") and whitespace, keeping accents: bge-m3
was trained on accented Spanish. normalize_for_index() is the copy a keyword index compares, never
what is displayed: lowercased, accents folded (á -> a, but ñ kept), quotes and dashes unified, so
"Artículo", "articulo" and "ARTÍCULO" match.
"""

from __future__ import annotations

import re
import unicodedata

_MOJIBAKE_RE = re.compile("Ã[\u0080-¿¡-¿]|Â[ -¿]|â€[\u0080-¿™œ˜”“¦]")
_QUOTES = str.maketrans({"«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "″": '"', "‘": "'", "’": "'",
                         "′": "'", "`": "'"})
_DASHES_RE = re.compile("[‐‑‒–—―]")
_LETTER_RE = re.compile(r"[^\W\d_]")
_SPANISH_HINT_RE = re.compile(r"[áéíóúñüÁÉÍÓÚÑÜ¿¡]")


def spanish_ratio(text: str) -> float:
    """Share of Latin letters among all letters (a rough "is this Spanish/Latin text" check)."""
    letters = _LETTER_RE.findall(text)
    if not letters:
        return 0.0
    return sum(1 for c in letters if "LATIN" in unicodedata.name(c, "")) / len(letters)


def _cp1252_bytes(text: str) -> bytes | None:
    out = bytearray()
    for ch in text:
        try:
            out += ch.encode("cp1252")
        except UnicodeEncodeError:
            if ord(ch) < 256:
                out.append(ord(ch))
            else:
                return None
    return bytes(out)


def _fix_mojibake(text: str) -> str | None:
    if len(_MOJIBAKE_RE.findall(text)) < 3:
        return None
    raw = _cp1252_bytes(text)
    if raw is None:
        return None
    fixed = raw.decode("utf-8", errors="replace")
    if fixed.count("�") > len(fixed) * 0.001 or len(_MOJIBAKE_RE.findall(fixed)) >= len(_MOJIBAKE_RE.findall(text)):
        return None
    return fixed


def repair_text(text: str) -> tuple[str, str, list[str]]:
    """(text, "ok" | "repaired" | "flagged", notes)."""
    notes: list[str] = []
    status = "ok"
    fixed = _fix_mojibake(text)
    if fixed is not None:
        text, status = fixed, "repaired"
        notes.append("encoding repaired: UTF-8 read as cp1252")
    elif len(_MOJIBAKE_RE.findall(text)) >= 3:
        status = "flagged"
        notes.append("looks mis-encoded (mojibake) but could not be repaired")
    if text.count("�") > max(3, len(text) * 0.001):
        status = "flagged"
        notes.append(f"{text.count(chr(0xFFFD))} replacement characters")
    return text, status, notes


def fold_accents(text: str) -> str:
    """á é í ó ú ü -> a e i o u; ñ/Ñ stays (it is a letter of its own in Spanish)."""
    out = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.category(ch) == "Mn":
            if ch == "̃" and out and out[-1] in "nN":  # the tilde of ñ
                out.append(ch)
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def normalize_quotes(text: str) -> str:
    return text.translate(_QUOTES)


def normalize_for_embedding(text: str, fold: bool = False) -> str:
    text = normalize_quotes(unicodedata.normalize("NFC", text))
    text = re.sub(r"[ \t ]+", " ", text)
    return fold_accents(text) if fold else text


def normalize_for_index(text: str) -> str:
    """The lexical copy: lowercased, accents folded, quotes ASCII, dashes as hyphens."""
    text = fold_accents(normalize_quotes(unicodedata.normalize("NFC", text))).lower()
    text = _DASHES_RE.sub("-", text)
    return re.sub(r"[ \t ]+", " ", text).strip()


def looks_spanish(text: str) -> bool:
    return bool(_SPANISH_HINT_RE.search(text)) or bool(
        re.search(r"(?i)\b(?:el|la|los|las|de|del|que|por|para|con|una?|es|artículo|ley)\b", text))
