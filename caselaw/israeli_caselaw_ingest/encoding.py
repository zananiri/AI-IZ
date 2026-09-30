"""Mojibake repair.

Some rows hold Hebrew cp1255 bytes decoded with the wrong code page: as Latin-1/cp1252
("á áéú äîùôè") or as Greek cp1253 ("α αιϊ δξωτθ"), both of which are "ב בית המשפט". A line is
suspect when enough of its letters are Latin-1-Supplement or Greek look-alikes; it is re-encoded
with each candidate code page and decoded as cp1255, and the candidate with the most Hebrew
letters (and fewest undecodable bytes) wins. ftfy is a last resort for what that leaves. Lines
are repaired one by one, so a document that is only partly garbled keeps its good lines."""

from __future__ import annotations

import re
import unicodedata

HEBREW_LETTER_RE = re.compile(r"[א-ת]")
LATIN1_SUPP_RE = re.compile(r"[À-ÿ]")
GREEK_RE = re.compile(r"[Ͱ-Ͽ]")
_LETTER_RE = re.compile(r"[^\W\d_]")

# (method name, code page the Hebrew bytes were wrongly decoded with)
CANDIDATES = (("latin1", "latin-1"), ("cp1252", "cp1252"), ("cp1253", "cp1253"))


def hebrew_count(text: str) -> int:
    return len(HEBREW_LETTER_RE.findall(text))


def suspect_count(text: str) -> int:
    return len(LATIN1_SUPP_RE.findall(text)) + len(GREEK_RE.findall(text))


def looks_mojibake(text: str, min_chars: int = 3, min_ratio: float = 0.5) -> bool:
    suspects = suspect_count(text)
    if suspects < min_chars:
        return False
    letters = len(_LETTER_RE.findall(text))
    return letters > 0 and suspects / letters >= min_ratio and hebrew_count(text) < suspects


def _redecode(text: str, codepage: str) -> str | None:
    try:
        raw = text.encode(codepage)
    except UnicodeEncodeError:
        try:
            raw = text.encode(codepage, errors="replace")
        except (UnicodeEncodeError, LookupError):
            return None
    return raw.decode("cp1255", errors="replace")


def repair_line(line: str, min_chars: int = 3, min_ratio: float = 0.5) -> tuple[str, str | None]:
    """(line, the method that repaired it or None)."""
    if not looks_mojibake(line, min_chars, min_ratio):
        return line, None
    best, best_method = line, None
    best_key = (hebrew_count(line), 0)
    for method, codepage in CANDIDATES:
        fixed = _redecode(line, codepage)
        if fixed is None:
            continue
        key = (hebrew_count(fixed), -fixed.count("�"))
        if key > best_key:
            best, best_method, best_key = fixed, method, key
    return best, best_method


def repair_text(text: str, min_chars: int = 3, min_ratio: float = 0.5, try_ftfy: bool = True) -> tuple[str, str]:
    """(repaired text, method): "none", one of latin1/cp1252/cp1253 (the most used, when lines
    needed different ones), "ftfy", or "unrepaired" when the text still looks garbled."""
    if not text or suspect_count(text) < min_chars:
        return text, "none"
    methods: dict[str, int] = {}
    lines = text.split("\n")
    for i, line in enumerate(lines):
        fixed, method = repair_line(line, min_chars, min_ratio)
        if method:
            lines[i] = fixed
            methods[method] = methods.get(method, 0) + 1
    repaired = "\n".join(lines)
    if methods:
        return repaired, max(methods, key=methods.get)
    if looks_mojibake(repaired, min_chars, min_ratio):
        if try_ftfy:
            try:
                import ftfy

                fixed = ftfy.fix_text(repaired)
                if hebrew_count(fixed) > hebrew_count(repaired):
                    return fixed, "ftfy"
            except ImportError:
                pass
        return repaired, "unrepaired"
    return repaired, "none"


# Directional marks and embeddings/overrides/isolates: invisible, and they break regexes and search.
BIDI_RE = re.compile("[‎‏‪-‮⁦-⁩]")
_SPACES_RE = re.compile(r"[ \t  -   　]+")
_BLANKS_RE = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    """NFC, no directional control marks, runs of spaces collapsed, lines trimmed, at most one
    blank line between paragraphs. Nothing else is changed (no nikud stripping)."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n").replace("\v", "\n")
    text = BIDI_RE.sub("", text)
    lines = [_SPACES_RE.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANKS_RE.sub("\n\n", "\n".join(lines)).strip()


def dedupe_key(text: str) -> str:
    """Whitespace- and mark-insensitive form used for the exact-duplicate hash."""
    text = BIDI_RE.sub("", unicodedata.normalize("NFC", text))
    return " ".join(text.split())
