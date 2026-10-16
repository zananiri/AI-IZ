"""Hebrew text hygiene for the corpus.

repair_text() detects and, when it can do so safely, repairs the two classic encoding faults of
older Hebrew documents -- flagging rather than guessing when it can't:
  * mojibake: UTF-8 bytes read as cp1252 ("×—×•×§" for "חוק"), or cp1255 bytes read as
    Latin-1 ("çå÷" for "חוק"). Repaired only if the result is mostly Hebrew letters.
  * visual order: text stored as displayed, right to left, so each line comes out reversed
    ("קוח" for "חוק"). Detected by final letters piling up at word starts
    (legal/sources.looks_visual_order); repaired line by line, keeping number and Latin runs
    in their order, and always noted.

normalize_for_embedding() strips niqqud and cantillation and unifies quote marks (״ for Hebrew
double quotes, as legal/chunking.normalize_hebrew_quotes does). normalize_for_index() also folds
final letters and quote marks -- the copy a keyword index compares, never what is displayed.
"""

from __future__ import annotations

import re

from docslides.legal.chunking import normalize_hebrew_quotes
from docslides.legal.sources import looks_visual_order

# Points and cantillation, but not maqaf (U+05BE), paseq (U+05C0), sof pasuq (U+05C3) or nun hafukha
# (U+05C6): those are punctuation, and dropping maqaf would glue words together ("חוק־יסוד").
_POINTS_RE = re.compile("[֑-ׇֽֿׁׂׅׄ]")
_HEBREW_LETTER_RE = re.compile("[א-ת]")
_LETTER_RE = re.compile(r"[^\W\d_]")
_UTF8_AS_CP1252_RE = re.compile("×[\u0080-¿ŒœŠšŸŽžƒˆ˜–-›€™]")
_CP1255_AS_LATIN1_RE = re.compile("[à-ú]{3,}")
_FINALS = str.maketrans("ךםןףץ", "כמנפצ")
_QUOTES_TO_HEBREW = str.maketrans({"“": '"', "”": '"', "„": '"', "″": '"', "‘": "'", "’": "'", "′": "'"})
_QUOTES_TO_ASCII = str.maketrans({"״": '"', "׳": "'", "“": '"', "”": '"', "„": '"', "″": '"', "‘": "'", "’": "'",
                                  "′": "'"})
_LTR_RUN_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z.,:/%\-]*")
_MIRROR = str.maketrans("()[]{}<>", ")(][}{><")


def hebrew_ratio(text: str) -> float:
    letters = _LETTER_RE.findall(text)
    return len(_HEBREW_LETTER_RE.findall(text)) / len(letters) if letters else 0.0


def _cp1252_bytes(text: str) -> bytes | None:
    out = bytearray()
    for ch in text:
        try:
            out += ch.encode("cp1252")
        except UnicodeEncodeError:
            if ord(ch) < 256:
                out.append(ord(ch))  # the 5 bytes cp1252 leaves undefined come back as C1 controls
            else:
                return None
    return bytes(out)


def _fix_mojibake(text: str) -> tuple[str, str] | None:
    """(repaired text, which fault) or None."""
    if len(_UTF8_AS_CP1252_RE.findall(text)) >= 3:
        raw = _cp1252_bytes(text)
        if raw is not None:
            fixed = raw.decode("utf-8", errors="replace")
            if fixed.count("�") <= len(fixed) * 0.001 and hebrew_ratio(fixed) >= 0.5:
                return fixed, "UTF-8 read as cp1252"
    if _CP1255_AS_LATIN1_RE.search(text) and hebrew_ratio(text) < 0.2:
        try:
            fixed = text.encode("latin-1").decode("cp1255")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return None
        if hebrew_ratio(fixed) >= 0.5:
            return fixed, "cp1255 read as Latin-1"
    return None


def visual_to_logical(line: str) -> str:
    """One visually ordered line in logical order: reversed, with number/Latin runs and
    bracket directions put back."""
    reversed_line = line[::-1].translate(_MIRROR)
    return _LTR_RUN_RE.sub(lambda m: m.group(0)[::-1], reversed_line)


def repair_text(text: str) -> tuple[str, str, list[str]]:
    """(text, "ok" | "repaired" | "flagged", notes)."""
    notes: list[str] = []
    status = "ok"
    fixed = _fix_mojibake(text)
    if fixed:
        text, fault = fixed
        status = "repaired"
        notes.append(f"encoding repaired: {fault}")
    elif len(_UTF8_AS_CP1252_RE.findall(text)) >= 3 or (_CP1255_AS_LATIN1_RE.search(text) and hebrew_ratio(text) < 0.2):
        status = "flagged"
        notes.append("looks mis-encoded (mojibake) but could not be repaired")
    if looks_visual_order(text):
        candidate = "\n".join(visual_to_logical(line) for line in text.splitlines())
        if not looks_visual_order(candidate):
            text = candidate
            status = "repaired" if status == "ok" else status
            notes.append("visual-order Hebrew reversed to logical order -- check before relying on it")
        else:
            status = "flagged"
            notes.append("visual-order Hebrew detected; line reversal did not fix it")
    if text.count("�") > max(3, len(text) * 0.001):
        status = "flagged"
        notes.append(f"{text.count(chr(0xFFFD))} replacement characters")
    return text, status, notes


def strip_points(text: str) -> str:
    return _POINTS_RE.sub("", text)


def normalize_for_embedding(text: str, fold_finals: bool = False) -> str:
    text = normalize_hebrew_quotes(strip_points(text).translate(_QUOTES_TO_HEBREW))
    return text.translate(_FINALS) if fold_finals else text


def normalize_for_index(text: str) -> str:
    """The lexical copy: no points, ASCII quote marks, maqaf as a hyphen, final letters folded."""
    text = strip_points(text).translate(_QUOTES_TO_ASCII).replace("־", "-")
    return re.sub(r"[ \t]+", " ", text.translate(_FINALS)).strip()
