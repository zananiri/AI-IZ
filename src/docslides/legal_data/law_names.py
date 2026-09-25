"""Law-name normalization and matching between Wikisource titles and Knesset OData names.

The same law is written 'חוק החוזים (חלק כללי), תשל״ג–1973' on Wikisource and
'חוק החוזים (חלק כללי), התשל"ג-1973' in OData: different quote marks, dashes and the optional
ה of the Hebrew year. name_key() erases exactly those differences and nothing else, so a
match is a match. Ambiguous keys (two registry ids) never match."""

from __future__ import annotations

import re
import unicodedata

from docslides.legal_data.hebrew import strip_points

_QUOTES = str.maketrans({"״": '"', "“": '"', "”": '"', "„": '"', "׳": "'", "‘": "'", "’": "'", "`": "'"})
_DASH_RE = re.compile(r"\s*[‐‑‒–—―־-]\s*")
# The Hebrew year before a Gregorian one, with the optional ה: 'התשל"ג-1973' / 'תשל"ג-1973'.
_HEBREW_YEAR_RE = re.compile(r"(?<![א-ת])ה?(ת[א-ת\"']{1,6}-\d{4})")


def normalize_name(name: str) -> str:
    text = strip_points(unicodedata.normalize("NFC", name)).translate(_QUOTES)
    text = _DASH_RE.sub("-", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    return re.sub(r"\s+", " ", text).strip()


def name_key(name: str) -> str:
    text = _HEBREW_YEAR_RE.sub(r"\1", normalize_name(name))
    text = text.replace('"', "").replace("'", "")
    return re.sub(r"[^\w]+", " ", text).strip()


def has_prefix(title: str, prefix: str) -> bool:
    return name_key(title).startswith(name_key(prefix))


class NameIndex:
    def __init__(self) -> None:
        self._ids: dict[str, set[int]] = {}

    def add(self, name: str | None, registry_id: int) -> None:
        if name:
            self._ids.setdefault(name_key(name), set()).add(registry_id)

    def lookup(self, name: str) -> int | None:
        ids = self._ids.get(name_key(name))
        return next(iter(ids)) if ids and len(ids) == 1 else None

    def __len__(self) -> int:
        return len(self._ids)
