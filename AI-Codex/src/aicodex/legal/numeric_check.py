"""Deterministic numeric grounding check for the Legal pipeline.

Every number in the final answer -- amounts, day counts, percentages, dates,
years, section numbers -- must appear in the evidence the answer cites (or in
the user's own question). A number that doesn't is exactly the failure mode
of an ungrounded model: a confident threshold, deadline or fine amount made
up from general knowledge. Unsupported numbers make the turn escalate and are
listed in the answer's notes; the check never edits the answer.

Matching is by value, not spelling, in both Spanish and English notation: "2,5 millones" ==
"2.500.000" == "2,500,000", "18 %" == "18", and digits inside article numbers ("1902.2") count as
present. Numbers written out in words ("veinte días") aren't checked.
"""

from __future__ import annotations

import re

_NUMBER_RE = re.compile(
    r"(?<![\d.,])(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)(?![\d])"
    r"\s*(%|por\s+ciento)?\s*(millones|millón|mil\s+millones|millardos?|mil)?",
    re.IGNORECASE,
)
_SCALE = {"mil": 1_000, "millón": 1_000_000, "millones": 1_000_000, "mil millones": 1_000_000_000,
          "millardo": 1_000_000_000, "millardos": 1_000_000_000}


def _parse(raw: str) -> list[float]:
    """Every value `raw` can mean: '1.500' is 1500 in Spanish, 1.5 in English; '2,5' is 2.5."""
    values: list[float] = []
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", raw):  # 2.500.000 / 1.500,50
        values.append(float(raw.replace(".", "").replace(",", ".")))
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", raw):  # 2,500,000
        values.append(float(raw.replace(",", "")))
    if re.fullmatch(r"\d+(?:[.,]\d+)?", raw):
        values.append(float(raw.replace(",", ".")))
    return values or [0.0]


def _scale(word: str | None) -> int | None:
    return _SCALE.get(re.sub(r"\s+", " ", word.lower())) if word else None


def _values(text: str) -> set[float]:
    values: set[float] = set()
    for match in _NUMBER_RE.finditer(text):
        scale = _scale(match.group(3))
        for base in _parse(match.group(1)):
            values.add(base)
            if scale:
                values.add(base * scale)
    return values


def unsupported_numbers(answer: str, evidence_texts: list[str], question: str = "") -> list[str]:
    """Numbers stated in `answer` (as written) whose value appears in neither
    the evidence nor the question, in order of first appearance."""
    allowed = _values(question)
    for text in evidence_texts:
        allowed |= _values(text)
    missing: list[str] = []
    for match in _NUMBER_RE.finditer(answer):
        scale = _scale(match.group(3))
        candidates = _parse(match.group(1))
        if any(v in allowed or (scale and v * scale in allowed) for v in candidates):
            continue
        written = match.group(0).strip()
        if written not in missing:
            missing.append(written)
    return missing
