"""Deterministic numeric grounding check for the Legal pipeline.

Every number in the final answer -- amounts, day counts, percentages, dates,
years, section numbers -- must appear in the evidence the answer cites (or in
the user's own question). A number that doesn't is exactly the failure mode
of an ungrounded model: a confident threshold, deadline or fine amount made
up from general knowledge. Unsupported numbers make the turn escalate and are
listed in the answer's notes; the check never edits the answer.

Matching is by value, not spelling: "2.5 מיליון" == "2,500,000", "18%" ==
"18", and digits embedded in section numbers ("116יז10") count as present.
Numbers written out in Hebrew words aren't checked.
"""

from __future__ import annotations

import re

_NUMBER_RE = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\d])\s*(%|אחוז)?\s*(מיליארד|מיליון|אלף|אלפים)?")
_SCALE = {"אלף": 1_000, "אלפים": 1_000, "מיליון": 1_000_000, "מיליארד": 1_000_000_000}


def _values(text: str) -> set[float]:
    values: set[float] = set()
    for match in _NUMBER_RE.finditer(text):
        base = float(match.group(1).replace(",", ""))
        values.add(base)
        if match.group(3):
            values.add(base * _SCALE[match.group(3)])
    return values


def unsupported_numbers(answer: str, evidence_texts: list[str], question: str = "") -> list[str]:
    """Numbers stated in `answer` (as written) whose value appears in neither
    the evidence nor the question, in order of first appearance."""
    allowed = _values(question)
    for text in evidence_texts:
        allowed |= _values(text)
    missing: list[str] = []
    for match in _NUMBER_RE.finditer(answer):
        base = float(match.group(1).replace(",", ""))
        scaled = base * _SCALE[match.group(3)] if match.group(3) else None
        if base in allowed or (scaled is not None and scaled in allowed):
            continue
        written = match.group(0).strip()
        if written not in missing:
            missing.append(written)
    return missing
