"""Cited precedents in a chunk of Hebrew judgment text.

Case numbers: a proceeding abbreviation with gershayim (ע"א, ע"פ, רע"א, בג"ץ, דנ"א, בש"פ, ת"א,
עת"מ ...), an optional court in parentheses (ת"א (ת"א) 1234/98, ע"פ (מחוזי ב"ש) 7/20), then the
number: 1234/99, or the newer 12345-06-15. Reports: פ"ד מט(3) 355, פ"ד ד 123, and the Supreme
Court's own series (תק-על 2004(2) 1234, פ"מ נא(1) 12). Output is normalised: straight quotes,
single spaces."""

from __future__ import annotations

import re

_Q = "[\"״“”]"  # " ״ “ ”
_ABBR = rf"(?:[א-ת]{{1,4}}{_Q}[א-ת]{{1,2}}|[א-ת]{{1,4}}{_Q}[א-ת]{{1,2}}{_Q}?[א-ת]?)"
_COURT = rf"(?:\s*\((?:[א-ת\s'׳\-]|{_Q}){{1,20}}\))?"
_NUMBER = r"\d{1,6}(?:/\d{2,4}|-\d{2}-\d{2,4})"
CASE_RE = re.compile(rf"(?<![א-ת]){_ABBR}{_COURT}\s*{_NUMBER}(?![\d/])")
REPORT_RE = re.compile(
    rf"(?<![א-ת])(?:פ{_Q}ד|פ{_Q}מ|פס{_Q}מ|תק[\-־]על|תק[\-־]מח|דינים[\-־]עליון)\s*"
    rf"(?:[א-ת]{{1,3}}{_Q}?|\d{{4}})\s*(?:\(\d{{1,2}}\))?\s*,?\s*\d{{1,5}}(?!\d)")

# Words that look like "abbreviation + number" but aren't case numbers.
_NOT_A_CASE = {'ס"ק', 'ש"ח', 'מע"מ', 'ת"ז', 'ח"כ', 'סה"כ', 'בס"ד', 'מ"מ', 'ע"י', 'ב"כ', 'עו"ד', 'יו"ר'}


def _normalize(citation: str) -> str:
    citation = re.sub(_Q, '"', citation)
    citation = citation.replace("־", "-")
    return " ".join(citation.split())


def extract_citations(text: str) -> list[str]:
    """Distinct citations in order of first appearance."""
    found: dict[str, None] = {}
    for m in CASE_RE.finditer(text):
        c = _normalize(m.group(0))
        if c.split()[0].split("(")[0] in _NOT_A_CASE:
            continue
        found.setdefault(c, None)
    for m in REPORT_RE.finditer(text):
        found.setdefault(_normalize(m.group(0)).replace(" ,", ","), None)
    return list(found)
