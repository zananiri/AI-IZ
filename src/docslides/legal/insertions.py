"""Provisions that an amending law inserts into another law (for chunking.py).

An amending section such as 'אחרי סעיף 132 יבוא: 132א. (א) ...' carries whole
new provisions of the amended law. Chunked as one unit, a long insertion (a
new chapter of three sections) sprawls across parts that every question about
any of it drags in together, and it can only be cited by the amending law's
own number ("6(4)") -- never by the number a lawyer asks about ("116יז10(ד)").
`find_insertion` picks those provisions out so each gets its own chunk,
labelled with both numbers.

Knesset PDFs mangle inserted numbers: a margin title and its number come out
merged ("הרכב ועדות 116 יז 11. הקלפי ..."), bidi-spaced ("116 יז 12(ג)"), or
lost entirely for the first section after an inserted chapter heading -- that
number is inferred from the next one (116יז11 -> 116יז10) and marked inferred.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_INSTRUCTION_RE = re.compile(r"יבוא\s*[:\-–]?\s*$")
_HEAD_RE = re.compile(r"^[\"״]?(?P<num>\d{1,4}[א-ת]{1,3}\d{0,3})\.(?:\s+(?P<rest>.*))?$")
_MERGED_HEAD_RE = re.compile(
    r"^(?P<pre>[^\d()\"״]*?)\s*(?P<num>\d{1,4}\s?[א-ת]{1,3}\s?\d{1,3})\.\s*(?P<post>[^\d()]*)$"
)
_CHAPTER_RE = re.compile(r"^[\"״]?(?:פרק|סימן)\s")
_CLOSE_RE = re.compile(r"(\.[\"״]|[\"״][;.,])\s*$")  # '...פרק זה.";' ends the quoted block
_SUBSECTION_START_RE = re.compile(r"^[\"״]?\([א-ת]{1,2}\d?\)")
_SPACED_NUMBER_RE = re.compile(r"(?<!\w)(\d{2,4}) ([א-ת]{1,2}) (\d{1,2})(?=[.(\s,;:\"״]|$)")
# Words that sit between two numbers in ordinary text ("6 או 7", "62 עד 64").
_NOT_SECTION_LETTERS = {"או", "עד", "ו", "ב", "ל", "מ", "ה", "ש", "כ"}


def join_spaced_section_numbers(text: str) -> str:
    """'116 יז 12(ג)' -> '116יז12(ג)' (the PDF's bidi spacing), leaving
    '6 או 7' and '62 עד 64' alone."""

    def join(match: re.Match) -> str:
        if match.group(2) in _NOT_SECTION_LETTERS:
            return match.group(0)
        return match.group(1) + match.group(2) + match.group(3)

    return _SPACED_NUMBER_RE.sub(join, text)


@dataclass
class InsertedProvision:
    number: str  # "116יז10"; "" if it couldn't be recovered
    title: str | None  # margin title, when the text kept it
    lines: list[str]
    inferred_number: bool = False


@dataclass
class Insertion:
    context: list[str]  # the amending instruction, plus an inserted chapter heading
    provisions: list[InsertedProvision] = field(default_factory=list)
    trailing: list[str] = field(default_factory=list)  # further instructions after the quoted block


def _head(line: str) -> tuple[str, str | None, str] | None:
    """(number, margin title, text after the number) if `line` opens an inserted section."""
    stripped = line.strip()
    match = _HEAD_RE.match(stripped)
    if match:
        return match.group("num"), None, (match.group("rest") or "").strip()
    if _SUBSECTION_START_RE.match(stripped):
        return None
    match = _MERGED_HEAD_RE.match(stripped)
    if match and match.group("pre").strip():  # a merged margin title, not a stray number
        title = " ".join(p for p in (match.group("pre").strip(), match.group("post").strip()) if p)
        return re.sub(r"\s", "", match.group("num")), title or None, ""
    return None


def _previous_number(number: str) -> str | None:
    match = re.match(r"^(.*?)(\d+)$", number)
    if not match or int(match.group(2)) <= 1:
        return None
    return f"{match.group(1)}{int(match.group(2)) - 1}"


def find_insertion(text: str) -> Insertion | None:
    """The provisions `text` (one amending section or subsection) inserts, or
    None when it inserts none -- replacing wording ('במקום "ה־40" יבוא
    "ה־43"') or appending a list item isn't inserting a provision."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for start, line in enumerate(lines):
        if not _INSTRUCTION_RE.search(line):
            continue
        body = lines[start + 1 :]
        heads = [i for i, candidate in enumerate(body) if _head(candidate)]
        if not heads:
            continue
        close = next((i for i in range(heads[0], len(body)) if _CLOSE_RE.search(body[i])), len(body) - 1)
        insertion = Insertion(context=lines[: start + 1], trailing=body[close + 1 :])
        body = body[: close + 1]
        i = 0
        if body and _CHAPTER_RE.match(body[0]):
            insertion.context.append(body[0])
            i = 1
            while i < heads[0] and not _SUBSECTION_START_RE.match(body[i]):
                insertion.context.append(body[i])
                i += 1
        if i < heads[0]:  # provision text before the first number: its heading was lost
            inferred = _previous_number(_head(body[heads[0]])[0])
            insertion.provisions.append(InsertedProvision(inferred or "", None, body[i : heads[0]], True))
        for k, h in enumerate(heads):
            number, title, first = _head(body[h])
            end = heads[k + 1] if k + 1 < len(heads) else len(body)
            insertion.provisions.append(InsertedProvision(number, title, ([first] if first else []) + body[h + 1 : end]))
        return insertion
    return None
