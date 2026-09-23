"""Structural parsing of Israeli statute / regulation / ruling text into
sections (סעיפים), for legal/chunking.py.

Plain-text line heuristics, not a published schema -- they're tuned to how
Knesset / Reshumot texts read once extracted to text:

  * Headings: חלק / פרק / סימן (Part / Chapter / Article in English
    translations) followed by an ordinal label ("פרק א'", "פרק שני",
    "Chapter 3"). A bare Hebrew word after "פרק" isn't accepted as a label,
    so body text like "פרק זמן סביר" never opens a chapter.
  * Sections: a line starting "12." / "5א." / "סעיף 12." / "Section 12.",
    optionally preceded by a short marginal title on the same line
    ("הגדרות 1. בחוק זה"). A short unpunctuated line directly above a
    section is also taken as its marginal title. To keep list items, years
    ("1973.") and the like from opening bogus sections, numbering must
    advance monotonically in small steps.
  * Subsections: "(א)" / "(1)" / "(a)" at line start. Only markers of the
    same kind as the section's first marker count as top-level, so "(1)"
    paragraphs nested inside "(א)" subsections stay with their parent.

Rulings use numbered paragraphs, which parse the same way (a "section" is
then a paragraph). Always preview a new source with
`scripts/ingest_legal.py stage --dry-run` before trusting its structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEB_ORDINALS = (
    "ראשון|שני|שלישי|רביעי|חמישי|שישי|שביעי|שמיני|תשיעי|עשירי|"
    "ראשונה|שנייה|שניה|שלישית|רביעית|חמישית|שישית|שביעית|שמינית|תשיעית|עשירית"
)
_HEB_LABEL = rf"(?:[א-ת]{{1,2}}['׳\"״]|[א-ת](?=\s*[:–—\-]|\s*$)|{_HEB_ORDINALS})"
_LATIN_LABEL = r"(?:\d{1,3}[A-Za-z]?|[IVXLC]{1,6}|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)"
_HEADING_TAIL = r"\s*[:.–—\-]?\s*(?P<title>.{0,120})$"


def _heading_re(hebrew_word: str, english_word: str) -> re.Pattern[str]:
    return re.compile(
        rf"^\s*(?:{hebrew_word}\s+(?P<heb>{_HEB_LABEL}|\d{{1,3}})|{english_word}\s+(?P<lat>{_LATIN_LABEL})\b)"
        + _HEADING_TAIL
    )


_DIVISION_RE = _heading_re("חלק", "Part")
_CHAPTER_RE = _heading_re("פרק", "Chapter")
_SUBCHAPTER_RE = _heading_re("סימן", "Article")

_SECTION_RE = re.compile(
    r"^\s*(?:(?P<title>[^\d()\n.:;]{2,60}?)\s+)?(?:(?:סעיף|Section)\s+)?"
    r"(?P<num>\d{1,4}[א-ת]{0,2})\.(?:\s+(?P<body>.*))?$"
)
_SUBSECTION_RE = re.compile(r"^\s*\((?P<label>[א-ת]{1,2}\d?|\d{1,3}|[a-z]{1,2})\)\s*(?P<rest>.*)$")

_XREF_RE = re.compile(
    r"(?:\b|(?<=[ובלמשכה]))(?:[ובלמשכה]{0,3}(?:סעיפים|סעיפי|סעיף)|[Ss]ections?)\s+"
    r"(?P<list>\d{1,4}[א-ת]?(?:\([^)]{1,4}\))*(?:\s*(?:,|ו[-־]?|או|עד|-|–|and|or|to)\s*\d{1,4}[א-ת]?(?:\([^)]{1,4}\))*)*)"
)
_XREF_EXTERNAL_RE = re.compile(r"^\s*(?:ל|ב)?(?:חוק|פקודת|פקודה|תקנות)(?!\s+זה)|^\s*of the\b|^\s*of (?!this)")
_XREF_RANGE_RE = re.compile(r"(\d{1,4})\s*(?:עד|-|–|to)\s*(\d{1,4})")

# Section titles marked by legal/pdf_text.py (margin notes): ⟦title⟧
_MARGIN_TITLE_RE = re.compile(r"^\s*⟦(.+)⟧\s*$")
# Amending laws quote whole new provisions: 'אחרי סעיף 24 יבוא:' then '24א. ...'
# up to a closing quote. Text in between is body of the amending section --
# never headings or sections of the amending law itself.
_QUOTE_OPEN_RE = re.compile(r"יבוא\s*[:\-–]?\s*$")
_QUOTE_CLOSE_RE = re.compile(r"(\.[\"״]|[\"״][;.,])\s*$")

_QUOTED = "\x00"  # internal prefix on lines inside a quoted amendment block

_MAX_TITLE_WORDS = 6
_MAX_FIRST_SECTION = 10
_MAX_SECTION_JUMP = 30


@dataclass
class Subsection:
    label: str
    text: str


@dataclass
class Section:
    number: str  # "12", "5א", or "preamble"
    title: str | None
    text: str  # full section body, subsections included
    division: str | None = None  # חלק / Part
    chapter: str | None = None  # פרק / Chapter
    subchapter: str | None = None  # סימן / Article
    intro: str = ""  # body text before the first top-level subsection
    subsections: list[Subsection] = field(default_factory=list)
    cross_refs: list[str] = field(default_factory=list)  # other section numbers in this law


def _section_key(number: str) -> tuple[int, str]:
    match = re.match(r"(\d+)(.*)", number)
    return (int(match.group(1)), match.group(2)) if match else (0, number)


def _is_marginal_title(line: str) -> bool:
    stripped = line.strip()
    return (
        bool(stripped)
        and len(stripped.split()) <= 5
        and stripped[-1] not in ".;:,–—-"
        and stripped[0] not in "\"'״׳"
        and not _SUBSECTION_RE.match(stripped)
    )


def _marker_kind(label: str) -> str:
    if label.isdigit():
        return "digit"
    if re.fullmatch(r"[a-z]{1,2}", label):
        return "latin"
    return "hebrew"


def _split_subsections(body_lines: list[str]) -> tuple[str, list[Subsection]]:
    top_kind: str | None = None
    intro: list[str] = []
    subsections: list[Subsection] = []
    for raw in body_lines:
        quoted = raw.startswith(_QUOTED)
        line = raw.lstrip(_QUOTED)
        match = None if quoted else _SUBSECTION_RE.match(line)  # quoted text's markers aren't ours
        if match:
            kind = _marker_kind(match.group("label"))
            if top_kind is None:
                top_kind = kind
            if kind == top_kind:
                subsections.append(Subsection(label=match.group("label"), text=line.strip()))
                continue
        if subsections:
            subsections[-1].text += "\n" + line.strip()
        else:
            intro.append(line.strip())
    return "\n".join(l for l in intro if l).strip(), subsections


def extract_cross_references(text: str, own_number: str | None = None) -> list[str]:
    """Section numbers of THIS law referenced from `text`, in first-seen
    order. References followed by another law's name ("סעיף 5 לחוק
    החוזים", "section 5 of the X Law") are external and skipped -- they
    can't be resolved to a chunk of this document."""
    refs: list[str] = []
    for match in _XREF_RE.finditer(text):
        if _XREF_EXTERNAL_RE.match(text[match.end() : match.end() + 30]):
            continue
        listing = re.sub(r"\([^)]*\)", "", match.group("list"))
        numbers = re.findall(r"\d{1,4}[א-ת]?", listing)
        for start, end in _XREF_RANGE_RE.findall(listing):
            lo, hi = int(start), int(end)
            if 0 < hi - lo <= 20:
                numbers.extend(str(n) for n in range(lo + 1, hi))
        for number in numbers:
            if number != own_number and number not in refs:
                refs.append(number)
    return refs


def parse_sections(text: str) -> list[Section]:
    division = chapter = subchapter = None
    sections: list[Section] = []
    preamble: list[str] = []
    current: Section | None = None
    current_lines: list[str] = []
    last_key: tuple[int, str] | None = None
    pending_title: str | None = None
    in_quote = False

    def close_current() -> None:
        if current is None:
            return
        while current_lines and not current_lines[-1].strip():
            current_lines.pop()
        current.intro, current.subsections = _split_subsections(current_lines)
        current.text = "\n".join(l.lstrip(_QUOTED).strip() for l in current_lines if l.strip(_QUOTED).strip())

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        target = current_lines if current is not None else preamble

        margin_title = _MARGIN_TITLE_RE.match(line)
        if in_quote:
            target.append(_QUOTED + (margin_title.group(1).strip() if margin_title else line))
            in_quote = not _QUOTE_CLOSE_RE.search(line)
            continue
        if margin_title:
            title_text = margin_title.group(1).strip()
            pending_title = f"{pending_title} {title_text}" if pending_title else title_text
            continue

        heading = None
        for pattern, level in ((_DIVISION_RE, "division"), (_CHAPTER_RE, "chapter"), (_SUBCHAPTER_RE, "subchapter")):
            if pattern.match(line) and len(line.split()) <= 14:
                heading = level
                break
        if heading:
            close_current()
            current, current_lines = None, []
            label = line.strip()
            if heading == "division":
                division, chapter, subchapter = label, None, None
            elif heading == "chapter":
                chapter, subchapter = label, None
            else:
                subchapter = label
            continue

        match = _SECTION_RE.match(line)
        if match:
            title = (match.group("title") or "").strip() or None
            key = _section_key(match.group("num"))
            title_ok = title is None or len(title.split()) <= _MAX_TITLE_WORDS
            if last_key is None:
                advances = key[0] <= _MAX_FIRST_SECTION
            else:
                advances = key > last_key and key[0] - last_key[0] <= _MAX_SECTION_JUMP
            if title_ok and advances:
                if pending_title:
                    title, pending_title = pending_title, None
                if title is None:
                    # Never the previous section's own first line -- that's its body.
                    pending, floor = (current_lines, 1) if current is not None else (preamble, 0)
                    if len(pending) > floor and not pending[-1].startswith(_QUOTED) and _is_marginal_title(pending[-1]):
                        title = pending.pop().strip()
                close_current()
                current = Section(
                    number=match.group("num"),
                    title=title,
                    text="",
                    division=division,
                    chapter=chapter,
                    subchapter=subchapter,
                )
                current_lines = [match.group("body") or ""]
                sections.append(current)
                last_key = key
                in_quote = bool(_QUOTE_OPEN_RE.search(line))
                continue

        if pending_title:  # a margin note beside running text, not a new section
            target.append(pending_title)
            pending_title = None
        target.append(line)
        in_quote = bool(_QUOTE_OPEN_RE.search(line))

    close_current()

    preamble_text = "\n".join(l.lstrip(_QUOTED).strip() for l in preamble if l.strip(_QUOTED).strip())
    if preamble_text:
        sections.insert(0, Section(number="preamble", title=None, text=preamble_text, intro=preamble_text))

    known = {s.number for s in sections}
    for section in sections:
        if (section.title or "").startswith("תיקון"):
            continue  # an amending section's "סעיף N" means the amended law's section N
        section.cross_refs = [
            ref for ref in extract_cross_references(section.text, section.number) if ref in known
        ]
    return sections
