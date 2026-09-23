"""Text extraction for Hebrew legal PDFs (Reshumot / Sefer HaChukim and the
like), rebuilt from character positions instead of trusting the PDF's text
order.

What plain extraction gets wrong on these files, and what this does about it:

  * Mixed-direction order. MuPDF returns Hebrew words in logical order but
    leaves number/Latin runs where they sit visually: "16 ביולי 2026" comes
    out as "2026 ביולי16", section "5." as ".5", "(1)" as ")1(", and a
    year jumps to the front of its title. Here every line is rebuilt from its
    characters' x positions: right-to-left for Hebrew-majority lines, with
    each run of digits/Latin (plus the . , / : % joining them) kept
    left-to-right. Lines that aren't Hebrew-majority are read left-to-right.
  * Duplicated pages. Some pages (mirrored-margin layouts) draw the whole
    page twice, one copy clipped out of view. A span repeated with the same
    text, size and baseline is dropped (first copy kept -- the texts are
    identical, only their position differs).
  * Margin notes (section titles, set smaller than the body beside it)
    would otherwise be glued into body lines. They're pulled out, joined
    across their own line breaks, and emitted as a separate line just above
    the body line they sit next to -- where the structure parser
    (legal/structure.py) takes a short unpunctuated line as a section title.
  * Noise: footnote reference markers (superscripts), the footnote block
    under the body (gazette page references), running headers/footers and
    page numbers, dotted leader lines ("¸ ¸ ¸"), zero-width characters.

Font sizes are relative to the page's dominant body size, so no single
gazette's typesetting is hard-coded.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_RTL_RE = re.compile(r"[֐-׿؀-ۿ]")
_LTR_RUN_CHAR = re.compile(r"[0-9A-Za-z]")
_RUN_JOINERS = set(".,/:%")
_DROP_CHARS = {"\ufeff", "\u200e", "\u200f", "\u200b", "¸"}  # BOM, LRM, RLM, ZWSP, leader dot
_BAND = 3.0  # baseline tolerance (pt) for grouping characters into one line
# Hebrew points and cantillation (not maqaf U+05BE / sof pasuq U+05C3, which
# are punctuation): laws add them only to disambiguate a new term, and a
# pointed word no longer matches its plain spelling in a query.
_NIKUD_RE = re.compile("[֑-ׇֽֿׁׂׅׄ]")
MARGIN_TITLE_OPEN, MARGIN_TITLE_CLOSE = "⟦", "⟧"  # ⟦title⟧ -- see legal/structure.py


@dataclass
class _Char:
    c: str
    x0: float
    x1: float
    baseline: float


@dataclass
class _Span:
    text: str
    size: float
    x0: float
    y0: float
    y1: float
    baseline: float
    chars: list[_Char]


def _raw_spans(page) -> list[_Span]:
    spans = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            for sp in line["spans"]:
                chars = [
                    _Char(ch["c"], ch["bbox"][0], ch["bbox"][2], ch["origin"][1])
                    for ch in sp["chars"]
                    if ch["c"] not in _DROP_CHARS and not _NIKUD_RE.match(ch["c"])
                ]
                text = "".join(ch.c for ch in chars)
                if text.strip():
                    spans.append(_Span(text, sp["size"], sp["bbox"][0], sp["bbox"][1], sp["bbox"][3], sp["origin"][1], chars))
    return spans


def _spans(page) -> list[_Span]:
    """The page's spans with a duplicated page copy removed. Pass 1 drops
    spans repeated verbatim (same text, size, baseline) and measures the
    copy's horizontal offset from those pairs. Pass 2 drops the stragglers
    the copy split differently (")ה(" vs ")" + "ה(") -- spans whose every
    character sits exactly one offset away from a kept character."""
    raw = _raw_spans(page)
    first_x: dict[tuple, float] = {}
    offsets: Counter = Counter()
    for sp in raw:
        key = (sp.text.strip(), round(sp.size, 1), round(sp.baseline))
        if key in first_x:
            offsets[round(sp.x0 - first_x[key])] += 1
        else:
            first_x[key] = sp.x0
    dx = offsets.most_common(1)[0][0] if offsets and offsets.most_common(1)[0][1] >= 3 else None

    kept: list[_Span] = []
    seen: set[tuple] = set()
    kept_chars: dict[tuple, list[float]] = {}
    for sp in raw:
        key = (sp.text.strip(), round(sp.size, 1), round(sp.baseline))
        if key in seen and len(sp.text.strip()) > 1:
            continue
        if dx and all(
            ch.c.isspace() or any(abs(x - (ch.x0 - dx)) < 1.5 for x in kept_chars.get((ch.c, round(ch.baseline)), []))
            for ch in sp.chars
        ):
            continue
        seen.add(key)
        kept.append(sp)
        for ch in sp.chars:
            kept_chars.setdefault((ch.c, round(ch.baseline)), []).append(ch.x0)
    return kept


_SECTION_NUMBER_RE = re.compile(r"^\.(\d{1,4}(?:[א-ת]{1,3}\d{0,3})?)\s")
_HEBREW_YEAR_RE = re.compile(r'([א-ת]"[א-ת])-\s*(,?)\s*(\d{4})')


def _normalize_line(line: str) -> str:
    """Tidies what character-level reordering leaves behind: section numbers
    with their period in front (".5 " -> "5. ", ".24 א " -> "24א. "),
    spaces inside brackets, a Hebrew-year hyphen split from its year
    ('התשי"א- , 1951' -> 'התשי"א-1951,'), a copy's doubled subsection marker,
    and spaces before punctuation."""
    line = _SECTION_NUMBER_RE.sub(r"\1. ", line)
    line = re.sub(r"\(\s+", "(", line)
    line = re.sub(r"\s+\)", ")", line)
    line = _HEBREW_YEAR_RE.sub(r"\1-\3\2", line)
    line = re.sub(r"(\([^()\s]{1,4}\))(?:\s+\1)+", r"\1", line)
    line = re.sub(r"\s+([,;:])", r"\1", line)
    line = re.sub(r"\s+\.(?=\s|$|[\"״])", ".", line)
    # "% 25" -> "25%" (the sign is read before its number right-to-left),
    # then no space between a number and its closing quote ("25% "" -> "25%"").
    line = re.sub(r"%\s*(\d+(?:\.\d+)?)", r"\1%", line)
    line = re.sub(r"(\d%?)\s+\"", r'\1"', line)
    # Section references split by the gap heuristic: "28 כה3(א)" -> "28כה3(א)",
    # "21 (א)(2)" -> "21(א)(2)".
    line = re.sub(r"\b(\d{1,4}) ([א-ת]{1,3}\d{1,3}\b|[א-ת]{1,3}(?=\())", r"\1\2", line)
    line = re.sub(r"(\d) (\([א-ת0-9]{1,2}\))", r"\1\2", line)
    line = re.sub(r"\b(\d{1,4}) ([א-ת])(?=\s|[,.;)]|$)", r"\1\2", line)  # "17 ב עד 17 ד" -> "17ב עד 17ד"
    line = re.sub(r"־\s+", "־", line)  # no space after a maqaf ("ו־ 28" -> "ו־28")
    return line.strip()


def _order_line(chars: list[_Char]) -> str:
    """Logical text of one visual line."""
    letters = [ch.c for ch in chars if ch.c.isalpha()]
    rtl = sum(bool(_RTL_RE.match(c)) for c in letters) * 2 > len(letters)
    ordered = sorted(chars, key=lambda ch: (ch.x0 + ch.x1) / 2, reverse=rtl)
    if not rtl:
        return _join(ordered)

    # Right-to-left walk: reverse back each maximal digit/Latin run (with the
    # punctuation joining its parts) so "2.5", "1992", "25%" read correctly.
    out: list[_Char] = []
    run: list[_Char] = []

    def flush() -> None:
        # Joiners trailing the run ("5." / "2,") aren't part of the number:
        # they were read after it, so they stay after it.
        trailing: list[_Char] = []
        while run and run[-1].c in _RUN_JOINERS:
            trailing.insert(0, run.pop())
        out.extend(reversed(run))
        out.extend(trailing)
        run.clear()

    for ch in ordered:
        if _LTR_RUN_CHAR.match(ch.c) or (ch.c in _RUN_JOINERS and run):
            run.append(ch)
        else:
            flush()
            out.append(ch)
    flush()
    return _join(out)


def _join(chars: list[_Char]) -> str:
    """Characters in reading order -> text, inserting spaces where the PDF
    left a visible gap instead of a space character."""
    pieces: list[str] = []
    prev: _Char | None = None
    for ch in chars:
        if prev is not None and ch.c != " " and prev.c != " ":
            gap = min(abs(ch.x0 - prev.x1), abs(prev.x0 - ch.x1))
            width = max(ch.x1 - ch.x0, 1.0)
            if gap > 0.6 * width:
                pieces.append(" ")
        pieces.append(ch.c)
        prev = ch
    return re.sub(r"\s+", " ", "".join(pieces)).strip()


def _group_lines(spans: list[_Span]) -> list[tuple[float, list[_Char]]]:
    chars = sorted((ch for sp in spans for ch in sp.chars), key=lambda ch: ch.baseline)
    lines: list[tuple[float, list[_Char]]] = []
    for ch in chars:
        if lines and ch.baseline - lines[-1][0] <= _BAND:
            lines[-1][1].append(ch)
        else:
            lines.append((ch.baseline, [ch]))
    return lines


def _page_text(page, body_size: float, repeated: set[str]) -> tuple[str, list[str]]:
    """(body text with marked margin titles, footnote lines) for one page."""
    height = page.rect.height
    spans = [sp for sp in _spans(page) if not (sp.y0 > height * 0.92 or sp.y1 < height * 0.05)]
    body = [sp for sp in spans if sp.size >= body_size * 0.92]
    small = [sp for sp in spans if sp.size < body_size * 0.92]
    if not body:
        lines = [_normalize_line(_order_line(chars)) for _, chars in _group_lines(spans)]
        return "\n".join(line for line in lines if line and line not in repeated), []

    body_bottom = max(sp.y1 for sp in body)
    body_chars = [ch for sp in body for ch in sp.chars]
    body_left = min(ch.x0 for ch in body_chars)
    body_right = max(ch.x1 for ch in body_chars)
    margin_notes, footnotes = [], []
    for sp in small:
        if sp.y0 >= body_bottom - 1:
            footnotes.append(sp)  # the footnote block under the body, markers included
        elif sp.size >= body_size * 0.75 and (sp.x0 >= body_right - 2 or max(ch.x1 for ch in sp.chars) <= body_left + 2):
            margin_notes.append(sp)
        # else: a superscript footnote reference inside the body -- dropped.

    # A margin note spans several short lines; join them into one title,
    # emitted (marked) just above the body line level with its first line.
    titles: list[list] = []  # [first baseline, last baseline, text]
    for baseline, chars in _group_lines(margin_notes):
        text = _normalize_line(_order_line(chars))
        if titles and baseline - titles[-1][1] <= body_size * 1.4:
            titles[-1][1] = baseline
            titles[-1][2] += f" {text}"
        else:
            titles.append([baseline, baseline, text])
    pending = [(first, f"{MARGIN_TITLE_OPEN}{text}{MARGIN_TITLE_CLOSE}") for first, _, text in titles]

    out: list[str] = []
    for baseline, chars in _group_lines(body):
        while pending and pending[0][0] <= baseline + _BAND:
            out.append(pending.pop(0)[1])
        line = _normalize_line(_order_line(chars))
        if line and line not in repeated:
            out.append(line)
    out.extend(text for _, text in pending)
    notes = [_normalize_line(_order_line(chars)) for _, chars in _group_lines(footnotes)]
    return "\n".join(out), [n for n in notes if n and n not in repeated]


def extract_pdf_text(path) -> tuple[str, list[int]]:
    """Returns (text, 1-based numbers of pages that look scanned).

    Footnotes are gathered into a block at the START of the text, so they
    land in the preamble chunk instead of being glued onto whichever section
    ends a page. They're worth keeping: the "*" note on a law's title carries
    the date the Knesset passed it."""
    import pymupdf

    with pymupdf.open(path) as doc:
        sizes: Counter = Counter()
        scanned = []
        for number, page in enumerate(doc, start=1):
            page_spans = _spans(page)
            for sp in page_spans:
                sizes[round(sp.size, 1)] += len(sp.text)
            if sum(len(re.findall(r"\w", sp.text)) for sp in page_spans) < 20 and page.get_images():
                scanned.append(number)
        body_size = sizes.most_common(1)[0][0] if sizes else 10.0

        pages = [_page_text(page, body_size, set()) for page in doc]
        # Lines repeated on most pages (running headers that escaped the bands).
        counts = Counter(line for text, _ in pages for line in set(text.splitlines()) if line.strip())
        repeated = {line for line, n in counts.items() if len(pages) >= 4 and n >= max(3, len(pages) // 2)}
        if repeated:
            pages = [_page_text(page, body_size, repeated) for page in doc]

    footnotes = [note for _, notes in pages for note in notes]
    parts = (["הערות שוליים:", *footnotes] if footnotes else []) + [text for text, _ in pages if text.strip()]
    return "\n".join(parts), scanned
