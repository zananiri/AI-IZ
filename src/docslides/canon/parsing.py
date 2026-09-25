"""Fetches and parses the two Canon GPT sources into `ProvisionRecord`s
(see canon/chunking.py for the shared record/chunk contract):

  * CIC 1983 (Italian) -- vatican.va/archive/cod-iuris-canonici/, one canon
    or § per <p>, e.g. `<p>Can. 748 - §1. Tutti gli uomini...</p>`, plus
    Book VI as a PDF.
  * CCEO 1990 (Latin, official Holy See text -- vatican.va has no Italian
    translation) --
    vatican.va/content/john-paul-ii/la/apost_constitutions/, a handful of
    long pages with many canons packed into shared <p> blocks separated by
    <br/>, e.g. `<b>Can. 7</b> - &sect; 1. Christifideles sunt...`.

Both markups were reverse-engineered from real sample pages fetched during
development, not from any published API/schema. If vatican.va changes its
templates, re-verify with
`scripts/ingest_canon_law.py --dry-run` before a full ingestion run.
"""

from __future__ import annotations

import re

import httpx
from bs4 import BeautifulSoup

from docslides.canon.chunking import ProvisionRecord
from docslides.logging_setup import get_logger

logger = get_logger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; docslides-canon-ingest/1.0)"}


def fetch(url: str) -> httpx.Response:
    resp = httpx.get(url, headers=_HEADERS, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    return resp


def _clean(fragment: str) -> str:
    text = BeautifulSoup(fragment, "html.parser").get_text()
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# CIC 1983 (Italian) -- vatican.va/archive/cod-iuris-canonici/cic_index_it.html
# links ~250 small HTML pages (one <p> per canon or §, e.g.
# `<p>Can. 748 - §1. Tutti gli uomini...</p>`), except Book VI (penal law,
# revised 2021), which is published only as a PDF (cic_libroVI_it.pdf).
# Amended canons carry an "n" after the number ("Can. 750n - §1."); the
# Italian text shown is already the current version.
# ---------------------------------------------------------------------------

CIC_INDEX_URL = "https://www.vatican.va/archive/cod-iuris-canonici/cic_index_it.html"
_CIC_BASE_URL = "https://www.vatican.va/archive/cod-iuris-canonici/"

_CIC_PAGE_HREF_RE = re.compile(r'href="([^"#]*cic_libro[^"#]*_it\.(?:html|pdf))')
_CIC_PAGE_START_RE = re.compile(r"_(\d+)(?:-\d+)?_it\.html$")
_CIC_CAN_RE = re.compile(r"^Can\.\s*(\d+)\s*n?\s*[-–]?\s*(?:§\s*(\d+)\.?\s*)?(.*)$", re.DOTALL)
_CIC_PARA_RE = re.compile(r"^§\s*(\d+)\.?\s*(.*)$", re.DOTALL)
_CIC_HEADING_LEVELS = ["LIBRO", "PARTE", "SEZIONE", "TITOLO", "CAPITOLO", "ARTICOLO"]
_CIC_HEADING_RE = re.compile(r"^(" + "|".join(_CIC_HEADING_LEVELS) + r")\b")
_CIC_RANGE_RE = re.compile(r"^\(\s*Cann?\.")  # "(Cann. 232 – 293)" under a heading
_PDF_FOOTER_RE = re.compile(r"^_+\s*\d+\s*$")


def dedupe_records(records: list[ProvisionRecord]) -> list[ProvisionRecord]:
    """First occurrence of each (code, number, paragraph) wins."""
    seen: set[tuple[str, str, str | None]] = set()
    out: list[ProvisionRecord] = []
    for r in records:
        key = (r.code, r.number, r.paragraph)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def discover_cic_page_urls(index_html: str) -> list[str]:
    """All Italian CIC page URLs (HTML + the Book VI PDF), in canon order.
    Some pages are linked only with a #fragment, so fragments are stripped
    before deduping."""
    urls = {
        href if href.startswith("http") else _CIC_BASE_URL + href.lstrip("/")
        for href in _CIC_PAGE_HREF_RE.findall(index_html)
    }
    # Book VI's old (pre-2021) HTML pages are still linked but are empty or
    # 404 -- the current text is the PDF.
    urls = {u.replace("http://", "https://", 1) for u in urls if "cic_libroVI_" not in u or u.endswith(".pdf")}

    def order(url: str) -> int:
        if url.endswith("cic_libroVI_it.pdf"):
            return 1311
        m = _CIC_PAGE_START_RE.search(url)
        return int(m.group(1)) if m else 10**6

    return sorted(urls, key=lambda u: (order(u), u))


def cic_html_blocks(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("td", attrs={"width": "99%"}) or soup.find(id="corpo") or soup
    return [
        text
        for p in content.find_all("p")
        if p.find_parent("li") is None and (text := _clean(p.get_text()))
    ]


def cic_pdf_blocks(pdf_bytes: bytes) -> list[str]:
    """Blank-line-separated blocks of the Book VI PDF, with page footers
    ("___ 2") removed and wrapped lines joined."""
    import fitz  # PyMuPDF -- part of the `canon` extra

    blocks: list[str] = []
    current: list[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            for line in page.get_text().splitlines():
                line = line.strip()
                if not line or _PDF_FOOTER_RE.match(line):
                    if current:
                        blocks.append(" ".join(current))
                        current = []
                    continue
                current.append(line)
    if current:
        blocks.append(" ".join(current))
    return [_clean(b) for b in blocks if b.strip()]


def _is_upper_line(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


class CicItalianParser:
    """Turns page blocks into records. Heading state carries across pages
    (fed in canon order), since a page doesn't always repeat its book/part/
    title headings."""

    def __init__(self) -> None:
        self.headings: dict[str, str] = {}
        self._pending_heading: str | None = None

    def breadcrumb(self) -> str:
        return " > ".join(self.headings[k] for k in _CIC_HEADING_LEVELS if k in self.headings)

    def parse(self, blocks: list[str], url: str) -> list[ProvisionRecord]:
        records: list[ProvisionRecord] = []
        number: str | None = None
        paragraph: str | None = None
        lines: list[str] = []
        breadcrumb = self.breadcrumb()

        def flush() -> None:
            nonlocal paragraph, lines
            text = " ".join(line for line in lines if line).strip()
            if number is not None and text:
                records.append(
                    ProvisionRecord(
                        code="cic", number=number, paragraph=paragraph, breadcrumb=breadcrumb,
                        text=text, source_url=url, language="it",
                    )
                )
            paragraph, lines = None, []

        for text in blocks:
            m_head = _CIC_HEADING_RE.match(text)
            if m_head:
                flush()
                number = None
                level = m_head.group(1)
                self.headings[level] = text
                for lower in _CIC_HEADING_LEVELS[_CIC_HEADING_LEVELS.index(level) + 1 :]:
                    self.headings.pop(lower, None)
                self._pending_heading = level
                continue
            if _CIC_RANGE_RE.match(text):
                continue
            if _is_upper_line(text):
                # A heading's descriptive title ("I FEDELI CRISTIANI"), or page
                # chrome ("CODICE DI DIRITTO CANONICO") -- never canon text.
                if self._pending_heading:
                    level = self._pending_heading
                    self.headings[level] = f"{self.headings[level]} - {text}"
                    self._pending_heading = None
                continue
            self._pending_heading = None

            m_can = _CIC_CAN_RE.match(text)
            if m_can:
                flush()
                number, paragraph = m_can.group(1), m_can.group(2)
                breadcrumb = self.breadcrumb()
                rest = m_can.group(3).strip()
                lines = [rest] if rest else []
                continue

            m_para = _CIC_PARA_RE.match(text)
            if m_para and number is not None:
                flush()
                paragraph = m_para.group(1)
                lines = [m_para.group(2).strip()]
                continue

            if number is not None:
                lines.append(text)  # "1º ..." items, wrapped continuations

        flush()
        return records



# ---------------------------------------------------------------------------
# CCEO (Latin) -- currently 3 long pages ("-1", "-2", "-3"); re-check for a
# "-4" etc. if a future ingestion run comes up suspiciously short.
# ---------------------------------------------------------------------------

CCEO_PAGE_URLS = [
    "https://www.vatican.va/content/john-paul-ii/la/apost_constitutions/documents/hf_jp-ii_apc_19901018_codex-can-eccl-orient-1.html",
    "https://www.vatican.va/content/john-paul-ii/la/apost_constitutions/documents/hf_jp-ii_apc_19901018_codex-can-eccl-orient-2.html",
    "https://www.vatican.va/content/john-paul-ii/la/apost_constitutions/documents/hf_jp-ii_apc_19901018_codex-can-eccl-orient-3.html",
]

_CCEO_HEADING_RE = re.compile(
    r'<a name="(TITULUS|CAPUT)_[^"]*"[^>]*>\s*</a>\s*([^<]+)',
    re.IGNORECASE,
)
# Canon headers vary: "<b>Can. 7</b> - ", "<b>Can. 66<i><sup>n</sup></i> </b>-"
# (amended), "<b>Can. 1409<i><sup>n</sup></i> - </b>", "<b>Can: 626</b> -",
# "<b>Can.</b> <b>1191</b> -", and unbolded "Can. 329 - ". Case-sensitive so
# in-text cross-references ("can. 181") never match.
_CCEO_CAN_START_RE = re.compile(
    r"(?:<b>\s*)?Can[.:]\s*(?:</b>\s*<b>\s*)?(?:&nbsp;|\s)*(\d+)(?:\s|&nbsp;|<[^>]*>|n)*?\s*-\s*(?:</b>\s*)?"
)
# A real paragraph marker is "§ 1." -- cross-references ("can. 181, § 1, 182")
# have no period and follow a comma, so exclude those.
_CCEO_PARA_RE = re.compile(r"(?<!,)(?<!,\s)§\s*(\d+)\.\s*")
_CCEO_EARLIER_VERSION_RE = re.compile(r"Versione precedente", re.IGNORECASE)


def parse_cceo_page(html: str, url: str) -> list[ProvisionRecord]:
    """Heading labels only (e.g. "TITULUS IV", "CAPUT II") -- earlier
    versions also tried to capture each heading's descriptive title from a
    following <p>, but that second regex hop was fragile against the actual
    markup (stray nested tags let it swallow hundreds of characters of
    unrelated canon text into the breadcrumb). The label alone is enough
    context to be useful and far more robust to parse correctly."""
    headings: list[tuple[int, str, str]] = []
    for m in _CCEO_HEADING_RE.finditer(html):
        kind = m.group(1).upper()
        label = _clean(m.group(2))[:80]  # defense in depth against any future runaway match
        headings.append((m.start(), kind, label))

    def breadcrumb_at(offset: int) -> str:
        titulus = caput = ""
        for pos, kind, text in headings:
            if pos > offset:
                break
            if kind == "TITULUS":
                titulus, caput = text, ""
            else:
                caput = text
        return " > ".join(p for p in (titulus, caput) if p)

    can_starts = list(_CCEO_CAN_START_RE.finditer(html))
    records: list[ProvisionRecord] = []

    for i, m in enumerate(can_starts):
        number = m.group(1)
        body_start = m.end()
        body_end = can_starts[i + 1].start() if i + 1 < len(can_starts) else len(html)
        body_html = html[body_start:body_end]
        if (ev := _CCEO_EARLIER_VERSION_RE.search(body_html)) is not None:
            body_html = body_html[: ev.start()]
        body_text = _clean(body_html)
        breadcrumb = breadcrumb_at(m.start())

        para_matches = list(_CCEO_PARA_RE.finditer(body_text))
        if not para_matches:
            if body_text:
                records.append(
                    ProvisionRecord(
                        code="cceo", number=number, paragraph=None, breadcrumb=breadcrumb,
                        text=body_text, source_url=url, language="la",
                    )
                )
            continue

        lead = body_text[: para_matches[0].start()].strip()
        for j, pm in enumerate(para_matches):
            seg_start = pm.end()
            seg_end = para_matches[j + 1].start() if j + 1 < len(para_matches) else len(body_text)
            seg_text = body_text[seg_start:seg_end].strip()
            if j == 0 and lead:
                seg_text = f"{lead} {seg_text}".strip()
            if seg_text:
                records.append(
                    ProvisionRecord(
                        code="cceo", number=number, paragraph=pm.group(1), breadcrumb=breadcrumb,
                        text=seg_text, source_url=url, language="la",
                    )
                )

    return dedupe_records(records)
