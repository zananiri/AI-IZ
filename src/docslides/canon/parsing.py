"""Fetches and parses the two Canon GPT sources into `ProvisionRecord`s
(see canon/chunking.py for the shared record/chunk contract):

  * CIC 1983 (English) -- vatican.va/archive/cod-iuris-canonici/, one canon
    per <p>, e.g. `<p>Can.&nbsp;1166 Sacramentals are...</p>`.
  * CCEO 1990 (Latin, official Holy See text -- see module docstring in
    scripts/ingest_canon_law.py for why not English) --
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
# CIC (English)
# ---------------------------------------------------------------------------

CIC_INDEX_URL = "http://www.vatican.va/archive/cod-iuris-canonici/cic_index_en.html"
_CIC_PAGE_HREF_RE = re.compile(r'href="([^"]*cic_lib\d[^"]*)"')

_CIC_CAN_RE = re.compile(r"^Can\.?\s*(\d+)\s*(?:§\s*(\d+)\.?\s*)?(.*)$", re.DOTALL)
_CIC_PARA_RE = re.compile(r"^§\s*(\d+)\.?\s*(.*)$", re.DOTALL)
_CIC_HEADING_LABEL_RE = re.compile(r"^(PART|TITLE)\b", re.IGNORECASE)


def discover_cic_page_urls(index_html: str) -> list[str]:
    """The index page (cic_index_en.html) links to ~45 per-book-range pages.
    Some are linked only via a #fragment-qualified entry (a specific title/
    chapter within the page), never as a bare URL on its own -- strip
    fragments before deduping, or those pages get silently dropped."""
    hrefs: set[str] = set()
    for m in _CIC_PAGE_HREF_RE.finditer(index_html):
        href = m.group(1).split("#", 1)[0]
        if href.endswith("_en.html"):
            hrefs.add(href)
    base = "http://www.vatican.va"
    return sorted(base + href if href.startswith("/") else href for href in hrefs)


def _cic_book_label(html: str) -> str:
    m = re.search(r"<title>\s*Code of Canon Law\s*-\s*(Book [^(<]+)", html)
    return m.group(1).strip() if m else ""


def parse_cic_page(html: str, url: str) -> list[ProvisionRecord]:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("td", attrs={"width": "99%"}) or soup.find(id="corpo") or soup
    book_label = _cic_book_label(html)

    heading_parts: dict[str, str] = {}
    heading_buffer: list[str] = []
    pending_chapter_title = False
    records: list[ProvisionRecord] = []

    number: str | None = None
    paragraph: str | None = None
    lines: list[str] = []
    breadcrumb = book_label

    def breadcrumb_string() -> str:
        parts = [book_label, heading_parts.get("part"), heading_parts.get("title"), heading_parts.get("chapter")]
        return " > ".join(p for p in parts if p)

    def absorb_heading_buffer() -> None:
        nonlocal heading_buffer
        buf = heading_buffer
        i = 0
        while i < len(buf):
            line = buf[i]
            m = _CIC_HEADING_LABEL_RE.match(line)
            if not m:
                i += 1
                continue
            kind = m.group(1).upper()
            desc = ""
            if i + 1 < len(buf) and not _CIC_HEADING_LABEL_RE.match(buf[i + 1]):
                desc = buf[i + 1]
                i += 1
            combined = f"{line} {desc}".strip(" :")
            if kind == "PART":
                heading_parts["part"] = combined
                heading_parts.pop("title", None)
                heading_parts.pop("chapter", None)
            else:
                heading_parts["title"] = combined
                heading_parts.pop("chapter", None)
            i += 1
        heading_buffer = []

    def flush() -> None:
        nonlocal number, paragraph, lines
        if number is not None:
            text = " ".join(line for line in lines if line).strip()
            if text:
                records.append(
                    ProvisionRecord(
                        code="cic",
                        number=number,
                        paragraph=paragraph,
                        breadcrumb=breadcrumb,
                        text=text,
                        source_url=url,
                        language="en",
                    )
                )
        number, paragraph, lines = None, None, []

    for p in content.find_all("p"):
        if p.find_parent("li") is not None:
            continue  # table-of-contents entry, not real content
        text = _clean(p.get_text())
        if not text:
            continue

        if p.get("align") == "center" and p.find("b") is not None:
            heading_buffer.append(text)
            continue
        if heading_buffer:
            absorb_heading_buffer()

        chapter_anchor = p.find("a", attrs={"name": re.compile(r"^CHAPTER", re.IGNORECASE)})
        if chapter_anchor:
            flush()
            heading_parts["chapter"] = text
            pending_chapter_title = True
            continue
        if pending_chapter_title and not _CIC_CAN_RE.match(text):
            heading_parts["chapter"] = f"{heading_parts.get('chapter', '')} {text}".strip()
            pending_chapter_title = False
            continue
        pending_chapter_title = False

        m_can = _CIC_CAN_RE.match(text)
        if m_can:
            flush()
            number, paragraph = m_can.group(1), m_can.group(2)
            breadcrumb = breadcrumb_string()
            rest = m_can.group(3).strip()
            lines = [rest] if rest else []
            continue

        m_para = _CIC_PARA_RE.match(text)
        if m_para and number is not None:
            flush()
            paragraph = m_para.group(1)
            breadcrumb = breadcrumb_string()
            lines = [m_para.group(2).strip()]
            continue

        if number is not None:
            lines.append(text)

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
_CCEO_CAN_START_RE = re.compile(r"<b>\s*Can\.\s*(\d+)\s*</b>\s*-\s*", re.IGNORECASE)
_CCEO_PARA_RE = re.compile(r"§\s*(\d+)\.?\s*")


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
        body_text = _clean(html[body_start:body_end])
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

    return records
