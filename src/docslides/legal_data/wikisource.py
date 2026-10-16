"""The Open Law Book (ספר החוקים הפתוח) of Hebrew Wikisource, from the official Wikimedia dump.

Dump: resolve_dump() picks the newest completed hewikisource dump (or a pinned date) from
dumpstatus.json, which also gives its size and sha1. iter_pages() streams pages out of the
.xml.bz2 -- from the downloaded file, or from a live download's first megabytes in a sample run.

Pages: the Open Law Book marks structure with `ח:` templates, one per line:
    {{ח:כותרת|חוק החוזים (חלק כללי), תשל״ג–1973}}
    {{ח:פתיח-התחלה}} {{ח:מאגר|2000292}} {{ח:תיבה|ס״ח תשל״ג, 118|...|https://fs.knesset.gov.il/...pdf}} ...
    {{ח:סוגר}}
    {{ח:מבוא}} בתוקף סמכותי לפי ...                   (regulations' preamble)
    {{ח:קטע2|פרק א|פרק א׳: כריתת החוזה}}             (קטע1 חלק, קטע2 פרק, קטע3 סימן, קטע4 deeper)
    {{ח:סעיף|3|חזרה מן ההצעה}}
    {{ח:ת}} body text   {{ח:תת|(א)}} subsection   {{ח:תתת|(1)}} / {{ח:תתתת|(א)}} nested items
Inline: {{ח:פנימי|anchor|text}}, {{ח:חיצוני|target|text}}, {{ח:הערה|...}}, {{ח:מוקטן|...}}.
`ח:מאגר` is the law's id in the Knesset registry (KNS_IsraelLaw.Id). `ח:תיבה` cites a gazette
publication and links its PDF; the citation is recorded, the link is never fetched from here.
Only namespace-0, non-redirect pages carrying `ח:` law templates count as laws: index, portal,
category and list pages don't have them. Templates the parser doesn't know are counted per page
(quality notes), not silently dropped.
"""

from __future__ import annotations

import bz2
import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from urllib.parse import quote, urljoin

from docslides.legal_data.http import FetchError, PoliteClient
from docslides.legal_data.records import SectionRecord, SubsectionRecord

LICENSE = "CC BY-SA 4.0"
ATTRIBUTION = ("Hebrew Wikisource contributors, the Open Law Book project (ויקיטקסט, ספר החוקים הפתוח); "
               "licensed CC BY-SA 4.0")

_HEADING_LEVELS = {"ח:קטע1": 1, "ח:קטע2": 2, "ח:קטע3": 3, "ח:קטע4": 4}
_ITEM_LEVELS = {"ח:תת": 1, "ח:תתת": 2, "ח:תתתת": 3, "ח:תתתתת": 4}
_NO_CONTENT = {"ח:התחלה", "ח:סוף", "ח:מפריד"}
_INLINE_TEXT = {"ח:הערה", "ח:מוקטן", "ח:מודגש", "ח:נטוי", "ח:גדול", "ח:קטן"}
_SCHEDULE_HEADING_RE = re.compile(r"^(?:ה)?(?:תוספת|תוספות|נספח|טופס|טפסים|לוח)")
_REPEALED_RE = re.compile(r"\((?:בוטל|בוטלה|בוטלו)\)|(?<![א-ת])(?:הישן|הישנה|הישנות|הישנים)(?![א-ת])")
_LAW_TEMPLATE_RE = re.compile(r"\{\{\s*ח:(?:התחלה|כותרת)")


# --- the dump -----------------------------------------------------------------------------


@dataclass
class DumpFile:
    date: str
    name: str
    url: str
    size: int
    sha1: str | None


def resolve_dump(http: PoliteClient, base: str, date: str = "latest") -> DumpFile:
    base = base.rstrip("/")
    if date == "latest":
        index = http.get(base + "/").text
        dates = sorted(set(re.findall(r'href="(\d{8})/"', index)), reverse=True)
    else:
        dates = [date]
    for candidate in dates[:4]:  # the newest may still be in progress
        response = http.get(f"{base}/{candidate}/dumpstatus.json", ok=(200, 404))
        if response.status_code == 404:
            continue
        job = response.json().get("jobs", {}).get("articlesdump", {})
        if job.get("status") != "done":
            continue
        for name, info in job.get("files", {}).items():
            if name.endswith("-pages-articles.xml.bz2"):
                return DumpFile(candidate, name, urljoin(base + "/", info["url"]), int(info.get("size") or 0),
                                info.get("sha1"))
    raise FetchError(f"no completed pages-articles dump found under {base} (tried {dates[:4]})")


@dataclass
class WikiPage:
    title: str
    ns: int
    page_id: int
    rev_id: int
    timestamp: str
    redirect: bool
    text: str


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(elem: ET.Element, name: str) -> ET.Element | None:
    return next((c for c in elem if _local(c.tag) == name), None)


def _page(elem: ET.Element) -> WikiPage:
    revision = _child(elem, "revision")

    def text_of(parent: ET.Element | None, name: str) -> str:
        node = _child(parent, name) if parent is not None else None
        return (node.text or "") if node is not None else ""

    return WikiPage(
        title=text_of(elem, "title"),
        ns=int(text_of(elem, "ns") or 0),
        page_id=int(text_of(elem, "id") or 0),
        rev_id=int(text_of(revision, "id") or 0),
        timestamp=text_of(revision, "timestamp"),
        redirect=_child(elem, "redirect") is not None,
        text=text_of(revision, "text"),
    )


def iter_pages(chunks: Iterable[bytes]) -> Iterator[WikiPage]:
    """Pages from a .xml.bz2 byte stream (multi-stream bz2 included), parsed incrementally."""
    parser = ET.XMLPullParser(events=("start", "end"))
    decompressor = bz2.BZ2Decompressor()
    root: ET.Element | None = None
    for chunk in chunks:
        data = chunk
        while data:
            parser.feed(decompressor.decompress(data))
            if decompressor.eof:
                data = decompressor.unused_data
                decompressor = bz2.BZ2Decompressor()
            else:
                data = b""
            for event, elem in parser.read_events():
                if event == "start":
                    if root is None:
                        root = elem
                    continue
                if _local(elem.tag) == "page":
                    yield _page(elem)
                    elem.clear()
                    if root is not None:
                        root.clear()  # drop finished pages so memory stays flat


def file_chunks(path, size: int = 1 << 20) -> Iterator[bytes]:
    with open(path, "rb") as f:
        yield from iter(lambda: f.read(size), b"")


def is_law_book_page(page: WikiPage) -> bool:
    return page.ns == 0 and not page.redirect and bool(_LAW_TEMPLATE_RE.search(page.text))


def page_url(wiki_base: str, title: str, rev_id: int | None = None) -> str:
    url = wiki_base + quote(title.replace(" ", "_"))
    return f"{url}?oldid={rev_id}" if rev_id else url


# --- the page -----------------------------------------------------------------------------


@dataclass
class ParsedLaw:
    full_title: str
    registry_id: int | None
    citations: list[dict]
    sections: list[SectionRecord]
    text: str
    unknown_templates: Counter
    repeal_marker: bool
    parse_errors: int

    @property
    def parsed(self) -> bool:
        return any(s.kind == "section" for s in self.sections)


def _template_name(template) -> str:
    name = str(template.name).strip().replace("_", " ")
    return name.split(":", 1)[1].strip() if name.startswith("תבנית:") else name


def _positional(template) -> list:
    return [p.value for p in template.params if not p.showkey]


class _Block:
    def __init__(self, kind: str, number: str, title: str | None, headings: dict[int, str], note: str | None = None):
        self.kind, self.number, self.title, self.note = kind, number, title, note
        self.division, self.chapter = headings.get(1), headings.get(2)
        self.subchapter = " > ".join(h for level, h in sorted(headings.items()) if level >= 3) or None
        self.intro: list[str] = []
        self.subsections: list[SubsectionRecord] = []

    def add_text(self, line: str) -> None:
        if not line:
            return
        if self.subsections:
            self.subsections[-1].text += "\n" + line
        else:
            self.intro.append(line)

    def record(self) -> SectionRecord:
        intro = "\n".join(self.intro).strip()
        text = "\n".join([intro, *(s.text for s in self.subsections)]).strip()
        return SectionRecord(kind=self.kind, number=self.number, title=self.title, division=self.division,
                             chapter=self.chapter, subchapter=self.subchapter, note=self.note, intro=intro,
                             subsections=self.subsections, text=text)


class _PageParser:
    def __init__(self, page_title: str) -> None:
        self.page_title = page_title
        self.full_title = ""
        self.registry_id: int | None = None
        self.citations: list[dict] = []
        self.unknown: Counter = Counter()
        self.categories: list[str] = []
        self.headings: dict[int, str] = {}
        self.items: list[tuple[str, object]] = []  # ("heading", str) | ("block", _Block), in page order
        self.current: _Block | None = None
        self.heading_since_block = False
        self.last_heading: tuple[str, str] | None = None  # (anchor, heading)
        self.in_box = False
        self.in_toc = False
        self.box_text: list[str] = []
        self.parse_errors = 0

    # inline rendering ----------------------------------------------------------------------

    def render(self, code) -> str:
        import mwparserfromhell.nodes as n

        if code is None:
            return ""
        out: list[str] = []
        for node in code.nodes:
            if isinstance(node, n.Text):
                out.append(str(node))
            elif isinstance(node, n.Template):
                out.append(self.render_template(node))
            elif isinstance(node, n.Wikilink):
                title = str(node.title).strip()
                if title.startswith(("קטגוריה:", "Category:")):
                    self.categories.append(title.split(":", 1)[1])
                elif not title.startswith(("קובץ:", "File:", "תמונה:")):
                    out.append(self.render(node.text) if node.text else title.split("#")[0])
            elif isinstance(node, n.ExternalLink):  # a bare URL shows as itself, [url text] as its text
                out.append(self.render(node.title) if node.title else str(node.url))
            elif isinstance(node, n.Tag):
                tag = str(node.tag).strip().lower()
                if tag in ("ref", "references"):
                    continue
                if tag == "br":
                    out.append("\n")
                elif node.contents is not None:
                    out.append(self.render(node.contents))
            elif isinstance(node, n.HTMLEntity):
                out.append(node.normalize())
            elif isinstance(node, n.Comment):
                continue
            elif isinstance(node, n.Heading):
                out.append(self.render(node.title))
            else:
                out.append(str(node))
        return "".join(out)

    def render_template(self, template) -> str:
        name = _template_name(template)
        params = _positional(template)

        def arg(i: int) -> str:
            return self.render(params[i]).strip() if len(params) > i else ""

        if name in ("ח:פנימי", "ח:חיצוני"):
            return arg(1) or arg(0)
        if name == "ח:מאגר":
            digits = re.search(r"\d+", arg(0))
            if digits and self.registry_id is None:
                self.registry_id = int(digits.group(0))
            return ""
        if name == "ח:תיבה":
            self.citations.append({"ref": arg(0), "name": arg(1), "url": arg(2)})
            return arg(0)
        if name in _INLINE_TEXT:
            return arg(0)
        self.unknown[name] += 1
        return arg(0) if len(params) == 1 else " ".join(a for a in (arg(i) for i in range(len(params))) if a)

    # structure -------------------------------------------------------------------------------

    def _new_block(self, kind: str, number: str, title: str | None, note: str | None = None) -> _Block:
        block = _Block(kind, number, title, self.headings, note)
        self.items.append(("block", block))
        self.current = block
        self.heading_since_block = False
        return block

    def _block_for_text(self) -> _Block | None:
        """Where free text goes: the open section; text under a heading outside any numbered
        section (schedules, forms, tables) gets a "schedule" block named by that heading; text
        before anything else is the preamble."""
        if self.heading_since_block or self.current is None:
            if self.last_heading and (self.heading_since_block or self.current is not None):
                anchor, heading = self.last_heading
                return self._new_block("schedule", anchor or heading, heading)
            if not any(kind == "block" for kind, _ in self.items):
                return self._new_block("preamble", "preamble", None)
        return self.current

    def text(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        if self.in_box:
            self.box_text.append(line)
            return
        block = self._block_for_text()
        if block is not None:
            block.add_text(line)

    def line(self, raw: str) -> None:
        import mwparserfromhell
        import mwparserfromhell.nodes as n

        stripped = raw.strip()
        if not stripped or stripped.startswith("__"):
            return
        if re.match(r"</?div\b", stripped):
            if self.in_toc or "law-toc" in stripped:
                return
            stripped = re.sub(r"</?div\b[^>]*>", "", stripped).strip()
            if not stripped:
                return
        try:
            nodes = list(mwparserfromhell.parse(stripped).nodes)
        except Exception:  # noqa: BLE001 -- a line the parser chokes on is kept as plain text
            self.parse_errors += 1
            self.text(re.sub(r"\{\{[^{}]*\}\}", "", stripped))
            return
        first = next((i for i, node in enumerate(nodes) if not (isinstance(node, n.Text) and not str(node).strip())), None)
        if first is None:
            return
        head = nodes[first]
        rest = mwparserfromhell.wikicode.Wikicode(nodes[first + 1 :])
        name = _template_name(head) if isinstance(head, n.Template) else ""
        params = _positional(head) if name else []

        def arg(i: int) -> str:
            return self.render(params[i]).strip() if len(params) > i else ""

        if self.in_toc and name not in _HEADING_LEVELS and name != "ח:סעיף":
            return
        if name == "ח:כותרת":
            self.full_title = arg(0)
        elif name == "ח:פתיח-התחלה":
            self.in_box = True
            self.text(self.render(rest))
        elif name == "ח:סוגר":
            self.in_box = False
            self.text(self.render(rest))
        elif name in _NO_CONTENT:
            self.text(self.render(rest))
        elif name == "ח:מבוא":
            self._new_block("preamble", "preamble", None)
            self.text(self.render(rest))
        elif name in _HEADING_LEVELS:
            anchor, heading = arg(0), arg(1) or arg(0)
            if "תוכן עניינים" in heading:
                self.in_toc = True
                return
            self.in_toc = False
            level = _HEADING_LEVELS[name]
            self.headings = {k: v for k, v in self.headings.items() if k < level}
            self.headings[level] = heading
            self.last_heading = (anchor, heading)
            self.heading_since_block = True
            self.items.append(("heading", heading))
        elif name == "ח:סעיף":
            self.in_toc = False
            notes = " ".join(a for a in (arg(i) for i in range(2, len(params))) if a) or None
            self._new_block("section", _clean_number(arg(0)), arg(1) or None, notes)
            self.text(self.render(rest))
        elif name == "ח:ת":
            self.text(self.render(rest))
        elif name in _ITEM_LEVELS:
            label, body = arg(0), self.render(rest).strip()
            line = f"{label} {body}".strip()
            block = self._block_for_text()
            if block is None or self.in_box:
                self.text(line)
            elif _ITEM_LEVELS[name] == 1:
                block.subsections.append(SubsectionRecord(label=_clean_label(label), text=line))
            else:
                block.add_text(line)
        elif name == "ח:חתימות":
            return  # signatories: not law text (schedules after them still are)
        else:
            self.text(self.render(mwparserfromhell.wikicode.Wikicode(nodes[first:])))

    def table(self, block: str) -> None:
        import mwparserfromhell

        rows: list[str] = []
        cells: list[str] = []
        for raw in block.splitlines():
            line = raw.strip()
            if line.startswith(("{|", "|}", "|+")):
                continue
            if line.startswith("|-"):
                if cells:
                    rows.append(" | ".join(cells))
                cells = []
                continue
            if line.startswith(("|", "!")):
                for cell in re.split(r"\|\||!!", line[1:]):
                    value = cell.split("|", 1)[-1] if "=" in cell.split("|", 1)[0] and "|" in cell else cell
                    cells.append(self.render(mwparserfromhell.parse(value)).strip())
            elif cells:
                cells[-1] += " " + self.render(mwparserfromhell.parse(line)).strip()
        if cells:
            rows.append(" | ".join(cells))
        for row in rows:
            self.text(row)

    def result(self) -> ParsedLaw:
        sections = [block.record() for kind, block in self.items if kind == "block"]
        title = self.full_title or self.page_title
        lines = [title, ""]
        for kind, item in self.items:
            if kind == "heading":
                lines += ["", item]
            else:
                record = item.record()
                if record.kind == "section":
                    lines.append(f"{record.number}. {record.title}" if record.title else f"{record.number}.")
                lines.append(record.text)  # a schedule's heading line is already in place
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        repealed = bool(_REPEALED_RE.search(self.page_title) or _REPEALED_RE.search(self.full_title)
                        or any("בוטל" in c for c in self.categories)
                        or any(re.search(r"\((?:בוטל|בוטלה|בוטלו)\b", t) for t in self.box_text))
        return ParsedLaw(title, self.registry_id, self.citations, sections, text, self.unknown, repealed,
                         self.parse_errors)


def _clean_number(value: str) -> str:
    value = re.sub(r"^סעיף\s+", "", value.strip()).rstrip(".").strip()
    return re.sub(r"\s+", "", value.replace("״", '"').replace("׳", "'"))


def _clean_label(value: str) -> str:
    return value.strip().strip("().").strip() or value.strip()


def _logical_lines(wikitext: str) -> Iterator[tuple[str, str]]:
    """("line", text) with multi-line templates joined, or ("table", {| ... |} block)."""
    lines = wikitext.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("{|"):
            block = [line]
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("|}"):
                block.append(lines[i])
                i += 1
            if i < len(lines):
                block.append(lines[i])
                i += 1
            yield "table", "\n".join(block)
            continue
        buffer = line
        i += 1
        while buffer.count("{{") > buffer.count("}}") and i < len(lines):
            buffer += "\n" + lines[i]
            i += 1
        yield "line", buffer


def parse_law_page(wikitext: str, page_title: str) -> ParsedLaw:
    parser = _PageParser(page_title)
    for kind, block in _logical_lines(wikitext):
        if kind == "table":
            parser.table(block)
        else:
            parser.line(block)
    return parser.result()
