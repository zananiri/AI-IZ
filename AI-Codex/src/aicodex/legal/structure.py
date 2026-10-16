"""Structural parsing of Spanish statute / regulation / ruling text into articles (artículos),
for legal/chunking.py.

Plain-text line heuristics tuned to how BOE / DOUE / autonomous-bulletin texts read once
extracted to text:

  * Headings: LIBRO / TÍTULO / CAPÍTULO / Sección / Subsección followed by an ordinal label
    ("TÍTULO PRELIMINAR", "CAPÍTULO II", "Sección 1.ª", "Sección primera"). Their name is often on
    the next line ("TÍTULO I" / "De los derechos y deberes fundamentales") and is joined to it.
    Libro and Título make the `division`, Capítulo the `chapter`, Sección/Subsección the
    `subchapter`.
  * Articles: a line starting "Artículo 14." / "Artículo 14 bis." / "Art. 14." / "Artículo único.",
    with the article's title and often its first words on the same line ("Artículo 14. Igualdad
    ante la ley. Los españoles son iguales..."). Dispositions ("Disposición adicional primera.",
    "Disposición transitoria segunda.", "Disposición derogatoria única.", "Disposición final
    tercera.") are sections of their own, numbered by their full name.
  * Article numbering must advance (a quoted article in an amending provision never opens one):
    text between « and » -- the wording an amending law inserts into another law -- is body text.
  * Apartados: "1." / "2." at line start are the top-level subsections; "a)" / "b)" letters are
    top-level only in an article without numbered apartados, otherwise they stay inside theirs.

Rulings use numbered paragraphs ("1.", "2.") and Antecedentes / Fundamentos jurídicos / Fallo
headings, which parse the same way. Always preview a new source with
`scripts/ingest_legal.py stage --dry-run` before trusting its structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_ORDINAL_WORDS = (
    "primer[oa]?|segund[oa]|tercer[oa]?|cuart[oa]|quint[oa]|sext[oa]|s[ée]ptim[oa]|octav[oa]|noven[oa]|d[ée]cim[oa]"
    "|und[ée]cim[oa]|duod[ée]cim[oa]|decimo\\w+|vig[ée]sim[oa]\\w*|[úu]nic[oa]|preliminar"
)
_LABEL = rf"(?:[IVXLC]{{1,7}}|\d{{1,3}}(?:\.?\s?[ªºa])?|{_ORDINAL_WORDS})"
_HEADING_RE = re.compile(
    rf"^\s*(?P<kind>LIBRO|Libro|T[ÍI]TULO|T[íi]tulo|CAP[ÍI]TULO|Cap[íi]tulo|SECCI[ÓO]N|Secci[óo]n|SUBSECCI[ÓO]N|Subsecci[óo]n)"
    rf"\s+(?P<label>{_LABEL})\b\.?\s*(?P<title>.{{0,160}})$",
    re.IGNORECASE,
)
_SUFFIXES = ("bis", "ter", "quater", "quinquies", "sexies", "septies", "octies", "nonies", "novies", "decies")
_SUFFIX_RE = "|".join(_SUFFIXES)
_ARTICLE_RE = re.compile(
    rf"^\s*(?:Art[íi]culo|ART[ÍI]CULO|Art\.)\s+(?P<num>\d{{1,4}}(?:\s+(?:{_SUFFIX_RE}))?|[úu]nico)\s*\.\s*(?P<rest>.*)$"
)
_DISPOSITION_RE = re.compile(
    r"^\s*(?P<num>Disposici[óo]n\s+(?:adicional|transitoria|derogatoria|final)\s+(?:[úu]nica|\w+))\s*\.\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
_APARTADO_RE = re.compile(r"^\s*(?P<label>\d{1,3})\.\s+(?P<rest>\S.*)$")
_LETTER_RE = re.compile(r"^\s*(?P<label>[a-zñ]{1,2})\)\s*(?P<rest>.*)$")

# References to articles: "artículo 5", "artículos 3 y 4", "art. 14.2", "arts. 3 a 5", "artículo 14 bis".
_NUM = rf"\d{{1,4}}(?:\s+(?:{_SUFFIX_RE}))?(?:\.\d{{1,3}})?(?:\.?[a-zñ]\))?"
_XREF_RE = re.compile(
    rf"\b(?:[Aa]rt[íi]culos?|[Aa]rts?\.)\s+(?P<list>{_NUM}(?:\s*(?:,|y|e|o|a|al|hasta)\s*{_NUM})*)"
)
_XREF_EXTERNAL_RE = re.compile(
    r"^\s*(?:,\s*)?(?:apartado\s+\d+\s*,?\s*)?(?:de|del)\s+(?:la\s+|el\s+)?(?!esta\b|este\b|la presente\b|el presente\b)"
    r"(?:Ley|Código|Real|Decreto|Constitución|Reglamento|Directiva|Estatuto|texto|Orden|Tratado|Carta|Convenio)"
    r"|^\s*(?:de\s+la\s+)?(?:CE|CC|CP|LEC|LECrim|LOPJ|LJCA|LRJS|ET|LGSS|LPAC|LRJSP|LAU|LSC|LGT|RGPD|TFUE|TUE)\b"
)
_XREF_RANGE_RE = re.compile(r"(\d{1,4})\s*(?:a|al|hasta)\s*(\d{1,4})")

# Section titles marked by legal/pdf_text.py (margin notes): ⟦title⟧
_MARGIN_TITLE_RE = re.compile(r"^\s*⟦(.+)⟧\s*$")
_QUOTED = "\x00"  # internal prefix on lines inside a quoted («...») amendment block

_MAX_FIRST_SECTION = 10
_MAX_SECTION_JUMP = 60


@dataclass
class Subsection:
    label: str
    text: str


@dataclass
class Section:
    number: str  # "12", "14 bis", "único", "Disposición adicional primera", or "preamble"
    title: str | None
    text: str  # full article body, apartados included
    division: str | None = None  # Libro / Título
    chapter: str | None = None  # Capítulo
    subchapter: str | None = None  # Sección / Subsección
    intro: str = ""  # body text before the first top-level apartado
    subsections: list[Subsection] = field(default_factory=list)
    cross_refs: list[str] = field(default_factory=list)  # other article numbers in this law


def is_article_number(number: str) -> bool:
    return bool(re.fullmatch(rf"\d{{1,4}}(?:\s+(?:{_SUFFIX_RE}))?|[úu]nico", number or ""))


def _section_key(number: str) -> tuple[int, int]:
    """Ordering of article numbers: 14 < 14 bis < 14 ter < 15."""
    match = re.match(rf"(\d+)(?:\s+({_SUFFIX_RE}))?", number or "")
    if not match:
        return (0, 0)
    suffix = _SUFFIXES.index(match.group(2)) + 1 if match.group(2) else 0
    return (int(match.group(1)), suffix)


def _quote_balance(line: str) -> int:
    return line.count("«") - line.count("»") + line.count("“") - line.count("”")


def _split_subsections(body_lines: list[str]) -> tuple[str, list[Subsection]]:
    """An article's intro and top-level apartados. A marker opens a new apartado only when it is
    the next one in sequence, so an inner numbered list stays with the apartado it belongs to, and
    quoted text («...») never opens one."""
    plain = [line.lstrip(_QUOTED) for line in body_lines if not line.startswith(_QUOTED)]
    numbered = any(_APARTADO_RE.match(line) for line in plain)
    pattern = _APARTADO_RE if numbered else _LETTER_RE
    intro: list[str] = []
    subsections: list[Subsection] = []

    def follows(label: str) -> bool:
        if not subsections:
            return label in ("1", "a")
        previous = subsections[-1].label
        if previous.isdigit() and label.isdigit():
            return int(label) == int(previous) + 1
        return len(label) == 1 == len(previous) and ord(label) == ord(previous) + 1

    for raw in body_lines:
        quoted = raw.startswith(_QUOTED)
        line = raw.lstrip(_QUOTED).strip()
        if not line:
            continue
        match = None if quoted else pattern.match(line)
        if match and follows(match.group("label")):
            subsections.append(Subsection(label=match.group("label"), text=line))
        elif subsections:
            subsections[-1].text += "\n" + line
        else:
            intro.append(line)
    return "\n".join(intro).strip(), subsections


def extract_cross_references(text: str, own_number: str | None = None) -> list[str]:
    """Article numbers of THIS law referenced from `text`, in first-seen order. References followed
    by another law's name ("artículo 1902 del Código Civil", "art. 24 CE") are external and skipped
    -- they can't be resolved to a chunk of this document. "Artículo 14.2" refers to article 14."""
    refs: list[str] = []
    for match in _XREF_RE.finditer(text):
        if _XREF_EXTERNAL_RE.match(text[match.end() : match.end() + 60]):
            continue
        listing = match.group("list")
        numbers = [re.sub(r"\s+", " ", m.group(1)) for m in
                   re.finditer(rf"(\d{{1,4}}(?:\s+(?:{_SUFFIX_RE}))?)(?:\.\d{{1,3}})?(?:\.?[a-zñ]\))?", listing)]
        for start, end in _XREF_RANGE_RE.findall(listing):
            lo, hi = int(start), int(end)
            if 0 < hi - lo <= 20:
                numbers.extend(str(n) for n in range(lo + 1, hi))
        for number in numbers:
            if number != own_number and number not in refs:
                refs.append(number)
    return refs


class _Hierarchy:
    def __init__(self) -> None:
        self.libro = self.titulo = self.capitulo = self.seccion = self.subseccion = None

    def set(self, kind: str, label: str) -> None:
        kind = kind.lower().replace("í", "i").replace("ó", "o")
        if kind == "libro":
            self.libro, self.titulo, self.capitulo, self.seccion, self.subseccion = label, None, None, None, None
        elif kind == "titulo":
            self.titulo, self.capitulo, self.seccion, self.subseccion = label, None, None, None
        elif kind == "capitulo":
            self.capitulo, self.seccion, self.subseccion = label, None, None
        elif kind == "seccion":
            self.seccion, self.subseccion = label, None
        else:
            self.subseccion = label

    def append_title(self, title: str) -> None:
        for name in ("subseccion", "seccion", "capitulo", "titulo", "libro"):
            value = getattr(self, name)
            if value:
                setattr(self, name, f"{value}. {title}")
                return

    @property
    def division(self) -> str | None:
        return " > ".join(p for p in (self.libro, self.titulo) if p) or None

    @property
    def subchapter(self) -> str | None:
        return " > ".join(p for p in (self.seccion, self.subseccion) if p) or None


def _split_title(rest: str) -> tuple[str | None, str]:
    """'Igualdad ante la ley. Los españoles son iguales...' -> (title, body). A rest without a
    short first sentence is all body."""
    rest = rest.strip()
    if not rest:
        return None, ""
    match = re.match(r"^(?P<title>[^.;:]{2,160}?)\.\s+(?P<body>\S.*)$", rest)
    if match and len(match.group("title").split()) <= 18 and not _APARTADO_RE.match(rest):
        return match.group("title").strip(), match.group("body").strip()
    if rest.endswith(".") and len(rest.split()) <= 14 and rest.count(".") == 1:
        return rest.rstrip(".").strip(), ""
    return None, rest


def parse_sections(text: str) -> list[Section]:
    hierarchy = _Hierarchy()
    sections: list[Section] = []
    preamble: list[str] = []
    current: Section | None = None
    current_lines: list[str] = []
    last_key: tuple[int, int] | None = None
    awaiting_heading_title = False
    quote_depth = 0
    pending_title: str | None = None

    def close_current() -> None:
        if current is None:
            return
        while current_lines and not current_lines[-1].strip(_QUOTED).strip():
            current_lines.pop()
        current.intro, current.subsections = _split_subsections(current_lines)
        current.text = "\n".join(l.lstrip(_QUOTED).strip() for l in current_lines if l.strip(_QUOTED).strip())

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        target = current_lines if current is not None else preamble
        if quote_depth > 0:
            target.append(_QUOTED + line)
            quote_depth = max(quote_depth + _quote_balance(line), 0)
            continue

        margin = _MARGIN_TITLE_RE.match(line)
        if margin:
            pending_title = margin.group(1).strip()
            continue

        heading = _HEADING_RE.match(line)
        if heading and len(line.split()) <= 24 and not line.rstrip().endswith((",", ";")):
            close_current()
            current, current_lines = None, []
            label = f"{heading.group('kind').upper()} {heading.group('label')}"
            title = heading.group("title").strip().rstrip(".")
            hierarchy.set(heading.group("kind"), f"{label}. {title}" if title else label)
            awaiting_heading_title = not title
            continue
        if awaiting_heading_title:
            awaiting_heading_title = False
            if not _ARTICLE_RE.match(line) and not _DISPOSITION_RE.match(line) and len(line.split()) <= 20:
                hierarchy.append_title(line.strip().rstrip("."))
                continue

        article = _ARTICLE_RE.match(line)
        disposition = None if article else _DISPOSITION_RE.match(line)
        if article:
            number = re.sub(r"\s+", " ", article.group("num")).lower()
            key = _section_key(number)
            if number in ("único", "unico"):
                advances = True
            elif last_key is None:
                advances = key[0] <= _MAX_FIRST_SECTION
            else:
                advances = key > last_key and key[0] - last_key[0] <= _MAX_SECTION_JUMP
            if not advances:
                article = None
        if article or disposition:
            match = article or disposition
            close_current()
            title, body = _split_title(match.group("rest"))
            if pending_title and not title:
                title = pending_title
            pending_title = None
            number = (re.sub(r"\s+", " ", match.group("num")).lower() if article else
                      re.sub(r"\s+", " ", match.group("num")).capitalize())
            current = Section(
                number=number,
                title=title,
                text="",
                division=hierarchy.division if article else None,
                chapter=hierarchy.capitulo if article else None,
                subchapter=hierarchy.subchapter if article else None,
            )
            current_lines = [body] if body else []
            sections.append(current)
            if article:
                last_key = _section_key(number)
            quote_depth = max(_quote_balance(body), 0)
            continue

        if pending_title:
            target.append(pending_title)
            pending_title = None
        target.append(line)
        quote_depth = max(_quote_balance(line), 0)

    close_current()

    preamble_text = "\n".join(l.lstrip(_QUOTED).strip() for l in preamble if l.strip(_QUOTED).strip())
    if preamble_text:
        sections.insert(0, Section(number="preamble", title=None, text=preamble_text, intro=preamble_text))

    known = {s.number for s in sections}
    for section in sections:
        if _amends_other_law(section):
            continue  # an amending article's "artículo N" means the amended law's article N
        section.cross_refs = [ref for ref in extract_cross_references(section.text, section.number) if ref in known]
    return sections


_AMENDING_RE = re.compile(r"(?i)^\s*(?:se\s+modifica|se\s+añade|se\s+a[ñn]aden|se\s+suprime|se\s+deroga|"
                          r"se\s+introduce|modificaci[óo]n\s+de)")


def _amends_other_law(section: Section) -> bool:
    return bool(_AMENDING_RE.match(section.title or "") or _AMENDING_RE.match(section.text[:200]))
