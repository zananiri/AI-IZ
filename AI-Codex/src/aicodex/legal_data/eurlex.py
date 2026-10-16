"""EU law in Spanish, from the Publications Office's CELLAR repository (the store behind EUR-Lex).

EU regulations apply directly in Spain and directives through Spanish transposition laws; either
way a Spanish lawyer needs their text. Acts are listed by CELEX number in config
(legal_data.eu_law.celex); for each one:

  1. SPARQL (publications.europa.eu/webapi/rdf/sparql) finds the newest consolidated version
     ("02016R0679-20160504"), when the act has been consolidated;
  2. content negotiation on http://publications.europa.eu/resource/celex/<CELEX> with
     Accept: application/xhtml+xml and Accept-Language: spa returns the Spanish XHTML.

That is the Publications Office's documented machine access -- EUR-Lex's own HTML pages sit behind
a bot challenge and are not used. Articles are parsed from the ELI-marked XHTML of both formats
(Official Journal "oj-ti-art" and consolidated "title-article-norm").

License: © European Union, https://eur-lex.europa.eu -- reuse authorised with acknowledgement
(Commission Decision 2011/833/EU). Only the Official Journal text is authentic; consolidated
versions are documentation.
"""

from __future__ import annotations

import re
from collections import Counter
from urllib.parse import quote

from aicodex.legal_data.http import FetchError
from aicodex.legal_data.progress import Progress
from aicodex.legal_data.records import CorpusRecord, JsonlWriter, Quality, SectionRecord, now_iso
from aicodex.legal_data.spanish import repair_text

SOURCE = "EUR-Lex / CELLAR (Publications Office of the EU)"
LICENSE = "© Unión Europea, https://eur-lex.europa.eu -- reutilización autorizada (Decisión 2011/833/UE)"
ATTRIBUTION = ("Fuente: EUR-Lex, Oficina de Publicaciones de la Unión Europea. Solo los textos publicados en el "
               "Diario Oficial de la Unión Europea son auténticos.")

_ART = "\ue000ART "
_STI = "\ue000STI "
_H1 = "\ue000H1 "
_H2 = "\ue000H2 "
_ART_CLASSES = {"title-article-norm", "oj-ti-art", "ti-art"}
_STI_CLASSES = {"stitle-article-norm", "oj-sti-art", "sti-art"}
_H1_CLASSES = {"title-division-1", "oj-ti-section-1", "ti-section-1"}
_H2_CLASSES = {"title-division-2", "oj-ti-section-2", "ti-section-2"}
_DROP_CLASSES = {"modref", "arrow", "oj-note-tag", "reference", "disclaimer", "hd-modifiers"}
_MARKER_LINE_RE = re.compile(r"^(?:\(?\d{1,3}[.)]|\(?[a-zñ]{1,3}\)|[ivxlc]{1,6}\)|—|-|•)$")
_ARTICLE_NUM_RE = re.compile(r"Art[íi]culo\s+(?P<num>\d{1,4}(?:\s*(?:bis|ter|quater))?|[úu]nico)", re.IGNORECASE)
_ANNEX_RE = re.compile(r"^ANEXO(?:\s+[IVXLC\d]+)?\b")

_KINDS = {"R": ("eu_regulation", "Reglamento"), "L": ("eu_directive", "Directiva"), "D": ("eu_decision", "Decisión"),
          "P": ("eu_treaty", "Carta de los Derechos Fundamentales"), "M": ("eu_treaty", "Tratado de la Unión Europea"),
          "E": ("eu_treaty", "Tratado de Funcionamiento de la Unión Europea")}

SPARQL_CONSOLIDATED = """PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
SELECT ?celex WHERE {{ ?w cdm:resource_legal_id_celex ?celex .
  FILTER(STRSTARTS(STR(?celex), "{prefix}-")) }}"""


def consolidated_prefix(celex: str) -> str | None:
    """'32016R0679' -> '02016R0679' (consolidated versions of sector-3 acts)."""
    return "0" + celex[1:] if re.fullmatch(r"3\d{4}[RLD]\d{4}", celex) else None


def latest_consolidated(client, sparql_url: str, celex: str) -> str | None:
    prefix = consolidated_prefix(celex)
    if not prefix:
        return None
    from urllib.parse import urlencode

    url = f"{sparql_url}?{urlencode({'query': SPARQL_CONSOLIDATED.format(prefix=prefix)})}"
    data = client.get_json(url, headers={"Accept": "application/sparql-results+json"})
    versions = [b["celex"]["value"] for b in data.get("results", {}).get("bindings", [])]
    return max(versions, key=lambda v: v.rsplit("-", 1)[-1]) if versions else None


def _mark(root) -> None:
    for element in list(root.iter()):
        if not isinstance(element.tag, str):
            continue
        classes = set((element.get("class") or "").split())
        if classes & _DROP_CLASSES:
            element.drop_tree()
            continue
        for marker, names in ((_ART, _ART_CLASSES), (_STI, _STI_CLASSES), (_H1, _H1_CLASSES), (_H2, _H2_CLASSES)):
            if classes & names:
                element.text = marker + (element.text or "")
                break


def _lines(root) -> list[str]:
    from aicodex.legal_data.tc import _text_of

    raw = [line for line in _text_of(root).splitlines() if line.strip()]
    lines: list[str] = []
    for line in raw:
        if lines and _MARKER_LINE_RE.match(lines[-1].strip()):  # "1." / "a)" on its own line: join it
            lines[-1] = f"{lines[-1].strip()} {line.strip()}"
        else:
            lines.append(line.strip())
    return lines


def parse_xhtml(html: bytes) -> tuple[str, list[SectionRecord]]:
    """(title, sections) of an EU act's Spanish XHTML."""
    import lxml.html

    root = lxml.html.fromstring(html, parser=lxml.html.HTMLParser(encoding="utf-8"))
    body = root.find(".//body")
    _mark(body if body is not None else root)
    lines = _lines(body if body is not None else root)
    title_lines: list[str] = []
    for element in root.iter():
        if isinstance(element.tag, str) and set((element.get("class") or "").split()) & {"title-doc-first", "oj-doc-ti"}:
            text = re.sub(r"\s+", " ", element.text_content()).strip()
            if text and text not in title_lines:
                title_lines.append(text)
            if len(title_lines) >= 3:
                break
    title = " ".join(title_lines[:3])

    sections: list[SectionRecord] = []
    preamble: list[str] = []
    heads: dict[str, str | None] = {"chapter": None, "section": None}
    last_level = "chapter"
    expect = None  # a heading marker whose text came out on the next line

    def heading(text: str, numeral: bool) -> None:
        nonlocal last_level
        if numeral:  # "CAPÍTULO III" / "TÍTULO II" / "Sección 3": opens a level
            last_level = "section" if re.match(r"(?i)secci[óo]n", text) else "chapter"
            if last_level == "chapter":
                heads["section"] = None
            heads[last_level] = text
        else:  # the level's name, on its own line
            heads[last_level] = f"{heads[last_level]}. {text}" if heads[last_level] else text

    current: SectionRecord | None = None
    annex: SectionRecord | None = None
    for line in lines:
        if line.startswith(_H1.strip()) or line.startswith(_H2.strip()):
            numeral = line.startswith(_H1.strip())
            text = line[len(_H1):].strip()
            if text:
                heading(text, numeral)
                expect = None
            else:
                expect = "h1" if numeral else "h2"
            continue
        if expect and not line.startswith("\ue000"):
            heading(line, expect == "h1")
            expect = None
            continue
        if line.startswith(_ART.strip()):
            match = _ARTICLE_NUM_RE.search(line)
            number = re.sub(r"\s+", " ", match.group("num")).lower() if match else line[len(_ART):].strip()
            current = SectionRecord(kind="section", number=number, chapter=heads["chapter"],
                                    subchapter=heads["section"], text="")
            sections.append(current)
            annex = None
            continue
        if line.startswith(_STI.strip()):
            if current is not None and not current.title:
                current.title = line[len(_STI):].strip()
            continue
        if _ANNEX_RE.match(line) and len(line) < 120:
            annex = SectionRecord(kind="schedule", number=line[:200], text="")
            sections.append(annex)
            current = None
            continue
        target = annex or current
        if target is None:
            preamble.append(line)
        else:
            target.text = f"{target.text}\n{line}".strip()
    if preamble:
        text = "\n".join(preamble)
        sections.insert(0, SectionRecord(kind="preamble", number="preamble", text=text, intro=text))
    from aicodex.legal_data.boe import _split_apartados

    for section in sections:
        if section.kind == "section":
            section.intro, section.subsections = _split_apartados(section.text.splitlines())
            section.text = section.text or "(sin contenido)"
    return title, [s for s in sections if s.text]


def build_record(celex: str, version: str, url: str, title: str, sections: list[SectionRecord]) -> CorpusRecord:
    from aicodex.legal_data.boe import render_text

    kind = celex[5] if len(celex) > 5 and celex[0] == "3" else (celex[5] if celex[0] == "1" and len(celex) > 5 else "")
    level, rank = _KINDS.get(kind, ("eu_decision", "Acto de la UE"))
    text, encoding, notes = repair_text(render_text(title, sections))
    date = re.search(r"-(\d{4})(\d{2})(\d{2})$", version)
    number = re.search(r"\b(\d{4}/\d{1,4})\b", title)
    record = CorpusRecord(
        id=f"eurlex:{celex.replace('/', '_')}",
        source=SOURCE,
        source_url=f"https://eur-lex.europa.eu/legal-content/ES/TXT/?uri=CELEX:{version}",
        license=LICENSE,
        attribution=ATTRIBUTION,
        category="eu_law",
        title=title or (f"{rank} (versión consolidada)" if level == "eu_treaty" else celex),
        authority_level=level,
        status="in_force",
        status_source="configured_list",
        effective_date=f"{date.group(1)}-{date.group(2)}-{date.group(3)}" if date else None,
        retrieved_at=now_iso(),
        language="es",
        text=text,
        quality=Quality(encoding=encoding, notes=notes + ([f"version {version}"] if version != celex else
                                                          ["original Official Journal text (no consolidated version)"])),
        law_id=celex,
        rank=rank,
        official_number=number.group(1) if number else None,
        scope="ue",
        journal="Diario Oficial de la Unión Europea",
        latest_amendment_date=f"{date.group(1)}-{date.group(2)}-{date.group(3)}" if date else None,
        source_updated_at=version,
        consolidated=version != celex,
        sections=sections,
    )
    return record.finalize()


def run_eurlex(ctx) -> None:
    cfg, args = ctx.cfg, ctx.args
    sources = cfg.sources
    acts = args.celex or list(cfg.eu_law.celex)
    if args.sample:
        acts = acts[: args.sample]
    ctx.manifest.source(SOURCE, sources.eurlex_cellar, LICENSE, ATTRIBUTION, acts=acts)
    ctx.log(f"EU law: {len(acts)} acts")
    if ctx.dry_run:
        ctx.manifest.set("estimates", "eu_law", {"acts": len(acts), "requests": len(acts) * 2,
                                                 "approx_mb": round(len(acts) * 0.6, 1)})
        return
    writer = JsonlWriter(ctx.root / "eu_law", "eu_law")
    progress = Progress("EU law", total=len(acts), unit="acts", log=ctx.log)
    counts: Counter = Counter()
    try:
        for celex in acts:
            version = celex
            if cfg.eu_law.prefer_consolidated:
                try:
                    version = latest_consolidated(ctx.client, sources.eurlex_sparql, celex) or celex
                except FetchError as exc:
                    ctx.manifest.warn(f"{celex}: consolidated-version lookup failed ({exc}); using the OJ text")
            url = f"{sources.eurlex_cellar.rstrip('/')}/{quote(version, safe='')}"
            try:
                html = ctx.client.get(url, headers={"Accept": "application/xhtml+xml", "Accept-Language": "spa"}).content
            except FetchError as exc:
                ctx.manifest.warn(f"{celex}: {exc}")
                counts["failed"] += 1
                continue
            title, sections = parse_xhtml(html)
            if not sections:
                ctx.manifest.warn(f"{celex}: no text parsed from {url}")
                counts["empty"] += 1
                continue
            writer.write(build_record(celex, version, url, title, sections))
            counts["consolidated" if version != celex else "original"] += 1
            progress.update(1, **counts)
    except BaseException:
        writer.close(commit=False)
        raise
    progress.done(**counts)
    writer.close()
    ctx.manifest.count("records_written", writer.count)
    ctx.log(f"eu_law: {writer.count} records ({dict(counts)})")
