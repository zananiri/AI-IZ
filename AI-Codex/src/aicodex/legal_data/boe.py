"""Spanish legislation from the BOE's open-data API (Agencia Estatal Boletín Oficial del Estado):
the consolidated texts ("legislación consolidada") of state and autonomous-community law, in force
and repealed, article by article and always in their current wording.

    GET {api}/legislacion-consolidada?offset=N&limit=M      catalogue (JSON; at most 10,000 per page)
    GET {api}/legislacion-consolidada/id/{id}/texto         full text (XML: <bloque> per article,
                                                             heading or disposition, each with its
                                                             <version>s; the last is the one in force)
    GET {api}/legislacion-consolidada/id/{id}/analisis      subjects (materias), with --with-subjects

This is the BOE's own machine interface for exactly this use, so it is preferred to scraping the
HTML pages: structured, versioned, and incremental (every catalogue entry has fecha_actualizacion;
a law whose stamp hasn't changed is never requested again -- its normalized record is cached under
<output>/boe/cache/).

Categories (classify()):
  laws           state rank-of-law norms: Constitución, Ley Orgánica, Ley, Real Decreto Legislativo,
                 Real Decreto-ley (and pre-constitutional Decreto Legislativo / Decreto-ley),
                 international agreements (Acuerdo Internacional)
  regulations    state regulations: Real Decreto, Decreto, Orden, Resolución, Circular, Instrucción ...
  regional_laws  everything with ámbito "Autonómico" (laws, decree-laws and regulations of the
                 autonomous communities), the community in `region`

Documents the BOE has withdrawn from its pages (its robots.txt lists them, e.g. after a right-to-be-
forgotten request) are skipped and counted, never fetched.

License: reuse of BOE content is authorised with attribution (Ley 37/2007 on the reuse of public
sector information, and the BOE's aviso legal). Consolidated texts are informative: only the
original publication in the BOE has official legal value -- said in every record's attribution.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from aicodex.legal_data.progress import Progress
from aicodex.legal_data.records import (
    CorpusRecord,
    JsonlWriter,
    Quality,
    SectionRecord,
    SubsectionRecord,
    now_iso,
)
from aicodex.legal_data.spanish import repair_text

SOURCE = "BOE"
LICENSE = "Reutilización autorizada con cita de la fuente (Ley 37/2007; aviso legal del BOE)"
ATTRIBUTION = ("Fuente: Agencia Estatal Boletín Oficial del Estado (www.boe.es), legislación consolidada, "
               "actualizada a {updated}. Texto consolidado de carácter informativo: solo la publicación "
               "oficial en el BOE tiene validez jurídica.")
PAGE_SIZE = 10_000  # the API's maximum per request
SHARD_SIZE = 5_000

_STATE_RANKS = {
    "Constitución": ("laws", "constitution"),
    "Ley Orgánica": ("laws", "organic_law"),
    "Ley": ("laws", "law"),
    "Real Decreto Legislativo": ("laws", "legislative_decree"),
    "Decreto Legislativo": ("laws", "legislative_decree"),
    "Real Decreto-ley": ("laws", "decree_law"),
    "Decreto-ley": ("laws", "decree_law"),
    "Acuerdo Internacional": ("laws", "international_treaty"),
    "Real Decreto": ("regulations", "regulation"),
    "Decreto": ("regulations", "regulation"),
    "Reglamento": ("regulations", "regulation"),
    "Orden": ("regulations", "order"),
    "Resolución": ("regulations", "resolution"),
    "Acuerdo": ("regulations", "resolution"),
    "Circular": ("regulations", "circular"),
    "Instrucción": ("regulations", "circular"),
}
_REGIONAL_LAW_RANKS = {"Ley", "Ley Foral"}
_REGIONAL_DECREE_LAW_RANKS = {"Decreto-ley", "Decreto-ley Foral", "Decreto Legislativo", "Decreto Foral Legislativo"}

_ARTICLE_RE = re.compile(
    r"^\s*Art(?:[íi]culo|\.)\s+(?P<num>\d{1,4}(?:\s*\.\s*\d{1,3})?(?:\s+(?:bis|ter|quater|quinquies|sexies|septies"
    r"|octies|nonies|novies|decies))?|[úu]nico|primero|segundo|tercero|cuarto|quinto|sexto|s[ée]ptimo|octavo"
    r"|noveno|d[ée]cimo)\s*\.?\s*(?P<title>.*)$",
    re.IGNORECASE,
)
_DISPOSITION_RE = re.compile(
    r"^\s*(?P<num>Disposici[óo]n\s+(?:adicional|transitoria|derogatoria|final)(?:\s+[\wáéíóúü]+)?)\s*\.?\s*(?P<title>.*)$",
    re.IGNORECASE,
)
_APARTADO_RE = re.compile(r"^\s*(?P<label>\d{1,3})\.\s+\S")
_LETTER_RE = re.compile(r"^\s*(?P<label>[a-zñ]{1,2})\)\s+\S")
_SKIP_CLASSES = {"imagen", "firma_rey", "firma_ministro", "firma"}
_NOTE_CLASSES = {"nota_pie", "nota_pie_2", "nota_pie_3"}
_HEADING_LEVELS = {
    "libro_num": "libro", "libro_tit": "libro", "libro": "libro",
    "titulo_num": "titulo", "titulo_tit": "titulo", "titulo": "titulo",
    "capitulo_num": "capitulo", "capitulo_tit": "capitulo", "capitulo": "capitulo",
    "seccion": "seccion", "subseccion": "subseccion",
    "anexo_num": "anexo", "anexo_tit": "anexo", "anexo": "anexo",
    "parte_num": "libro", "parte_tit": "libro",
}


def iso(stamp: str | None) -> str | None:
    """'20230124' / '20261016T130533Z' -> '2023-01-24'."""
    match = re.match(r"(\d{4})(\d{2})(\d{2})", stamp or "")
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}" if match else None


def _text(item: dict | str | None) -> str:
    return (item.get("texto") if isinstance(item, dict) else item) or ""


# Codes and laws enacted in the form of a (Real) Decreto that "publishes" or "approves" them: the
# Código Civil (RD de 24 de julio de 1889), the Código de Comercio, the Ley de Enjuiciamiento
# Criminal, the Ley Hipotecaria, older textos refundidos. They have the rank of law.
_LAW_IN_DECREE_RE = re.compile(
    r"(?i)por\s+(?:el|la)\s+que\s+se\s+(?:publica|aprueba)\s+(?:el|la)\s+(?:nuevo\s+|nueva\s+)?"
    r"(?:C[óo]digo\s+(?:Civil|Penal|de\s+Comercio)|Ley\b|texto\s+refundido\s+de\s+la\s+Ley|redacci[óo]n\s+oficial\s+de\s+la\s+Ley)"
)


def classify(item: dict) -> tuple[str, str] | None:
    """(category, authority_level) of a catalogue entry, or None for a rank we don't know."""
    rank = _text(item.get("rango")).strip()
    if rank in ("Real Decreto", "Decreto") and _LAW_IN_DECREE_RE.search(item.get("titulo") or "") \
            and not _text(item.get("ambito")).startswith("Auton"):
        return "laws", "law" if "texto refundido" not in (item.get("titulo") or "").lower() else "legislative_decree"
    if _text(item.get("ambito")).startswith("Auton"):
        if rank in _REGIONAL_LAW_RANKS:
            return "regional_laws", "regional_law"
        if rank in _REGIONAL_DECREE_LAW_RANKS:
            return "regional_laws", "regional_decree_law"
        return "regional_laws", "regional_regulation"
    return _STATE_RANKS.get(rank)


def iter_catalogue(client, api: str, log=None) -> Iterator[dict]:
    """Every consolidated act, newest update first, paging through the catalogue."""
    offset = 0
    while True:
        url = f"{api}/legislacion-consolidada?offset={offset}&limit={PAGE_SIZE}"
        data = client.get_json(url).get("data") or []
        if log:
            log(f"BOE catalogue: {offset + len(data):,} entries")
        yield from data
        if len(data) < PAGE_SIZE:
            return
        offset += PAGE_SIZE


# --- the consolidated text ------------------------------------------------------------------------


def _xml_root(xml: bytes):
    from lxml import etree

    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True, remove_comments=True)
    return etree.fromstring(xml, parser)


def _clean(text: str) -> str:
    return re.sub(r"[ \t ]+", " ", text).strip()


def _element_lines(element) -> list[tuple[str, str]]:
    """(class, text) for every paragraph-like child of a <version>, tables flattened row by row."""
    out: list[tuple[str, str]] = []
    for child in element:
        tag = child.tag if isinstance(child.tag, str) else ""
        css = child.get("class") or ""
        if tag == "p":
            text = _clean("".join(child.itertext()))
            if text:
                out.append((css, text))
        elif tag == "blockquote":
            out += [(css or c, t) for c, t in _element_lines(child)]
        elif tag == "table":
            for row in child.iter("tr"):
                cells = [_clean("".join(cell.itertext())) for cell in row if cell.tag in ("td", "th")]
                if any(cells):
                    out.append(("tabla", " | ".join(cells)))
        elif tag in ("div", "ul", "ol", "li"):
            out += _element_lines(child)
    return out


def _split_apartados(lines: list[str]) -> tuple[str, list[SubsectionRecord]]:
    """An article's intro and top-level apartados ("1.", "2." ...; or "a)", "b)" when it has no
    numbered ones). A marker counts only when it continues the sequence, so a nested "1." list
    inside apartado 3 stays with apartado 3."""
    numbered = any(_APARTADO_RE.match(line) for line in lines)
    pattern = _APARTADO_RE if numbered else _LETTER_RE
    intro: list[str] = []
    subs: list[SubsectionRecord] = []

    def follows(label: str) -> bool:
        if not subs:
            return label in ("1", "a")
        previous = subs[-1].label
        if previous.isdigit() and label.isdigit():
            return int(label) == int(previous) + 1
        return len(label) == 1 == len(previous) and ord(label) == ord(previous) + 1

    for line in lines:
        match = pattern.match(line)
        if match and follows(match.group("label")):
            subs.append(SubsectionRecord(label=match.group("label"), text=line))
        elif subs:
            subs[-1].text += "\n" + line
        else:
            intro.append(line)
    return "\n".join(intro).strip(), subs


@dataclass
class _Headings:
    libro: str | None = None
    titulo: str | None = None
    capitulo: str | None = None
    seccion: str | None = None
    subseccion: str | None = None
    anexo: str | None = None

    def set(self, level: str, text: str) -> None:
        order = ["libro", "titulo", "capitulo", "seccion", "subseccion"]
        if level == "anexo":
            self.libro = self.titulo = self.capitulo = self.seccion = self.subseccion = None
            self.anexo = text
            return
        for lower in order[order.index(level) + 1 :]:
            setattr(self, lower, None)
        setattr(self, level, text)

    @property
    def division(self) -> str | None:
        parts = [p for p in (self.libro, self.titulo) if p]
        return " > ".join(parts) or None

    @property
    def subchapter(self) -> str | None:
        parts = [p for p in (self.seccion, self.subseccion) if p]
        return " > ".join(parts) or None


def _heading_text(lines: list[tuple[str, str]]) -> tuple[str | None, str, list[str]]:
    """(level, 'TÍTULO I. De los derechos ...', body lines that aren't part of the heading) of an
    encabezado block. Without heading classes, the first line is the heading."""
    level = None
    heading = [t for c, t in lines if c in _HEADING_LEVELS]
    for css, _ in lines:
        level = level or _HEADING_LEVELS.get(css)
    if not heading:
        heading = [lines[0][1]]
    rest = [t for c, t in lines if c not in _HEADING_LEVELS][0 if level else 1 :]
    joined = ". ".join(p.rstrip(".") for p in heading if p)
    return level, joined, rest


def parse_consolidated_text(xml: bytes) -> tuple[list[SectionRecord], dict]:
    """Sections (articles, dispositions, annexes, preamble) in their current wording, from the BOE's
    /texto XML, plus stats {blocks, versions, latest_version}."""
    root = _xml_root(xml)
    sections: list[SectionRecord] = []
    headings = _Headings()
    stats: Counter = Counter()
    latest = ""
    annex: SectionRecord | None = None
    group: str | None = None  # "DISPOSICIONES TRANSITORIAS": the heading over unnumbered dispositions
    today = date.today().strftime("%Y%m%d")
    for block in root.iter("bloque"):
        stats["blocks"] += 1
        kind = block.get("tipo") or ""
        versions = block.findall("version")
        if not versions or kind == "firma":
            continue
        expired = (block.get("fecha_caducidad") or "")[:8]
        if expired and expired <= today:  # a heading or provision that no longer exists
            stats["expired_blocks"] += 1
            continue
        version = versions[-1]
        stats["versions"] += len(versions)
        version_date = iso(version.get("fecha_vigencia") or version.get("fecha_publicacion"))
        latest = max(latest, version_date or "")
        lines = [(c, t) for c, t in _element_lines(version) if c not in _SKIP_CLASSES]
        notes = [t for c, t in lines if c in _NOTE_CLASSES]
        body = [(c, t) for c, t in lines if c not in _NOTE_CLASSES]
        if not body:
            continue
        if kind == "encabezado" or (body[0][0] in _HEADING_LEVELS and kind != "preambulo"):
            level, text, rest = _heading_text(body)
            group = text[:120] if re.match(r"(?i)disposici[oó]n(?:es)?\b", text) else None
            if rest and level != "anexo":  # text directly under the heading (e.g. an intro to the transitional rules)
                sections.append(SectionRecord(kind="disposition", number=text[:200], text="\n".join(rest),
                                              intro="\n".join(rest), version_date=version_date))
            if level == "anexo" or (block.get("id") or "").startswith("an"):
                headings.set("anexo", text)
                annex = SectionRecord(kind="schedule", number=text[:200], title=None, text="", version_date=version_date)
                sections.append(annex)
            else:
                headings.set(level or "titulo", text)
            continue
        if kind in ("preambulo", "parte_dispositiva") and block.get("id", "").startswith(("pr", "preambulo")):
            text = "\n".join(t for _, t in body)
            sections.append(SectionRecord(kind="preamble", number="preamble", text=text, intro=text,
                                          version_date=version_date))
            continue
        first_class, first = body[0]
        title_line = first if first_class in ("articulo", "sangrado_articulo") else (block.get("titulo") or "")
        article = _ARTICLE_RE.match(title_line)
        disposition = None if article else _DISPOSITION_RE.match(title_line)
        rest = [t for _, t in (body[1:] if first_class in ("articulo", "sangrado_articulo") else body)]
        annex_wrapper = annex is not None and not annex.text
        if article or disposition:
            if annex_wrapper:  # the "anexo" was a wrapper around the main text (CÓDIGO CIVIL, a texto refundido)
                sections.remove(annex)
                headings.anexo = None
            annex = None
            match = article or disposition
            number = re.sub(r"\s+", " ", match.group("num")).strip()
            number = number[0].upper() + number[1:] if disposition else number.lower()
            title = match.group("title").strip().rstrip(".") or None
            intro, subs = _split_apartados(rest)
            text = "\n".join(rest).strip()
            sections.append(SectionRecord(
                kind="section" if article else "disposition", number=number, title=title,
                division=headings.division if not disposition else None,
                chapter=headings.capitulo if not disposition else None,
                subchapter=headings.subchapter if not disposition else None,
                note=" ".join(notes) or None, intro=intro, subsections=subs, text=text or "(sin contenido)",
                version_date=version_date,
            ))
            continue
        text = "\n".join(t for _, t in body)
        if annex is not None:  # annex content
            annex.text = (annex.text + "\n" + text).strip()
            continue
        # a block with no article heading ("Primera" under DISPOSICIONES TRANSITORIAS): its own title
        label = (block.get("titulo") or block.get("id") or "").strip()
        if group and label:
            label = f"{group} — {label}"
        sections.append(SectionRecord(kind="disposition", number=label[:200] or "texto", text=text,
                                      intro=text, note=" ".join(notes) or None, version_date=version_date))
    stats["latest_version"] = latest
    sections = [s for s in sections if s.kind != "schedule" or s.text]
    return sections, dict(stats)


def render_text(title: str, sections: list[SectionRecord]) -> str:
    """The whole act as plain text: headings once each, then every provision."""
    lines = [title]
    seen: tuple = ()
    for section in sections:
        path = (section.division, section.chapter, section.subchapter)
        if section.kind == "section" and path != seen:
            lines += [p for p, old in zip(path, seen or (None, None, None)) if p and p != old]
            seen = path
        if section.kind == "preamble":
            lines.append(section.text)
        elif section.kind == "schedule":
            lines += [section.number, section.text]
        else:
            label = f"Artículo {section.number}" if section.kind == "section" else section.number
            lines.append(f"{label}." + (f" {section.title}." if section.title else ""))
            lines.append(section.text)
    return "\n".join(line for line in lines if line).strip()


def _number_of(item: dict) -> str | None:
    number = (item.get("numero_oficial") or "").strip()
    return number or None


def build_record(item: dict, category: str, level: str, sections: list[SectionRecord], stats: dict,
                 subjects: list[str] | None = None) -> CorpusRecord:
    title = (item.get("titulo") or "").strip().rstrip(".")
    text = render_text(title, sections)
    text, encoding, notes = repair_text(text)
    updated = iso(item.get("fecha_actualizacion")) or ""
    regional = category == "regional_laws"
    record = CorpusRecord(
        id=f"boe:{item['identificador']}",
        source=SOURCE,
        source_url=item.get("url_html_consolidada") or f"https://www.boe.es/buscar/act.php?id={item['identificador']}",
        license=LICENSE,
        attribution=ATTRIBUTION.format(updated=updated),
        category=category,
        title=title,
        authority_level=level,
        status="repealed" if item.get("vigencia_agotada") == "S" else "in_force",
        status_source="boe_vigencia_agotada",
        effective_date=iso(item.get("fecha_vigencia")),
        retrieved_at=now_iso(),
        language="es",
        text=text,
        quality=Quality(encoding=encoding, extraction="n/a", notes=notes + (
            [] if sections else ["no structured blocks in the consolidated text"])),
        law_id=item["identificador"],
        rank=_text(item.get("rango")) or None,
        official_number=_number_of(item),
        department=_text(item.get("departamento")) or None,
        scope="autonomico" if regional else "estatal",
        region=_text(item.get("departamento")) if regional else None,
        eli_url=item.get("url_eli") or None,
        journal=item.get("diario") or None,
        journal_number=item.get("diario_numero") or None,
        disposition_date=iso(item.get("fecha_disposicion")),
        publication_date=iso(item.get("fecha_publicacion")),
        latest_amendment_date=stats.get("latest_version") or None,
        source_updated_at=item.get("fecha_actualizacion"),
        consolidated=_text(item.get("estado_consolidacion")) == "Finalizado",
        subjects=subjects or [],
        sections=sections,
    )
    return record.finalize()


def fetch_subjects(client, api: str, boe_id: str) -> list[str]:
    root = _xml_root(client.get(f"{api}/legislacion-consolidada/id/{boe_id}/analisis",
                                headers={"Accept": "application/xml"}).content)
    return [_clean(m.text or "") for m in root.iter("materia") if (m.text or "").strip()]


# --- the run ----------------------------------------------------------------------------------------


class RecordCache:
    """<output>/boe/cache/<id>.json.gz: the normalized record of each act, with the catalogue stamp it
    was built from. An unchanged stamp means no request at all on the next run."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, boe_id: str) -> Path:
        return self.directory / boe_id[:9] / f"{boe_id}.json.gz"  # BOE-A-YYY / ... keeps folders small

    def get(self, boe_id: str, stamp: str) -> dict | None:
        path = self._path(boe_id)
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        return data["record"] if data.get("stamp") == stamp else None

    def has(self, boe_id: str) -> bool:
        return self._path(boe_id).exists()

    def put(self, boe_id: str, stamp: str, record: dict) -> None:
        path = self._path(boe_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump({"stamp": stamp, "record": record}, f, ensure_ascii=False)
        tmp.replace(path)


def _selected(item: dict, args, cfg) -> tuple[str, str] | None:
    kind = classify(item)
    if kind is None:
        return None
    category, _ = kind
    if category not in args.categories:
        return None
    if args.ids and item["identificador"] not in args.ids:
        return None
    if not args.include_repealed and item.get("vigencia_agotada") == "S":
        return None
    if category == "regional_laws" and cfg.boe.regions:
        if not any(r.lower() in _text(item.get("departamento")).lower() for r in cfg.boe.regions):
            return None
    rank = _text(item.get("rango"))
    if rank in cfg.boe.exclude_ranks:
        return None
    if args.since and (item.get("fecha_actualizacion") or "")[:8] < args.since:
        return None
    return kind


def run_boe(ctx) -> None:
    """Catalogue -> select -> fetch changed texts -> one JSONL per category."""
    cfg, args = ctx.cfg, ctx.args
    api = cfg.sources.boe_api.rstrip("/")
    ctx.manifest.source("BOE datos abiertos: legislación consolidada", f"{api}/legislacion-consolidada", LICENSE,
                        "Agencia Estatal Boletín Oficial del Estado")
    catalogue = list(iter_catalogue(ctx.client, api, ctx.log))
    ctx.manifest.count("catalogue_entries", len(catalogue))
    cache = RecordCache(ctx.main_root / "boe" / "cache")

    selected: dict[str, list[tuple[dict, str, str]]] = {c: [] for c in args.categories}
    reasons: Counter = Counter()
    # Acts the BOE has withdrawn from its pages (right to be forgotten): listed in its robots.txt.
    withdrawn = ctx.client.disallowed_tokens(f"{api}/", r"BOE-[A-Z]-\d{4}-\d+")
    ctx.log(f"BOE robots.txt withdraws {len(withdrawn):,} documents; they are never fetched")
    for item in catalogue:
        kind = classify(item)
        if kind is None:
            reasons[f"unknown rank: {_text(item.get('rango'))}"] += 1
            continue
        chosen = _selected(item, args, cfg)
        if chosen is None:
            reasons["not selected"] += 1
            continue
        if item["identificador"] in withdrawn:
            reasons["withdrawn (robots.txt)"] += 1
            continue
        selected[chosen[0]].append((item, *chosen))
    if args.sample:
        selected = {c: items[: args.sample] for c, items in selected.items()}
    ctx.manifest.data["filters"] = {"skipped": dict(reasons)}
    counts = {c: len(items) for c, items in selected.items()}
    stale = {c: sum(1 for item, *_ in items if cache.get(item["identificador"], item.get("fecha_actualizacion") or "") is None)
             for c, items in selected.items()}
    ctx.log(f"BOE: selected {counts}; to fetch (new or updated) {stale}; skipped {dict(reasons)}")
    if ctx.dry_run:
        requests = sum(stale.values()) * (2 if args.with_subjects else 1)
        ctx.manifest.set("estimates", "boe", {
            "selected": counts, "to_fetch": stale, "requests": requests,
            "hours_at_min_interval": round(requests * max(cfg.min_interval_s, 1.0) / 3600, 1),
            "approx_mb_xml": round(sum(stale.values()) * 0.12, 1),
        })
        ctx.log(f"dry run: {requests:,} requests, ~{requests * max(cfg.min_interval_s, 1.0) / 3600:.1f} h at "
                f"{cfg.min_interval_s}s per request")
        return

    for category, items in selected.items():
        if not items:
            continue
        writer = JsonlWriter(ctx.root / category, category, shard_size=SHARD_SIZE)
        progress = Progress(f"BOE {category}", total=len(items), unit="acts", log=ctx.log)
        fetched = reused = failed = 0
        try:
            for item, category_, level in items:
                boe_id, stamp = item["identificador"], item.get("fecha_actualizacion") or ""
                record = None if args.force else cache.get(boe_id, stamp)
                if record is None:
                    try:
                        xml = ctx.client.get(f"{api}/legislacion-consolidada/id/{boe_id}/texto",
                                             headers={"Accept": "application/xml"}).content
                        sections, stats = parse_consolidated_text(xml)
                        subjects = fetch_subjects(ctx.client, api, boe_id) if args.with_subjects else []
                    except Exception as exc:  # noqa: BLE001 -- one bad act is reported, the run goes on
                        from aicodex.legal_data.http import FetchStopped

                        if isinstance(exc, FetchStopped):
                            raise
                        failed += 1
                        ctx.manifest.warn(f"{boe_id}: {type(exc).__name__}: {exc}")
                        continue
                    record = build_record(item, category_, level, sections, stats, subjects).model_dump(mode="json")
                    cache.put(boe_id, stamp, record)
                    fetched += 1
                else:
                    reused += 1
                writer.write(record)
                progress.update(1, fetched=fetched, cached=reused, failed=failed)
        except BaseException:
            writer.close(commit=False)
            raise
        progress.done(fetched=fetched, cached=reused, failed=failed)
        paths = writer.close()
        ctx.manifest.count(f"{category}_records", writer.count)
        ctx.manifest.count(f"{category}_fetched", fetched)
        ctx.manifest.count(f"{category}_failed", failed)
        ctx.log(f"{category}: {writer.count} records ({fetched} fetched, {reused} from cache, {failed} failed) "
                f"-> {', '.join(str(p) for p in paths)}")
