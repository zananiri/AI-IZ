"""Constitutional case law: the resolutions of the Tribunal Constitucional (sentencias, autos and
declaraciones) from its own public search system, "Sistema HJ" (hj.tribunalconstitucional.es).

    GET {hj}/es/Resolucion/Show/{n}       one resolution per sequential id n (a missing id redirects)

Each page carries the complete text (cabecera, antecedentes, fundamentos jurídicos, fallo, votos
particulares), the ECLI and the court's own "ficha técnica" (chamber, judges, BOE publication,
procedure and number, date, síntesis). Only these pages are requested -- no search pages -- one
per second, with the same robots.txt and stop-on-block rules as every other fetcher.

The TC's doctrine binds every court (art. 5.1 LOPJ), and its sentencias are published in the BOE,
so this is the most authoritative case law that is openly reusable. The Tribunal Supremo and the
other courts publish through CENDOJ (poderjudicial.es), whose terms forbid bulk downloading
without CGPJ authorisation: it is deliberately not crawled (see scripts/legal_data/README.md).

Privacy: the TC already anonymises minors and victims. With legal_data.constitutional_court.
anonymize_parties (on by default), every other natural person introduced as "don/doña ..." is
replaced by "[persona N]" -- the magistrates excepted -- because a research corpus needs the
doctrine, not the parties' names (RGPD art. 5.1.c, data minimisation). A heuristic, not a guarantee.

Incremental: every id fetched is cached (<output>/constitutional_court/cache/), including the ones
skipped for their type, so a re-run only requests ids it has never seen; the scan stops after
`stop_after_missing` consecutive missing ids.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from aicodex.legal_data.progress import Progress
from aicodex.legal_data.records import CorpusRecord, JsonlWriter, Quality, now_iso
from aicodex.legal_data.spanish import repair_text

SOURCE = "Tribunal Constitucional (Sistema HJ)"
LICENSE = "Reutilización autorizada con cita de la fuente (Ley 37/2007; Tribunal Constitucional de España)"
ATTRIBUTION = "Fuente: Tribunal Constitucional de España, buscador de jurisprudencia constitucional (hj.tribunalconstitucional.es)"
SHARD_SIZE = 5_000

_TITLE_RE = re.compile(r"(?P<type>SENTENCIA|AUTO|DECLARACI[ÓO]N)\s+(?P<num>\d+/\d{4})", re.IGNORECASE)
_DATE_RE = re.compile(r"(\d{2})/(\d{2})/(\d{4})")
_BOE_RE = re.compile(r"\(\s*(BOE[^)]*)\)")
_ECLI_RE = re.compile(r"ECLI:ES:TC:\d{4}:\d+[A-Z]?")
_BLOCK_TAGS = ("p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "br", "table", "fieldset", "legend")

# "don Miguel Ángel González Ortiz", "doña María de la O López"
_NAME_WORD = r"(?:[A-ZÁÉÍÓÚÑÜ][\wáéíóúñü'’.-]+)"
_PERSON_RE = re.compile(
    rf"\b(?P<title>[Dd]on|[Dd]oña|D\.|D\.ª|Dña\.)\s+(?P<name>{_NAME_WORD}(?:\s+(?:(?:de|del|de la|de los|y|i)\s+)?{_NAME_WORD}){{0,5}})"
)


def resolution_url(base: str, n: int) -> str:
    return f"{base.rstrip('/')}/es/Resolucion/Show/{n}"


def _text_of(element) -> str:
    """Text with a line break after every block element, whitespace tidied."""
    if element is None:
        return ""
    for node in element.iter():  # line breaks in the HTML source are spaces, not paragraph breaks
        if node.text:
            node.text = re.sub(r"\s*\n\s*", " ", node.text)
        if node.tail:
            node.tail = re.sub(r"\s*\n\s*", " ", node.tail)
    for node in element.iter():
        if isinstance(node.tag, str) and node.tag.lower() in _BLOCK_TAGS:
            node.tail = "\n" + (node.tail or "")
    for node in element.iter("script", "style"):
        node.drop_tree()
    text = element.text_content()
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines()]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


def _ficha(root) -> dict[str, str]:
    """The "ficha técnica" table: header -> value."""
    ficha = root.get_element_by_id("ficha-tecnica", None)
    fields: dict[str, str] = {}
    if ficha is None:
        return fields
    for row in ficha.iter("tr"):
        cells = [c for c in row if isinstance(c.tag, str) and c.tag in ("th", "td")]
        if len(cells) >= 2:
            key = re.sub(r"\s+", " ", cells[0].text_content()).strip()
            value = re.sub(r"\s+", " ", cells[1].text_content()).strip()
            if key:
                fields[key] = value
    return fields


def _iso(value: str | None) -> str | None:
    match = _DATE_RE.search(value or "")
    return f"{match.group(3)}-{match.group(2)}-{match.group(1)}" if match else None


def _judges(value: str) -> list[str]:
    value = re.sub(r"\.$", "", value or "")
    parts = re.split(r",\s*|\s+y\s+", value)
    names = [re.sub(r"^(?:[Dd]on|[Dd]oña)\s+", "", p).strip() for p in parts]
    return [n for n in names if n]


class Anonymizer:
    """Natural persons introduced as don/doña -> [persona N], consistently within one resolution;
    the names in `keep` (the magistrates) stay."""

    def __init__(self, keep: list[str]) -> None:
        self.keep = {self._key(k) for k in keep}
        self.names: dict[str, str] = {}

    @staticmethod
    def _key(name: str) -> str:
        return re.sub(r"\s+", " ", name).strip().lower()

    def __call__(self, text: str) -> tuple[str, int]:
        replaced = 0

        def swap(match: re.Match) -> str:
            nonlocal replaced
            name = match.group("name")
            key = self._key(name)
            if key in self.keep or any(key in k or k in key for k in self.keep if len(k) > 8):
                return match.group(0)
            before = match.string[max(0, match.start() - 30) : match.start()].lower()
            if "magistrad" in before:  # a magistrate of the court not sitting on this bench
                self.keep.add(key)
                return match.group(0)
            label = self.names.setdefault(key, f"[persona {len(self.names) + 1}]")
            replaced += 1
            return f"{match.group('title')} {label}"

        return _PERSON_RE.sub(swap, text), replaced


def parse_resolution(html: bytes | str, url: str, source_id: int, anonymize: bool) -> dict | None:
    """A resolution page -> the fields of a CorpusRecord (None if the page holds no resolution)."""
    import lxml.html

    if isinstance(html, str):
        html = html.encode("utf-8")
    root = lxml.html.fromstring(html, parser=lxml.html.HTMLParser(encoding="utf-8"))
    title = root.findtext(".//title") or ""
    head = _TITLE_RE.search(title)
    if not head:
        return None
    doc_type = head.group("type").upper().replace("DECLARACION", "DECLARACIÓN")
    number = head.group("num")
    identifier = _text_of(root.get_element_by_id("resolucion-identifier", None))
    raw = html.decode("utf-8", errors="replace")
    ecli = _ECLI_RE.search(identifier) or _ECLI_RE.search(raw)
    boe = _BOE_RE.search(identifier)
    ficha = _ficha(root)
    body = _text_of(root.get_element_by_id("complete_resolucion", None))
    if not body:
        return None
    judges = _judges(ficha.get("Magistrados", ""))
    procedure_full = ficha.get("Tipo y número de registro", "")
    procedure = re.sub(r"\s+[\d/ -]+(?:\s*y\s*[\d/ -]+)*(?:\s*\(.*\))?$", "", procedure_full).strip() or None
    summary = " ".join(v for k, v in ficha.items() if k.lower().startswith("síntesis")) or None
    replaced = 0
    if anonymize:
        anonymizer = Anonymizer(judges)
        body, replaced = anonymizer(body)
        if summary:
            summary, n = anonymizer(summary)
            replaced += n
    decision_date = _iso(ficha.get("Fecha de resolución")) or None
    headline = (identifier.splitlines() or [""])[0].strip() or f"{doc_type} {number}"
    return {
        "doc_type": doc_type,
        "resolution_number": number,
        "title": f"{headline}" if headline.upper().startswith(doc_type[:4]) else f"{doc_type} {number}",
        "ecli": ecli.group(0) if ecli else None,
        "boe_publication": boe.group(1).strip() if boe else None,
        "chamber": ficha.get("Órgano") or None,
        "judges": judges,
        "case_number": procedure_full or None,
        "procedure_type": procedure,
        "decision_date": decision_date,
        "summary": summary,
        "text": body,
        "anonymized": bool(anonymize),
        "replacements": replaced,
        "source_url": url,
        "source_case_id": str(source_id),
    }


def to_record(fields: dict) -> CorpusRecord:
    text, encoding, notes = repair_text(fields["text"])
    year = int(fields["resolution_number"].split("/")[1]) if "/" in (fields.get("resolution_number") or "") else None
    record = CorpusRecord(
        id=f"tc:{fields['ecli'] or fields['doc_type'] + '-' + fields['resolution_number']}",
        source=SOURCE,
        source_url=fields["source_url"],
        license=LICENSE,
        attribution=ATTRIBUTION,
        category="constitutional_court",
        title=fields["title"],
        authority_level="judgment" if fields["doc_type"] == "SENTENCIA" else "decision",
        status="in_force",
        status_source="tc_published",
        effective_date=fields.get("decision_date"),
        retrieved_at=now_iso(),
        language="es",
        text=text,
        quality=Quality(encoding=encoding, notes=notes),
        ecli=fields.get("ecli"),
        case_number=fields.get("case_number"),
        court="Tribunal Constitucional",
        chamber=fields.get("chamber"),
        judges=fields.get("judges") or [],
        decision_date=fields.get("decision_date"),
        doc_type=fields["doc_type"],
        resolution_number=fields["resolution_number"],
        procedure_type=fields.get("procedure_type"),
        boe_publication=fields.get("boe_publication"),
        summary=fields.get("summary"),
        year=year,
        source_case_id=fields.get("source_case_id"),
        anonymized=fields.get("anonymized", False),
    )
    return record.finalize()


class IdCache:
    """<output>/constitutional_court/cache/<bucket>/<id>.json.gz -- {"status": "record"|"skipped"|"missing", ...}."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, n: int) -> Path:
        return self.directory / f"{n // 1000:03d}" / f"{n}.json.gz"

    def get(self, n: int) -> dict | None:
        path = self._path(n)
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def put(self, n: int, data: dict) -> None:
        path = self._path(n)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp.replace(path)

    def ids(self) -> list[int]:
        return sorted(int(p.name.split(".")[0]) for p in self.directory.glob("*/*.json.gz"))


def _fetch(client, url: str):
    """(status, html): 'ok' with the page, or 'missing' when the id redirects (no such resolution)."""
    response = client.get(url, ok=(200, 301, 302, 303), follow_redirects=False)
    if response.status_code != 200:
        return "missing", None
    return "ok", response.content


def find_last_id(client, base: str, known: int, log=None) -> int:
    """The highest existing id, by galloping then bisecting from `known` (an id that exists)."""
    lo, step = known, 256
    while _fetch(client, resolution_url(base, lo + step))[0] == "ok":
        lo, step = lo + step, step * 2
    hi = lo + step
    while hi - lo > 1:
        mid = (lo + hi) // 2
        status = _fetch(client, resolution_url(base, mid))[0]
        if status == "ok":
            lo = mid
        else:
            # an isolated gap must not end the search early: look a few ids further
            if any(_fetch(client, resolution_url(base, mid + k))[0] == "ok" for k in (1, 2, 3)):
                lo = mid
            else:
                hi = mid
    if log:
        log(f"TC: highest resolution id {lo:,}")
    return lo


def run_tc(ctx) -> None:
    cfg, args = ctx.cfg, ctx.args
    tc_cfg = cfg.constitutional_court
    base = cfg.sources.tc_hj.rstrip("/")
    doc_types = {t.upper() for t in (args.doc_types or tc_cfg.doc_types)}
    anonymize = tc_cfg.anonymize_parties if args.anonymize is None else args.anonymize
    ctx.manifest.source(SOURCE, base, LICENSE, ATTRIBUTION, doc_types=sorted(doc_types), anonymize=anonymize)
    cache = IdCache(ctx.main_root / "constitutional_court" / "cache")
    first = args.from_id or tc_cfg.first_id
    if args.to_id:
        last = args.to_id
    else:
        known = max([i for i in cache.ids() if (cache.get(i) or {}).get("status") != "missing"] or [tc_cfg.known_id])
        last = find_last_id(ctx.client, base, known, ctx.log)
    ids = list(range(first, last + 1))
    if args.sample:
        ids = ids[-args.sample :]  # the newest resolutions
    todo = [n for n in ids if args.force or cache.get(n) is None]
    ctx.log(f"TC: ids {ids[0] if ids else '-'}..{ids[-1] if ids else '-'} ({len(ids):,}); {len(todo):,} not fetched yet")
    if ctx.dry_run:
        hours = len(todo) * max(cfg.min_interval_s, 1.0) / 3600
        ctx.manifest.set("estimates", "constitutional_court", {"ids": len(ids), "to_fetch": len(todo),
                                                               "hours_at_min_interval": round(hours, 1),
                                                               "approx_mb_html": round(len(todo) * 0.06, 1)})
        ctx.log(f"dry run: {len(todo):,} pages, ~{hours:.1f} h at {cfg.min_interval_s}s per request")
        return

    counts: Counter = Counter()
    progress = Progress("TC resolutions", total=len(todo), unit="pages", log=ctx.log)
    missing_run = 0
    for n in todo:
        url = resolution_url(base, n)
        status, html = _fetch(ctx.client, url)
        if status == "missing":
            cache.put(n, {"status": "missing", "checked_at": now_iso()})
            counts["missing"] += 1
            missing_run += 1
            if not args.to_id and missing_run >= tc_cfg.stop_after_missing:
                ctx.log(f"TC: {missing_run} missing ids in a row -- stopping the scan at {n}")
                break
            progress.update(1, **counts)
            continue
        missing_run = 0
        fields = parse_resolution(html, url, n, anonymize)
        if fields is None:
            cache.put(n, {"status": "missing", "checked_at": now_iso(), "note": "no resolution on the page"})
            counts["unparsed"] += 1
        elif fields["doc_type"] not in doc_types:
            cache.put(n, {"status": "skipped", "doc_type": fields["doc_type"], "number": fields["resolution_number"]})
            counts[f"skipped_{fields['doc_type'].lower()}"] += 1
        else:
            record = to_record(fields).model_dump(mode="json")
            cache.put(n, {"status": "record", "record": record, "doc_type": fields["doc_type"]})
            counts[f"new_{fields['doc_type'].lower()}"] += 1
            counts["anonymization_replacements"] += fields["replacements"]
        progress.update(1, **{k: v for k, v in counts.items() if k != "anonymization_replacements"})
    progress.done()

    # Assemble the category from every cached record of the selected types (newest first).
    writer = JsonlWriter(ctx.root / "constitutional_court", "constitutional_court", shard_size=SHARD_SIZE)
    written: Counter = Counter()
    try:
        for n in sorted((i for i in cache.ids() if i in set(ids)) if args.sample else cache.ids(), reverse=True):
            data = cache.get(n) or {}
            if data.get("status") != "record" or data.get("doc_type") not in doc_types:
                continue
            writer.write(data["record"])
            written[data["doc_type"]] += 1
    except BaseException:
        writer.close(commit=False)
        raise
    paths = writer.close()
    ctx.manifest.data["filters"] = dict(counts)
    for key, value in written.items():
        ctx.manifest.count(f"written_{key.lower()}", value)
    ctx.manifest.count("records_written", writer.count)
    ctx.log(f"constitutional_court: {writer.count} records {dict(written)} -> {len(paths)} shard(s); this run: {dict(counts)}"
            f" ({datetime.now():%Y-%m-%d})")
