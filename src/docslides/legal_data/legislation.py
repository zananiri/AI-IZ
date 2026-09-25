"""Laws and regulations: Open Law Book pages (legal_data/wikisource.py) joined to the Knesset
registry that scripts/legal_data/fetch_metadata.py saved under metadata/knesset/.

    laws              חוק, חוק-יסוד (or KNS_IsraelLaw.IsBasicLaw), פקודה/פקודת (Mandate era included)
    procedural_rules  everything else in the Open Law Book: תקנות, צו, כללים, הודעה, אכרזה ...

Laws join on the page's `ח:מאגר` registry id (KNS_IsraelLaw.Id), falling back to the normalized
name (KNS_IsraelLaw.Name and KNS_IsraelLawName). Regulations have no registry id on their pages
and join KNS_SecondaryLaw by name; KNS_SecLawAuthorizingLaw gives their authorizing laws.
Status comes from KNS_IsraelLaw.LawValidityDesc; failing that, a repeal marker on the page
("(בוטל)", "... הישנות") marks it repealed; otherwise it is unknown -- never guessed.

PDFs are downloaded only from the Knesset file server (legal_data.sources.knesset_pdf_host), from
the links in KNS_DocumentIsraelLaw / KNS_DocumentSecondaryLaw. The PDF links on Wikisource pages
point partly to other sites and are recorded as citations, not fetched -- except, with
--wikisource-pdfs, those on the Knesset file server.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from urllib.parse import quote, urlsplit

from docslides.legal_data import wikisource
from docslides.legal_data.cli import RunContext
from docslides.legal_data.hebrew import repair_text
from docslides.legal_data.http import FetchError
from docslides.legal_data.law_names import NameIndex, has_prefix, normalize_name
from docslides.legal_data.pdf_quality import extract_pdf_with_quality
from docslides.legal_data.progress import Progress
from docslides.legal_data.records import CorpusRecord, JsonlWriter, Quality, now_iso, read_jsonl

LAW_LEVELS = ("basic_law", "law", "ordinance")
_SAMPLE_MAX_BYTES = 300 * 1024 * 1024  # a sample run reads at most this much of the dump
_PDF_ESTIMATE_PROBES = 20


def classify_title(title: str) -> str:
    name = normalize_name(title)
    if re.match(r"חוק[- ]?יסוד", name):
        return "basic_law"
    if re.match(r"(?:פקודת|פקודה)(?![א-ת])", name):
        return "ordinance"
    if re.match(r"(?:חוק|חוקת)(?![א-ת])", name):
        return "law"
    return "regulation"


def category_for(level: str) -> str:
    return "laws" if level in LAW_LEVELS else "procedural_rules"


def _get(row: dict, *names: str):
    return next((row[n] for n in names if row.get(n) is not None), None)


def _iso_date(value) -> str | None:
    if not value:
        return None
    return str(value)[:10]


def law_status(validity: str | None) -> str:
    text = (validity or "").strip()
    if not text:
        return "unknown"
    if re.search(r"בטל|פקע|לא ב?תוקף|הוחלף|מבוטל", text):
        return "repealed"
    if text.startswith("תקף") or text == "בתוקף":
        return "in_force"
    return "unknown"


@dataclass
class Registry:
    laws: dict[int, dict] = field(default_factory=dict)
    law_names: NameIndex = field(default_factory=NameIndex)
    secondary: dict[int, dict] = field(default_factory=dict)
    secondary_names: NameIndex = field(default_factory=NameIndex)
    authorizing: dict[int, list[int]] = field(default_factory=dict)
    law_documents: dict[int, list[dict]] = field(default_factory=dict)
    secondary_documents: dict[int, list[dict]] = field(default_factory=dict)
    loaded_tables: list[str] = field(default_factory=list)


def metadata_dir(ctx: RunContext) -> Path:
    """The Knesset tables: this run's own (a sample run's), else the main run's."""
    for root in (ctx.root, ctx.main_root):
        directory = root / "metadata" / "knesset"
        if (directory / "KNS_IsraelLaw.jsonl").exists():
            return directory
    return ctx.main_root / "metadata" / "knesset"


def load_registry(directory: Path) -> Registry:
    registry = Registry()

    def rows(table: str):
        path = directory / f"{table}.jsonl"
        if not path.exists():
            return []
        registry.loaded_tables.append(table)
        return read_jsonl(path)

    for row in rows("KNS_IsraelLaw"):
        registry.laws[row["Id"]] = row
        registry.law_names.add(row.get("Name"), row["Id"])
    for row in rows("KNS_IsraelLawName"):
        law_id = _get(row, "IsraelLawID", "IsraelLawId")
        if law_id:
            registry.law_names.add(row.get("Name"), law_id)
    for row in rows("KNS_SecondaryLaw"):
        registry.secondary[row["Id"]] = row
        registry.secondary_names.add(row.get("Name"), row["Id"])
    for row in rows("KNS_SecLawAuthorizingLaw"):
        secondary, authorizing = _get(row, "SecondaryLawID", "SecondaryLawId"), _get(row, "AuthorizingLawID", "AuthorizingLawId")
        if secondary and authorizing:
            registry.authorizing.setdefault(secondary, []).append(authorizing)
    for row in rows("KNS_DocumentIsraelLaw"):
        law_id = _get(row, "IsraelLawID", "IsraelLawId")
        if law_id:
            registry.law_documents.setdefault(law_id, []).append(row)
    for row in rows("KNS_DocumentSecondaryLaw"):
        secondary = _get(row, "SecondaryLawId", "SecondaryLawID")
        if secondary:
            registry.secondary_documents.setdefault(secondary, []).append(row)
    return registry


# --- PDFs ------------------------------------------------------------------------------------


def knesset_file_url(file_path: str, host: str) -> str | None:
    """A registry FilePath ('https://fs.knesset.gov.il/\\20\\SecondaryLaw\\20_scl_bg_491805.pdf') as a
    clean URL, or None if it isn't a PDF on the Knesset file server."""
    if not file_path:
        return None
    url = file_path.strip().replace("\\", "/")
    parts = urlsplit(url)
    if parts.hostname != host or not parts.path.lower().endswith(".pdf"):
        return None
    path = re.sub(r"/{2,}", "/", parts.path)
    return f"https://{host}{quote(path, safe='/%')}"


@dataclass
class PdfJob:
    url: str
    dest: Path
    owner_id: int
    group_type: str | None


def download_pdfs(ctx: RunContext, jobs: list[PdfJob], label: str) -> dict[int, list[dict]]:
    """owner id -> [{url, path, sha256, group_type}] for the files now on disk."""
    by_owner: dict[int, list[dict]] = {}
    if ctx.dry_run:
        estimate_pdfs(ctx, jobs, label)
        return by_owner
    progress = Progress(f"{label} PDFs", total=len(jobs), unit="files", log=ctx.log)
    outcomes: Counter = Counter()
    for job in jobs:
        try:
            result = ctx.client.download(job.url, job.dest, ctx.ledger)
        except FetchError as exc:
            ctx.manifest.warn(f"{label} PDF {job.url}: {exc}")
            ctx.manifest.count(f"{label}_pdfs_failed")
            outcomes["failed"] += 1
            progress.update(1, **outcomes)
            continue
        ctx.manifest.file(job.url, result.path, result.bytes, result.sha256, result.status)
        ctx.manifest.count(f"{label}_pdfs_{result.status}")
        by_owner.setdefault(job.owner_id, []).append({
            "url": job.url, "path": str(result.path.relative_to(ctx.main_root)).replace("\\", "/"),
            "sha256": result.sha256, "group_type": job.group_type,
        })
        outcomes[result.status] += 1
        progress.update(1, **outcomes)
    progress.done(**outcomes)
    return by_owner


def estimate_pdfs(ctx: RunContext, jobs: list[PdfJob], label: str) -> None:
    present = sum(job.dest.exists() for job in jobs)
    probes = random.Random(0).sample(jobs, min(_PDF_ESTIMATE_PROBES, len(jobs)))
    sizes = []
    for job in probes:
        response = ctx.client.head(job.url)
        if response is not None and response.headers.get("content-length", "").isdigit():
            sizes.append(int(response.headers["content-length"]))
    average = sum(sizes) / len(sizes) if sizes else 0
    estimate = {"pdfs": len(jobs), "already_present": present, "sampled_sizes": len(sizes),
                "estimated_total_mb": round(average * (len(jobs) - present) / 1e6, 1),
                "estimated_hours_at_rate_limit": round((len(jobs) - present) * ctx.cfg.min_interval_s / 3600, 1)}
    ctx.manifest.set("estimates", f"{label}_pdfs", estimate)
    ctx.log(f"{label} PDFs (dry run): {estimate}")


# --- records ------------------------------------------------------------------------------------


def build_record(page: wikisource.WikiPage, parsed: wikisource.ParsedLaw, level: str, registry: Registry,
                 ctx: RunContext, dump: wikisource.DumpFile, pdfs: dict[int, list[dict]]) -> CorpusRecord:
    text, encoding, notes = repair_text(parsed.text)
    if not parsed.parsed:
        notes.append("no ח:סעיף sections recognized: kept as plain text")
    if parsed.unknown_templates:
        notes.append("unrecognized templates: " + ", ".join(f"{k}×{v}" for k, v in parsed.unknown_templates.most_common(8)))
    if parsed.parse_errors:
        notes.append(f"{parsed.parse_errors} line(s) could not be parsed and were kept as plain text")
    record = CorpusRecord(
        id=f"wikisource:{page.page_id}",
        source=f"he.wikisource.org Open Law Book (dump {dump.date})",
        source_url=wikisource.page_url(ctx.cfg.sources.wikisource_wiki, page.title, page.rev_id),
        license=wikisource.LICENSE,
        attribution=wikisource.ATTRIBUTION,
        category=category_for(level),
        title=parsed.full_title,
        authority_level=level,
        retrieved_at=now_iso(),
        text=text,
        quality=Quality(encoding=encoding, extraction="n/a", notes=notes),
        sections=parsed.sections,
        gazette_citations=parsed.citations,
        wikisource_title=page.title,
        wikisource_pageid=page.page_id,
        wikisource_revid=page.rev_id,
        wikisource_timestamp=page.timestamp,
    )
    if level in LAW_LEVELS:
        _join_law(record, parsed, registry, pdfs)
    else:
        _join_regulation(record, registry, pdfs)
    if record.status == "unknown" and parsed.repeal_marker:
        record.status, record.status_source = "repealed", "wikisource_page"
    if not record.effective_date:
        year = re.search(r"(?<!\d)(1[89]\d\d|20\d\d)(?!\d)", normalize_name(record.title))
        if year:
            record.effective_date = f"{year.group(1)}-01-01"
            record.quality.notes.append("effective_date approximated from the year in the title")
    return record.finalize()


def _join_law(record: CorpusRecord, parsed: wikisource.ParsedLaw, registry: Registry, pdfs: dict) -> None:
    law_id, source = parsed.registry_id, "wikisource_registry_id"
    if law_id is None or (registry.laws and law_id not in registry.laws):
        matched = registry.law_names.lookup(record.title)
        if matched is not None:
            law_id, source = matched, "name_match"
    record.law_id, record.law_id_source = law_id, source if law_id is not None else None
    row = registry.laws.get(law_id) if law_id is not None else None
    if row is None:
        if law_id is not None and registry.laws:
            record.quality.notes.append(f"registry id {law_id} is not in KNS_IsraelLaw")
        return
    record.knesset_name = row.get("Name")
    record.is_basic_law = row.get("IsBasicLaw")
    if record.is_basic_law:
        record.authority_level = "basic_law"
    record.status, record.status_source = law_status(row.get("LawValidityDesc")), "odata"
    record.publication_date = _iso_date(row.get("PublicationDate"))
    record.latest_amendment_date = _iso_date(row.get("LatestPublicationDate"))
    record.effective_date = _iso_date(row.get("ValidityStartDate")) or record.publication_date
    record.pdf_files = pdfs.get(law_id, [])


def _join_regulation(record: CorpusRecord, registry: Registry, pdfs: dict) -> None:
    secondary_id = registry.secondary_names.lookup(record.title)
    if secondary_id is None:
        return
    row = registry.secondary[secondary_id]
    record.law_id, record.law_id_source = secondary_id, "name_match"
    record.knesset_name = row.get("Name")
    record.authorizing_law_ids = sorted(set(registry.authorizing.get(secondary_id, [])))
    record.publication_date = _iso_date(row.get("PublicationDate"))
    record.effective_date = record.publication_date
    record.pdf_files = pdfs.get(secondary_id, [])


def pdf_record(ctx: RunContext, secondary_id: int, row: dict, files: list[dict]) -> CorpusRecord | None:
    """A regulation Wikisource doesn't have, from its registry PDFs (configured group types only)."""
    texts, qualities = [], []
    for info in files:
        text, quality = extract_pdf_with_quality(ctx.main_root / info["path"])
        if text.strip():
            texts.append(text)
            qualities.append(quality)
    if not texts:
        return None
    low = any(q.extraction == "low" for q in qualities)
    flagged = any(q.encoding == "flagged" for q in qualities)
    repaired = any(q.encoding == "repaired" for q in qualities)
    record = CorpusRecord(
        id=f"knesset-secondary:{secondary_id}",
        source="Knesset OData KNS_SecondaryLaw + KNS_DocumentSecondaryLaw PDF",
        source_url=files[0]["url"],
        license="Knesset open data (terms per knesset.gov.il)",
        attribution="הכנסת -- מאגר החקיקה (OData)",
        category="procedural_rules",
        title=row.get("Name") or f"KNS_SecondaryLaw {secondary_id}",
        authority_level="regulation",
        retrieved_at=now_iso(),
        text="\n\n".join(texts),
        quality=Quality(encoding="flagged" if flagged else "repaired" if repaired else "ok",
                        extraction="low" if low else "ok",
                        notes=["text extracted from the registry PDF(s)"] + [n for q in qualities for n in q.notes]),
        law_id=secondary_id,
        law_id_source="odata",
        knesset_name=row.get("Name"),
        publication_date=_iso_date(row.get("PublicationDate")),
        effective_date=_iso_date(row.get("PublicationDate")),
        pdf_files=files,
    )
    return record.finalize()


# --- the run ---------------------------------------------------------------------------------------


def run_legislation(ctx: RunContext, category: str) -> None:
    manifest, sources = ctx.manifest, ctx.cfg.sources
    dump = wikisource.resolve_dump(ctx.client, sources.wikisource_dumps, sources.wikisource_dump_date)
    manifest.source("Hebrew Wikisource dump (Open Law Book pages)", dump.url, wikisource.LICENSE,
                    wikisource.ATTRIBUTION, dump_date=dump.date, sha1=dump.sha1, bytes=dump.size)
    dump_path = ctx.main_root / "laws" / "raw" / "wikisource" / dump.name
    registry = load_registry(metadata_dir(ctx))
    if not registry.laws:
        manifest.warn("no Knesset registry tables found -- run fetch_metadata.py first; records won't be joined")
    manifest.set("registry", "tables_loaded", registry.loaded_tables)

    pdf_jobs = _pdf_jobs(ctx, category, registry)
    if ctx.dry_run:
        present = dump_path.exists()
        manifest.set("estimates", "dump", {"file": dump.name, "mb": round(dump.size / 1e6, 1), "already_present": present})
        ctx.log(f"dump {dump.name}: {dump.size / 1e6:.0f} MB{' (already downloaded)' if present else ''}")
        download_pdfs(ctx, pdf_jobs, category)
        return

    pdfs = download_pdfs(ctx, pdf_jobs, category)
    if ctx.sample:
        ctx.log(f"sample: streaming the start of {dump.name} (at most {_SAMPLE_MAX_BYTES // 2**20} MB)")
        chunks = _capped(ctx.client.stream(dump.url), _SAMPLE_MAX_BYTES)
    else:
        result = ctx.client.download(dump.url, dump_path, ctx.ledger, expected_sha1=dump.sha1,
                                     progress_label=f"download {dump.name}")
        manifest.file(dump.url, result.path, result.bytes, result.sha256, result.status)
        ctx.log(f"dump {result.status}: {dump_path}")
        chunks = wikisource.file_chunks(dump_path)
    total = _SAMPLE_MAX_BYTES if ctx.sample else dump_path.stat().st_size
    progress = Progress(f"{category}: reading {dump.name}", total=total, unit="bytes", log=ctx.log)
    reader = _counted(chunks, progress)

    writer = JsonlWriter(ctx.root / category, category)
    counts: Counter = Counter()
    joined: set[int] = set()
    wikisource_pdf_urls: list[tuple[int, str]] = []
    try:
        for page in wikisource.iter_pages(reader):
            counts["dump_pages"] += 1
            progress.extra.update(pages=counts["dump_pages"], law_book_pages=counts["open_law_book_pages"],
                                  records=writer.count)
            if not wikisource.is_law_book_page(page):
                continue
            counts["open_law_book_pages"] += 1
            parsed = wikisource.parse_law_page(page.text, page.title)
            level = classify_title(parsed.full_title)
            if category_for(level) != category:
                continue
            if ctx.sample and parsed.registry_id and parsed.registry_id not in registry.laws:
                _lookup_registry_law(ctx, registry, parsed.registry_id)
                counts["sample_registry_lookups"] += 1
            record = build_record(page, parsed, level, registry, ctx, dump, pdfs)
            writer.write(record)
            counts[f"level_{record.authority_level}"] += 1
            counts[f"status_{record.status}"] += 1
            counts[f"joined_{record.law_id_source or 'none'}"] += 1
            counts["parsed_sections" if parsed.parsed else "plain_text_only"] += 1
            counts[f"encoding_{record.quality.encoding}"] += 1
            if record.law_id is not None:
                joined.add(record.law_id)
            if getattr(ctx.args, "wikisource_pdfs", False):
                wikisource_pdf_urls += [(page.page_id, c["url"]) for c in parsed.citations
                                        if urlsplit(c.get("url") or "").hostname == sources.knesset_pdf_host]
            if ctx.sample and writer.count >= ctx.sample:
                break
        if category == "procedural_rules":
            _pdf_only_regulations(ctx, registry, joined, pdfs, writer, counts)
    except BaseException:
        writer.close(commit=False)
        raise
    finally:
        reader.close()
        chunks.close()  # a sample run stops mid-dump: release the HTTP stream now
    progress.done(pages=counts["dump_pages"], law_book_pages=counts["open_law_book_pages"], records=writer.count)
    paths = writer.close()
    for key, value in counts.items():
        manifest.count(key, value)
    manifest.count("records_written", writer.count)
    ctx.log(f"{category}: {writer.count} records -> {', '.join(str(p) for p in paths)}")

    if wikisource_pdf_urls:
        jobs = [PdfJob(url, ctx.root / "laws" / "raw" / "pdf" / "wikisource" / Path(urlsplit(url).path).name,
                       page_id, "wikisource citation") for page_id, url in dict.fromkeys(wikisource_pdf_urls)]
        download_pdfs(ctx, jobs, "laws_wikisource")
    if category == "laws":
        _report_unmatched_laws(ctx, registry, joined)
    else:
        _coverage_report(ctx, paths)


def _lookup_registry_law(ctx: RunContext, registry: Registry, law_id: int) -> None:
    """Sample runs only: one KNS_IsraelLaw row by id, so a sample's joins mean something without
    the full registry (fetch_metadata.py --sample holds just N rows per table)."""
    from docslides.legal_data.knesset_odata import KnessetOData

    odata = KnessetOData(ctx.client, ctx.cfg.sources.knesset_odata)
    for row in ctx.client.get_json(odata.url("KNS_IsraelLaw", {"$filter": f"Id eq {int(law_id)}"})).get("value", []):
        registry.laws[row["Id"]] = row
        registry.law_names.add(row.get("Name"), row["Id"])


def _counted(chunks, progress: Progress):
    for chunk in chunks:
        progress.update(len(chunk))
        yield chunk


def _capped(chunks, limit: int):
    read = 0
    for chunk in chunks:
        yield chunk
        read += len(chunk)
        if read >= limit:
            return


def _pdf_jobs(ctx: RunContext, category: str, registry: Registry) -> list[PdfJob]:
    choice = getattr(ctx.args, "pdfs", "all")
    if choice == "none":
        return []
    host = ctx.cfg.sources.knesset_pdf_host
    documents = registry.law_documents if category == "laws" else registry.secondary_documents
    folder = ctx.root / category / "raw" / "pdf"
    jobs = []
    for owner, rows in sorted(documents.items()):
        for row in rows:
            url = knesset_file_url(row.get("FilePath") or "", host)
            if url:
                jobs.append(PdfJob(url, folder / str(owner) / f"{row['Id']}.pdf", owner, row.get("GroupTypeDesc")))
    if category == "laws" and not registry.law_documents and "KNS_DocumentIsraelLaw" in registry.loaded_tables:
        ctx.manifest.warn("KNS_DocumentIsraelLaw is empty in OData V4: no registry PDFs for laws "
                          "(the law text comes from Wikisource; --wikisource-pdfs fetches the fs.knesset.gov.il "
                          "PDFs the pages cite)")
    groups = Counter(job.group_type for job in jobs)
    ctx.manifest.set("pdf_group_types", category, dict(groups))
    if ctx.sample:
        jobs = jobs[: ctx.sample]
    return jobs


def _pdf_only_regulations(ctx, registry: Registry, joined: set[int], pdfs: dict, writer: JsonlWriter,
                          counts: Counter) -> None:
    wanted = set(ctx.cfg.secondary_text_group_types)
    if not wanted:
        return
    for secondary_id, files in pdfs.items():
        if secondary_id in joined or secondary_id not in registry.secondary:
            continue
        files = [f for f in files if f.get("group_type") in wanted]
        record = pdf_record(ctx, secondary_id, registry.secondary[secondary_id], files) if files else None
        if record:
            writer.write(record)
            counts["pdf_only_regulations"] += 1
            counts[f"pdf_extraction_{record.quality.extraction}"] += 1


def _report_unmatched_laws(ctx: RunContext, registry: Registry, joined: set[int]) -> None:
    if not registry.laws or ctx.sample:
        return
    missing = [{"id": law_id, "name": row.get("Name"), "validity": row.get("LawValidityDesc")}
               for law_id, row in sorted(registry.laws.items()) if law_id not in joined]
    ctx.manifest.set("counts", "registry_laws", len(registry.laws))
    ctx.manifest.set("counts", "registry_laws_without_wikisource_page", len(missing))
    path = ctx.root / "laws" / "unmatched_registry_laws.json"
    path.write_text(json.dumps(missing, ensure_ascii=False, indent=1), encoding="utf-8")
    ctx.log(f"laws: {len(joined)} of {len(registry.laws)} registry laws matched; unmatched list: {path}")


def _coverage_report(ctx: RunContext, paths: list[Path]) -> None:
    titles = [record["title"] for path in paths for record in read_jsonl(path)]
    report = {"sample_run": bool(ctx.sample), "generated": date.today().isoformat(), "required": {}}
    for label, prefixes in ctx.cfg.required_regulations.items():
        found = sorted({t for t in titles for p in prefixes if has_prefix(t, p)})
        report["required"][label] = {"prefixes": prefixes, "found": found}
        if not found:
            ctx.manifest.warn(f"required regulations missing: {label} ({'; '.join(prefixes)})"
                              + (" -- expected in a sample run" if ctx.sample else ""))
        else:
            ctx.log(f"required {label}: {len(found)} found (e.g. {found[0]})")
    path = ctx.root / "procedural_rules" / "coverage_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    ctx.manifest.set("filters", "coverage", {k: len(v["found"]) for k, v in report["required"].items()})
