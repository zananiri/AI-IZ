#!/usr/bin/env python3
"""Registry metadata: Knesset OData tables and ISCD metadata files, stored as-is.

    python scripts/legal_data/fetch_metadata.py --dry-run          # row counts, ISCD file list + sizes
    python scripts/legal_data/fetch_metadata.py --sample 20        # 20 rows per table -> legal_txt/_sample/
    python scripts/legal_data/fetch_metadata.py                    # delta since the last run (LastUpdatedDate)
    python scripts/legal_data/fetch_metadata.py --full             # every row again
    python scripts/legal_data/fetch_metadata.py --only iscd --import-dir <folder>   # ISCD files you downloaded

Knesset: each table in legal_data.sources.knesset_tables -> legal_txt/metadata/knesset/<Table>.jsonl,
upserted by Id; per-table LastUpdatedDate watermarks in _state.json. Table names are checked against
the OData V4 service each run: KNS_DocumentLaw is fetched as KNS_DocumentIsraelLaw, and KNS_Law (not
in V4) is reported and skipped. Rows deleted at the source aren't seen by a delta run -- use --full.

ISCD: the metadata files linked from iscd.huji.ac.il/data (data files and codebooks) -> legal_txt/
metadata/iscd/, as-is. Only files on iscd.huji.ac.il itself are fetched.

Run first: fetch_laws.py and fetch_procedural_rules.py join to these tables.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from docslides.legal_data import cli
from docslides.legal_data.http import file_sha256
from docslides.legal_data.knesset_odata import KnessetOData
from docslides.legal_data.records import now_iso, read_jsonl

KNESSET_LICENSE = "Knesset open parliamentary data (OData); terms of use per knesset.gov.il"
ISCD_EXTENSIONS = (".csv", ".xlsx", ".xls", ".zip", ".sav", ".dta", ".rds", ".rdata", ".json", ".pdf", ".docx", ".doc")
ISCD_HOST = "iscd.huji.ac.il"


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _later(a: str | None, b: str | None) -> str | None:
    if not a or not b:
        return a or b
    return a if _parse_time(a) >= _parse_time(b) else b


def knesset(ctx: cli.RunContext) -> None:
    sources = ctx.cfg.sources
    odata = KnessetOData(ctx.client, sources.knesset_odata)
    ctx.manifest.source("Knesset OData V4 (ParliamentInfo)", sources.knesset_odata, KNESSET_LICENSE,
                        "הכנסת -- מאגר המידע הפרלמנטרי")
    out = ctx.root / "metadata" / "knesset"
    state_path = out / "_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    available = set(odata.entity_sets())
    for requested in sources.knesset_tables:
        table = sources.knesset_table_aliases.get(requested, requested)
        if table not in available:
            ctx.manifest.warn(f"{requested}: not a table of the OData V4 service -- skipped")
            ctx.manifest.set("tables", requested, {"status": "unavailable"})
            continue
        since = None if (ctx.args.full or ctx.sample) else state.get(table, {}).get("watermark")
        if ctx.dry_run:
            total = odata.count(table)
            pending = odata.count(table, since) if since else total
            ctx.manifest.set("tables", requested, {"table": table, "rows": total, "to_fetch": pending, "since": since})
            ctx.log(f"{table}: {total} rows, {pending} to fetch{f' (since {since})' if since else ''}")
            continue
        path = out / f"{table}.jsonl"
        rows = {} if (ctx.args.full or ctx.sample or not path.exists()) else {r["Id"]: r for r in read_jsonl(path)}
        fetched, watermark = 0, since
        for row in odata.rows(table, since=since, top=ctx.sample):
            rows[row["Id"]] = row
            fetched += 1
            watermark = _later(watermark, row.get("LastUpdatedDate"))
            if fetched % 5000 == 0:
                ctx.log(f"{table}: {fetched} rows")
        out.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for key in sorted(rows, key=lambda k: (isinstance(k, str), k)):
                f.write(json.dumps(rows[key], ensure_ascii=False) + "\n")
        tmp.replace(path)
        state[table] = {"watermark": watermark, "rows": len(rows), "fetched_at": now_iso(), "requested_as": requested}
        ctx.manifest.set("tables", requested, {"table": table, "fetched": fetched, "rows_stored": len(rows),
                                               "watermark": watermark, "file": str(path)})
        ctx.manifest.count("odata_rows_fetched", fetched)
        ctx.log(f"{table}: fetched {fetched}, stored {len(rows)} -> {path}")
    if not ctx.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def _safe_name(url: str) -> str:
    name = unquote(Path(urlsplit(url).path).name) or "index"
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)


def iscd(ctx: cli.RunContext) -> None:
    out = ctx.root / "metadata" / "iscd"
    page_url = ctx.cfg.sources.iscd_data_page
    if ctx.args.import_dir:
        source = Path(ctx.args.import_dir)
        ctx.manifest.source("ISCD metadata (manual download)", page_url, "see iscd.huji.ac.il/data",
                            "The Israeli Supreme Court Database (ISCD), Hebrew University")
        for path in sorted(p for p in source.iterdir() if p.is_file()):
            if ctx.dry_run:
                ctx.log(f"would import {path.name} ({path.stat().st_size} bytes)")
                continue
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out / path.name)
            ctx.manifest.file(f"manual-import:{path.name}", out / path.name, path.stat().st_size,
                              file_sha256(out / path.name), "imported")
        return
    html = ctx.client.get(page_url).text
    terms = re.findall(r"[^<>\n]{0,160}(?:license|licence|terms|cite|citation|רישיון|תנאי שימוש|ציטוט)[^<>\n]{0,160}",
                       html, flags=re.IGNORECASE)
    ctx.manifest.source("ISCD metadata files", page_url, "see the terms on iscd.huji.ac.il/data",
                        "The Israeli Supreme Court Database (ISCD), Hebrew University", page_terms=terms[:5])
    links = []
    for href in re.findall(r'href\s*=\s*["\']([^"\']+)["\']', html, flags=re.IGNORECASE):
        url = urljoin(page_url, href)
        host = urlsplit(url).hostname or ""
        if (host == ISCD_HOST or host.endswith("." + ISCD_HOST)) and urlsplit(url).path.lower().endswith(ISCD_EXTENSIONS):
            links.append(url)
    links = list(dict.fromkeys(links))
    ctx.manifest.set("counts", "iscd_files_listed", len(links))
    if not links:
        ctx.manifest.warn("no metadata files linked from the ISCD data page -- download them manually and use --import-dir")
        return
    for url in links[: ctx.sample] if ctx.sample else links:
        if ctx.dry_run:
            head = ctx.client.head(url)
            size = head.headers.get("content-length") if head is not None else None
            ctx.log(f"would download {url} ({size or '?'} bytes)")
            continue
        result = ctx.client.download(url, out / _safe_name(url), ctx.ledger)
        ctx.manifest.file(url, result.path, result.bytes, result.sha256, result.status)
        ctx.log(f"ISCD {result.status}: {result.path.name}")


def main() -> int:
    parser = cli.base_parser(__doc__)
    parser.add_argument("--only", choices=["knesset", "iscd"], help="fetch one source only")
    parser.add_argument("--full", action="store_true", help="ignore the watermarks: fetch every row again")
    parser.add_argument("--import-dir", help="ISCD files downloaded by hand: copy them into metadata/iscd as-is")
    args = parser.parse_args()

    def body(ctx: cli.RunContext) -> None:
        if args.only != "iscd":
            knesset(ctx)
        if args.only != "knesset":
            iscd(ctx)

    return cli.run("fetch_metadata", args, body)


if __name__ == "__main__":
    raise SystemExit(main())
