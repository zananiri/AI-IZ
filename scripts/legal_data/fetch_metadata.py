#!/usr/bin/env python3
"""Registry metadata: the Knesset OData tables, stored as-is.

    python scripts/legal_data/fetch_metadata.py --dry-run          # row counts
    python scripts/legal_data/fetch_metadata.py --sample 20        # 20 rows per table -> legal_txt/_sample/
    python scripts/legal_data/fetch_metadata.py                    # delta since the last run (LastUpdatedDate)
    python scripts/legal_data/fetch_metadata.py --full             # every row again
    python scripts/legal_data/fetch_metadata.py --tables KNS_IsraelLaw,KNS_IsraelLawName,KNS_SecondaryLaw,KNS_SecLawAuthorizingLaw

Knesset: each table in legal_data.sources.knesset_tables -> legal_txt/metadata/knesset/<Table>.jsonl,
upserted by Id; per-table LastUpdatedDate watermarks in _state.json. Table names are checked against
the OData V4 service each run: KNS_DocumentLaw is fetched as KNS_DocumentIsraelLaw, and KNS_Law (not
in V4) is reported and skipped. Rows deleted at the source aren't seen by a delta run -- use --full.

Run first: fetch_laws.py and fetch_procedural_rules.py join to these tables.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from docslides.legal_data import cli
from docslides.legal_data.knesset_odata import KnessetOData
from docslides.legal_data.progress import Progress
from docslides.legal_data.records import now_iso, read_jsonl

KNESSET_LICENSE = "Knesset open parliamentary data (OData); terms of use per knesset.gov.il"


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
    wanted = [t.strip() for t in ctx.args.tables.split(",") if t.strip()] if ctx.args.tables else sources.knesset_tables
    for n, requested in enumerate(wanted, 1):
        ctx.log(f"table {n}/{len(wanted)}: {requested}")
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
        total = ctx.sample or odata.count(table, since)
        progress = Progress(table, total=total, unit="rows", log=ctx.log)
        for row in odata.rows(table, since=since, top=ctx.sample):
            rows[row["Id"]] = row
            fetched += 1
            watermark = _later(watermark, row.get("LastUpdatedDate"))
            progress.update(1)
        progress.done(stored=len(rows))
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


def main() -> int:
    parser = cli.base_parser(__doc__)
    parser.add_argument("--full", action="store_true", help="ignore the watermarks: fetch every row again")
    parser.add_argument("--tables", help="comma-separated subset of legal_data.sources.knesset_tables, e.g. "
                                         "KNS_IsraelLaw,KNS_IsraelLawName,KNS_SecondaryLaw,KNS_SecLawAuthorizingLaw "
                                         "(the tables fetch_laws.py / fetch_procedural_rules.py join to)")
    args = parser.parse_args()
    return cli.run("fetch_metadata", args, knesset)


if __name__ == "__main__":
    raise SystemExit(main())
