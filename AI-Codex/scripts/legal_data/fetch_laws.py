#!/usr/bin/env python3
"""Spanish legislation from the BOE open-data API (consolidated texts, current wording, article by
article): state laws, state regulations and the autonomous communities' law.

    python scripts/legal_data/fetch_laws.py --dry-run                     # catalogue counts, requests, hours
    python scripts/legal_data/fetch_laws.py --sample 20                   # 20 acts per category -> legal_txt/_sample/
    python scripts/legal_data/fetch_laws.py                               # laws + regulations + regional_laws
    python scripts/legal_data/fetch_laws.py --categories laws             # only rank-of-law norms (~1,200 acts)
    python scripts/legal_data/fetch_laws.py --ids BOE-A-1889-4763 BOE-A-2000-323    # specific acts
    python scripts/legal_data/fetch_laws.py --since 20260101              # only acts updated since a date

Output: legal_txt/<category>/<category>.jsonl (sharded per 5,000 records) and legal_txt/boe/cache/
(one normalized record per act, keyed by the catalogue's fecha_actualizacion: a re-run fetches only
new and updated acts). Rules and parsing: src/aicodex/legal_data/boe.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aicodex.legal_data import cli
from aicodex.legal_data.boe import run_boe

BOE_CATEGORIES = ("laws", "regulations", "regional_laws")


def main() -> int:
    parser = cli.base_parser(__doc__)
    parser.add_argument("--categories", default=",".join(BOE_CATEGORIES),
                        help=f"comma-separated subset of {', '.join(BOE_CATEGORIES)}")
    parser.add_argument("--ids", nargs="*", help="only these BOE identifiers (BOE-A-...)")
    parser.add_argument("--since", help="only acts whose consolidated text was updated on/after YYYYMMDD")
    parser.add_argument("--exclude-repealed", dest="include_repealed", action="store_false",
                        help="leave out acts whose vigencia is agotada (default: kept, status 'repealed')")
    parser.add_argument("--with-subjects", action="store_true", help="also fetch each act's materias (1 more request each)")
    parser.add_argument("--force", action="store_true", help="re-fetch acts even when the cache is current")
    args = parser.parse_args()
    args.categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    unknown = set(args.categories) - set(BOE_CATEGORIES)
    if unknown:
        parser.error(f"unknown categories: {', '.join(sorted(unknown))}")
    args.ids = set(args.ids or [])
    return cli.run("fetch_laws", args, run_boe)


if __name__ == "__main__":
    raise SystemExit(main())
