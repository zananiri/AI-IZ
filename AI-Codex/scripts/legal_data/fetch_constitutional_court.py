#!/usr/bin/env python3
"""Tribunal Constitucional case law (sentencias by default; autos and declaraciones on request) from
the court's public system hj.tribunalconstitucional.es, with private parties' names anonymized.

    python scripts/legal_data/fetch_constitutional_court.py --dry-run          # newest id, pages to fetch, hours
    python scripts/legal_data/fetch_constitutional_court.py --sample 50        # the 50 newest resolutions -> legal_txt/_sample/
    python scripts/legal_data/fetch_constitutional_court.py                    # everything not fetched yet (resumable)
    python scripts/legal_data/fetch_constitutional_court.py --doc-types SENTENCIA AUTO
    python scripts/legal_data/fetch_constitutional_court.py --from-id 25000    # a range of ids

A full first run is ~32,000 pages (~9 h at one request per second); it can be interrupted and
resumed at any point, and later runs only fetch new resolutions. Output:
legal_txt/constitutional_court/constitutional_court-00001.jsonl ... Rules: src/aicodex/legal_data/tc.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aicodex.legal_data import cli
from aicodex.legal_data.tc import run_tc


def main() -> int:
    parser = cli.base_parser(__doc__)
    parser.add_argument("--doc-types", nargs="*", help="SENTENCIA, AUTO, DECLARACIÓN (default: config)")
    parser.add_argument("--from-id", type=int, help="first HJ id to scan (default: config first_id)")
    parser.add_argument("--to-id", type=int, help="last HJ id to scan (default: the newest)")
    parser.add_argument("--force", action="store_true", help="re-fetch ids already cached")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--anonymize", dest="anonymize", action="store_true", default=None,
                       help="replace private parties' names with [persona N] (default: config, on)")
    group.add_argument("--no-anonymize", dest="anonymize", action="store_false")
    args = parser.parse_args()
    return cli.run("fetch_constitutional_court", args, run_tc)


if __name__ == "__main__":
    raise SystemExit(main())
