#!/usr/bin/env python3
"""Laws: Basic Laws, Knesset laws and ordinances (Mandate era included) from the Open Law Book
(Hebrew Wikisource, official Wikimedia dump), joined to the Knesset registry.

    python scripts/legal_data/fetch_laws.py --dry-run      # dump size, PDF count + estimated size
    python scripts/legal_data/fetch_laws.py --sample 20    # 20 laws from the start of the dump -> legal_txt/_sample/laws/
    python scripts/legal_data/fetch_laws.py                # the dump (sha1-verified, skipped if unchanged) -> legal_txt/laws/laws.jsonl
    python scripts/legal_data/fetch_laws.py --wikisource-pdfs   # also the fs.knesset.gov.il PDFs the pages cite

Needs legal_txt/metadata/knesset/ from fetch_metadata.py for law_id, status (in force / repealed),
Basic Law flag and dates. Also writes legal_txt/laws/unmatched_registry_laws.json: registry laws
with no Open Law Book page. Rules: src/docslides/legal_data/legislation.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from docslides.legal_data import cli
from docslides.legal_data.legislation import run_legislation


def main() -> int:
    parser = cli.base_parser(__doc__)
    parser.add_argument("--pdfs", choices=["all", "none"], default="all",
                        help="registry PDFs (KNS_DocumentIsraelLaw) -- empty in OData V4 as of Sept 2026")
    parser.add_argument("--wikisource-pdfs", action="store_true",
                        help="download the fs.knesset.gov.il PDFs the Wikisource pages cite (off by default)")
    args = parser.parse_args()
    return cli.run("fetch_laws", args, lambda ctx: run_legislation(ctx, "laws"))


if __name__ == "__main__":
    raise SystemExit(main())
