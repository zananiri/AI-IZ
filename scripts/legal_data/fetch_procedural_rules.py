#!/usr/bin/env python3
"""Procedural rules and all other secondary legislation (תקנות, צווים, כללים, הודעות ...) from the
Open Law Book, joined to KNS_SecondaryLaw / KNS_SecLawAuthorizingLaw, plus the registry PDFs from
KNS_DocumentSecondaryLaw.

    python scripts/legal_data/fetch_procedural_rules.py --dry-run     # dump size, PDF count + estimated size
    python scripts/legal_data/fetch_procedural_rules.py --sample 20   # -> legal_txt/_sample/procedural_rules/
    python scripts/legal_data/fetch_procedural_rules.py               # -> legal_txt/procedural_rules/procedural_rules.jsonl
    python scripts/legal_data/fetch_procedural_rules.py --pdfs none   # skip the registry PDFs

Writes procedural_rules/coverage_report.json and warns in the manifest if any of
legal_data.required_regulations is missing: the Civil Procedure Regulations 2018, the criminal
procedure regulations and the court-fee regulations.

Regulations Wikisource lacks get a record from their registry PDFs only for the PDF group types
listed in legal_data.secondary_text_group_types (empty by default: the registry mostly holds
committee background material, so pick the right group types from a sample run's manifest,
"pdf_group_types", first). Shares the dump with fetch_laws.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from docslides.legal_data import cli
from docslides.legal_data.legislation import run_legislation


def main() -> int:
    parser = cli.base_parser(__doc__)
    parser.add_argument("--pdfs", choices=["all", "none"], default="all", help="KNS_DocumentSecondaryLaw PDFs")
    args = parser.parse_args()
    return cli.run("fetch_procedural_rules", args, lambda ctx: run_legislation(ctx, "procedural_rules"))


if __name__ == "__main__":
    raise SystemExit(main())
