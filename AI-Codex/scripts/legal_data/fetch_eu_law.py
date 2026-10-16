#!/usr/bin/env python3
"""EU law in Spanish (regulations, directives, the Charter and Treaties) from the Publications
Office's CELLAR, newest consolidated version when there is one.

    python scripts/legal_data/fetch_eu_law.py --dry-run
    python scripts/legal_data/fetch_eu_law.py                          # the acts in legal_data.eu_law.celex
    python scripts/legal_data/fetch_eu_law.py --celex 32016R0679 32024R1689

Output: legal_txt/eu_law/eu_law.jsonl. Rules: src/aicodex/legal_data/eurlex.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aicodex.legal_data import cli
from aicodex.legal_data.eurlex import run_eurlex


def main() -> int:
    parser = cli.base_parser(__doc__)
    parser.add_argument("--celex", nargs="*", help="CELEX numbers (default: legal_data.eu_law.celex)")
    args = parser.parse_args()
    return cli.run("fetch_eu_law", args, run_eurlex)


if __name__ == "__main__":
    raise SystemExit(main())
