#!/usr/bin/env python3
"""Checks that the core Spanish codes and laws (legal_data.required_laws) are in the fetched corpus,
in force, and parsed into articles -- run after fetch_laws.py and before vectorize.py.

    python scripts/legal_data/check_corpus_coverage.py
    python scripts/legal_data/check_corpus_coverage.py --input legal_txt/_sample

Exit code 1 when a required law is missing, repealed or has no articles.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aicodex.config import get_config
from aicodex.legal_data.records import LEGISLATION_CATEGORIES, read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", help="corpus root (default: legal_data.output_dir)")
    args = parser.parse_args()
    cfg = get_config().legal_data
    root = Path(args.input or cfg.output_dir)
    found: dict[str, dict] = {}
    for category in LEGISLATION_CATEGORIES:
        for path in sorted((root / category).glob(f"{category}*.jsonl")):
            for record in read_jsonl(path):
                if record.get("law_id"):
                    found[record["law_id"]] = {"category": category, "status": record.get("status"),
                                               "articles": len(record.get("section_numbers") or []),
                                               "title": record.get("title", "")[:80]}
    problems = 0
    for label, boe_id in cfg.required_laws.items():
        info = found.get(boe_id)
        if info is None:
            print(f"MISSING   {label:10s} {boe_id}")
            problems += 1
        elif info["status"] == "repealed" or info["articles"] == 0:
            print(f"PROBLEM   {label:10s} {boe_id}: status {info['status']}, {info['articles']} articles -- {info['title']}")
            problems += 1
        else:
            print(f"ok        {label:10s} {boe_id}: {info['articles']} articles ({info['category']}) -- {info['title']}")
    print(f"{len(found):,} acts in {root}; {problems} problem(s) among {len(cfg.required_laws)} required laws")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
