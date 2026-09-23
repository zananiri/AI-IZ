#!/usr/bin/env python3
"""Index law PDFs dropped into ./legal_txt (config.legal.ingestion.legal_txt_dir).

Each new or changed file is parsed, chunked, embedded and signed into the
Legal tab's index in one step (official source, auto-approved). An older
version of the same law is superseded automatically: its end date is set
and its status becomes "amended". PDFs without a <file>.meta.json get one
derived from the title; check it, edit it and re-run to correct. See
src/docslides/legal/folder_ingest.py for the rules.

    python scripts/ingest_legal_txt.py --dry-run     # preview metadata + chunking, write nothing
    python scripts/ingest_legal_txt.py               # index new/changed files
    python scripts/ingest_legal_txt.py --watch       # keep watching the folder (polls)
    python scripts/ingest_legal_txt.py --prune       # also retract laws whose PDF was deleted

Needs DOCSLIDES_LEGAL_BUNDLE_KEY (the key the index is signed with -- the API
process needs the same value) and the `legal` extra: pip install -e ".[legal]"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from docslides.legal import bundle, folder_ingest


def _report(results: list[folder_ingest.FileResult], verbose_unchanged: bool) -> int:
    failures = 0
    for r in results:
        if r.action == "unchanged" and not verbose_unchanged:
            continue
        line = f"[{r.action}] {r.path.name}"
        if r.law_name:
            line += f" -- {r.law_name}"
        if r.chunk_count:
            line += f" ({r.chunk_count} chunks)"
        print(line)
        if r.message:
            print(f"    {r.message}")
        if r.derived_fields:
            print(f"    guessed metadata: {', '.join(r.derived_fields)} -- check {r.path.name}.meta.json")
        for change in r.superseded:
            print(f"    superseded: {change}")
        failures += r.action == "failed"
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="derive metadata and chunk, but write nothing")
    parser.add_argument("--prune", action="store_true", help="retract laws whose file was removed from the folder")
    parser.add_argument("--watch", action="store_true", help="keep polling the folder")
    parser.add_argument("--interval", type=int, default=60, help="seconds between polls with --watch")
    args = parser.parse_args()

    try:
        while True:
            results = folder_ingest.run(dry_run=args.dry_run, prune=args.prune)
            failures = _report(results, verbose_unchanged=not args.watch)
            if not args.watch:
                if not results:
                    print("No files in legal_txt/.")
                return 1 if failures else 0
            time.sleep(args.interval)
    except bundle.BundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
