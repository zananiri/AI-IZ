#!/usr/bin/env python3
"""Every source, in order: BOE legislation, EU law, Tribunal Constitucional. Each step is the
script of the same name, run with the same --dry-run / --sample / --output flags; a step that is
stopped (403, robots.txt, host unavailable) stops the run with its exit code.

    python scripts/legal_data/fetch_all.py --dry-run
    python scripts/legal_data/fetch_all.py --sample 10
    python scripts/legal_data/fetch_all.py                 # then: vectorize.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ("fetch_laws.py", "fetch_eu_law.py", "fetch_constitutional_court.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample", type=int)
    parser.add_argument("--output")
    parser.add_argument("--skip", nargs="*", default=[], help="steps to skip, e.g. fetch_constitutional_court.py")
    args = parser.parse_args()
    shared = (["--dry-run"] if args.dry_run else []) + (["--sample", str(args.sample)] if args.sample else []) \
        + (["--output", args.output] if args.output else [])
    for step in STEPS:
        if step in args.skip:
            continue
        print(f"=== {step}", flush=True)
        code = subprocess.call([sys.executable, str(HERE / step), *shared])
        if code:
            print(f"=== {step} exited with {code}; stopping", flush=True)
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
