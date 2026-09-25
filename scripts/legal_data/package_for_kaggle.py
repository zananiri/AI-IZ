#!/usr/bin/env python3
"""Zip the corpus JSONL (not the raw dump, PDFs or parquet) for upload as a private Kaggle Dataset,
which notebooks/kaggle_legal_corpus_vectorize.ipynb then vectorizes.

    python scripts/legal_data/package_for_kaggle.py                      # all categories
    python scripts/legal_data/package_for_kaggle.py --categories supreme_court
    python scripts/legal_data/package_for_kaggle.py --output legal_txt/_sample   # a sample run's output

Writes <output>/_kaggle_upload/legal_corpus_<categories>.zip, laid out as laws/laws.jsonl,
procedural_rules/..., supreme_court/supreme_court-00001.jsonl ... -- upload it at
kaggle.com/datasets (New Dataset; Kaggle unpacks it) and attach it to the notebook.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from docslides.config import get_config

CATEGORIES = ("laws", "procedural_rules", "supreme_court")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=get_config().legal_data.output_dir, help="the fetchers' output root")
    parser.add_argument("--categories", default=",".join(CATEGORIES))
    args = parser.parse_args()
    root = Path(args.output)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    target = root / "_kaggle_upload" / f"legal_corpus_{'_'.join(categories)}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    files = [p for c in categories for p in sorted((root / c).glob("*.jsonl"))]
    files += [p for c in categories for p in (root / c).glob("*_report.json")]
    if not files:
        print(f"no JSONL under {root} for {categories}")
        return 1
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
            print(f"  + {path.relative_to(root).as_posix()} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"{target} ({target.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
