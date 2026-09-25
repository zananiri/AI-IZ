#!/usr/bin/env python3
"""Install the corpus vector store built on Kaggle (notebooks/kaggle_legal_corpus_build.ipynb) on
this machine.

    python scripts/legal_data/install_corpus.py C:\\Users\\me\\Downloads\\legal_corpus_vectordb.zip
    python scripts/legal_data/install_corpus.py legal_corpus_vectordb.zip --replace     # over an older install

Unzips into legal.corpus.vectordb_dir (data/legal_corpus_vectordb): one folder per category (laws/,
procedural_rules/), each a Chroma store, plus _build_info.json. Then opens every collection and checks
its chunk count against the build info. No model is loaded and nothing is sent anywhere.

The store must be read by the chromadb version that wrote it (or a compatible one): the script warns
if yours differs -- `pip install chromadb==<version it names>` fixes that. Queries must embed with
the model named in _build_info.json (BAAI/bge-m3, the config default).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from docslides.config import get_config
from docslides.legal_data.progress import Progress


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"refusing to extract {info.filename!r}: it points outside the target folder")
        members.append(info)
    return members


def _major_minor(version: str) -> tuple[str, ...]:
    return tuple(version.split(".")[:2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zip", help="legal_corpus_vectordb.zip from the Kaggle notebook's output")
    parser.add_argument("--target", default=get_config().legal.corpus.vectordb_dir)
    parser.add_argument("--replace", action="store_true", help="delete an existing install of the same categories first")
    args = parser.parse_args()
    target = Path(args.target)

    with zipfile.ZipFile(args.zip) as archive:
        members = _safe_members(archive)
        categories = sorted({PurePosixPath(m.filename).parts[0] for m in members
                             if len(PurePosixPath(m.filename).parts) > 1})
        print(f"{args.zip}: categories {categories}, {sum(m.file_size for m in members) / 1e6:,.0f} MB unpacked")
        existing = [c for c in categories if (target / c).exists()]
        if existing and not args.replace:
            raise SystemExit(f"{target} already holds {existing}: re-run with --replace to overwrite them")
        for category in existing:
            shutil.rmtree(target / category)
        target.mkdir(parents=True, exist_ok=True)
        progress = Progress("extracting", total=sum(m.file_size for m in members), unit="bytes")
        for member in members:
            archive.extract(member, target)
            progress.update(member.file_size)
        progress.done(files=len(members))

    info_path = target / "_build_info.json"
    if not info_path.exists():
        raise SystemExit(f"{info_path} is missing: this zip wasn't made by the build notebook")
    info = json.loads(info_path.read_text(encoding="utf-8"))

    import chromadb

    built, local = info.get("chromadb_version", "?"), chromadb.__version__
    if _major_minor(built) != _major_minor(local):
        print(f"WARNING: the store was written by chromadb {built}, this machine has {local}. "
              f"If opening it fails: pip install chromadb=={built}")

    problems = 0
    for category, expected in sorted(info.get("categories", {}).items()):
        if category not in categories:
            continue
        client = chromadb.PersistentClient(path=str(target / category))
        try:
            count = client.get_collection(expected["collection"]).count()
        except Exception as exc:  # noqa: BLE001 -- reported, the others are still checked
            print(f"  {category}: could not open {expected['collection']}: {exc}")
            problems += 1
            continue
        ok = count == expected["chunks"]
        problems += not ok
        status = "OK" if ok else f"-- expected {expected['chunks']:,}"
        print(f"  {category}: {count:,} chunks in {expected['collection']} "
              f"({expected.get('records', 0):,} records) {status}")

    print(f"\ninstalled in {target.resolve()} (embedding model: {info.get('embedding_model')}, built {info.get('built_at')})")
    print("try it:  python scripts/legal_data/vectorize.py --probe \"מה דינו של חוזה שנכרת בטעות?\" --category laws")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
