#!/usr/bin/env python3
"""Offline ingestion for the Canon GPT tab: rebuilds the canon-law vector
store from scratch with both codes as published on vatican.va:

  * CIC 1983 (Code of Canon Law) in Italian -- ~250 HTML pages plus Book VI
    (penal law, revised 2021) as a PDF. The Italian text is the current,
    amended version.
  * CCEO 1990 (Code of Canons of the Eastern Churches) in Latin -- the only
    free official text; vatican.va has no Italian translation.

Steps:
  1. Fetch and parse both codes at canon/§ granularity (docslides.canon.parsing).
  2. Chunk (docslides.canon.chunking) and check that no chunk ids repeat.
  3. Delete ALL existing data in the canon collection. This only happens after
     steps 1-2 succeed, so a failed download never leaves you with an empty
     or partial index.
  4. Embed and upsert into the local ChromaDB store (docslides.canon.retrieval).

vatican.va's markup was reverse-engineered from sample pages, not from a
published schema. If a run comes back with suspiciously few provisions,
check the parser against a fresh page first. `--dry-run` is for exactly this.

Requires the optional `canon` dependency group: pip install -e ".[canon]"

Usage:
    python scripts/ingest_canon_law.py            # wipe canon DB, fetch, embed, store
    python scripts/ingest_canon_law.py --dry-run  # fetch/parse/chunk only; DB untouched
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from docslides.canon import parsing
from docslides.canon.chunking import Chunk, ProvisionRecord, chunk_provisions
from docslides.logging_setup import configure_logging, get_logger

logger = get_logger(__name__)

# Canon counts in each code, to flag parser gaps.
EXPECTED_CANONS = {"cic": 1752, "cceo": 1546}


class DownloadFailed(Exception):
    pass


def ingest_cic() -> list[ProvisionRecord]:
    index_html = parsing.fetch(parsing.CIC_INDEX_URL).text
    page_urls = parsing.discover_cic_page_urls(index_html)
    logger.info("cic_pages_discovered", count=len(page_urls))

    parser = parsing.CicItalianParser()
    records: list[ProvisionRecord] = []
    for url in page_urls:
        try:
            resp = parsing.fetch(url)
        except Exception as exc:  # noqa: BLE001
            raise DownloadFailed(f"{url}: {exc}") from exc
        if url.endswith(".pdf"):
            blocks = parsing.cic_pdf_blocks(resp.content)
        else:
            blocks = parsing.cic_html_blocks(resp.text)
        page_records = parser.parse(blocks, url)
        if not page_records:
            logger.warning("cic_page_no_canons_parsed", url=url)
        records.extend(page_records)
    return parsing.dedupe_records(records)


def ingest_cceo() -> list[ProvisionRecord]:
    records: list[ProvisionRecord] = []
    for url in parsing.CCEO_PAGE_URLS:
        try:
            html = parsing.fetch(url).text
        except Exception as exc:  # noqa: BLE001
            raise DownloadFailed(f"{url}: {exc}") from exc
        page_records = parsing.parse_cceo_page(html, url)
        if not page_records:
            logger.warning("cceo_page_no_canons_parsed", url=url)
        records.extend(page_records)
    return parsing.dedupe_records(records)


SOURCES = {"cic": ("CIC 1983, Italian", ingest_cic), "cceo": ("CCEO 1990, Latin", ingest_cceo)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch, parse and chunk, but don't touch the vector store; print counts and samples.",
    )
    args = parser.parse_args()
    configure_logging()

    all_chunks: list[Chunk] = []
    for code, (label, ingest) in SOURCES.items():
        print(f"== {label} ==")
        try:
            records = ingest()
        except DownloadFailed as exc:
            # We're about to wipe the DB -- refuse to replace it with a partial index.
            sys.exit(f"\nDownload failed ({exc}). Nothing written; re-run.")
        numbers = {int(r.number) for r in records}
        missing = sorted(set(range(1, EXPECTED_CANONS[code] + 1)) - numbers)
        print(f"  {len(records)} provisions from {len(numbers)} of {EXPECTED_CANONS[code]} canons")
        if missing:
            print(f"  WARNING: {len(missing)} canons not found: {', '.join(map(str, missing[:30]))}")
        chunks = chunk_provisions(records)
        print(f"  {len(chunks)} chunks produced")
        if chunks:
            print(f"  sample id: {chunks[0].id!r}")
            print(f"  sample text: {chunks[0].text[:300]!r}")
        all_chunks.extend(chunks)

    # Chroma rejects duplicate ids in one upsert -- check before wiping the
    # DB and before the (slow) embedding step.
    dup_ids = [cid for cid, n in Counter(c.id for c in all_chunks).items() if n > 1]
    if dup_ids:
        sys.exit(f"\n{len(dup_ids)} duplicate chunk ids (parser bug?), e.g. {sorted(dup_ids)[:10]}. Nothing written.")
    if not all_chunks:
        sys.exit("\nNo chunks produced. Nothing written.")

    if args.dry_run:
        print(f"\nDry run: {len(all_chunks)} chunks total. Canon DB untouched.")
        return

    from docslides.canon.retrieval import collection_count, reset_collection, upsert_chunks

    print("\nClearing canon DB...")
    reset_collection()
    print(f"Embedding and storing {len(all_chunks)} chunks (slow on CPU)...")
    upsert_chunks(all_chunks)
    print(f"Done. Canon collection now has {collection_count()} chunks.")


if __name__ == "__main__":
    main()
