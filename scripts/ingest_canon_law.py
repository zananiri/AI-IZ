#!/usr/bin/env python3
"""Offline ingestion for the Canon GPT tab: fetches the Code of Canon Law
(CIC 1983, English), the Code of Canons of the Eastern Churches (CCEO 1990,
Latin), and Vatican City State civil law (Italian PDFs) from vatican.va/
vaticanstate.va, chunks them at canon/article granularity
(docslides.canon.chunking), embeds them, and upserts into the local ChromaDB
store (docslides.canon.retrieval). See config.yaml's `canon:` section for
the vector store location and embedding model.

CCEO is Latin-only: the only free/official Holy See text is Latin -- the
standard English translation is a copyrighted Canon Law Society of America
publication and isn't scraped here.

vatican.va/vaticanstate.va's markup was reverse-engineered from real sample
pages during development (docslides.canon.parsing), not from a published
schema. If a run comes back with suspiciously few provisions for a source,
re-verify that source's parser against a fresh sample page before trusting
the result -- `--dry-run` is for exactly this.

Run manually/offline, not part of the FastAPI app. Safe to re-run any time
to refresh the index after a Vatican text update -- upsert is idempotent,
keyed by each chunk's stable id.

Requires the optional `canon` dependency group: pip install -e ".[canon]"

Usage:
    python scripts/ingest_canon_law.py                # ingest everything
    python scripts/ingest_canon_law.py --only cic      # one source at a time (repeatable)
    python scripts/ingest_canon_law.py --dry-run       # parse/chunk only, print samples, write nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from docslides.canon import parsing
from docslides.canon.chunking import Chunk, ProvisionRecord, chunk_provisions
from docslides.logging_setup import configure_logging, get_logger

logger = get_logger(__name__)


def ingest_cic() -> list[ProvisionRecord]:
    index_html = parsing.fetch(parsing.CIC_INDEX_URL).text
    page_urls = parsing.discover_cic_page_urls(index_html)
    logger.info("cic_pages_discovered", count=len(page_urls))

    records: list[ProvisionRecord] = []
    for url in page_urls:
        try:
            html = parsing.fetch(url).text
        except Exception as exc:  # noqa: BLE001
            logger.warning("cic_page_fetch_failed", url=url, error=str(exc))
            continue
        page_records = parsing.parse_cic_page(html, url)
        if not page_records:
            logger.warning("cic_page_no_canons_parsed", url=url)
        records.extend(page_records)
    return records


def ingest_cceo() -> list[ProvisionRecord]:
    records: list[ProvisionRecord] = []
    for url in parsing.CCEO_PAGE_URLS:
        try:
            html = parsing.fetch(url).text
        except Exception as exc:  # noqa: BLE001
            logger.warning("cceo_page_fetch_failed", url=url, error=str(exc))
            continue
        page_records = parsing.parse_cceo_page(html, url)
        if not page_records:
            logger.warning("cceo_page_no_canons_parsed", url=url)
        records.extend(page_records)
    return records


def ingest_vcs_law() -> list[ProvisionRecord]:
    records: list[ProvisionRecord] = []
    for doc in parsing.VCS_DOCUMENTS:
        try:
            resp = parsing.fetch(doc["url"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("vcs_pdf_fetch_failed", law_name=doc["law_name"], error=str(exc))
            continue
        doc_records = parsing.parse_vcs_pdf(resp.content, doc["law_name"], doc["url"])
        if not doc_records:
            logger.warning("vcs_pdf_no_articles_parsed", law_name=doc["law_name"])
        records.extend(doc_records)
    return records


INGESTORS = {"cic": ingest_cic, "cceo": ingest_cceo, "vcs_law": ingest_vcs_law}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--only",
        choices=sorted(INGESTORS),
        action="append",
        dest="only",
        help="Ingest only this source (repeatable, e.g. --only cic --only cceo). Default: all three.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk but don't touch the vector store; print counts and a sample chunk per source.",
    )
    args = parser.parse_args()

    configure_logging()
    sources = args.only or sorted(INGESTORS)

    all_chunks: list[Chunk] = []
    for source in sources:
        print(f"== {source} ==")
        records = INGESTORS[source]()
        print(f"  {len(records)} provisions parsed")
        chunks = chunk_provisions(records)
        print(f"  {len(chunks)} chunks produced")
        if chunks:
            sample = chunks[0]
            print(f"  sample id: {sample.id!r}")
            print(f"  sample text: {sample.text[:300]!r}")
        all_chunks.extend(chunks)

    if args.dry_run:
        print(f"\nDry run: {len(all_chunks)} chunks total across {len(sources)} source(s). Nothing written.")
        return

    from docslides.canon.retrieval import (
        collection_count,
        upsert_chunks,
    )

    upsert_chunks(all_chunks)
    print(f"\nUpserted {len(all_chunks)} chunks. Collection now has {collection_count()} chunks total.")


if __name__ == "__main__":
    main()
