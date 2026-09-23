#!/usr/bin/env python3
"""Stage 0 for the Legal tab: stage, review, approve and sign Israeli legal
sources into the local index (src/docslides/legal/).

Nothing reaches the index without going through a staged batch:

    # 1. Official sources: put the text (.txt/.md/.docx/text-layer .pdf) plus a
    #    <file>.meta.json sidecar in data/legal/sources/ -- see legal/sources.py.
    python scripts/ingest_legal.py stage data/legal/sources/contracts.txt --dry-run   # preview structure
    python scripts/ingest_legal.py stage data/legal/sources/contracts.txt
    python scripts/ingest_legal.py stage-sources           # every source file in sources_dir

    # 2. Uploads (memos, firm materials, client documents): drop files in uploads/
    python scripts/ingest_legal.py stage-uploads           # one scan
    python scripts/ingest_legal.py stage-uploads --watch   # keep watching (polls)

    # 3. Review and approve (approval embeds, upserts and re-signs the bundle)
    python scripts/ingest_legal.py list --status pending
    python scripts/ingest_legal.py show <batch_id> --limit 10
    python scripts/ingest_legal.py approve <batch_id> --reviewer "Adv. Name"   # required for uploads
    python scripts/ingest_legal.py reject <batch_id> --reviewer "Adv. Name" --reason "..."

    # 4. Check the index against the signed bundle
    python scripts/ingest_legal.py verify
    python scripts/ingest_legal.py dups              # near-duplicate chunks, to tune dedupe

Approving needs DOCSLIDES_LEGAL_BUNDLE_KEY set (the HMAC key the bundle is
signed with). Set the same value for the API process so it can verify it.

Needs the optional `legal` dependency group: pip install -e ".[legal]"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from docslides.config import get_config
from docslides.legal import bundle, staging
from docslides.legal.sources import is_source_file, load_official_source


def _print_batch_summary(batch: staging.Batch) -> None:
    signoff = " [needs named reviewer]" if batch.requires_signoff and batch.status == "pending" else ""
    reviewer = f" by {batch.reviewed_by}" if batch.reviewed_by else ""
    print(
        f"{batch.batch_id}  {batch.status}{reviewer}{signoff}\n"
        f"    {batch.law_name}  ({batch.source_type}/{batch.source_origin}, from {batch.effective_date_start})\n"
        f"    {len(batch.chunks)} chunks  <- {batch.source_path}"
    )
    for change in batch.superseded:
        print(f"    superseded: {change}")


def _print_chunks(chunks, limit: int) -> None:
    for chunk in chunks[:limit]:
        meta = chunk.metadata
        refs = f"  xrefs={meta.cross_references}" if meta.cross_references else ""
        print(f"\n--- {meta.chunk_id}  ({meta.part_index}/{meta.part_count}){refs}")
        print(chunk.text[:800] + (" [...]" if len(chunk.text) > 800 else ""))
    if len(chunks) > limit:
        print(f"\n... {len(chunks) - limit} more chunk(s)")


def cmd_stage(args) -> int:
    source = load_official_source(Path(args.file), Path(args.meta) if args.meta else None)
    if args.dry_run:
        chunks = staging.build_chunks(source)
        sections = {c.metadata.section_key for c in chunks}
        print(f"{source.meta.law_name}: {len(sections)} sections -> {len(chunks)} chunks (dry run, nothing written)")
        _print_chunks(chunks, args.limit)
        return 0
    _print_batch_summary(staging.stage(source))
    return 0


def cmd_stage_sources(args) -> int:
    sources_dir = Path(get_config().legal.ingestion.sources_dir)
    failures = 0
    for path in sorted(p for p in sources_dir.rglob("*") if is_source_file(p)):
        try:
            _print_batch_summary(staging.stage(load_official_source(path)))
        except Exception as exc:  # noqa: BLE001 -- report and continue with the rest
            failures += 1
            print(f"FAILED {path}: {exc}", file=sys.stderr)
    return 1 if failures else 0


def cmd_stage_uploads(args) -> int:
    while True:
        staged, failed = staging.scan_uploads()
        for batch in staged:
            _print_batch_summary(batch)
        for path, error in failed:
            print(f"FAILED {path}: {error}", file=sys.stderr)
        if not args.watch:
            return 1 if failed else 0
        time.sleep(args.interval)


def cmd_list(args) -> int:
    batches = staging.list_batches(args.status)
    if not batches:
        print("No batches.")
    for batch in batches:
        _print_batch_summary(batch)
    return 0


def cmd_show(args) -> int:
    batch = staging.load_batch(args.batch_id)
    _print_batch_summary(batch)
    _print_chunks(batch.chunks, args.limit)
    return 0


def cmd_approve(args) -> int:
    batch = staging.approve(args.batch_id, args.reviewer)
    _print_batch_summary(batch)
    return 0


def cmd_reject(args) -> int:
    _print_batch_summary(staging.reject(args.batch_id, args.reviewer, args.reason))
    return 0


def cmd_retract(args) -> int:
    for change in staging.retract_version(args.law_id, args.effective_date_start, actor=args.reviewer):
        print(f"re-chained: {change}")
    print(f"Retracted {args.law_id} @ {args.effective_date_start}")
    return 0


def cmd_dups(args) -> int:
    """Chunk pairs of the same law version whose embeddings are suspiciously
    close -- to eyeball when tuning dedupe. Exact duplicates are already
    dropped at chunking; what shows up here is near-duplicates (a page printed
    twice with a margin note moved) or legitimately similar provisions."""
    import numpy as np

    from docslides.legal.retrieval import _get_collection

    data = _get_collection().get(include=["embeddings", "metadatas", "documents"])
    ids, metas, docs = data["ids"], data["metadatas"], data["documents"]
    if not ids:
        print("Index is empty.")
        return 0
    vectors = np.array(data["embeddings"], dtype=float)
    sims = vectors @ vectors.T
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            same_version = (metas[i]["law_id"], metas[i]["effective_date_start"]) == (
                metas[j]["law_id"], metas[j]["effective_date_start"])
            if same_version and metas[i]["source_id"] != metas[j]["source_id"] and args.low <= sims[i, j] <= args.high:
                pairs.append((sims[i, j], i, j))
    for sim, i, j in sorted(pairs, reverse=True)[: args.limit]:
        print(f"\n{sim:.3f}  {ids[i]}  <->  {ids[j]}")
        for k in (i, j):
            print("   ", docs[k].split("\n\n", 1)[-1][:160].replace("\n", " "))
    print(f"\n{len(pairs)} pair(s) with similarity in [{args.low}, {args.high}]")
    return 0


def cmd_verify(args) -> int:
    from docslides.legal import retrieval
    from docslides.legal.models import content_hash

    entries, level = bundle.verify()
    print(f"Bundle: {len(entries)} approved chunks, signature check: {level}")
    stored = retrieval.all_chunks()
    stored_ids = {chunk_id for chunk_id, _, _ in stored}
    bad = [cid for cid, text, meta in stored if entries.get(cid, {}).get("sha256") != content_hash(text, meta)]
    missing = sorted(set(entries) - stored_ids)
    print(f"Index: {len(stored)} chunks; {len(bad)} not matching the bundle; {len(missing)} approved but missing")
    for cid in bad[:20]:
        print(f"  not in bundle / hash mismatch (excluded at query time): {cid}")
    for cid in missing[:20]:
        print(f"  approved but missing from index: {cid}")
    return 1 if bad or missing else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("stage", help="stage one official source file")
    p.add_argument("file")
    p.add_argument("--meta", help="metadata JSON (default: <file>.meta.json)")
    p.add_argument("--dry-run", action="store_true", help="parse and chunk only; print, write nothing")
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(func=cmd_stage)

    p = sub.add_parser("stage-sources", help="stage every official source file in sources_dir")
    p.set_defaults(func=cmd_stage_sources)

    p = sub.add_parser("stage-uploads", help="stage new/changed files in the uploads folder")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=int, default=30, help="seconds between scans with --watch")
    p.set_defaults(func=cmd_stage_uploads)

    p = sub.add_parser("list")
    p.add_argument("--status", choices=["pending", "approved", "rejected"])
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show")
    p.add_argument("batch_id")
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("approve", help="embed, index and sign a pending batch")
    p.add_argument("batch_id")
    p.add_argument("--reviewer", help="name of the reviewer signing off (required for uploads)")
    p.set_defaults(func=cmd_approve)

    p = sub.add_parser("reject")
    p.add_argument("batch_id")
    p.add_argument("--reviewer", required=True)
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_reject)

    p = sub.add_parser("retract", help="remove one version of a law and re-chain the remaining versions")
    p.add_argument("law_id")
    p.add_argument("effective_date_start", help="YYYY-MM-DD, as in the version's chunk ids")
    p.add_argument("--reviewer")
    p.set_defaults(func=cmd_retract)

    p = sub.add_parser("dups", help="list near-duplicate chunk pairs within each law version")
    p.add_argument("--low", type=float, default=0.90)
    p.add_argument("--high", type=float, default=0.999)
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_dups)

    p = sub.add_parser("verify", help="check the index against the signed bundle")
    p.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    try:
        return args.func(args)
    except (staging.StagingError, bundle.BundleError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
