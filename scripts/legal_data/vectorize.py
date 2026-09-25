#!/usr/bin/env python3
"""Vectorize the legal corpus (legal_txt/*/*.jsonl) into its own Chroma stores: one per category,
<legal.corpus.vectordb_dir>/<category>/ holding collection <legal.corpus.collection_prefix>_<category>,
separate from the Legal tab's signed index. Same embedding model (legal.retrieval.embedding_model, bge-m3).
One directory per category keeps each Kaggle output (and download) to one category.

    python scripts/legal_data/vectorize.py --dry-run                        # records, chunks, size/time estimate
    python scripts/legal_data/vectorize.py --categories laws,procedural_rules --device cpu
    python scripts/legal_data/vectorize.py --input /kaggle/input/legal-corpus --device cuda --sample 200
    python scripts/legal_data/vectorize.py --categories supreme_court --device cuda --shard 0/2   # session 1 of 2
    python scripts/legal_data/vectorize.py --probe "מה דינו של חוזה שנכרת בטעות?" --category laws \
        --where '{"status": "in_force"}'
    python scripts/legal_data/vectorize.py --export-lexical                 # lexical_<category>.jsonl for a keyword index
    python scripts/legal_data/vectorize.py --merge-from <other vectordb dir> --categories supreme_court
                                                                            # add another session's shard, no re-embedding

Incremental: a record whose content hash is unchanged is skipped; a changed one is re-chunked
and its old chunks replaced; --prune deletes records no longer in the input. State is committed
after every batch, so an interrupted run (e.g. Kaggle's 12-hour limit) resumes where it stopped.
--shard i/n indexes a fixed 1/n of the records (by id hash) -- for splitting a category across
sessions; don't combine it with --prune.

Chunking and metadata: src/docslides/legal_data/corpus_chunking.py. Metadata filters available at
query time: category, authority_level, status, effective_ymd / decision_ymd (integers YYYYMMDD),
law_id, court, case_number, doc_type, record_id, section_number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from docslides.config import get_config
from docslides.legal_data.corpus_chunking import CorpusChunk, chunk_record
from docslides.legal_data.hebrew import normalize_for_embedding
from docslides.legal_data.progress import Progress
from docslides.legal_data.records import read_jsonl, record_hash

CATEGORIES = ("laws", "procedural_rules", "supreme_court")
FLUSH_CHUNKS = 2048  # embed + write + commit state every this many chunks
BYTES_PER_CHUNK_OVERHEAD = 1024 * 4 * 2 + 600  # float32 vector (index + store) + metadata, roughly


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class Embedder:
    def __init__(self, model_name: str, devices: list[str], batch_size: int) -> None:
        from docslides.rag.embedding import get_embedder

        self.batch_size = batch_size
        self.model = get_embedder(model_name, devices[0] if devices else None)
        self.pool = None
        if len(devices) > 1:
            try:
                self.pool = self.model.start_multi_process_pool(target_devices=devices)
                log(f"embedding on {devices}")
            except Exception as exc:  # noqa: BLE001 -- one device still works
                log(f"multi-device pool failed ({exc}); using {devices[0]} only")

    def encode(self, texts: list[str]):
        if self.pool is not None:
            try:
                return self.model.encode(texts, pool=self.pool, batch_size=self.batch_size, normalize_embeddings=True)
            except TypeError:  # sentence-transformers < 5
                return self.model.encode_multi_process(texts, self.pool, batch_size=self.batch_size,
                                                       normalize_embeddings=True)
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True)

    def close(self) -> None:
        if self.pool is not None:
            self.model.stop_multi_process_pool(self.pool)


def input_files(root: Path, category: str) -> list[Path]:
    folder = root / category
    return sorted(folder.glob(f"{category}*.jsonl")) if folder.exists() else []


def in_shard(record_id: str, shard: tuple[int, int] | None) -> bool:
    if shard is None:
        return True
    index, total = shard
    return int(hashlib.sha1(record_id.encode("utf-8")).hexdigest(), 16) % total == index


def index_category(category: str, args, corpus_cfg, model_name: str, embedder_factory) -> dict:
    from docslides.legal_data.corpus_index import CorpusCollection, CorpusState

    files = input_files(Path(args.input), category)
    stats: Counter = Counter()
    if not files:
        log(f"{category}: no input under {Path(args.input) / category}")
        return dict(stats)
    vectordb = Path(args.vectordb) / category
    state = CorpusState(vectordb / "_corpus_state.sqlite")
    collection = None if args.dry_run else CorpusCollection(vectordb, f"{corpus_cfg.collection_prefix}_{category}")
    embedder = None
    pending: list[tuple[str, str, list[CorpusChunk], list[str]]] = []  # record id, hash, chunks, old chunk ids
    seen: set[str] = set()

    def flush() -> None:
        nonlocal embedder
        chunks = [c for _, _, record_chunks, _ in pending for c in record_chunks]
        if chunks:
            embedder = embedder or embedder_factory()
            log(f"{category}: embedding {len(chunks):,} chunks ({stats['chunks_indexed']:,} done so far)")
            vectors = embedder.encode([c.embed_text for c in chunks])
            stale = [cid for _, _, _, old in pending for cid in old]
            if stale:
                collection.delete(stale)
            collection.upsert(chunks, vectors)
        for record_id, rhash, record_chunks, _ in pending:
            state.put(category, record_id, rhash, record_chunks)
        state.commit()
        stats["chunks_indexed"] += len(chunks)
        progress.extra.update(chunks_indexed=stats["chunks_indexed"])
        pending.clear()

    total = sum(1 for path in files for line in open(path, encoding="utf-8") if line.strip())
    if args.shard:
        total = total // args.shard[1] + 1  # about 1/n of the records fall in a shard
    if args.sample:
        total = min(total, args.sample)
    progress = Progress(f"{category}: records", total=total, unit="records", log=log)
    try:
        for path in files:
            for record in read_jsonl(path):
                record_id = record["id"]
                if not in_shard(record_id, args.shard):
                    continue
                seen.add(record_id)
                stats["records"] += 1
                progress.update(1, chunked=stats["chunks"], unchanged=stats["records_unchanged"])
                rhash = record_hash(record)
                previous = state.get(category, record_id)
                if previous and previous.record_hash == rhash and not args.rebuild:
                    stats["records_unchanged"] += 1
                    continue
                chunks = chunk_record(record, rhash, args.chunk_tokens, args.overlap_tokens,
                                      corpus_cfg.fold_final_letters_for_embedding)
                stats["records_indexed"] += 1
                stats["chunks"] += len(chunks)
                stats["chunk_chars"] += sum(len(c.text) for c in chunks)
                if args.dry_run:
                    continue
                pending.append((record_id, rhash, chunks, previous.chunk_ids if previous else []))
                if sum(len(p[2]) for p in pending) >= FLUSH_CHUNKS:
                    flush()
                if args.sample and stats["records_indexed"] >= args.sample:
                    break
            if args.sample and stats["records_indexed"] >= args.sample:
                break
        if not args.dry_run:
            flush()
            if args.prune and not args.sample and args.shard is None:
                for record_id in state.record_ids(category) - seen:
                    old = state.get(category, record_id)
                    collection.delete(old.chunk_ids if old else [])
                    state.delete(category, record_id)
                    stats["records_pruned"] += 1
                state.commit()
            stats["collection_count"] = collection.count()
        progress.done(chunks=stats["chunks"], chunks_indexed=stats["chunks_indexed"], unchanged=stats["records_unchanged"])
    finally:
        if embedder is not None:
            embedder.close()
        state.close()
    if args.dry_run:
        chunks = stats["chunks"]
        stats["estimated_index_gb"] = round(chunks * (BYTES_PER_CHUNK_OVERHEAD + 2.5 * stats["chunk_chars"] / max(chunks, 1)) / 1e9, 2)
        stats["estimated_gpu_hours_one_t4"] = round(chunks / args.rate / 3600, 1)
    return dict(stats)


def probe(args, corpus_cfg, model_name: str) -> None:
    from docslides.legal_data.corpus_index import CorpusCollection

    embedder = Embedder(model_name, parse_devices(args), args.batch_size)
    vector = embedder.encode([normalize_for_embedding(args.probe, corpus_cfg.fold_final_letters_for_embedding)])[0]
    where = json.loads(args.where) if args.where else None
    for category in args.categories:
        collection = CorpusCollection(Path(args.vectordb) / category, f"{corpus_cfg.collection_prefix}_{category}")
        result = collection.query(vector, args.top_k, where)
        print(f"\n=== {category} ({collection.count()} chunks) where={where}")
        for distance, meta, document in zip(result["distances"][0], result["metadatas"][0], result["documents"][0]):
            label = meta.get("section_number") or meta.get("case_number") or ""
            print(f"- {distance:.3f} | {meta.get('title', '')[:80]} | {label} | {meta.get('status')} "
                  f"| {meta.get('effective_ymd') or meta.get('decision_ymd') or ''}")
            print("    " + document[:300].replace("\n", " "))
    embedder.close()


def export_lexical(args) -> None:
    from docslides.legal_data.corpus_index import CorpusState

    for category in args.categories:
        directory = Path(args.vectordb) / category
        if not (directory / "_corpus_state.sqlite").exists():
            log(f"{category}: not indexed yet")
            continue
        state = CorpusState(directory / "_corpus_state.sqlite")
        path = directory / f"lexical_{category}.jsonl"
        count = 0
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for chunk_id, record_id, text in state.lexical(category):
                f.write(json.dumps({"chunk_id": chunk_id, "record_id": record_id, "text": text}, ensure_ascii=False) + "\n")
                count += 1
        state.close()
        log(f"{category}: {count} chunks -> {path}")


def merge(args, corpus_cfg) -> None:
    """Copies another run's collection and state into this one (vectors included, nothing re-embedded):
    how the shards of a category indexed in separate Kaggle sessions become one store."""
    import sqlite3

    from docslides.legal_data.corpus_index import CorpusCollection, CorpusState

    for category in args.categories:
        source_dir, target_dir = Path(args.merge_from) / category, Path(args.vectordb) / category
        if not (source_dir / "_corpus_state.sqlite").exists():
            log(f"{category}: nothing to merge in {source_dir}")
            continue
        name = f"{corpus_cfg.collection_prefix}_{category}"
        source, target = CorpusCollection(source_dir, name), CorpusCollection(target_dir, name)
        total, offset, page = source.count(), 0, 2000
        while offset < total:
            batch = source.collection.get(include=["embeddings", "documents", "metadatas"], limit=page, offset=offset)
            if not batch["ids"]:
                break
            target.collection.upsert(ids=batch["ids"], embeddings=batch["embeddings"], documents=batch["documents"],
                                     metadatas=batch["metadatas"])
            offset += len(batch["ids"])
            log(f"{category}: merged {offset}/{total}")
        CorpusState(target_dir / "_corpus_state.sqlite").close()  # creates the tables in a new target
        db = sqlite3.connect(target_dir / "_corpus_state.sqlite")
        db.execute("ATTACH DATABASE ? AS other", (str(source_dir / "_corpus_state.sqlite"),))
        db.execute("INSERT OR REPLACE INTO records SELECT * FROM other.records")
        db.execute("INSERT OR REPLACE INTO chunks SELECT * FROM other.chunks")
        db.commit()
        db.close()
        log(f"{category}: {target.count()} chunks after merging {source_dir}")


def write_build_info(args, corpus_cfg, model_name: str, results: dict) -> None:
    """<vectordb>/_build_info.json: what install_corpus.py checks on the machine the store is copied to --
    the Chroma version that wrote it (a different major version may not open it), the embedding model
    queries must use, and each category's chunk count."""
    import chromadb
    import sentence_transformers

    path = Path(args.vectordb) / "_build_info.json"
    info = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"categories": {}}
    info.update(built_at=time.strftime("%Y-%m-%dT%H:%M:%S"), chromadb_version=chromadb.__version__,
                sentence_transformers_version=sentence_transformers.__version__, embedding_model=model_name,
                collection_prefix=corpus_cfg.collection_prefix, chunk_tokens=args.chunk_tokens,
                overlap_tokens=args.overlap_tokens)
    for category, stats in results.items():
        if "collection_count" in stats:
            info["categories"][category] = {"collection": f"{corpus_cfg.collection_prefix}_{category}",
                                            "chunks": stats["collection_count"], "records": stats.get("records", 0)}
    path.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"build info: {path}")


def parse_devices(args) -> list[str]:
    if args.devices:
        return [d.strip() for d in args.devices.split(",") if d.strip()]
    return [args.device] if args.device else []


def main() -> int:
    cfg = get_config()
    corpus_cfg = cfg.legal.corpus
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=cfg.legal_data.output_dir, help="folder holding laws/, procedural_rules/ ...")
    parser.add_argument("--vectordb", default=corpus_cfg.vectordb_dir)
    parser.add_argument("--categories", default=",".join(CATEGORIES))
    parser.add_argument("--dry-run", action="store_true", help="count records and chunks; embed nothing")
    parser.add_argument("--sample", type=int, metavar="N", help="index at most N changed records per category")
    parser.add_argument("--shard", help="i/n: only the i-th of n record groups (0-based)")
    parser.add_argument("--rebuild", action="store_true", help="re-index records even if unchanged")
    parser.add_argument("--prune", action="store_true", help="delete records no longer in the input")
    parser.add_argument("--device", help="cuda, cuda:0, cpu ... (default: sentence-transformers' choice)")
    parser.add_argument("--devices", help="several devices for one run, e.g. cuda:0,cuda:1")
    parser.add_argument("--batch-size", type=int, default=corpus_cfg.embed_batch_size)
    parser.add_argument("--chunk-tokens", type=int, default=corpus_cfg.chunk_max_tokens)
    parser.add_argument("--overlap-tokens", type=int, default=corpus_cfg.chunk_overlap_tokens)
    parser.add_argument("--merge-from", help="another vectordb root to copy into this one (per category)")
    parser.add_argument("--rate", type=float, default=30.0, help="chunks/s per T4 for --dry-run estimates")
    parser.add_argument("--probe", help="run one query instead of indexing")
    parser.add_argument("--category", help="with --probe: which category's collection")
    parser.add_argument("--where", help='with --probe: a Chroma filter, e.g. \'{"status": "in_force"}\'')
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--export-lexical", action="store_true")
    args = parser.parse_args()
    args.categories = [c.strip() for c in (args.category or args.categories).split(",") if c.strip()]
    if unknown := [c for c in args.categories if c not in CATEGORIES]:
        parser.error(f"unknown categories: {unknown}")
    if args.shard:
        index, total = (int(x) for x in args.shard.split("/"))
        if not 0 <= index < total:
            parser.error("--shard i/n needs 0 <= i < n")
        args.shard = (index, total)
    model_name = cfg.legal.retrieval.embedding_model

    if args.probe:
        probe(args, corpus_cfg, model_name)
        return 0
    if args.export_lexical:
        export_lexical(args)
        return 0
    if args.merge_from:
        merge(args, corpus_cfg)
        return 0

    log(f"input {Path(args.input).resolve()} -> {Path(args.vectordb).resolve()} ({model_name})"
        + (" [dry run]" if args.dry_run else ""))
    results = {}
    for category in args.categories:
        results[category] = index_category(
            category, args, corpus_cfg, model_name,
            lambda: Embedder(model_name, parse_devices(args), args.batch_size),
        )
        log(f"{category}: {results[category]}")
    if not args.dry_run:
        write_build_info(args, corpus_cfg, model_name, results)
    summary = Path(args.vectordb) / f"_vectorize_{time.strftime('%Y%m%dT%H%M%S')}{'_dry_run' if args.dry_run else ''}.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({"args": {k: v for k, v in vars(args).items()}, "results": results},
                                  ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log(f"summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
