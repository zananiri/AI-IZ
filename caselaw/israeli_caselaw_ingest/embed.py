"""Stage 5 (optional): embed every chunk, build the offline vector store, and the BM25 index.

Chunks are read in one fixed order (chunk.chunk_files) and cut into shards of embed.shard_size;
shard i is written as embeddings/shard-NNNNN.npy (float16, L2-normalised) + shard-NNNNN.ids.parquet
and skipped when present, so a Colab disconnect costs at most one shard. Within a shard texts are
sorted by length to cut padding. Before the full run, embed.estimate_sample chunks are timed and
the time and size estimated; above embed.max_hours_without_confirm hours it stops unless --yes.

Vector store: LanceDB (one folder, no server; IVF_PQ index above lancedb_index_min_rows), or FAISS
IndexHNSWFlat + meta.parquet. BM25: bm25s shards over context_prefix + text (index.py)."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .chunk import CHUNK_SCHEMA, chunk_files
from .config import paths_for
from .encoders import get_encoder, instructions
from .state import atomic_write_json, atomic_write_parquet, log, mark_done, read_json, stage_done


def count_chunks(files: list[Path]) -> int:
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


def iter_shards(files: list[Path], shard_size: int, columns: list[str] | None = None, start_shard: int = 0):
    """(shard index, table) over the chunk files in order, shard_size rows each (the last may be
    shorter). Shards before start_shard are skipped without reading their files' contents."""
    rows_before = start_shard * shard_size
    pending: list[pa.Table] = []
    pending_rows = 0
    index = start_shard
    seen = 0
    for f in files:
        n = pq.ParquetFile(f).metadata.num_rows
        if seen + n <= rows_before:
            seen += n
            continue
        table = pq.read_table(f, columns=columns, schema=CHUNK_SCHEMA if columns is None else None)
        if seen < rows_before:
            table = table.slice(rows_before - seen)
        seen += n
        pending.append(table)
        pending_rows += table.num_rows
        while pending_rows >= shard_size:
            merged = pa.concat_tables(pending)
            yield index, merged.slice(0, shard_size)
            rest = merged.slice(shard_size)
            pending, pending_rows = ([rest] if rest.num_rows else []), rest.num_rows
            index += 1
    if pending_rows:
        yield index, pa.concat_tables(pending)


def embed_texts(table: pa.Table, passage_prefix: str, include_prefix: bool) -> list[str]:
    texts = table.column("text").to_pylist()
    prefixes = table.column("context_prefix").to_pylist() if include_prefix else [""] * len(texts)
    return [f"{passage_prefix}{p}\n{t}" if p else f"{passage_prefix}{t}" for p, t in zip(prefixes, texts)]


def encode_sorted(encoder, texts: list[str], batch_size: int) -> np.ndarray:
    order = np.argsort([len(t) for t in texts])[::-1]  # longest first: an OOM shows up at once
    vecs = encoder.encode([texts[i] for i in order], batch_size=batch_size)
    out = np.empty_like(vecs)
    out[order] = vecs
    return out


def _save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, array)
    os.replace(tmp, path)


def estimate(cfg: dict, encoder, files: list[Path], total: int, remaining: int, yes: bool) -> None:
    ecfg = cfg["embed"]
    n = min(ecfg.get("estimate_sample", 2000), total)
    if n == 0 or remaining == 0:
        return
    passage, _ = instructions(cfg)
    sample = next(iter_shards(files, n, columns=["chunk_id", "context_prefix", "text"]))[1]
    texts = embed_texts(sample, passage, ecfg.get("include_prefix", True))
    encode_sorted(encoder, texts[: min(64, len(texts))], ecfg["batch_size"])  # warm-up
    started = time.monotonic()
    vecs = encode_sorted(encoder, texts, ecfg["batch_size"])
    rate = len(texts) / max(time.monotonic() - started, 1e-6)
    hours = remaining / rate / 3600
    dim = vecs.shape[1]
    gb = remaining * dim * 2 / 1e9
    log(f"embed estimate: {rate:,.0f} chunks/s on {len(texts):,} chunks -> {remaining:,} chunks in "
        f"{hours:.1f} h; vectors {gb:.1f} GB float16 (+ about the same again in the vector store)")
    if hours > ecfg.get("max_hours_without_confirm", 3) and not yes:
        raise SystemExit(f"the estimate ({hours:.1f} h) exceeds {ecfg.get('max_hours_without_confirm', 3)} h, "
                         "longer than a free Colab session: re-run with --yes to go ahead (it resumes shard by "
                         "shard after a disconnect), or use a smaller sample_n")


def run_embed(cfg: dict, yes: bool = False, force: bool = False, store: str | None = None) -> dict:
    paths = paths_for(cfg)
    if not stage_done(paths.state, "chunk"):
        raise SystemExit("run the chunk stage first")
    ecfg = cfg["embed"]
    files = chunk_files(paths.chunks)
    total = count_chunks(files)
    shard_size = ecfg["shard_size"]
    n_shards = math.ceil(total / shard_size)
    out = paths.embeddings
    out.mkdir(parents=True, exist_ok=True)
    settings = {"model": ecfg["model"], "shard_size": shard_size, "include_prefix": ecfg.get("include_prefix", True),
                "chunks": total, "chunk_stage": stage_done(paths.state, "chunk").get("config_hash")}
    marker = read_json(out / "_settings.json")
    if force or (marker and marker != settings):
        log("embed: settings or chunks changed: starting the embeddings over")
        for p in out.glob("shard-*"):
            p.unlink()
    atomic_write_json(out / "_settings.json", settings)

    done_shards = [i for i in range(n_shards) if (out / f"shard-{i:05d}.npy").exists()
                   and (out / f"shard-{i:05d}.ids.parquet").exists()]
    first_missing = next((i for i in range(n_shards) if i not in done_shards), n_shards)
    remaining = total - sum(min(shard_size, total - i * shard_size) for i in done_shards)
    log(f"embed: {total:,} chunks, {n_shards} shards of {shard_size:,}; {len(done_shards)} done")
    encoder = None
    if remaining:
        encoder = get_encoder(cfg, paths)
        estimate(cfg, encoder, files, total, remaining, yes)
        passage, _ = instructions(cfg)
        started = time.monotonic()
        done_now = 0
        for i, table in iter_shards(files, shard_size, columns=["chunk_id", "context_prefix", "text"],
                                    start_shard=first_missing):
            if i in done_shards:
                continue
            texts = embed_texts(table, passage, ecfg.get("include_prefix", True))
            vecs = encode_sorted(encoder, texts, ecfg["batch_size"])
            _save_npy(out / f"shard-{i:05d}.npy", vecs)
            atomic_write_parquet(out / f"shard-{i:05d}.ids.parquet", table.select(["chunk_id"]))
            done_now += len(texts)
            rate = done_now / max(time.monotonic() - started, 1e-6)
            left = remaining - done_now
            log(f"embed: shard {i + 1}/{n_shards} ({done_now:,}/{remaining:,} this run, {rate:,.0f}/s, "
                f"~{left / max(rate, 1e-6) / 60:.0f} min left)")
    # Saved by the run that encoded; a run with nothing left to encode doesn't load the model.
    if encoder is not None and ecfg.get("save_model", True) and hasattr(encoder, "save"):
        target = paths.models / ecfg["model"].replace("/", "__")
        if not (target / "config.json").exists():
            encoder.save(target)
            log(f"embed: model saved to {target} for offline query encoding")

    write_id_map(out, n_shards)
    store = store or ecfg.get("store", "lancedb")
    if store == "lancedb":
        rows = build_lancedb(cfg, files, total)
    elif store == "faiss":
        rows = build_faiss(cfg, files, total)
    else:
        raise SystemExit(f"unknown embed.store {store}; choose lancedb or faiss")
    mark_done(paths.state, "embed", chunks=total, shards=n_shards, store=store, rows=rows, model=ecfg["model"])
    return {"chunks": total, "store": store, "rows": rows}


def write_id_map(out: Path, n_shards: int) -> None:
    tables = []
    for i in range(n_shards):
        ids = pq.read_table(out / f"shard-{i:05d}.ids.parquet")
        tables.append(ids.append_column("shard", pa.array([i] * ids.num_rows, pa.int32()))
                      .append_column("row", pa.array(range(ids.num_rows), pa.int32())))
    if tables:
        atomic_write_parquet(out / "id_map.parquet", pa.concat_tables(tables))


def _load_shard(out: Path, i: int) -> np.ndarray:
    return np.load(out / f"shard-{i:05d}.npy")


def build_lancedb(cfg: dict, files: list[Path], total: int) -> int:
    import lancedb

    paths = paths_for(cfg)
    ecfg = cfg["embed"]
    shard_size = ecfg["shard_size"]
    db = lancedb.connect(str(paths.lancedb))
    name = ecfg.get("lancedb_table", "chunks")
    try:
        existing = db.open_table(name).count_rows()
    except (ValueError, FileNotFoundError, RuntimeError):  # no table yet
        existing = 0
    if existing % shard_size and existing != total:
        log(f"lancedb: {existing:,} rows is not a whole number of shards: rebuilding the table")
        db.drop_table(name)
        existing = 0
    if existing > total:
        db.drop_table(name)
        existing = 0
    start = existing // shard_size
    table = db.open_table(name) if existing else None
    for i, meta in iter_shards(files, shard_size, start_shard=start):
        vecs = _load_shard(paths.embeddings, i)
        if len(vecs) != meta.num_rows:
            raise SystemExit(f"shard {i}: {len(vecs)} vectors for {meta.num_rows} chunks; re-run embed with --force")
        ids = pq.read_table(paths.embeddings / f"shard-{i:05d}.ids.parquet").column(0)
        if not ids.equals(meta.column("chunk_id")):
            raise SystemExit(f"shard {i}: chunk ids differ from the chunk files; re-run embed with --force")
        vector = pa.FixedSizeListArray.from_arrays(pa.array(vecs.reshape(-1), pa.float16()), vecs.shape[1])
        data = meta.append_column("vector", vector)
        if table is None:
            table = db.create_table(name, data, mode="overwrite")
        else:
            table.add(data)
        log(f"lancedb: shard {i} added ({table.count_rows():,}/{total:,} rows)")
    if table is None:
        raise SystemExit("lancedb: no chunks to index")
    rows = table.count_rows()
    if rows >= ecfg.get("lancedb_index_min_rows", 100000):
        dim = table.schema.field("vector").type.list_size
        try:
            table.create_index(metric="cosine", vector_column_name="vector", index_type="IVF_PQ",
                               num_partitions=max(16, int(math.sqrt(rows))), num_sub_vectors=max(1, dim // 16),
                               replace=True)
            log(f"lancedb: IVF_PQ index built over {rows:,} rows")
        except Exception as exc:  # noqa: BLE001 -- brute-force search still works
            log(f"lancedb: index not built ({type(exc).__name__}: {exc}); searches will scan")
    return rows


def build_faiss(cfg: dict, files: list[Path], total: int) -> int:
    import faiss

    paths = paths_for(cfg)
    ecfg = cfg["embed"]
    out = paths.faiss
    out.mkdir(parents=True, exist_ok=True)
    first = _load_shard(paths.embeddings, 0)
    dim = first.shape[1]
    need_gb = total * dim * 4 * 1.3 / 1e9
    log(f"faiss: HNSW over {total:,} x {dim} float32 needs about {need_gb:.1f} GB of RAM")
    index = faiss.IndexHNSWFlat(dim, ecfg.get("faiss_hnsw_m", 32), faiss.METRIC_INNER_PRODUCT)
    tmp_meta = out / "meta.parquet.tmp"
    with pq.ParquetWriter(tmp_meta, CHUNK_SCHEMA, compression="zstd") as writer:
        for i, meta in iter_shards(files, ecfg["shard_size"]):
            index.add(_load_shard(paths.embeddings, i).astype(np.float32))
            writer.write_table(meta)
    faiss.write_index(index, str(out / "index.faiss.tmp"))
    os.replace(out / "index.faiss.tmp", out / "index.faiss")
    os.replace(tmp_meta, out / "meta.parquet")
    return index.ntotal
