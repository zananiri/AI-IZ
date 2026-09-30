"""BM25 indexing and offline search (BM25, dense, and hybrid by reciprocal-rank fusion).

Nothing here touches the network: the dense side loads the model saved under <root>/models/.

BM25 (bm25s) is built in shards of bm25.shard_size chunks to bound RAM; a query is scored in every
shard and the hits merged by score (IDF is per shard, close enough at 50k chunks per shard).
Tokens keep gershayim and slashes, so ע"א and 6821/93 are terms of their own; with
strip_prefixes, a Hebrew word of four letters or more is also indexed without a leading ו/ה/ב/ל/מ/ש/כ."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from .chunk import chunk_files
from .config import paths_for
from .embed import count_chunks, iter_shards
from .state import atomic_write_json, atomic_write_parquet, log, mark_done, read_json, stage_done

_TOKEN_RE = re.compile(r"[0-9A-Za-zא-ת]+(?:[\"'/\-.][0-9A-Za-zא-ת]+)*")
_PREFIX_RE = re.compile(r"^[והבלמשכ][א-ת]{3,}$")


def tokenize(text: str, strip_prefixes: bool = True) -> list[str]:
    text = text.replace("״", '"').replace("׳", "'").replace("”", '"').replace("“", '"').lower()
    tokens = []
    for tok in _TOKEN_RE.findall(text):
        tokens.append(tok)
        if strip_prefixes and _PREFIX_RE.match(tok):
            tokens.append(tok[1:])
    return tokens


def run_bm25(cfg: dict, force: bool = False) -> dict:
    import bm25s

    paths = paths_for(cfg)
    chunk_state = stage_done(paths.state, "chunk")
    if not chunk_state:
        raise SystemExit("run the chunk stage first")
    bcfg = cfg["bm25"]
    files = chunk_files(paths.chunks)
    total = count_chunks(files)
    out = paths.bm25
    settings = {"shard_size": bcfg["shard_size"], "strip_prefixes": bcfg.get("strip_prefixes", True),
                "chunks": total, "chunk_stage": chunk_state.get("config_hash")}
    if force or read_json(out / "_settings.json") not in (None, settings):
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "_settings.json", settings)
    n = 0
    for i, table in iter_shards(files, bcfg["shard_size"], columns=["chunk_id", "context_prefix", "text"]):
        shard_dir = out / f"shard-{i:05d}"
        n += table.num_rows
        if (shard_dir / "done.json").exists():
            continue
        corpus = [tokenize(f"{p}\n{t}", bcfg.get("strip_prefixes", True))
                  for p, t in zip(table.column("context_prefix").to_pylist(), table.column("text").to_pylist())]
        retriever = bm25s.BM25()
        retriever.index(corpus, show_progress=False)
        shutil.rmtree(shard_dir, ignore_errors=True)
        retriever.save(str(shard_dir))
        atomic_write_parquet(shard_dir / "ids.parquet", table.select(["chunk_id"]))
        atomic_write_json(shard_dir / "done.json", {"chunks": table.num_rows})
        log(f"bm25: shard {i} indexed ({n:,}/{total:,} chunks)")
    mark_done(paths.state, "bm25", chunks=total, shards=len(list(out.glob("shard-*"))))
    return {"chunks": total}


class Searcher:
    """Offline search over the built artifacts under a run directory."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.paths = paths_for(cfg)
        self._bm25 = None
        self._table = None
        self._encoder = None

    # -- BM25 ---------------------------------------------------------------------------------
    def bm25_shards(self):
        if self._bm25 is None:
            import bm25s

            self._bm25 = []
            for shard in sorted(self.paths.bm25.glob("shard-*")):
                if (shard / "done.json").exists():
                    ids = pq.read_table(shard / "ids.parquet").column(0).to_pylist()
                    self._bm25.append((bm25s.BM25.load(str(shard), mmap=True), ids))
        return self._bm25

    def bm25(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        tokens = tokenize(query, self.cfg["bm25"].get("strip_prefixes", True))
        hits = []
        for retriever, ids in self.bm25_shards():
            known = [t for t in tokens if t in retriever.vocab_dict]
            if not known:
                continue
            docs, scores = retriever.retrieve([known], k=min(k, len(ids)), show_progress=False)
            hits += [(ids[int(d)], float(s)) for d, s in zip(docs[0], scores[0]) if s > 0]
        return sorted(hits, key=lambda h: -h[1])[:k]

    # -- dense ---------------------------------------------------------------------------------
    def table(self):
        if self._table is None:
            import lancedb

            db = lancedb.connect(str(self.paths.lancedb))
            self._table = db.open_table(self.cfg["embed"].get("lancedb_table", "chunks"))
        return self._table

    def encoder(self):
        if self._encoder is None:
            from .encoders import get_encoder

            self._encoder = get_encoder(self.cfg, self.paths)
        return self._encoder

    def dense(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        from .encoders import instructions

        _, query_prefix = instructions(self.cfg)
        vec = self.encoder().encode([query_prefix + query])[0].astype("float32")
        rows = self.table().search(vec, vector_column_name="vector").metric("cosine").limit(k) \
            .select(["chunk_id", "_distance"]).to_list()
        return [(r["chunk_id"], 1.0 - float(r["_distance"])) for r in rows]

    def hybrid(self, query: str, k: int = 5, pool: int = 50, rrf_k: int = 60) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for ranking in (self.bm25(query, pool), self.dense(query, pool)):
            for rank, (cid, _) in enumerate(ranking):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        return sorted(scores.items(), key=lambda kv: -kv[1])[:k]

    # -- rows ------------------------------------------------------------------------------------
    def rows(self, chunk_ids: list[str], columns: list[str] | None = None) -> dict[str, dict]:
        columns = columns or ["chunk_id", "case_citation", "doc_type", "decision_date", "section", "text", "source_url"]
        if not chunk_ids:
            return {}
        import pyarrow.compute as pc
        import pyarrow.dataset as ds

        dataset = ds.dataset([str(f) for f in chunk_files(self.paths.chunks)], format="parquet")
        table = dataset.to_table(columns=columns, filter=pc.field("chunk_id").isin(chunk_ids))
        return {r["chunk_id"]: r for r in table.to_pylist()}


def paths_exist(path: Path) -> bool:
    return path.exists() and any(path.iterdir())
