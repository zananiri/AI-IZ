"""Israeli Supreme Court case law as a second evidence source, next to the statute corpus.

The index is the output of caselaw/israeli_caselaw_ingest (judgments decided before 2022, from the
Hugging Face dataset LevMuchnik/SupremeCourtOfIsrael): a LanceDB table of chunks with bge-m3
vectors, and optionally bm25s shards. legal.corpus.caselaw_dir points at the folder holding them.

Per question: dense search with the question and each planned issue (the same bge-m3 model the
statute retrieval has loaded, so no second copy), BM25 on the same texts when the shards are there,
the candidates fused by reciprocal rank, header chunks dropped, then the cross-encoder reranks them
against the question and at most one excerpt per judgment is kept, within caselaw_max_tokens.

The excerpts go into the prompt as their own <case_law> block, labelled as judgments, never mixed
into <context>: the statute text stays what the answer must rest on.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from docslides.cleaning.tokens import count_tokens
from docslides.config import get_config
from docslides.legal.retrieval import cpu_on_gpu_oom
from docslides.logging_setup import get_logger

logger = get_logger(__name__)

COLUMNS = ["chunk_id", "doc_id", "case_citation", "case_name", "doc_type", "decision_date", "section",
           "is_holding", "text", "source_url"]
_TOKEN_RE = re.compile(r"[0-9A-Za-zא-ת]+(?:[\"'/\-.][0-9A-Za-zא-ת]+)*")
_PREFIX_RE = re.compile(r"^[והבלמשכ][א-ת]{3,}$")


def tokenize(text: str) -> list[str]:
    """The index's BM25 tokenizer (israeli_caselaw_ingest.search.tokenize, strip_prefixes on)."""
    text = text.replace("״", '"').replace("׳", "'").replace("”", '"').replace("“", '"').lower()
    tokens = []
    for tok in _TOKEN_RE.findall(text):
        tokens.append(tok)
        if _PREFIX_RE.match(tok):
            tokens.append(tok[1:])
    return tokens


class CaseLawIndex:
    def __init__(self, root: Path):
        import lancedb

        self.root = root
        state = root / "state" / "embed.done.json"
        self.model = json.loads(state.read_text(encoding="utf-8")).get("model") if state.exists() else None
        self.table = lancedb.connect(str(root / "lancedb")).open_table("chunks")
        self.bm25 = []
        for shard in sorted((root / "bm25").glob("shard-*")):
            if (shard / "done.json").exists():
                import bm25s
                import pyarrow.parquet as pq

                ids = pq.read_table(shard / "ids.parquet").column(0).to_pylist()
                self.bm25.append((bm25s.BM25.load(str(shard), mmap=True), ids))
        logger.info("caselaw_index_loaded", root=str(root), rows=self.table.count_rows(), model=self.model,
                    bm25_shards=len(self.bm25))

    def dense(self, vector: list[float], k: int) -> list[dict]:
        return (self.table.search(vector, vector_column_name="vector").metric("cosine").limit(k)
                .where("section != 'header'", prefilter=True).select(COLUMNS + ["_distance"]).to_list())

    def lexical(self, text: str, k: int) -> list[str]:
        tokens = tokenize(text)
        hits = []
        for retriever, ids in self.bm25:
            known = [t for t in tokens if t in retriever.vocab_dict]
            if not known:
                continue
            docs, scores = retriever.retrieve([known], k=min(k, len(ids)), show_progress=False)
            hits += [(ids[int(d)], float(s)) for d, s in zip(docs[0], scores[0]) if s > 0]
        return [cid for cid, _ in sorted(hits, key=lambda h: -h[1])[:k]]

    def rows(self, chunk_ids: list[str]) -> list[dict]:
        if not chunk_ids:
            return []
        quoted = ", ".join("'" + c.replace("'", "''") + "'" for c in chunk_ids)
        return self.table.search().where(f"chunk_id IN ({quoted})").select(COLUMNS).limit(len(chunk_ids)).to_list()


def caselaw_stats() -> dict | None:
    """Judgments/chunks in the installed case-law index (legal.corpus.caselaw_dir) and when it was
    last built, read from the ingest's state files -- no LanceDB connection. None if it's off or
    not built."""
    root = get_config().legal.corpus.caselaw_dir
    if not root:
        return None
    state = Path(root) / "state"
    chunk_path, embed_path = state / "chunk.done.json", state / "embed.done.json"
    if not chunk_path.exists() or not (Path(root) / "lancedb").exists():
        return None
    chunked = json.loads(chunk_path.read_text(encoding="utf-8"))
    embedded = json.loads(embed_path.read_text(encoding="utf-8")) if embed_path.exists() else {}
    return {
        "judgments": chunked.get("documents", 0),
        "chunks": embedded.get("chunks") or chunked.get("chunks", 0),
        "built_at": embedded.get("finished_at") or chunked.get("finished_at"),
    }


@lru_cache(maxsize=2)
def open_index(root: str) -> CaseLawIndex | None:
    path = Path(root)
    if not (path / "lancedb").exists():
        logger.warning("caselaw_index_missing", root=root)
        return None
    try:
        return CaseLawIndex(path)
    except Exception as exc:  # noqa: BLE001 -- answer from the statutes alone rather than fail
        logger.warning("caselaw_index_unavailable", root=root, error=f"{type(exc).__name__}: {exc}")
        return None


@cpu_on_gpu_oom
def search_caselaw(question: str, issues: list[str]) -> list[dict]:
    """Up to caselaw_top_k judgment excerpts for the question (one per judgment), best first."""
    legal_cfg = get_config().legal
    cfg = legal_cfg.corpus
    if not cfg.caselaw_dir:
        return []
    index = open_index(cfg.caselaw_dir)
    if index is None:
        return []
    queries = [question] + [i for i in issues if i]
    pool: dict[str, dict] = {}

    def add(rows: list[dict], source: str) -> None:
        for rank, row in enumerate(rows):
            kept = pool.setdefault(row["chunk_id"], {**row, "sources": set(), "fused": 0.0})
            kept["sources"].add(source)
            kept["fused"] += 1.0 / (60 + rank)

    k = cfg.caselaw_candidates
    if index.model is None or index.model == legal_cfg.retrieval.embedding_model:
        from docslides.rag.embedding import embed_texts

        for qi, vector in enumerate(embed_texts(legal_cfg.retrieval.embedding_model, queries,
                                                device=legal_cfg.retrieval.device)):
            add(index.dense(vector, k), f"d{qi}")
    else:
        logger.warning("caselaw_model_mismatch", index_model=index.model, query_model=legal_cfg.retrieval.embedding_model)
    if index.bm25:
        ranked = [index.lexical(text, k // 2) for text in queries]
        need = list(dict.fromkeys(cid for ids in ranked for cid in ids if cid not in pool))
        rows = {**{r["chunk_id"]: r for r in index.rows(need)}, **pool}
        for qi, ids in enumerate(ranked):
            add([rows[cid] for cid in ids if cid in rows], f"k{qi}")
    candidates = [c for c in pool.values() if c.get("section") != "header"]
    candidates.sort(key=lambda c: -c["fused"])
    candidates = candidates[:k]

    from docslides.legal.retrieval import _reranker, rerank_limit, reranker_device

    reranker = _reranker(legal_cfg.retrieval.reranker_model, reranker_device()) \
        if legal_cfg.retrieval.reranker_model else None
    if reranker is not None and candidates:
        rerank_query = question + ("\n" + "; ".join(issues) if issues else "")
        n = rerank_limit(len(candidates))  # on the CPU the rest keep their fused order after these
        scored, rest = candidates[:n], candidates[n:]
        for c, score in zip(scored, reranker.predict([(rerank_query, c["text"]) for c in scored])):
            c["score"] = float(score)
        candidates = sorted(scored, key=lambda c: -c["score"]) + rest

    picked, docs, spent = [], set(), 0
    for c in candidates:
        if c["doc_id"] in docs:
            continue
        tokens = count_tokens(c["text"])
        if spent + tokens > cfg.caselaw_max_tokens:
            continue
        picked.append(c)
        docs.add(c["doc_id"])
        spent += tokens
        if len(picked) >= cfg.caselaw_top_k:
            break
    return picked


def render_caselaw(hits: list[dict]) -> str:
    """The excerpts as a block of their own, marked as judgments (not statute text)."""
    if not hits:
        return ""
    blocks = []
    for i, h in enumerate(hits, 1):
        date = h.get("decision_date")
        head = " | ".join(str(x) for x in (h.get("case_citation"), h.get("case_name"), h.get("doc_type"),
                                           date.isoformat() if hasattr(date, "isoformat") else date,
                                           "הכרעה" if h.get("is_holding") else None) if x)
        blocks.append(f"[C{i}] {head}\n{h['text']}")
    return ("<case_law>\nExcerpts from judgments of the Israeli Supreme Court (decided before 2022), found by "
            "search: they show how the court has read and applied the law. Rely on <context> for what the law "
            "says; use a judgment where it bears on the question, cite it by its case number (e.g. ע\"א 6821/93), "
            "and don't treat an excerpt as binding unless it states the court's ruling.\n\n"
            + "\n\n".join(blocks) + "\n</case_law>")


def caselaw_record(hits: list[dict]) -> list[dict]:
    """What an answers.jsonl row keeps about the excerpts used."""
    return [{"chunk_id": h["chunk_id"], "citation": h.get("case_citation"), "section": h.get("section"),
             "score": round(h["score"], 4) if "score" in h else None, "source_url": h.get("source_url")}
            for h in hits]
