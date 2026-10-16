"""Tribunal Constitucional case law as a second evidence source, next to the statute corpus.

The index is the corpus category legal.corpus.caselaw_category (default "constitutional_court"):
the TC's sentencias fetched by scripts/legal_data/fetch_constitutional_court.py and vectorized by
scripts/legal_data/vectorize.py into its own Chroma store, with its BM25 copy
(vectorize.py --export-lexical). None turns case law off.

Per question: dense search with the question and each planned issue (the same bge-m3 model the
statute retrieval has loaded, so no second copy), BM25 on the same texts when the lexical file is
there, the candidates fused by reciprocal rank, then the cross-encoder reranks them against the
question and at most one excerpt per judgment is kept, within caselaw_max_tokens.

The excerpts go into the prompt as their own <case_law> block, labelled as judgments, never mixed
into <context>: the statute text stays what the answer must rest on.
"""

from __future__ import annotations

import json
from pathlib import Path

from aicodex.cleaning.tokens import count_tokens
from aicodex.config import get_config
from aicodex.legal.retrieval import cpu_on_gpu_oom
from aicodex.logging_setup import get_logger

logger = get_logger(__name__)


def caselaw_stats() -> dict | None:
    """Judgments/chunks of the case-law category in the installed corpus, from _build_info.json --
    no Chroma connection. None if case law is off or not built."""
    cfg = get_config().legal.corpus
    category = cfg.caselaw_category
    if not category:
        return None
    info_path = Path(cfg.vectordb_dir) / "_build_info.json"
    if not info_path.exists() or not (Path(cfg.vectordb_dir) / category).exists():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8")).get("categories", {}).get(category)
    if not info:
        return None
    return {"judgments": info.get("records", 0), "chunks": info.get("chunks", 0),
            "built_at": json.loads(info_path.read_text(encoding="utf-8")).get("built_at")}


def _collection(category: str):
    from aicodex.legal_data.corpus_index import CorpusCollection

    cfg = get_config().legal.corpus
    path = Path(cfg.vectordb_dir) / category
    if not path.exists():
        return None
    try:
        collection = CorpusCollection(path, f"{cfg.collection_prefix}_{category}")
    except Exception as exc:  # noqa: BLE001 -- answer from the statutes alone rather than fail
        logger.warning("caselaw_index_unavailable", path=str(path), error=f"{type(exc).__name__}: {exc}")
        return None
    return collection if collection.count() else None


def _row(chunk_id: str, document: str, meta: dict, distance: float | None) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": meta.get("record_id") or chunk_id.split(":judgment")[0],
        "case_citation": meta.get("title"),
        "ecli": meta.get("ecli"),
        "case_name": meta.get("case_number") or meta.get("procedure_type"),
        "chamber": meta.get("chamber"),
        "doc_type": meta.get("doc_type"),
        "decision_date": str(meta.get("decision_ymd") or ""),
        "text": document,
        "distance": distance,
        "source_url": meta.get("source_url"),
    }


@cpu_on_gpu_oom
def search_caselaw(question: str, issues: list[str]) -> list[dict]:
    """Up to caselaw_top_k judgment excerpts for the question (one per judgment), best first."""
    legal_cfg = get_config().legal
    cfg = legal_cfg.corpus
    category = cfg.caselaw_category
    if not category:
        return []
    collection = _collection(category)
    if collection is None:
        return []
    queries = [question] + [i for i in issues if i]
    pool: dict[str, dict] = {}

    def add(rows: list[dict], source: str) -> None:
        for rank, row in enumerate(rows):
            kept = pool.setdefault(row["chunk_id"], {**row, "sources": set(), "fused": 0.0})
            kept["sources"].add(source)
            kept["fused"] += 1.0 / (60 + rank)

    from aicodex.legal_data.spanish import normalize_for_embedding
    from aicodex.rag.embedding import embed_texts

    k = cfg.caselaw_candidates
    vectors = embed_texts(legal_cfg.retrieval.embedding_model, [normalize_for_embedding(q) for q in queries],
                          device=legal_cfg.retrieval.device)
    for qi, vector in enumerate(vectors):
        result = collection.query(vector, k)
        add([_row(cid, doc, meta, dist) for cid, doc, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0])], f"d{qi}")

    from aicodex.legal.corpus_lexical import lexical_index

    lexical = lexical_index(cfg.vectordb_dir, category) if legal_cfg.retrieval.corpus_lexical else None
    if lexical is not None:
        for qi, text in enumerate(queries):
            ids = [cid for cid, _ in lexical.search(text, k // 2)]
            missing = [cid for cid in ids if cid not in pool]
            if missing:
                got = collection.collection.get(ids=missing, include=["documents", "metadatas"])
                rows = {cid: _row(cid, doc, meta, None)
                        for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])}
            else:
                rows = {}
            add([pool.get(cid) or rows[cid] for cid in ids if cid in pool or cid in rows], f"k{qi}")

    candidates = sorted(pool.values(), key=lambda c: -c["fused"])[:k]

    from aicodex.legal.retrieval import _reranker, rerank_limit, reranker_device

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


def _date(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}" if len(value) == 8 and value.isdigit() else value


def render_caselaw(hits: list[dict]) -> str:
    """The excerpts as a block of their own, marked as judgments (not statute text)."""
    if not hits:
        return ""
    blocks = []
    for i, h in enumerate(hits, 1):
        head = " | ".join(str(x) for x in (h.get("case_citation"), h.get("ecli"), h.get("chamber"), h.get("case_name"),
                                           _date(h.get("decision_date") or "")) if x)
        body = h["text"].split("\n\n", 1)[-1]  # the chunk's own header repeats the line above
        blocks.append(f"[C{i}] {head}\n{body}")
    return ("<case_law>\nExcerpts from judgments of the Spanish Tribunal Constitucional, found by search: they show "
            "how the court has interpreted the Constitution and applied the law, and its doctrine binds every court "
            "(art. 5.1 LOPJ). Rely on <context> for what the law says; use a judgment where it bears on the question, "
            "cite it by its number and ECLI (e.g. STC 84/2025, ECLI:ES:TC:2025:84), and don't treat an excerpt as the "
            "court's ruling unless it states it (fundamentos jurídicos, fallo).\n\n" + "\n\n".join(blocks) + "\n</case_law>")


def caselaw_record(hits: list[dict]) -> list[dict]:
    """What an answers.jsonl row keeps about the excerpts used."""
    return [{"chunk_id": h["chunk_id"], "citation": h.get("case_citation"), "ecli": h.get("ecli"),
             "score": round(h["score"], 4) if "score" in h else None, "source_url": h.get("source_url")}
            for h in hits]
