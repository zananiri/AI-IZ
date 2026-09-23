"""Retrieval over the Legal tab's Israeli-law vector store (a ChromaDB
collection populated only by approving staged batches -- see
legal/staging.py and scripts/ingest_legal.py).

A query fetches `fetch_k` candidates, drops any whose normalized body
duplicates a closer one, and picks up to `top_k` by maximal marginal
relevance -- so duplicates that slipped through ingestion can't fill the
context. Hits much farther from the question than the best one are then
cut (`relevance_margin`). It also pulls in:
  * the other parts of any multi-part provision that was hit, when they are
    close to the question too (`sibling_margin`; with it off, every part);
  * the sections that relevant hits cross-reference (ingestion spec 0.1),
    capped at config.legal.retrieval.max_cross_refs, closest chunks first.
Everything shares one token budget (`max_evidence_tokens`), filled
best-first, and the best hit is always kept: a small model given a pile of
loosely related provisions drops or garbles the one that answers. What was
cut is reported in `trimmed_chunk_ids`.

Every returned chunk is checked against the signed bundle (legal/bundle.py);
anything missing from it or whose hash doesn't match is dropped and reported
in `rejected_chunk_ids`.

chromadb is imported lazily -- it's in the optional `legal` dependency group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from docslides.cleaning.tokens import count_tokens
from docslides.config import get_config
from docslides.legal import bundle
from docslides.legal.chunking import normalized_body
from docslides.legal.models import ChunkMetadata, LegalChunk, content_hash
from docslides.logging_setup import get_logger
from docslides.rag import embedding as rag_embedding

logger = get_logger(__name__)

_COLLECTION_NAME = "israeli_law"


@dataclass
class RetrievedLegalChunk:
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    distance: float | None
    via: Literal["search", "sibling_part", "cross_reference"]


@dataclass
class RetrievalResult:
    chunks: list[RetrievedLegalChunk]
    low_relevance: bool
    best_distance: float | None
    bundle_verification: bundle.VerificationLevel
    rejected_chunk_ids: list[str] = field(default_factory=list)
    duplicate_chunk_ids: list[str] = field(default_factory=list)  # same body as a closer hit
    trimmed_chunk_ids: list[str] = field(default_factory=list)  # cut by a relevance margin or the token budget

    def by_source_id(self) -> dict[str, list[RetrievedLegalChunk]]:
        grouped: dict[str, list[RetrievedLegalChunk]] = {}
        for chunk in self.chunks:
            grouped.setdefault(chunk.metadata.source_id, []).append(chunk)
        for parts in grouped.values():
            parts.sort(key=lambda c: c.metadata.part_index)
        return grouped


@lru_cache(maxsize=1)
def _get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=get_config().legal.retrieval.vectordb_dir)
    # embedding_function=None: every write/query passes bge-m3 vectors explicitly; never let
    # Chroma fall back to (and download) its own default embedder.
    return client.get_or_create_collection(
        _COLLECTION_NAME, metadata={"hnsw:space": "cosine"}, embedding_function=None
    )


def _embed(texts: list[str]) -> list[list[float]]:
    return rag_embedding.embed_texts(get_config().legal.retrieval.embedding_model, texts)


def upsert_chunks(chunks: list[LegalChunk]) -> None:
    if not chunks:
        return
    _get_collection().upsert(
        ids=[c.metadata.chunk_id for c in chunks],
        embeddings=_embed([c.text for c in chunks]),
        documents=[c.text for c in chunks],
        metadatas=[c.metadata.to_chroma() for c in chunks],
    )


def delete_chunks(chunk_ids: list[str]) -> None:
    if chunk_ids:
        _get_collection().delete(ids=chunk_ids)


def chunk_ids_for_version(law_id: str, effective_date_start: str) -> list[str]:
    result = _get_collection().get(
        where={"$and": [{"law_id": law_id}, {"effective_date_start": effective_date_start}]}, include=[]
    )
    return list(result.get("ids") or [])


def chunks_for_law(law_id: str) -> list[tuple[str, str, dict]]:
    """Every stored chunk of every version of `law_id`: (chunk_id, text, flat metadata)."""
    return _get_where({"law_id": law_id})


def update_metadatas(chunk_ids: list[str], metadatas: list[dict]) -> None:
    """Metadata-only update (text and embedding untouched) -- used when a
    newer version supersedes an older one (legal/staging.py)."""
    if chunk_ids:
        _get_collection().update(ids=chunk_ids, metadatas=metadatas)


_amendment_cache: dict = {}


def amendment_index() -> dict:
    """target law key -> amendments indexed for it (legal/amendments.py),
    rebuilt only when the signed bundle changes (every approval rewrites it)."""
    from pathlib import Path

    from docslides.legal import amendments

    manifest = Path(get_config().legal.ingestion.bundle_manifest)
    stamp = (manifest.stat().st_mtime if manifest.exists() else 0.0, collection_count())
    if _amendment_cache.get("stamp") != stamp:
        metadatas = _get_collection().get(include=["metadatas"]).get("metadatas") or []
        _amendment_cache.update(stamp=stamp, index=amendments.build_index(metadatas))
    return _amendment_cache["index"]


def collection_count() -> int:
    return _get_collection().count()


def all_chunks() -> list[tuple[str, str, dict]]:
    result = _get_collection().get(include=["documents", "metadatas"])
    return list(zip(result.get("ids") or [], result.get("documents") or [], result.get("metadatas") or []))


def _get_where(where: dict) -> list[tuple[str, str, dict]]:
    result = _get_collection().get(where=where, include=["documents", "metadatas"])
    return list(zip(result.get("ids") or [], result.get("documents") or [], result.get("metadatas") or []))


def _mmr(pool: list[tuple], k: int, lam: float) -> list[tuple]:
    """Maximal marginal relevance over (id, text, meta, cosine distance,
    embedding) candidates: each pick maximizes lam * relevance - (1 - lam) *
    its highest similarity to anything already picked. Near-duplicate chunks
    (the same provision with a margin note moved) score high on the second
    term, so they can't crowd out a second, different section a multi-hop
    question needs."""
    import numpy as np

    if len(pool) <= k:
        return pool
    vectors = np.array([np.asarray(c[4], dtype=float) for c in pool])
    relevance = np.array([1.0 - c[3] for c in pool])
    selected: list[int] = []
    remaining = list(range(len(pool)))
    while remaining and len(selected) < k:
        if selected:
            redundancy = (vectors[remaining] @ vectors[selected].T).max(axis=1)
        else:
            redundancy = np.zeros(len(remaining))
        scores = lam * relevance[remaining] - (1 - lam) * redundancy
        best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)
    return [pool[i] for i in selected]


def _get_where_embedded(where: dict) -> list[tuple[str, str, dict, list[float]]]:
    result = _get_collection().get(where=where, include=["documents", "metadatas", "embeddings"])
    embeddings = result.get("embeddings")
    if embeddings is None:  # a numpy array when present -- never test its truth value
        embeddings = []
    return list(zip(result.get("ids") or [], result.get("documents") or [], result.get("metadatas") or [], embeddings))


def _nearest_first(rows: list[tuple[str, str, dict, list[float]]], query_vector) -> list[tuple[str, str, dict, float]]:
    """(id, text, meta, embedding) rows as (id, text, meta, cosine distance to
    the query), closest first -- so a budget cut drops the least related."""
    import numpy as np

    query = np.asarray(query_vector, dtype=float)
    scored = []
    for chunk_id, text, meta, embedding in rows:
        vector = np.asarray(embedding, dtype=float)
        norm = float(np.linalg.norm(query) * np.linalg.norm(vector)) or 1.0
        scored.append((chunk_id, text, meta, 1.0 - float(query @ vector) / norm))
    return sorted(scored, key=lambda row: row[3])


class _TokenBudget:
    def __init__(self, limit: int | None) -> None:
        self.limit, self.used = limit, 0

    def take(self, text: str, force: bool = False) -> bool:
        tokens = count_tokens(text)
        if force or self.limit is None or self.used + tokens <= self.limit:
            self.used += tokens
            return True
        return False


def retrieve(query: str) -> RetrievalResult:
    cfg = get_config().legal.retrieval
    entries, verification = bundle.verified_entries()
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return RetrievalResult(chunks=[], low_relevance=True, best_distance=None, bundle_verification=verification)

    query_vector = _embed([query])[0]
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(max(cfg.fetch_k, cfg.top_k), count),
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    candidates = list(
        zip(
            (result.get("ids") or [[]])[0],
            (result.get("documents") or [[]])[0],
            (result.get("metadatas") or [[]])[0],
            (result.get("distances") or [[]])[0],
            (result.get("embeddings") or [[]])[0],
        )
    )

    chunks: dict[str, RetrievedLegalChunk] = {}
    rejected: list[str] = []
    duplicates: list[str] = []
    trimmed: list[str] = []
    budget = _TokenBudget(cfg.max_evidence_tokens)

    def verified(chunk_id: str, text: str, meta: dict) -> bool:
        if chunk_id in rejected:
            return False
        entry = entries.get(chunk_id)
        if entry is None or entry.get("sha256") != content_hash(text, meta):
            rejected.append(chunk_id)
            logger.warning("legal_chunk_rejected_by_bundle", chunk_id=chunk_id, in_manifest=entry is not None)
            return False
        return True

    def admit(chunk_id: str, text: str, meta: dict, distance: float | None, via) -> None:
        if chunk_id in chunks or not verified(chunk_id, text, meta):
            return
        if not budget.take(text, force=not chunks):  # the best hit always goes in
            trimmed.append(chunk_id)
            return
        chunks[chunk_id] = RetrievedLegalChunk(chunk_id, text, ChunkMetadata.from_chroma(meta), distance, via)

    # Candidate pool: bundle-verified, and one copy per normalized body (closest kept),
    # so a provision indexed twice can't take two context slots.
    pool, seen_bodies = [], set()
    for chunk_id, text, meta, distance, embedding in sorted(candidates, key=lambda c: c[3]):
        if not verified(chunk_id, text, meta):
            continue
        body = normalized_body(text)
        if body in seen_bodies:
            duplicates.append(chunk_id)
            continue
        seen_bodies.add(body)
        pool.append((chunk_id, text, meta, distance, embedding))

    picked = _mmr(pool, cfg.top_k, cfg.mmr_lambda)
    best = min((c[3] for c in picked), default=None)
    # Thin coverage is judged on the whole selection, before the margin below trims it.
    relevant_count = sum(1 for c in picked if c[3] <= cfg.low_relevance_distance)
    for chunk_id, text, meta, distance, _ in picked:
        if cfg.relevance_margin is not None and distance > best + cfg.relevance_margin:
            trimmed.append(chunk_id)
            continue
        admit(chunk_id, text, meta, distance, "search")

    search_hits = list(chunks.values())
    relevant = [c for c in search_hits if c.distance is not None and c.distance <= cfg.low_relevance_distance]

    multipart = sorted({c.metadata.source_id for c in search_hits if c.metadata.part_count > 1})
    if multipart:
        rows = _nearest_first(_get_where_embedded({"source_id": {"$in": multipart}}), query_vector)
        for chunk_id, text, meta, distance in rows:
            if chunk_id in chunks:
                continue
            if cfg.sibling_margin is not None and distance > best + cfg.sibling_margin:
                trimmed.append(chunk_id)
                continue
            admit(chunk_id, text, meta, distance, "sibling_part")

    present_sections = {c.metadata.section_key for c in chunks.values()}
    referenced: list[str] = []
    for hit in relevant:
        for key in hit.metadata.cross_references:
            if key not in present_sections and key not in referenced:
                referenced.append(key)
    referenced = referenced[: cfg.max_cross_refs]
    if referenced:
        rows = _nearest_first(_get_where_embedded({"section_key": {"$in": referenced}}), query_vector)
        for chunk_id, text, meta, distance in rows:
            admit(chunk_id, text, meta, distance, "cross_reference")

    return RetrievalResult(
        chunks=list(chunks.values()),
        low_relevance=relevant_count < cfg.min_relevant_chunks,
        best_distance=best,
        bundle_verification=verification,
        rejected_chunk_ids=rejected,
        duplicate_chunk_ids=duplicates,
        # A chunk cut at one stage can still come in at a later one (a far hit that a
        # relevant hit cross-references), so report only what stayed out.
        trimmed_chunk_ids=[cid for cid in dict.fromkeys(trimmed) if cid not in chunks],
    )
