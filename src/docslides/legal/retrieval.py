"""Retrieval over the Legal tab's Israeli-law vector store (a ChromaDB
collection populated only by approving staged batches -- see
legal/staging.py and scripts/ingest_legal.py).

A query fetches `fetch_k` candidates, drops any whose normalized body
duplicates a closer one, and picks the final `top_k` by maximal marginal
relevance -- so duplicates that slipped through ingestion can't fill the
context. It then also pulls in:
  * the sibling parts of any multi-part provision that was hit, so claims are
    checked against the whole provision rather than a fragment of it;
  * the sections that relevant hits cross-reference (ingestion spec 0.1),
    capped at config.legal.retrieval.max_cross_refs.

Every returned chunk is checked against the signed bundle (legal/bundle.py);
anything missing from it or whose hash doesn't match is dropped and reported
in `rejected_chunk_ids`.

chromadb is imported lazily -- it's in the optional `legal` dependency group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

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


def retrieve(query: str) -> RetrievalResult:
    cfg = get_config().legal.retrieval
    entries, verification = bundle.verified_entries()
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return RetrievalResult(chunks=[], low_relevance=True, best_distance=None, bundle_verification=verification)

    result = collection.query(
        query_embeddings=_embed([query]),
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
        if chunk_id not in chunks and verified(chunk_id, text, meta):
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

    for chunk_id, text, meta, distance, _ in _mmr(pool, cfg.top_k, cfg.mmr_lambda):
        admit(chunk_id, text, meta, distance, "search")

    search_hits = list(chunks.values())
    relevant = [c for c in search_hits if c.distance is not None and c.distance <= cfg.low_relevance_distance]

    multipart = sorted({c.metadata.source_id for c in search_hits if c.metadata.part_count > 1})
    if multipart:
        for chunk_id, text, meta in _get_where({"source_id": {"$in": multipart}}):
            admit(chunk_id, text, meta, None, "sibling_part")

    present_sections = {c.metadata.section_key for c in chunks.values()}
    referenced: list[str] = []
    for hit in relevant:
        for key in hit.metadata.cross_references:
            if key not in present_sections and key not in referenced:
                referenced.append(key)
    referenced = referenced[: cfg.max_cross_refs]
    if referenced:
        for chunk_id, text, meta in _get_where({"section_key": {"$in": referenced}}):
            admit(chunk_id, text, meta, None, "cross_reference")

    distances = [c.distance for c in search_hits if c.distance is not None]
    return RetrievalResult(
        chunks=list(chunks.values()),
        low_relevance=len(relevant) < cfg.min_relevant_chunks,
        best_distance=min(distances) if distances else None,
        bundle_verification=verification,
        rejected_chunk_ids=rejected,
        duplicate_chunk_ids=duplicates,
    )
