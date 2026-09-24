"""Retrieval over the Legal tab's Israeli-law vector store (a ChromaDB
collection populated only by approving staged batches -- see
legal/staging.py and scripts/ingest_legal.py).

A query fetches `fetch_k` candidates, drops any whose normalized body
duplicates a closer one, and picks up to `top_k` by maximal marginal
relevance -- so duplicates that slipped through ingestion can't fill the
context. Candidates come from two rankings fused by reciprocal rank --
embedding distance and Hebrew-aware keyword search (legal/keyword.py),
which catches exact identifiers embeddings blur -- and, when a reranker is
configured, a cross-encoder reorders them and scores how directly each
answers (`rerank_margin` keeps hits close to the best score;
`min_rerank_score` sets the thin-coverage flag). Without one, hits much
farther than the best by embedding distance are cut (`relevance_margin`).
Sections the question names ("סעיף 132א") are looked up by number first.
It also pulls in:
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

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from docslides.cleaning.tokens import count_tokens
from docslides.config import get_config
from docslides.legal import bundle
from docslides.legal.chunking import normalized_body
from docslides.legal.insertions import join_spaced_section_numbers
from docslides.legal.keyword import KeywordIndex
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
    via: Literal["search", "section_lookup", "sibling_part", "cross_reference"]
    score: float | None = None  # reranker relevance, 0-1, when the reranker ran


@dataclass
class RetrievalResult:
    chunks: list[RetrievedLegalChunk]
    low_relevance: bool
    best_distance: float | None
    bundle_verification: bundle.VerificationLevel
    rejected_chunk_ids: list[str] = field(default_factory=list)
    duplicate_chunk_ids: list[str] = field(default_factory=list)  # same body as a closer hit
    trimmed_chunk_ids: list[str] = field(default_factory=list)  # cut by a relevance margin or the token budget
    best_rerank_score: float | None = None  # how directly the best provision answers, 0-1

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


def _get_embedded(where: dict | None = None, ids: list[str] | None = None) -> list[tuple[str, str, dict, list[float]]]:
    if ids is not None and not ids:
        return []
    kwargs: dict = {"include": ["documents", "metadatas", "embeddings"]}
    if where is not None:
        kwargs["where"] = where
    if ids is not None:
        kwargs["ids"] = ids
    result = _get_collection().get(**kwargs)
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


# --- derived indexes: keyword search and section-number lookup -------------------

_derived_cache: dict = {}


def _derived_indexes() -> dict:
    """Keyword index and section-number index over every stored chunk,
    rebuilt only when the collection or the signed bundle changes."""
    from pathlib import Path

    manifest = Path(get_config().legal.ingestion.bundle_manifest)
    collection = _get_collection()
    stamp = (id(collection), manifest.stat().st_mtime if manifest.exists() else 0.0, collection.count())
    if _derived_cache.get("stamp") != stamp:
        rows = all_chunks()
        _derived_cache.update(
            stamp=stamp,
            keyword=KeywordIndex([(chunk_id, text) for chunk_id, text, _ in rows]),
            sections=_section_index(rows),
        )
    return _derived_cache


def _section_index(rows: list[tuple[str, str, dict]]) -> dict[str, list[str]]:
    """Section number -> chunks that hold it: a law's own sections, the
    provisions an amending chunk inserts ("116יז10"), and the sections it
    amends ("62" for 'בסעיף 62(ג) ... במקום')."""
    from docslides.legal import amendments

    index: dict[str, list[str]] = {}
    for chunk_id, _, meta in rows:
        numbers = set()
        own = meta.get("section_number") or ""
        if own and own != "preamble":
            numbers.add(own)
        if meta.get("inserted_section"):
            numbers.add(amendments._base_section(meta["inserted_section"]))
        for ref in amendments.decode(meta.get("amends") or "[]"):
            numbers.update(amendments._base_section(s) for s in ref.sections)
        for number in numbers:
            index.setdefault(number, []).append(chunk_id)
    return index


_QUERY_SECTION_RE = re.compile(r"סעי(?:ף|פים)\s+(?P<num>\d{1,4}(?:[א-ת]{1,3}\d{0,3})?)(?:\((?P<sub>[^)]{1,3})\))?")


def named_sections(query: str) -> list[tuple[str, str | None]]:
    """(section number, subsection) pairs a question names: "סעיף 62(ג)" ->
    ("62", "ג"), "בסעיף 116 יז 10" -> ("116יז10", None)."""
    found: list[tuple[str, str | None]] = []
    for match in _QUERY_SECTION_RE.finditer(join_spaced_section_numbers(query)):
        pair = (match.group("num"), match.group("sub"))
        if pair not in found:
            found.append(pair)
    return found


@lru_cache(maxsize=2)
def _reranker(model_name: str):
    """The cross-encoder, or None if it can't be loaded (then retrieval ranks
    by embedding distance alone, as before the reranker existed)."""
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(model_name, max_length=get_config().legal.retrieval.rerank_max_length)
    except Exception as exc:  # noqa: BLE001 -- a missing model must not block answering
        logger.warning("legal_reranker_unavailable", model=model_name, error=str(exc))
        return None


@dataclass
class _Candidate:
    chunk_id: str
    text: str
    meta: dict
    distance: float
    embedding: list[float]
    dense_rank: int | None = None
    keyword_rank: int | None = None
    score: float | None = None

    def fused(self) -> float:
        """Reciprocal-rank fusion of the embedding and keyword rankings."""
        return sum(1.0 / (60 + rank) for rank in (self.dense_rank, self.keyword_rank) if rank is not None)


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
    candidates: dict[str, _Candidate] = {}
    for rank, (chunk_id, text, meta, distance, embedding) in enumerate(
        zip(
            (result.get("ids") or [[]])[0],
            (result.get("documents") or [[]])[0],
            (result.get("metadatas") or [[]])[0],
            (result.get("distances") or [[]])[0],
            (result.get("embeddings") or [[]])[0],
        )
    ):
        candidates[chunk_id] = _Candidate(chunk_id, text, meta, distance, embedding, dense_rank=rank)

    if cfg.keyword_search:
        hits = _derived_indexes()["keyword"].search(query, cfg.fetch_k)
        missing = [chunk_id for chunk_id, _ in hits if chunk_id not in candidates]
        for chunk_id, text, meta, embedding in _get_embedded(ids=missing):
            candidates[chunk_id] = _Candidate(chunk_id, text, meta, _nearest_first(
                [(chunk_id, text, meta, embedding)], query_vector)[0][3], embedding)
        for rank, (chunk_id, _) in enumerate(hits):
            if chunk_id in candidates:
                candidates[chunk_id].keyword_rank = rank

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

    def admit(chunk_id: str, text: str, meta: dict, distance: float | None, via, score: float | None = None) -> None:
        if chunk_id in chunks or not verified(chunk_id, text, meta):
            return
        if not budget.take(text, force=not chunks):  # the first chunk always goes in
            trimmed.append(chunk_id)
            return
        chunks[chunk_id] = RetrievedLegalChunk(chunk_id, text, ChunkMetadata.from_chroma(meta), distance, via, score)

    # Candidate pool, best fused rank first: bundle-verified, and one copy per normalized
    # body, so a provision indexed twice can't take two context slots.
    pool: list[_Candidate] = []
    seen_bodies: set[str] = set()
    for candidate in sorted(candidates.values(), key=lambda c: (-c.fused(), c.distance)):
        if not verified(candidate.chunk_id, candidate.text, candidate.meta):
            continue
        # Same law version + same body = the same provision indexed twice. Identical wording in two
        # different laws (a shared definition) is two provisions and both must stay citable.
        body = (candidate.meta.get("law_id"), candidate.meta.get("effective_date_start"),
                normalized_body(candidate.text))
        if body in seen_bodies:
            duplicates.append(candidate.chunk_id)
            continue
        seen_bodies.add(body)
        pool.append(candidate)

    reranker = _reranker(cfg.reranker_model) if cfg.reranker_model else None
    best_score: float | None = None
    if reranker is not None and pool:
        # A cross-encoder reads question and provision together: its score says how directly
        # the provision answers, which embedding distance can't (an unanswerable question's
        # nearest provision is often as close as an answerable one's).
        head = pool[: cfg.rerank_candidates]
        for candidate, score in zip(head, reranker.predict([(query, c.text) for c in head])):
            candidate.score = float(score)
        head.sort(key=lambda c: c.score, reverse=True)
        best_score = head[0].score
        picked = head[: cfg.top_k]
        low_relevance = best_score < cfg.min_rerank_score
        kept = [
            c for c in picked
            if c is picked[0]
            or ((cfg.rerank_margin is None or c.score >= best_score - cfg.rerank_margin) and c.score >= cfg.rerank_floor)
        ]
    else:
        if cfg.keyword_search:
            picked = pool[: cfg.top_k]  # fused order
        else:
            by_distance = sorted(pool, key=lambda c: c.distance)
            mmr = _mmr([(c.chunk_id, c.text, c.meta, c.distance, c.embedding) for c in by_distance], cfg.top_k,
                       cfg.mmr_lambda)
            picked = [candidates[row[0]] for row in mmr]
        best = min((c.distance for c in picked), default=None)
        # Thin coverage is judged on the whole selection, before the margin below trims it.
        low_relevance = sum(1 for c in picked if c.distance <= cfg.low_relevance_distance) < cfg.min_relevant_chunks
        kept = [
            c for c in picked
            if cfg.relevance_margin is None or c.distance <= best + cfg.relevance_margin
            or c.keyword_rank == 0  # the best exact-term match stays even when embeddings rank it far
        ]
    trimmed.extend(c.chunk_id for c in picked if c not in kept)

    # Sections the question names come first: an exact identifier beats any similarity.
    if cfg.section_lookup_max:
        index = _derived_indexes()["sections"] if named_sections(query) else {}
        for number, sub in named_sections(query):
            rows = _nearest_first(_get_embedded(ids=index.get(number, [])), query_vector)
            if sub:  # "סעיף 62(ג)": a chunk naming that very subsection first
                wanted = f"{number}({sub})"
                rows.sort(key=lambda row: wanted not in join_spaced_section_numbers(row[1]))
            for chunk_id, text, meta, distance in rows[: cfg.section_lookup_max]:
                admit(chunk_id, text, meta, distance, "section_lookup")
                low_relevance = False  # the question's own section is in hand

    for candidate in kept:
        admit(candidate.chunk_id, candidate.text, candidate.meta, candidate.distance, "search", candidate.score)

    search_hits = [c for c in chunks.values() if c.via in ("search", "section_lookup")]
    best = min((c.distance for c in search_hits if c.distance is not None), default=None)
    relevant = [c for c in search_hits if c.distance is not None and c.distance <= cfg.low_relevance_distance]

    multipart = sorted({c.metadata.source_id for c in search_hits if c.metadata.part_count > 1})
    if multipart:
        rows = _nearest_first(_get_embedded(where={"source_id": {"$in": multipart}}), query_vector)
        for chunk_id, text, meta, distance in rows:
            if chunk_id in chunks:
                continue
            if cfg.sibling_margin is not None and best is not None and distance > best + cfg.sibling_margin:
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
        rows = _nearest_first(_get_embedded(where={"section_key": {"$in": referenced}}), query_vector)
        for chunk_id, text, meta, distance in rows:
            admit(chunk_id, text, meta, distance, "cross_reference")

    return RetrievalResult(
        chunks=list(chunks.values()),
        low_relevance=low_relevance,
        best_distance=best,
        bundle_verification=verification,
        rejected_chunk_ids=rejected,
        duplicate_chunk_ids=duplicates,
        # A chunk cut at one stage can still come in at a later one (a far hit that a
        # relevant hit cross-references), so report only what stayed out.
        trimmed_chunk_ids=[cid for cid in dict.fromkeys(trimmed) if cid not in chunks],
        best_rerank_score=best_score,
    )
