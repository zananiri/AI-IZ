"""Retrieval over the local canon-law vector store: a ChromaDB persistent
collection populated offline by scripts/ingest_canon_law.py. Embeddings are
computed in-process with sentence-transformers -- no model server needed,
so retrieval works even when vLLM/Ollama isn't up (only the generation step
in canon/pipeline.py needs the chat model). See config.canon.

Both `chromadb` and `sentence_transformers` are imported lazily (inside the
cached getters below) since they're part of the optional `canon` dependency
group -- importing this module shouldn't fail just because the extra isn't
installed; only actually calling `retrieve`/`upsert_chunks` should.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from docslides.config import get_config
from docslides.rag import embedding as rag_embedding

_COLLECTION_NAME = "canon_law"


@dataclass
class RetrievedChunk:
    text: str
    code: str
    number: str
    paragraph: str | None
    breadcrumb: str
    source_url: str
    language: str
    distance: float


@lru_cache(maxsize=1)
def _get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=get_config().canon.vectordb_dir)
    return client.get_or_create_collection(_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def reset_collection() -> None:
    """Delete the whole canon collection (all codes) and recreate it empty."""
    import chromadb

    client = chromadb.PersistentClient(path=get_config().canon.vectordb_dir)
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:  # noqa: BLE001 -- didn't exist yet
        pass
    _get_collection.cache_clear()


def embed_texts(texts: list[str]) -> list[list[float]]:
    return rag_embedding.embed_texts(get_config().canon.embedding_model, texts)


def upsert_chunks(chunks) -> None:
    """`chunks`: list[canon.chunking.Chunk]. Embeds and upserts into Chroma,
    keyed by each chunk's stable id -- safe to re-run after a source-text
    update (see scripts/ingest_canon_law.py)."""
    if not chunks:
        return
    collection = _get_collection()
    embeddings = embed_texts([c.text for c in chunks])
    collection.upsert(
        ids=[c.id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "code": c.code,
                "number": c.number,
                "paragraph": c.paragraph or "",
                "breadcrumb": c.breadcrumb,
                "source_url": c.source_url,
                "language": c.language,
            }
            for c in chunks
        ],
    )


def collection_count() -> int:
    return _get_collection().count()


def retrieve(query: str, top_k: int | None = None, codes: list[str] | None = None) -> list[RetrievedChunk]:
    collection = _get_collection()
    top_k = top_k or get_config().canon.top_k
    where = {"code": {"$in": codes}} if codes else None
    query_embedding = embed_texts([query])[0]

    result = collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where)

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    return [
        RetrievedChunk(
            text=doc,
            code=meta["code"],
            number=meta["number"],
            paragraph=meta["paragraph"] or None,
            breadcrumb=meta["breadcrumb"],
            source_url=meta["source_url"],
            language=meta["language"],
            distance=dist,
        )
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]
