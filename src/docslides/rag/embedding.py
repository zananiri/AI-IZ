"""In-process sentence-transformers embeddings shared by the RAG tabs (Canon
GPT, Legal GPT). Cached per model name, so two tabs configured with the same
model (both default to BAAI/bge-m3) share one loaded copy instead of each
holding ~2GB of weights.

`sentence_transformers` is imported lazily: it lives in the optional `canon`
/`legal` dependency groups, and importing this module must not require it.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4)
def _get_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(model_name: str, texts: list[str]) -> list[list[float]]:
    """bge-m3 has no query/passage prefix convention (unlike e.g. e5), so
    query and passage text are embedded identically."""
    return _get_embedder(model_name).encode(list(texts), normalize_embeddings=True).tolist()
