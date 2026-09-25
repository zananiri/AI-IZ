"""In-process sentence-transformers embeddings for the Legal tab and the legal corpus
(scripts/legal_data/vectorize.py). Cached per model name and device, so callers configured with
the same model (BAAI/bge-m3 by default) share one loaded copy instead of each holding ~2GB of weights.

`sentence_transformers` is imported lazily: it lives in the optional `legal` dependency group,
and importing this module must not require it.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4)
def _get_embedder(model_name: str, device: str | None = None):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    if device and device.startswith("cuda"):
        model.half()  # bulk corpus indexing on GPU (scripts/legal_data/vectorize.py): fp16 halves time and memory
    return model


def get_embedder(model_name: str, device: str | None = None):
    return _get_embedder(model_name, device)


def embed_texts(
    model_name: str, texts: list[str], batch_size: int | None = None, device: str | None = None
) -> list[list[float]]:
    """bge-m3 has no query/passage prefix convention (unlike e.g. e5), so
    query and passage text are embedded identically."""
    kwargs = {"batch_size": batch_size} if batch_size else {}
    return _get_embedder(model_name, device).encode(list(texts), normalize_embeddings=True, **kwargs).tolist()
