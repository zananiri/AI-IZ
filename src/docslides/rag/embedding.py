"""In-process sentence-transformers embeddings for the Legal tab and the legal corpus
(scripts/legal_data/vectorize.py). Cached per model name and device, so callers configured with
the same model (BAAI/bge-m3 by default) share one loaded copy instead of each holding ~2GB of weights.

`sentence_transformers` is imported lazily: it lives in the optional `legal` dependency group,
and importing this module must not require it.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4)
def _get_embedder(model_name: str, device: str | None = None, full_precision: bool = False):
    from sentence_transformers import SentenceTransformer

    from docslides.config import get_config
    from docslides.system_memory import has_room

    # Out of RAM, loading doesn't raise: the Rust tokenizer aborts the whole app. Raise instead
    # (not cached, so the next call tries again).
    needed = get_config().legal.retrieval.min_free_ram_gb_for_embedder
    ok, free = has_room(needed)
    if not ok:
        raise MemoryError(
            f"Not enough free RAM to load the embedding model {model_name}: "
            f"{(free or 0) / 1024**3:.1f} GB free, {needed} GB needed. Close other programs "
            "(or use a smaller LLM in Ollama) and try again."
        )

    if device and device.startswith("cuda") and not full_precision:
        # fp16 halves time and memory on a GPU. Cast on the CPU, then move: loading in fp32 on a GPU
        # the LLM has nearly filled can fail before the cast frees anything (the 30 Sep reranker OOM).
        return SentenceTransformer(model_name, device="cpu").half().to(device)
    return SentenceTransformer(model_name, device=device)


def get_embedder(model_name: str, device: str | None = None, full_precision: bool = False):
    """`full_precision` keeps fp32 on a GPU too: vectorize.py re-encodes, with it, the rare texts whose
    fp16 vectors come out with NaN/Inf (Chroma rejects a whole batch holding one)."""
    return _get_embedder(model_name, device, full_precision)


def embed_texts(
    model_name: str, texts: list[str], batch_size: int | None = None, device: str | None = None
) -> list[list[float]]:
    """bge-m3 has no query/passage prefix convention (unlike e.g. e5), so
    query and passage text are embedded identically."""
    kwargs = {"batch_size": batch_size} if batch_size else {}
    return _get_embedder(model_name, device).encode(list(texts), normalize_embeddings=True, **kwargs).tolist()
