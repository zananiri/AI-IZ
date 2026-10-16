"""In-process sentence-transformers embeddings for the Legal tab and the legal corpus
(scripts/legal_data/vectorize.py). Cached per model name and device, so callers configured with
the same model share one loaded copy instead of each holding its weights twice.

Queries and documents are encoded differently when the model expects it: instruction-tuned
embedders (Qwen3-Embedding) take a task instruction before the query only; e5-style models
prefix "query: " / "passage: "; a model whose sentence-transformers config defines a "query"
prompt (e.g. SINAI's Spanish legal embedder) gets it on queries. bge-m3 has no convention, so it
encodes both identically. `query_prompt` decides; legal.retrieval.query_instruction overrides the
instruction text.

`sentence_transformers` is imported lazily: it lives in the optional `legal` dependency group,
and importing this module must not require it.
"""

from __future__ import annotations

from functools import lru_cache

# The retrieval task, for instruction-tuned embedders. Qwen recommends writing instructions in English.
LEGAL_TASK = ("Given a legal question about Spanish or EU law, retrieve the articles of statutes, regulations and "
              "court judgments that answer it")


@lru_cache(maxsize=4)
def _get_embedder(model_name: str, device: str | None = None, full_precision: bool = False):
    from sentence_transformers import SentenceTransformer

    if device and device.startswith("cuda") and not full_precision:
        # fp16 halves time and memory on a GPU. Cast on the CPU, then move: loading in fp32 on a GPU
        # the LLM has nearly filled can fail before the cast frees anything (the 30 Sep reranker OOM).
        return SentenceTransformer(model_name, device="cpu").half().to(device)
    return SentenceTransformer(model_name, device=device)


def get_embedder(model_name: str, device: str | None = None, full_precision: bool = False):
    """`full_precision` keeps fp32 on a GPU too: vectorize.py re-encodes, with it, the rare texts whose
    fp16 vectors come out with NaN/Inf (Chroma rejects a whole batch holding one)."""
    return _get_embedder(model_name, device, full_precision)


def _configured_instruction() -> str | None:
    try:
        from aicodex.config import get_config

        return get_config().legal.retrieval.query_instruction
    except Exception:  # noqa: BLE001 -- no config (tests, scripts): the default
        return None


def query_prompt(model_name: str, model=None) -> tuple[str | None, str | None]:
    """(prompt text, prompt_name) to encode a query with: at most one of them is set."""
    name = model_name.lower()
    if "qwen3-embedding" in name or "gte-qwen" in name:
        return f"Instruct: {_configured_instruction() or LEGAL_TASK}\nQuery:", None
    if "e5" in name.split("/")[-1].split("-"):
        return "query: ", None
    prompts = getattr(model, "prompts", None) or {}
    if "query" in prompts:
        return None, "query"
    return None, None


def document_prompt(model_name: str) -> str | None:
    name = model_name.lower()
    return "passage: " if "e5" in name.split("/")[-1].split("-") else None


def _encode(model, texts: list[str], prompt: str | None, prompt_name: str | None, **kwargs):
    if prompt:
        kwargs["prompt"] = prompt
    elif prompt_name:
        kwargs["prompt_name"] = prompt_name
    return model.encode(list(texts), normalize_embeddings=True, **kwargs)


def encode_queries(model_name: str, texts: list[str], batch_size: int | None = None, device: str | None = None):
    model = _get_embedder(model_name, device)
    prompt, prompt_name = query_prompt(model_name, model)
    kwargs = {"batch_size": batch_size} if batch_size else {}
    return _encode(model, texts, prompt, prompt_name, **kwargs).tolist()


def encode_documents(model_name: str, texts: list[str], batch_size: int | None = None, device: str | None = None,
                     max_seq_length: int | None = None):
    model = _get_embedder(model_name, device)
    if max_seq_length:
        model.max_seq_length = max_seq_length
    kwargs = {"batch_size": batch_size} if batch_size else {}
    return _encode(model, texts, document_prompt(model_name), None, **kwargs).tolist()


def embed_texts(
    model_name: str, texts: list[str], batch_size: int | None = None, device: str | None = None
) -> list[list[float]]:
    """Query-side encoding (every caller of this embeds questions and search phrases): the model's
    query prompt, if it has one (query_prompt)."""
    return encode_queries(model_name, texts, batch_size=batch_size, device=device)
