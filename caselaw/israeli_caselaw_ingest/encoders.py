"""Text encoders: a sentence-transformers model (bge-m3 by default), and a hashing encoder used by
the tests and offline dry runs (embed.model: "hashing-test:<dim>")."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from .state import log, quiet_hf


def instructions(cfg: dict) -> tuple[str, str]:
    """(passage prefix, query prefix): E5 models need "passage: " / "query: "; bge-m3 needs none."""
    ecfg = cfg["embed"]
    model = ecfg["model"].lower()
    auto_passage, auto_query = ("passage: ", "query: ") if "e5" in model else ("", "")
    passage = ecfg.get("passage_instruction", "auto")
    query = ecfg.get("query_instruction", "auto")
    return (auto_passage if passage == "auto" else passage or ""), (auto_query if query == "auto" else query or "")


class HashEncoder:
    """Character-trigram hashing: deterministic, no weights, for tests only."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.name = f"hashing-test:{dim}"

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            t = re.sub(r"\s+", " ", t)
            for j in range(max(len(t) - 2, 1)):
                h = int.from_bytes(hashlib.blake2b(t[j:j + 3].encode(), digest_size=4).digest(), "little")
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.maximum(norms, 1e-9)).astype(np.float16)


class STEncoder:
    def __init__(self, model_name: str, device: str = "auto", fp16: bool = True, max_seq_length: int = 1024,
                 local_dir=None):
        quiet_hf()
        import torch
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        source = str(local_dir) if local_dir is not None and (local_dir / "config.json").exists() else model_name
        self.model = SentenceTransformer(source, device=device)
        if fp16 and device == "cuda":
            self.model.half()
        self.model.max_seq_length = max_seq_length
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        self.name = model_name
        log(f"encoder: {model_name} on {device}{' fp16' if fp16 and device == 'cuda' else ''}, dim {self.dim}")

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        import torch

        while True:
            try:
                vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                         convert_to_numpy=True, show_progress_bar=False)
                self.batch_size = batch_size
                return vecs.astype(np.float16)
            except torch.cuda.OutOfMemoryError:
                if batch_size <= 1:
                    raise
                torch.cuda.empty_cache()
                batch_size //= 2
                log(f"encoder: CUDA out of memory, batch size -> {batch_size}")

    def save(self, path) -> None:
        self.model.save(str(path))


def get_encoder(cfg: dict, paths=None):
    ecfg = cfg["embed"]
    name = ecfg["model"]
    if name.startswith("hashing-test"):
        dim = int(name.split(":")[1]) if ":" in name else 64
        return HashEncoder(dim)
    local = paths.models / name.replace("/", "__") if paths is not None else None
    return STEncoder(name, ecfg.get("device", "auto"), ecfg.get("fp16", True), ecfg.get("max_seq_length", 1024), local)
