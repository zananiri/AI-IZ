"""Token counting with the embedding model's tokenizer (chunk sizes are in its tokens), with a
regex word counter as the fallback for offline dry runs and tests."""

from __future__ import annotations

import re
import warnings

from .state import log, quiet_hf

_WORD_RE = re.compile(r"\w+|[^\w\s]")


class RegexTokenizer:
    """Words and punctuation marks: about 0.7 of an XLM-R (bge-m3) count on Hebrew legal text."""

    name = "regex"

    def count(self, texts: list[str]) -> list[int]:
        return [len(_WORD_RE.findall(t)) for t in texts]


class HFTokenizer:
    def __init__(self, model_name: str):
        quiet_hf()
        from transformers import AutoTokenizer

        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.model_max_length = 10**9  # counting only: no truncation, no length warning

    def count(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ids = self.tok(list(texts), add_special_tokens=False, return_attention_mask=False,
                           return_token_type_ids=False)["input_ids"]
        return [len(x) for x in ids]


_CACHE: dict[str, object] = {}


def get_tokenizer(cfg: dict):
    choice = cfg["chunk"].get("tokenizer", "auto")
    model = cfg["embed"]["model"]
    key = f"{choice}:{model}"
    if key in _CACHE:
        return _CACHE[key]
    if choice == "regex" or (choice == "auto" and model.startswith("hashing-test")):
        tok = RegexTokenizer()
    else:
        name = model if choice == "auto" else choice
        try:
            tok = HFTokenizer(name)
        except Exception as exc:  # noqa: BLE001 -- offline / not installed
            log(f"WARNING: tokenizer {name} unavailable ({type(exc).__name__}: {exc}); "
                "counting words with a regex -- chunk sizes will not match the embedding model")
            tok = RegexTokenizer()
    _CACHE[key] = tok
    return tok
