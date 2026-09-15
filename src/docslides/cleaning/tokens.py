"""Token counting for chunk-budget decisions.

Uses the real Qwen tokenizer via `transformers` when it's available locally
(recommended -- download it alongside the model weights, no internet needed
at inference time), falling back to a conservative heuristic otherwise so
chunking still works before that dependency is set up.
"""

from __future__ import annotations

from docslides.config import get_config

_tokenizer = None
_tokenizer_load_attempted = False


def _get_tokenizer():
    global _tokenizer, _tokenizer_load_attempted
    if _tokenizer_load_attempted:
        return _tokenizer
    _tokenizer_load_attempted = True
    try:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(get_config().llm.model)
    except Exception:  # noqa: BLE001 -- optional; heuristic fallback below covers this
        _tokenizer = None
    return _tokenizer


def count_tokens(text: str) -> int:
    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    # Heuristic fallback: ~4 chars/token for Latin scripts, ~2.5 for Arabic/Hebrew
    # (denser subword tokenization for non-Latin scripts).
    has_non_latin = any(ord(ch) > 0x0590 for ch in text)
    divisor = 2.5 if has_non_latin else 4.0
    return max(1, int(len(text) / divisor))
