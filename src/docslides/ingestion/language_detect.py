"""Per-page/segment language detection.

Runs fastText's lid.176 model when available (higher accuracy, handles short
segments better), falling back to py3langid (pure-Python, no model download
required beyond the pip package) so language detection never blocks on a
missing binary model file.

Detection happens per page (and later, per sentence-segment during cleaning)
-- we never assume one language for a whole document.
"""

from __future__ import annotations

from pathlib import Path

from docslides.config import get_config
from docslides.logging_setup import get_logger

logger = get_logger(__name__)

try:
    import fasttext  # type: ignore

    _FASTTEXT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTTEXT_AVAILABLE = False

try:
    import py3langid as langid  # type: ignore

    _LANGID_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LANGID_AVAILABLE = False

_fasttext_model = None


def _get_fasttext_model():
    global _fasttext_model
    if _fasttext_model is not None:
        return _fasttext_model
    model_path = Path(get_config().languages.fasttext_model_path)
    if not model_path.exists():
        return None
    _fasttext_model = fasttext.load_model(str(model_path))
    return _fasttext_model


def detect_language(text: str) -> str | None:
    """Return an ISO 639-1 code among the configured supported languages, or
    None if the text is too short/ambiguous to classify confidently."""
    text = text.strip().replace("\n", " ")
    if len(text) < 3:
        return None

    supported = set(get_config().languages.supported)

    if _FASTTEXT_AVAILABLE:
        model = _get_fasttext_model()
        if model is not None:
            labels, probs = model.predict(text, k=1)
            lang = labels[0].replace("__label__", "")
            if lang in supported:
                return lang
            logger.debug("fasttext_detected_unsupported_language", lang=lang)

    if _LANGID_AVAILABLE:
        langid.set_languages(list(supported))
        lang, _score = langid.classify(text)
        if lang in supported:
            return lang

    logger.warning("language_detection_unavailable_or_inconclusive", text_preview=text[:50])
    return None


def detect_languages_per_page(pages_text: list[str]) -> list[str | None]:
    return [detect_language(t) for t in pages_text]
