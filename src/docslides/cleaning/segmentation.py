"""Per-language sentence segmentation.

spaCy provides small pipelines for en/fr/es/it/de; Stanza covers ar/he (spaCy
has no small pipeline for either). Both are loaded lazily and cached per
language so the app doesn't pay startup cost for languages that never appear
in a given document.
"""

from __future__ import annotations

from docslides.config import get_config
from docslides.logging_setup import get_logger

logger = get_logger(__name__)

_spacy_pipelines: dict[str, object] = {}
_stanza_pipelines: dict[str, object] = {}


def _get_spacy_pipeline(lang: str):
    if lang in _spacy_pipelines:
        return _spacy_pipelines[lang]
    import spacy

    model_name = get_config().languages.spacy_models[lang]
    try:
        nlp = spacy.load(model_name, exclude=["ner", "lemmatizer", "tagger", "attribute_ruler"])
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' not installed. Run: python -m spacy download {model_name}"
        ) from exc
    if "senter" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    _spacy_pipelines[lang] = nlp
    return nlp


def _get_stanza_pipeline(lang: str):
    if lang in _stanza_pipelines:
        return _stanza_pipelines[lang]
    import stanza

    stanza_lang = get_config().languages.stanza_models[lang]
    try:
        nlp = stanza.Pipeline(lang=stanza_lang, processors="tokenize", download_method=None)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Stanza model for '{stanza_lang}' not installed. Run: "
            f"python -c \"import stanza; stanza.download('{stanza_lang}')\" while online, once."
        ) from exc
    _stanza_pipelines[lang] = nlp
    return nlp


def segment_sentences(text: str, lang: str) -> list[str]:
    """Split `text` into sentences using the correct per-language model."""
    if not text.strip():
        return []

    cfg = get_config().languages
    if lang in cfg.spacy_models:
        nlp = _get_spacy_pipeline(lang)
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if lang in cfg.stanza_models:
        nlp = _get_stanza_pipeline(lang)
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sentences if sent.text.strip()]

    logger.warning("no_segmentation_model_for_language_using_naive_split", lang=lang)
    return [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
