"""Mask non-translatable spans before sending text to the LLM, and restore
them afterwards.

Categories, applied in this order so more-specific patterns are masked
before more-generic ones can swallow part of them:
  1. formula     -- LaTeX-style $...$/$$...$$ and inline code spans
  2. placeholder -- {{var}}, {var}, %s/%(name)s, <VAR>, [PLACEHOLDER]
  3. unit        -- a number immediately followed by a recognized unit
  4. number      -- remaining bare numeric literals
  5. named_entity -- via per-language NER (spaCy where available, Stanza
                     otherwise), run last over whatever text remains

Each masked span becomes an indexed token like [[NUM_0]], [[UNIT_1]],
[[ENT_2]], [[PLACEHOLDER_3]], [[FORMULA_4]]. The LLM is instructed (see
llm/prompts.py) to copy these tokens through unchanged; `restore_masks`
substitutes the originals back in after generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_FORMULA_RE = re.compile(r"(\${1,2}[^$]+\${1,2}|`[^`]+`)")
_PLACEHOLDER_RE = re.compile(
    r"(\{\{[^{}]+\}\}|\{[^{}\s]+\}|%\([a-zA-Z_]+\)s|%[sd]|<[A-Z_]+>|\[[A-Z_]+\])"
)
# Common unit abbreviations across the supported languages (units themselves
# are not translated -- only spelled-out words like "kilograms" would be).
_UNITS = (
    r"kg|g|mg|km|m|cm|mm|mi|ft|in|lb|lbs|oz|l|ml|kWh|MWh|GW|MW|kW|W|"
    r"°C|°F|%|EUR|USD|GBP|ILS|\$|€|£|₪|Hz|kHz|MHz|GHz|s|ms|min|hrs?|hz"
)
_UNIT_RE = re.compile(
    rf"(?<![\w.])(\d[\d.,]*)\s?({_UNITS})(?![\w])"
)
_NUMBER_RE = re.compile(r"(?<![\w.])\d[\d.,]*%?(?![\w])")


@dataclass
class MaskedSpan:
    token: str
    original: str
    category: str


@dataclass
class MaskedText:
    text: str
    spans: list[MaskedSpan] = field(default_factory=list)


def _mask_with_regex(text: str, pattern: re.Pattern, category: str, counter: list[int], spans: list[MaskedSpan]) -> str:
    def _replace(match: re.Match) -> str:
        idx = counter[0]
        counter[0] += 1
        token = f"[[{category.upper()}_{idx}]]"
        spans.append(MaskedSpan(token=token, original=match.group(0), category=category))
        return token

    return pattern.sub(_replace, text)


def mask_non_translatable_spans(text: str, lang: str, run_ner: bool = True) -> MaskedText:
    spans: list[MaskedSpan] = []
    counters = {"formula": [0], "placeholder": [0], "unit": [0], "number": [0], "entity": [0]}

    text = _mask_with_regex(text, _FORMULA_RE, "formula", counters["formula"], spans)
    text = _mask_with_regex(text, _PLACEHOLDER_RE, "placeholder", counters["placeholder"], spans)
    text = _mask_with_regex(text, _UNIT_RE, "unit", counters["unit"], spans)
    text = _mask_with_regex(text, _NUMBER_RE, "number", counters["number"], spans)

    if run_ner:
        text = _mask_named_entities(text, lang, counters["entity"], spans)

    return MaskedText(text=text, spans=spans)


def _mask_named_entities(text: str, lang: str, counter: list[int], spans: list[MaskedSpan]) -> str:
    from docslides.cleaning.segmentation import _get_spacy_pipeline, _get_stanza_pipeline
    from docslides.config import get_config

    cfg = get_config().languages
    entities: list[tuple[int, int, str]] = []

    try:
        if lang in cfg.spacy_models:
            import spacy

            model_name = cfg.spacy_models[lang]
            nlp = spacy.load(model_name, exclude=["parser", "lemmatizer", "tagger"])
            if "sentencizer" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            doc = nlp(text)
            entities = [(ent.start_char, ent.end_char, ent.text) for ent in doc.ents]
        elif lang in cfg.stanza_models:
            import stanza

            stanza_lang = cfg.stanza_models[lang]
            nlp = stanza.Pipeline(lang=stanza_lang, processors="tokenize,ner", download_method=None)
            doc = nlp(text)
            entities = [
                (ent.start_char, ent.end_char, ent.text)
                for sent in doc.sentences
                for ent in sent.ents
            ]
    except Exception as exc:  # noqa: BLE001 -- NER model may be missing; degrade gracefully
        from docslides.logging_setup import get_logger

        get_logger(__name__).warning("ner_masking_unavailable", lang=lang, error=str(exc))
        return text

    if not entities:
        return text

    # Rebuild text with entities replaced back-to-front so earlier offsets stay valid.
    for start, end, ent_text in sorted(entities, key=lambda e: e[0], reverse=True):
        idx = counter[0]
        counter[0] += 1
        token = f"[[ENT_{idx}]]"
        spans.append(MaskedSpan(token=token, original=ent_text, category="named_entity"))
        text = text[:start] + token + text[end:]

    return text


def restore_masks(text: str, spans: list[MaskedSpan]) -> str:
    for span in spans:
        text = text.replace(span.token, span.original)
    return text
