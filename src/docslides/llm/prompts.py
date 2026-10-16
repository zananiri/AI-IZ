"""System prompt templates for each LLM call site."""

from __future__ import annotations

from docslides.llm.schemas import GlossaryTerm

LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "de": "German",
    "ar": "Arabic",
    "he": "Hebrew",
}


def translation_system_prompt(
    source_lang: str, target_lang: str, glossary: list[GlossaryTerm]
) -> str:
    src_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)

    glossary_block = ""
    if glossary:
        lines = "\n".join(f"- {g.source_term} -> {g.target_term}" for g in glossary)
        glossary_block = (
            "\n\nUse the following glossary consistently. These terms have already been "
            f"translated elsewhere in this document and must remain consistent:\n{lines}"
        )

    return (
        f"You are a professional {src_name}-to-{tgt_name} document translator. "
        "Translate the user's text faithfully, preserving meaning, tone, and register. "
        "The text may contain indexed placeholder tokens like [[NUM_0]], [[ENT_2]], [[UNIT_1]], "
        "[[FORMULA_0]] -- copy these tokens through to the output UNCHANGED, in the position "
        "grammatically appropriate for the target language. Do not translate, alter, or drop them. "
        "Do not add explanations or commentary; output only the translation and any newly "
        "identified glossary terms as requested by the schema."
        f"{glossary_block}"
    )


def translategemma_prompt(source_lang: str, target_lang: str, text: str) -> str:
    """TranslateGemma's own prompt, verbatim from its model card (ollama.com/library/translategemma),
    including the two blank lines before the text: it is trained on exactly this."""
    src_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    return (
        f"You are a professional {src_name} ({source_lang}) to {tgt_name} ({target_lang}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {src_name} text "
        f"while adhering to {tgt_name} grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the {tgt_name} translation, without any additional explanations or commentary. "
        f"Please translate the following {src_name} text into {tgt_name}:\n\n\n"
        f"{text}"
    )


def glossary_build_system_prompt(source_lang: str, target_lang: str, max_terms: int) -> str:
    src_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    return (
        f"You prepare a terminology glossary for translating the user's {src_name} document into "
        f"{tgt_name}. List at most {max_terms} terms that must be translated the same way every time "
        "they appear: domain-specific terms, names of places, institutions, programmes and groups, and "
        "recurring key phrases. Give each its standard, idiomatic "
        f"{tgt_name} translation (keep a person's name as it is). Skip ordinary words that any "
        "translator would render the same way anyway. Output only the JSON the schema asks for."
    )


def glossary_fix_system_prompt(source_lang: str, target_lang: str, terms: list[GlossaryTerm]) -> str:
    src_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    lines = "\n".join(f"- {g.source_term} -> {g.target_term}" for g in terms)
    return (
        f"You are a {tgt_name} copy editor. The user gives you a {src_name} source text and its "
        f"{tgt_name} translation. The translation does not use the glossary below for terms that "
        "occur in the source. Return the translation with those terms rendered as the glossary says, "
        "inflected to fit the sentence, and nothing else changed: keep every other word, the "
        f"sentence order and the line breaks exactly as they are.\n\nGlossary:\n{lines}"
    )


def outline_system_prompt(target_lang: str, num_source_chunks: int) -> str:
    tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    return (
        "You are a presentation architect. You will be given a structured summary of a "
        f"document (from {num_source_chunks} source chunks) and must design a slide-by-slide "
        f"outline for a PowerPoint deck in {tgt_name}. "
        "For each slide choose the layout_type that best fits its content "
        "(title_bullets, two_column, section_header, image_caption, quote), a concise title, "
        "a one-phrase intent describing what the slide should communicate, and a realistic "
        "target_bullet_count (0 for section_header/quote slides). "
        "Cover the document's structure faithfully without inventing facts. "
        "Think through the document's structure before committing to the outline."
    )


def slide_fill_system_prompt(target_lang: str) -> str:
    tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    return (
        f"You are writing the final content for one slide of a {tgt_name}-language "
        "PowerPoint deck. You will be given the slide's planned title, intent, target bullet "
        "count, layout_type, and the relevant source content. "
        "Write concise, presentation-appropriate bullets (not full paragraphs), a short "
        "speaker-notes paragraph, and keep the given layout_type. "
        "Do not exceed the requested bullet count by more than one. "
        "Output strictly follows the required JSON schema with no extra commentary."
    )


def chunk_summary_system_prompt() -> str:
    return (
        "Summarize the following document chunk faithfully and concisely, preserving its key "
        "facts, figures, and structure, so it can be used to plan a presentation outline "
        "without access to the full original text. Extract 2-5 key points as short phrases."
    )


def glossary_extraction_note() -> str:
    return (
        "Additionally, identify any domain-specific or proper-noun terms in this chunk that "
        "are not already in the glossary and should be translated consistently for the rest "
        "of the document, with their target-language translation."
    )
