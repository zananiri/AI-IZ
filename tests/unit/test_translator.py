"""Translation flow: TranslateGemma translates, Gemma 4 fixes glossary terms it missed."""

from __future__ import annotations

import asyncio

from docslides.cleaning.chunking import Chunk
from docslides.config import reload_config
from docslides.llm.prompts import translategemma_prompt
from docslides.llm.schemas import GlossaryTerm, TranslationRevision
from docslides.translation import translator as tr
from docslides.translation.glossary import Glossary


class FakeTranslator:
    model = "translategemma:12b"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def complete_text(self, messages, call_site, sampling=None, enable_thinking=None):
        self.prompts.append(messages[-1].content)
        return self.reply


class FakeGemma:
    model = "gemma4:12b"

    def __init__(self, revision: str) -> None:
        self.revision = revision
        self.calls = 0

    async def complete_json(self, messages, call_site, schema, sampling=None, **_):
        self.calls += 1
        return TranslationRevision(translated_text=self.revision)


def _glossary(*pairs: tuple[str, str]) -> Glossary:
    glossary = Glossary("doc_de")
    glossary.add_terms([GlossaryTerm(source_term=s, target_term=t) for s, t in pairs])
    return glossary


CHUNK = Chunk(index=0, text="Only 541 Christians remain in the Holy Land, rate of $54 %$.", language="en", token_count=20)


def test_translategemma_gets_its_own_prompt_with_ocr_math_unwrapped():
    translator = FakeTranslator("Nur 541 Christen bleiben im Heiligen Land, eine Quote von 54 %.")
    gemma = FakeGemma("unused")
    glossary = _glossary(("Holy Land", "Heiliges Land"), ("Christians", "Christen"))

    result = asyncio.run(tr.translate_chunk(gemma, CHUNK, "de", glossary, translator))

    assert translator.prompts == [
        translategemma_prompt("en", "de", "Only 541 Christians remain in the Holy Land, rate of 54%.")
    ]
    assert "into German:\n\n\nOnly 541" in translator.prompts[0]
    # "Heiligen Land" is an inflection of "Heiliges Land": nothing to fix, Gemma 4 isn't called.
    assert gemma.calls == 0
    assert result.translated_text.startswith("Nur 541 Christen")


def test_a_missed_glossary_term_goes_to_gemma_for_a_fix():
    draft = "Nur 541 Christen bleiben in the Holy Land, eine Quote von 54 %."
    fixed = "Nur 541 Christen bleiben im Heiligen Land, eine Quote von 54 %."
    gemma = FakeGemma(fixed)
    glossary = _glossary(("Holy Land", "Heiliges Land"))

    result = asyncio.run(tr.translate_chunk(gemma, CHUNK, "de", glossary, FakeTranslator(draft)))

    assert gemma.calls == 1
    assert result.translated_text == fixed


def test_a_fix_that_drops_text_is_ignored():
    draft = "Nur 541 Christen bleiben in the Holy Land, eine Quote von 54 %."
    gemma = FakeGemma("Heiliges Land")
    result = asyncio.run(
        tr.translate_chunk(gemma, CHUNK, "de", _glossary(("Holy Land", "Heiliges Land")), FakeTranslator(draft))
    )
    assert result.translated_text == draft


def test_terms_not_in_the_source_are_not_checked():
    glossary = _glossary(("Latin Patriarchate", "Lateinisches Patriarchat"))
    assert tr._missed_terms(CHUNK.text, "irrelevant", glossary) == []


def test_translator_env_overrides_build_a_non_thinking_deployment(monkeypatch):
    for name in ("MAX_MODEL_LEN", "REQUEST_TIMEOUT_S"):
        monkeypatch.delenv(f"DOCSLIDES_TRANSLATOR_{name}", raising=False)
    monkeypatch.setenv("DOCSLIDES_TRANSLATOR_BACKEND", "ollama")
    monkeypatch.setenv("DOCSLIDES_TRANSLATOR_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("DOCSLIDES_TRANSLATOR_MODEL", "translategemma:12b")
    try:
        translator = reload_config().translation.translator
        assert translator is not None
        assert (translator.backend, translator.model, translator.supports_thinking) == (
            "ollama",
            "translategemma:12b",
            False,
        )
    finally:
        monkeypatch.undo()
        reload_config()
