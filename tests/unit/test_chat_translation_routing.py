"""Chat routing: every translation, including a summary asked for in another language, goes
through the translator (TranslateGemma)."""

from __future__ import annotations

import asyncio

import pytest

from docslides.api import routes_chat as rc
from docslides.llm.schemas import ChatIntent


class FakeGemma:
    model = "gemma4:12b"

    def __init__(self, intent: ChatIntent, answer: str = "") -> None:
        self.intent = intent
        self.answer = answer
        self.answer_prompts: list[list] = []
        self.streamed = False

    async def complete_json(self, messages, call_site, schema, **_):
        if schema is ChatIntent:
            return self.intent
        raise AssertionError(f"unexpected JSON call: {schema.__name__}")

    async def complete_text(self, messages, call_site, **_):
        self.answer_prompts.append(messages)
        return self.answer

    async def stream_chat(self, messages, call_site, **_):
        self.streamed = True
        yield type("Delta", (), {"kind": "content", "text": "streamed by Gemma"})()


class FakeTranslateGemma:
    model = "translategemma:12b"

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def complete_text(self, messages, call_site, **_):
        prompt = messages[-1].content
        self.texts.append(prompt.split("\n\n\n", 1)[1])
        return "ÜBERSETZT"


LANGS = {"Fasse den Text zusammen.": "de", "The summary.": "en", "Zusammenfassung.": "de"}


@pytest.fixture
def run(monkeypatch):
    events: list[tuple[str, str]] = []

    async def publish(job_id, event):
        events.append((event.kind, event.data.get("text", "")))

    async def publish_status(job_id, message, **_):
        events.append(("status", message))

    async def publish_done(job_id, **_):
        events.append(("done", ""))

    async def publish_error(job_id, message):
        events.append(("error", message))

    for name, fn in [("publish", publish), ("publish_status", publish_status),
                     ("publish_done", publish_done), ("publish_error", publish_error)]:
        monkeypatch.setattr(rc.event_bus, name, fn)
    monkeypatch.setattr(rc, "detect_language", lambda text: LANGS.get(text.strip(), "en"))

    def _run(gemma, translator, message, attachment=None):
        monkeypatch.setattr(rc, "get_client", lambda: gemma)
        monkeypatch.setattr(rc, "get_translator_client", lambda: translator)
        req = rc.ChatRequest(messages=[{"role": "user", "content": message}], attachment_path=attachment)
        asyncio.run(rc._run_chat_turn("job-1", req))
        return events

    return _run


def _content(events):
    return "".join(text for kind, text in events if kind == "content_delta")


def test_pasted_text_to_translate_goes_to_translategemma_without_the_instruction(run):
    intent = ChatIntent(wants_slides=False, wants_translation=True, target_lang="de",
                        instruction="Translate this into German:")
    gemma, translator = FakeGemma(intent), FakeTranslateGemma()
    events = run(gemma, translator, "Translate this into German:\nThe Holy Land gave the world hope.")
    assert translator.texts == ["The Holy Land gave the world hope."]
    assert _content(events) == "ÜBERSETZT"
    assert not gemma.streamed and events[-1] == ("done", "")


def test_a_summary_in_another_language_is_written_by_gemma_then_translated(run):
    intent = ChatIntent(wants_slides=False, wants_translation=False, target_lang="de")
    gemma, translator = FakeGemma(intent, answer="The summary."), FakeTranslateGemma()
    events = run(gemma, translator, "Summarize this in German: The Holy Land gave the world hope.")
    asked = gemma.answer_prompts[0][-1].content
    assert "Write your reply in English" in asked  # Gemma answers in the material's language
    assert translator.texts == ["The summary."]
    assert _content(events) == "ÜBERSETZT"


def test_an_answer_already_in_the_target_language_is_not_translated_again(run):
    intent = ChatIntent(wants_slides=False, wants_translation=False, target_lang="de")
    gemma, translator = FakeGemma(intent, answer="Zusammenfassung."), FakeTranslateGemma()
    events = run(gemma, translator, "Summarize this in German: The Holy Land gave the world hope.")
    assert translator.texts == []
    assert _content(events) == "Zusammenfassung."


def test_a_reply_in_the_messages_own_language_streams_from_gemma(run):
    gemma, translator = FakeGemma(ChatIntent(wants_slides=False)), FakeTranslateGemma()
    events = run(gemma, translator, "What is the capital of France?")
    assert gemma.streamed and translator.texts == []
    assert _content(events) == "streamed by Gemma"


def test_an_instruction_the_model_did_not_copy_exactly_drops_the_first_line():
    assert rc._text_to_translate("Please translate to French\nLine one.\nLine two.", "translate it") == (
        "Line one.\nLine two."
    )
    assert rc._text_to_translate("Translate: Hello world", "Translate:") == "Hello world"
