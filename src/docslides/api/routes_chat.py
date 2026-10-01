"""Chat completion (streaming, proxied to the LLM backend) and tone-controlled
rewrite.

Every chat turn is first classified by Gemma 4: does it want a PowerPoint deck
generated, the text itself translated, and in which language should the reply
be? A chat turn may carry an `attachment_path` (a file already uploaded via
/api/upload), whose extracted text is the material; otherwise the material is
whatever the user pasted. Slide requests go to the full pipeline (translate ->
outline -> fill -> PPTX assembly). Translation always goes through
TranslateGemma (translation/translator.py): a request to translate the
document or the pasted text is translated chunk by chunk, and any other reply
asked for in another language (a summary, an answer) is written by Gemma 4 in
the material's own language and then translated. Everything else is answered
inline by Gemma 4, with the document's text as context. A tone rewrite asked
for in another language is rewritten by Gemma 4 in the text's own language,
then translated by TranslateGemma. Both paths stream over the same job's SSE queue (status events during
slide generation, reasoning_delta/content_delta for normal chat text) so the
UI only needs one event handler -- see ui/gradio_app.py's `_stream_job`.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from docslides.api.events import Event, event_bus, job_outputs
from docslides.cleaning.chunking import chunk_document
from docslides.cleaning.tokens import count_tokens
from docslides.config import get_config
from docslides.ingestion.language_detect import detect_language
from docslides.llm.client import (
    ChatMessage,
    LLMCallSite,
    SamplingParams,
    get_client,
    get_translator_client,
)
from docslides.llm.prompts import LANGUAGE_NAMES
from docslides.llm.schemas import ChatIntent
from docslides.pipeline.orchestrator import extract_document_text, run_pipeline
from docslides.tone.tone_control import (
    ToneSettings,
    compose_rewrite_system_prompt,
    resolve_sampling_params,
)
from docslides.translation.glossary import Glossary
from docslides.translation.translator import prepare_glossary, translate_chunk

router = APIRouter(prefix="/api", tags=["chat"])

# The slides pipeline chunks+translates a document properly regardless of
# size (see pipeline/orchestrator.py). This "just chat about the attached
# document" path -- and the plain-pasted-text path below, e.g. pasting a
# large block of text into the message box to rewrite/translate/summarize --
# has no such chunking; it's one prompt, so cap how much text we inject,
# leaving headroom in max_model_len for the prompt wrapper, conversation
# history, and the response itself. Oversized input is truncated (with a
# visible marker so it isn't silent) rather than erroring on the LLM call;
# ask for slides instead if you need an arbitrarily long document processed
# in full.
_ATTACHMENT_CONTEXT_TOKEN_FRACTION = 0.5


def _fit_to_token_budget(text: str, max_tokens: int) -> str:
    tokens = count_tokens(text)
    if tokens <= max_tokens:
        return text
    ratio = max_tokens / tokens
    cut = max(1, int(len(text) * ratio * 0.95))  # extra safety margin, count_tokens is approximate
    return text[:cut] + "\n\n[... truncated for length ...]"


class ChatRequest(BaseModel):
    messages: list[dict]  # [{"role": "user"|"assistant"|"system", "content": str}]
    attachment_path: str | None = None  # server-side path from a prior /api/upload call


class ToneRewriteRequest(BaseModel):
    text: str = ""  # ignored if attachment_path is set -- the attachment's text is rewritten instead
    attachment_path: str | None = None  # server-side path from a prior /api/upload call
    task_description: str = "Rewrite the following text."
    professionalism: int
    creativity: int


# Long pasted text: the classifier sees its start and end, where the instruction usually sits.
_CLASSIFY_HEAD_CHARS = 1500
_CLASSIFY_TAIL_CHARS = 500


def _classification_view(message: str) -> str:
    if len(message) <= _CLASSIFY_HEAD_CHARS + _CLASSIFY_TAIL_CHARS:
        return message
    return f"{message[:_CLASSIFY_HEAD_CHARS]}\n[...]\n{message[-_CLASSIFY_TAIL_CHARS:]}"


async def _classify_intent(client, user_message: str, has_attachment: bool) -> ChatIntent:
    material = (
        "The user has attached a document to this chat and sent a request about it."
        if has_attachment
        else "The user sent a chat message, which may include a block of pasted text."
    )
    return await client.complete_json(
        [
            ChatMessage(
                role="system",
                content=(
                    f"{material} Decide whether they explicitly asked for a PowerPoint / slide deck / "
                    "presentation to be generated from an attached document -- translation, rewriting, "
                    "summarizing, or answering questions are NOT slide requests. Separately, decide "
                    "whether they asked for the text itself (the attached document, or the text pasted "
                    "in the message) to be translated; a summary or answer in another language is NOT "
                    "a translation request. Extract the language they want the reply in as an ISO "
                    "639-1 code, if they named one or wrote the request in a language different from "
                    "the material. If the message starts with an instruction followed by the text it "
                    "applies to (to translate or rewrite), copy that instruction verbatim."
                ),
            ),
            ChatMessage(role="user", content=_classification_view(user_message)),
        ],
        LLMCallSite("chat_general"),
        schema=ChatIntent,
        sampling=SamplingParams(temperature=0.1, max_tokens=300),
        enable_thinking=False,
    )


def _text_to_translate(message: str, instruction: str) -> str:
    """The pasted text of a "translate this" message, without the instruction in front of it."""
    instruction = instruction.strip()
    if instruction and instruction in message:
        text = message.replace(instruction, "", 1)
    else:
        first, _, rest = message.partition("\n")
        text = rest if rest.strip() and len(first) <= 300 else message
    return text.strip().lstrip(":").strip()


async def _stream_translation(
    job_id: str,
    client,
    text: str,
    source_lang: str | None,
    target_lang: str,
    glossary_id: str | None = None,
) -> None:
    """Translate `text` with TranslateGemma chunk by chunk (Gemma 4 keeps the terms consistent)
    and stream each chunk. A document's glossary (`glossary_id`) is saved for its next
    translation; text pasted into the chat, or a model answer, gets a throwaway one."""
    source_lang = source_lang or detect_language(text) or "en"
    chunks = chunk_document([text], [source_lang])
    if glossary_id is not None or len(chunks) > 1:
        await event_bus.publish_status(job_id, "Preparing the terminology glossary")
        glossary = await prepare_glossary(
            client,
            glossary_id or f"chat-{job_id}",
            text,
            chunks[0].language if chunks else source_lang,
            target_lang,
        )
    else:
        glossary = Glossary(f"chat-{job_id}_{target_lang}")
    translator = get_translator_client()
    for chunk in chunks:
        await event_bus.publish_status(job_id, f"Translating part {chunk.index + 1} of {len(chunks)}")
        result = await translate_chunk(client, chunk, target_lang, glossary, translator)
        separator = "\n\n" if chunk.index else ""
        await event_bus.publish(
            job_id, Event(kind="content_delta", data={"text": separator + result.translated_text})
        )
    if glossary_id is not None:
        glossary.save()
    await event_bus.publish_done(job_id)


def _write_in_note(answer_lang: str, what: str) -> str:
    return (
        f"Write your {what} in {LANGUAGE_NAMES.get(answer_lang, answer_lang)}, whatever language the "
        "request asks for: a dedicated translator renders it into that language afterwards."
    )


async def _publish_translated(
    job_id: str, client, text: str, written_lang: str | None, target_lang: str
) -> None:
    """Publish `text` (Gemma 4's reply or rewrite) in `target_lang`: translated by TranslateGemma,
    unless it already came back in the target language."""
    written_in = detect_language(text) or written_lang
    if written_in == target_lang:
        await event_bus.publish(job_id, Event(kind="content_delta", data={"text": text}))
        await event_bus.publish_done(job_id)
        return
    await _stream_translation(job_id, client, text, written_in, target_lang)


async def _run_chat_turn(job_id: str, req: ChatRequest) -> None:
    client = get_client()
    try:
        user_message = req.messages[-1]["content"] if req.messages and req.messages[-1]["role"] == "user" else ""
        await event_bus.publish_status(job_id, "Reading your request")
        intent = await _classify_intent(client, user_message, has_attachment=bool(req.attachment_path))
        target_lang = intent.target_lang

        if req.attachment_path:
            document_text, doc_lang = await extract_document_text(job_id, req.attachment_path)

            if intent.wants_slides:
                slides_lang = target_lang or doc_lang or "en"
                await event_bus.publish_status(
                    job_id, f"Generating a presentation (target language: {slides_lang})"
                )
                try:
                    output_path = await run_pipeline(job_id, req.attachment_path, slides_lang)
                except Exception:  # noqa: BLE001 -- run_pipeline already published its own error event
                    return
                job_outputs[job_id] = str(output_path)
                return  # run_pipeline already published "done"

            if intent.wants_translation and target_lang:
                # A whole document doesn't fit one prompt or one reply: translate it chunk by chunk.
                await _stream_translation(
                    job_id, client, document_text, doc_lang, target_lang, Path(req.attachment_path).stem
                )
                return

            source_lang = doc_lang
            budget = int(get_config().llm.max_model_len * _ATTACHMENT_CONTEXT_TOKEN_FRACTION)
            document_text = _fit_to_token_budget(document_text, budget)
            messages = [
                ChatMessage(
                    role="system",
                    content=(
                        "The user has attached a document; its extracted text follows. Use it to "
                        "answer, rewrite, summarize, or otherwise fulfill their request "
                        "exactly as asked below.\n\n--- DOCUMENT START ---\n"
                        f"{document_text}\n--- DOCUMENT END ---"
                    ),
                ),
                *[ChatMessage(role=m["role"], content=m["content"]) for m in req.messages],
            ]
        else:
            if intent.wants_translation and target_lang:
                text = _text_to_translate(user_message, intent.instruction)
                if text:
                    await _stream_translation(job_id, client, text, None, target_lang)
                    return

            source_lang = detect_language(user_message)
            messages = [ChatMessage(role=m["role"], content=m["content"]) for m in req.messages]
            # A turn with no attachment can still carry a large block of pasted
            # text (paste-to-rewrite/summarize) as the latest user message --
            # cap it the same way an attachment's extracted text is capped,
            # rather than sending it through uncapped and risking a raw
            # context-length error from the LLM backend.
            if messages and messages[-1].role == "user":
                budget = int(get_config().llm.max_model_len * _ATTACHMENT_CONTEXT_TOKEN_FRACTION)
                messages[-1] = ChatMessage(
                    role="user", content=_fit_to_token_budget(messages[-1].content, budget)
                )

        if target_lang and target_lang != source_lang:
            # A summary or answer in another language: Gemma 4 writes it in the material's own
            # language, TranslateGemma translates it.
            # (Gemma's chat template takes a system message only first, so the note joins the
            # user's turn.)
            answer_lang = source_lang or "en"
            note = f"({_write_in_note(answer_lang, 'reply')})"
            if messages and messages[-1].role == "user":
                messages[-1] = ChatMessage(role="user", content=f"{messages[-1].content}\n\n{note}")
            else:
                messages = [*messages, ChatMessage(role="user", content=note)]
            await event_bus.publish_status(job_id, "Writing the answer")
            answer = await client.complete_text(messages, LLMCallSite("chat_general"))
            await _publish_translated(job_id, client, answer, answer_lang, target_lang)
            return

        await event_bus.publish_status(job_id, "Waiting for model response")
        async for delta in client.stream_chat(messages, LLMCallSite("chat_general")):
            kind = "reasoning_delta" if delta.kind == "reasoning" else "content_delta"
            await event_bus.publish(job_id, Event(kind=kind, data={"text": delta.text}))
        await event_bus.publish_done(job_id)
    except Exception as exc:  # noqa: BLE001
        await event_bus.publish_error(job_id, str(exc))


async def _run_tone_rewrite(job_id: str, req: ToneRewriteRequest) -> None:
    """Rewrites either the pasted `text` or, when the same message box's
    attach button was used instead, the attached document's extracted text --
    the two are mutually exclusive inputs to the same "what am I rewriting"
    slot, not separate features.

    A rewrite asked for in another language is done in two steps: Gemma 4
    rewrites in the text's own language (with the tone settings), then
    TranslateGemma translates the rewrite into the target language."""
    client = get_client()
    try:
        task_description = req.task_description
        budget = int(get_config().llm.max_model_len * _ATTACHMENT_CONTEXT_TOKEN_FRACTION)

        target_lang = None
        instruction = ""
        if req.text:
            await event_bus.publish_status(job_id, "Reading your request")
            intent = await _classify_intent(client, req.text, has_attachment=bool(req.attachment_path))
            target_lang = intent.target_lang
            instruction = intent.instruction.strip()

        if req.attachment_path:
            document_text, source_lang = await extract_document_text(job_id, req.attachment_path)
            text_to_rewrite = _fit_to_token_budget(document_text, budget)
            if req.text:
                # Typed text alongside an attachment is extra instruction, not the rewrite target.
                task_description = f"{req.task_description}\n\nAdditional instructions: {req.text}"
        else:
            text = req.text
            if target_lang and instruction and instruction in text:
                # "Rewrite this in German: <text>": the instruction is not part of what gets rewritten.
                text = _text_to_translate(text, instruction)
                task_description = f"{req.task_description}\n\nAdditional instructions: {instruction}"
            # Large pasted text (paste-to-rewrite) gets the same cap an
            # attachment's extracted text gets -- see _run_chat_turn above.
            text_to_rewrite = _fit_to_token_budget(text, budget)
            source_lang = detect_language(text_to_rewrite)

        translate = bool(target_lang) and target_lang != source_lang
        if translate:
            task_description = f"{task_description}\n\n{_write_in_note(source_lang or 'en', 'rewrite')}"

        tone = ToneSettings(professionalism=req.professionalism, creativity=req.creativity)
        system_prompt = compose_rewrite_system_prompt(tone, task_description)
        sampling = resolve_sampling_params(tone)
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=text_to_rewrite),
        ]

        if translate:
            await event_bus.publish_status(job_id, "Rewriting")
            rewrite = await client.complete_text(messages, LLMCallSite("tone_rewrite"), sampling=sampling)
            await _publish_translated(job_id, client, rewrite, source_lang or "en", target_lang)
            return

        await event_bus.publish_status(job_id, "Waiting for model response")
        async for delta in client.stream_chat(messages, LLMCallSite("tone_rewrite"), sampling=sampling):
            kind = "reasoning_delta" if delta.kind == "reasoning" else "content_delta"
            await event_bus.publish(job_id, Event(kind=kind, data={"text": delta.text}))
        await event_bus.publish_done(job_id)
    except Exception as exc:  # noqa: BLE001
        await event_bus.publish_error(job_id, str(exc))


@router.post("/chat")
async def chat(req: ChatRequest) -> dict:
    job_id = uuid.uuid4().hex[:12]
    event_bus.create(job_id)

    asyncio.create_task(_run_chat_turn(job_id, req))
    return {"job_id": job_id}


@router.post("/tone-rewrite")
async def tone_rewrite(req: ToneRewriteRequest) -> dict:
    job_id = uuid.uuid4().hex[:12]
    event_bus.create(job_id)

    asyncio.create_task(_run_tone_rewrite(job_id, req))
    return {"job_id": job_id}


@router.get("/chat-events/{job_id}")
async def chat_events(job_id: str) -> EventSourceResponse:
    return EventSourceResponse(event_bus.stream(job_id))
