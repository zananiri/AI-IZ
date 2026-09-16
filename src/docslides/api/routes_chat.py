"""Chat completion (streaming, proxied to the LLM backend) and tone-controlled
rewrite.

A chat turn may carry an `attachment_path` (a file already uploaded via
/api/upload). When present, we extract its text and ask the model a quick
classification question: does this turn want a PowerPoint deck generated, or
something else (translate/rewrite/summarize/ask-a-question)? Slide requests
are handed off to the existing full pipeline (translate -> outline -> fill ->
PPTX assembly, unchanged from the old dedicated tab); everything else is
answered inline in the chat with the document's text as context, so
"translate this" or "make this more formal" just works as a normal chat
turn. Both paths stream over the same job's SSE queue (status events during
slide generation, reasoning_delta/content_delta for normal chat text) so the
UI only needs one event handler -- see ui/gradio_app.py's `_stream_job`.
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from docslides.api.events import Event, event_bus, job_outputs
from docslides.cleaning.tokens import count_tokens
from docslides.config import get_config
from docslides.llm.client import ChatMessage, LLMCallSite, SamplingParams, get_client
from docslides.llm.schemas import ChatIntent
from docslides.pipeline.orchestrator import extract_document_text, run_pipeline
from docslides.tone.tone_control import ToneSettings, compose_rewrite_system_prompt, resolve_sampling_params

router = APIRouter(prefix="/api", tags=["chat"])

# The slides pipeline chunks+translates a document properly regardless of
# size (see pipeline/orchestrator.py). This "just chat about the attached
# document" path has no such chunking -- it's one prompt -- so cap how much
# of the extracted text we inject, leaving headroom in max_model_len for the
# prompt wrapper, conversation history, and the response itself. Long
# documents get silently truncated here rather than erroring on the LLM
# call; ask for slides instead if you need the whole thing processed.
_ATTACHMENT_CONTEXT_TOKEN_FRACTION = 0.5


def _fit_to_token_budget(text: str, max_tokens: int) -> str:
    tokens = count_tokens(text)
    if tokens <= max_tokens:
        return text
    ratio = max_tokens / tokens
    cut = max(1, int(len(text) * ratio * 0.95))  # extra safety margin, count_tokens is approximate
    return text[:cut] + "\n\n[... document truncated for length ...]"


class ChatRequest(BaseModel):
    messages: list[dict]  # [{"role": "user"|"assistant"|"system", "content": str}]
    attachment_path: str | None = None  # server-side path from a prior /api/upload call


class ToneRewriteRequest(BaseModel):
    text: str
    task_description: str = "Rewrite the following text."
    professionalism: int
    creativity: int


async def _classify_intent(client, user_message: str) -> ChatIntent:
    return await client.complete_json(
        [
            ChatMessage(
                role="system",
                content=(
                    "The user has attached a document to this chat and sent a request about it. "
                    "Decide only whether they explicitly asked for a PowerPoint / slide deck / "
                    "presentation to be generated from it -- translation, rewriting, summarizing, "
                    "or answering questions about the document are NOT slide requests. If it is a "
                    "slide request, also extract the target output language they asked for as an "
                    "ISO 639-1 code, if they named one."
                ),
            ),
            ChatMessage(role="user", content=user_message),
        ],
        LLMCallSite("chat_general"),
        schema=ChatIntent,
        sampling=SamplingParams(temperature=0.1, max_tokens=200),
        enable_thinking=False,
    )


async def _run_chat_turn(job_id: str, req: ChatRequest) -> None:
    client = get_client()
    try:
        if req.attachment_path:
            document_text, doc_lang = await extract_document_text(job_id, req.attachment_path)
            user_message = req.messages[-1]["content"] if req.messages else ""

            await event_bus.publish_status(job_id, "Reading your request")
            intent = await _classify_intent(client, user_message)

            if intent.wants_slides:
                target_lang = intent.target_lang or doc_lang or "en"
                await event_bus.publish_status(
                    job_id, f"Generating a presentation (target language: {target_lang})"
                )
                try:
                    output_path = await run_pipeline(job_id, req.attachment_path, target_lang)
                except Exception:  # noqa: BLE001 -- run_pipeline already published its own error event
                    return
                job_outputs[job_id] = str(output_path)
                return  # run_pipeline already published "done"

            budget = int(get_config().llm.max_model_len * _ATTACHMENT_CONTEXT_TOKEN_FRACTION)
            document_text = _fit_to_token_budget(document_text, budget)
            messages = [
                ChatMessage(
                    role="system",
                    content=(
                        "The user has attached a document; its extracted text follows. Use it to "
                        "answer, translate, rewrite, summarize, or otherwise fulfill their request "
                        "exactly as asked below.\n\n--- DOCUMENT START ---\n"
                        f"{document_text}\n--- DOCUMENT END ---"
                    ),
                ),
                *[ChatMessage(role=m["role"], content=m["content"]) for m in req.messages],
            ]
        else:
            messages = [ChatMessage(role=m["role"], content=m["content"]) for m in req.messages]

        async for delta in client.stream_chat(messages, LLMCallSite("chat_general")):
            kind = "reasoning_delta" if delta.kind == "reasoning" else "content_delta"
            await event_bus.publish(job_id, Event(kind=kind, data={"text": delta.text}))
        await event_bus.publish_done(job_id)
    except Exception as exc:  # noqa: BLE001
        await event_bus.publish_error(job_id, str(exc))


async def _stream_and_publish(job_id: str, messages: list[ChatMessage], call_site: LLMCallSite, sampling=None) -> None:
    client = get_client()
    try:
        async for delta in client.stream_chat(messages, call_site, sampling=sampling):
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

    tone = ToneSettings(professionalism=req.professionalism, creativity=req.creativity)
    system_prompt = compose_rewrite_system_prompt(tone, req.task_description)
    sampling = resolve_sampling_params(tone)
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=req.text),
    ]

    asyncio.create_task(
        _stream_and_publish(job_id, messages, LLMCallSite("tone_rewrite"), sampling=sampling)
    )
    return {"job_id": job_id}


@router.get("/chat-events/{job_id}")
async def chat_events(job_id: str) -> EventSourceResponse:
    return EventSourceResponse(event_bus.stream(job_id))
