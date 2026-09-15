"""Chat completion (streaming, proxied to vLLM) and tone-controlled rewrite.

Both endpoints stream over SSE using the same "reasoning_delta"/"content_delta"
event split so the UI's collapsible reasoning panel works identically for
plain chat turns and tone-rewrite turns.
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from docslides.api.events import Event, event_bus
from docslides.llm.client import ChatMessage, LLMCallSite, get_client
from docslides.tone.tone_control import ToneSettings, compose_rewrite_system_prompt, resolve_sampling_params

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    messages: list[dict]  # [{"role": "user"|"assistant"|"system", "content": str}]


class ToneRewriteRequest(BaseModel):
    text: str
    task_description: str = "Rewrite the following text."
    professionalism: int
    creativity: int


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
    messages = [ChatMessage(role=m["role"], content=m["content"]) for m in req.messages]

    asyncio.create_task(_stream_and_publish(job_id, messages, LLMCallSite("chat_general")))
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
