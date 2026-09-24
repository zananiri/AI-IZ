"""SSE event plumbing, kept as two distinct event kinds per the UI contract:

  * "status" -- pipeline stage progress (e.g. "OCR: Arabic page 6/12 --
    PaddleOCR-VL fallback", "Translating chunk 3/9 -- glossary: 42 terms").
    Rendered as a persistent status line during pipeline runs.
  * "reasoning_delta" / "content_delta" -- for chat turns run with thinking
    enabled, the model's <think>...</think> block streamed separately from
    the final answer. Rendered as a collapsible reasoning panel above the
    streaming final answer.
  * "citations" -- Legal / Canon tabs: structured source citations,
    rendered in a side panel rather than the chat bubble.
  * "legal_report" -- Legal tab only: the pipeline's structured output
    (research memorandum, escalation, coverage gaps) -- see
    api/routes_legal.py.

Each job/chat-turn gets its own asyncio.Queue so multiple concurrent
requests don't cross-talk.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

import orjson


@dataclass
class Event:
    kind: Literal["status", "reasoning_delta", "content_delta", "citations", "legal_report", "done", "error"]
    data: dict
    ts: float = field(default_factory=time.time)

    def to_sse_message(self) -> dict:
        """`sse_starlette`'s `EventSourceResponse` expects each yielded item to
        be a dict (or `ServerSentEvent`/bytes) that IT formats into SSE wire
        format -- NOT an already-formatted "event: ...\\ndata: ...\\n\\n"
        string. Handing it a pre-formatted string gets treated as one opaque
        multi-line `data` value, and since SSE requires multi-line data to be
        split into repeated `data: ` lines, sse_starlette re-wraps every line
        of our own formatting with another `data: ` prefix -- silently
        corrupting the stream (every line comes out as `data: event: ...`,
        `data: data: {...}`, etc.), which is exactly what happened here."""
        return {"event": self.kind, "data": orjson.dumps(self.data).decode("utf-8")}


class EventBus:
    """Registry of per-job event queues."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[Event]] = {}

    def create(self, job_id: str) -> None:
        self._queues[job_id] = asyncio.Queue()

    def get(self, job_id: str) -> asyncio.Queue[Event]:
        if job_id not in self._queues:
            self.create(job_id)
        return self._queues[job_id]

    async def publish(self, job_id: str, event: Event) -> None:
        await self.get(job_id).put(event)

    async def publish_status(self, job_id: str, message: str, **extra) -> None:
        await self.publish(job_id, Event(kind="status", data={"message": message, **extra}))

    async def publish_done(self, job_id: str, **extra) -> None:
        await self.publish(job_id, Event(kind="done", data=extra))

    async def publish_error(self, job_id: str, message: str) -> None:
        await self.publish(job_id, Event(kind="error", data={"message": message}))

    async def stream(self, job_id: str) -> AsyncIterator[dict]:
        queue = self.get(job_id)
        while True:
            event = await queue.get()
            yield event.to_sse_message()
            if event.kind in ("done", "error"):
                self._queues.pop(job_id, None)
                break


event_bus = EventBus()

# Job-id -> generated .pptx path. Shared between routes_pptx.py (kicked off
# from the old dedicated tab, kept for direct API use) and routes_chat.py
# (kicked off when a chat turn's attached file is classified as a slides
# request) so /api/download/{job_id} works the same way regardless of which
# route started the job.
job_outputs: dict[str, str] = {}
