"""SSE event plumbing, kept as two distinct event kinds per the UI contract:

  * "status" -- pipeline stage progress (e.g. "OCR: Arabic page 6/12 --
    PaddleOCR-VL fallback", "Translating chunk 3/9 -- glossary: 42 terms").
    Rendered as a persistent status line during pipeline runs.
  * "reasoning_delta" / "content_delta" -- for chat turns run with thinking
    enabled, the model's <think>...</think> block streamed separately from
    the final answer. Rendered as a collapsible reasoning panel above the
    streaming final answer.

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
    kind: Literal["status", "reasoning_delta", "content_delta", "done", "error"]
    data: dict
    ts: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        payload = orjson.dumps(self.data).decode("utf-8")
        return f"event: {self.kind}\ndata: {payload}\n\n"


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

    async def stream(self, job_id: str) -> AsyncIterator[str]:
        queue = self.get(job_id)
        while True:
            event = await queue.get()
            yield event.to_sse()
            if event.kind in ("done", "error"):
                self._queues.pop(job_id, None)
                break


event_bus = EventBus()
