"""Legal tab endpoints.

  * POST /api/legal-chat + GET /api/legal-events/{job_id}: runs one turn of
    the grounded Israeli-law pipeline (legal/pipeline.py) and streams it:
    "status" per stage, the answer as "content_delta" (citation tokens
    rendered as [n] footnote markers), a "citations" event with the
    footnotes (law, section, effective range, relation, verified or not),
    and a "legal_report" event with the spec section 9 output (research
    memorandum, escalation, coverage gaps).
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from docslides.api.events import Event, event_bus
from docslides.legal.pipeline import run_legal_turn

router = APIRouter(prefix="/api", tags=["legal"])


class LegalChatRequest(BaseModel):
    messages: list[dict]  # [{"role": "user"|"assistant"|"system", "content": str}]


async def _run_legal_turn(job_id: str, req: LegalChatRequest) -> None:
    try:
        query = req.messages[-1]["content"] if req.messages else ""

        async def status(message: str) -> None:
            await event_bus.publish_status(job_id, message)

        result = await run_legal_turn(query, job_id, status)

        await event_bus.publish(job_id, Event(kind="content_delta", data={"text": result.display_answer}))
        await event_bus.publish(job_id, Event(kind="citations", data={"citations": result.footnotes}))
        await event_bus.publish(
            job_id,
            Event(
                kind="legal_report",
                data={
                    **result.output,
                    "escalation_reasons": result.escalation_reasons,
                    "reply_language": result.reply_language,
                    "notes": result.notes,
                    "audit_path": result.audit_path,
                },
            ),
        )
        await event_bus.publish_done(job_id)
    except Exception as exc:  # noqa: BLE001
        await event_bus.publish_error(job_id, str(exc))


@router.post("/legal-chat")
async def legal_chat(req: LegalChatRequest) -> dict:
    job_id = uuid.uuid4().hex[:12]
    event_bus.create(job_id)

    asyncio.create_task(_run_legal_turn(job_id, req))
    return {"job_id": job_id}


@router.get("/legal-events/{job_id}")
async def legal_events(job_id: str) -> EventSourceResponse:
    return EventSourceResponse(event_bus.stream(job_id))

