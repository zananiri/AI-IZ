"""Legal tab endpoints.

  * POST /api/legal-chat + GET /api/legal-events/{job_id}: runs one turn of
    the grounded Israeli-law pipeline (legal/pipeline.py) and streams it:
    "status" per stage, the answer as "content_delta" (citation tokens
    rendered as [n] footnote markers), a "citations" event with the
    footnotes (law, section, effective range, relation, verified or not),
    and a "legal_report" event with the spec section 9 output (research
    memorandum, escalation, coverage gaps) plus which DictaLM tier/model
    handled the turn.
  * GET /api/legal/dicta-tiers: the RAM/VRAM check behind the tab's tier
    suggestion banner (legal/resources.py). Measured on this API host.
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from docslides.api.events import Event, event_bus
from docslides.config import get_config
from docslides.legal.pipeline import run_legal_turn
from docslides.legal.resources import tier_report

router = APIRouter(prefix="/api", tags=["legal"])


class LegalChatRequest(BaseModel):
    messages: list[dict]  # [{"role": "user"|"assistant"|"system", "content": str}]
    dicta_tier: str | None = None  # key of config.legal.dicta_tiers; default tier if omitted


async def _run_legal_turn(job_id: str, req: LegalChatRequest) -> None:
    try:
        query = req.messages[-1]["content"] if req.messages else ""

        async def status(message: str) -> None:
            await event_bus.publish_status(job_id, message)

        result = await run_legal_turn(query, req.dicta_tier, job_id, status)

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
                    "polish_status": result.polish_status,
                    "dicta_tier": result.dicta_tier,
                    "dicta_model": result.dicta_model,
                    "dicta_used": result.dicta_used,
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
    try:
        get_config().legal.dicta_tier(req.dicta_tier)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = uuid.uuid4().hex[:12]
    event_bus.create(job_id)

    asyncio.create_task(_run_legal_turn(job_id, req))
    return {"job_id": job_id}


@router.get("/legal-events/{job_id}")
async def legal_events(job_id: str) -> EventSourceResponse:
    return EventSourceResponse(event_bus.stream(job_id))


@router.get("/legal/dicta-tiers")
async def legal_dicta_tiers(selected: str | None = None) -> dict:
    try:
        return await asyncio.to_thread(tier_report, selected)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
