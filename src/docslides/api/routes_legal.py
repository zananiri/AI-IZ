"""Legal tab chat endpoint: runs the 3-step pipeline in legal/pipeline.py --
an orchestrator model (Qwen) plans the research and reformulates the user's
question in Hebrew, a Hebrew legal-domain model (DictaLM) analyzes it in
Hebrew, and the orchestrator verifies the result and translates the final
answer back into whatever language the user asked in. Citations and
relevant laws are published as their own SSE event (see api/events.py) so
the UI can show them in a side panel separate from the answer, rather than
folded into the answer's prose.
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from docslides.api.events import Event, event_bus
from docslides.ingestion.language_detect import detect_language
from docslides.legal.pipeline import analyze_legal_query, run_hebrew_legal_analysis, verify_and_finalize
from docslides.llm.client import ChatMessage, get_legal_hebrew_client, get_legal_orchestrator_client

router = APIRouter(prefix="/api", tags=["legal"])


class LegalChatRequest(BaseModel):
    messages: list[dict]  # [{"role": "user"|"assistant"|"system", "content": str}]


async def _run_legal_turn(job_id: str, req: LegalChatRequest) -> None:
    orchestrator = get_legal_orchestrator_client()
    hebrew_client = get_legal_hebrew_client()
    try:
        user_message = req.messages[-1]["content"] if req.messages else ""
        user_lang = detect_language(user_message) or "en"
        messages = [ChatMessage(role=m["role"], content=m["content"]) for m in req.messages]

        await event_bus.publish_status(job_id, "Analyzing your question")
        plan = await analyze_legal_query(orchestrator, messages)

        await event_bus.publish_status(job_id, f"Researching Israeli law: {plan.topic_summary}")
        findings = await run_hebrew_legal_analysis(hebrew_client, plan.hebrew_query)

        await event_bus.publish_status(job_id, "Verifying the answer")
        final = await verify_and_finalize(orchestrator, user_message, user_lang, plan.hebrew_query, findings)

        await event_bus.publish(job_id, Event(kind="content_delta", data={"text": final.answer}))
        await event_bus.publish(
            job_id,
            Event(kind="citations", data={"citations": final.citations, "relevant_laws": final.relevant_laws}),
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
