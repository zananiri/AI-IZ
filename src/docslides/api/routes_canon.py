"""Canon GPT tab chat endpoint: runs the RAG pipeline in canon/pipeline.py --
an orchestrator model reformulates the question and guesses relevant
code(s), retrieval fetches matching canon-law provisions from the
local vector store (see canon/retrieval.py, populated offline by
scripts/ingest_canon_law.py), and the orchestrator answers grounded in that
retrieved text. Citations published on the SSE stream come straight from
retrieval metadata (real vatican.va source URLs), not
model-invented text -- see api/events.py's "citations" event kind and
routes_legal.py for the parallel (non-RAG) Legal tab pattern this mirrors.
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from docslides.api.events import Event, event_bus
from docslides.canon.chunking import citation_label
from docslides.canon.pipeline import answer_from_context, plan_canon_query, retrieve_context
from docslides.canon.retrieval import RetrievedChunk
from docslides.ingestion.language_detect import detect_language
from docslides.llm.client import ChatMessage, get_canon_generation_client

router = APIRouter(prefix="/api", tags=["canon"])


class CanonChatRequest(BaseModel):
    messages: list[dict]  # [{"role": "user"|"assistant"|"system", "content": str}]


def _citation_payload(chunk: RetrievedChunk) -> dict:
    return {
        "label": citation_label(chunk.code, chunk.number, chunk.paragraph),
        "url": chunk.source_url,
        "breadcrumb": chunk.breadcrumb,
    }


async def _run_canon_turn(job_id: str, req: CanonChatRequest) -> None:
    client = get_canon_generation_client()
    try:
        user_message = req.messages[-1]["content"] if req.messages else ""
        user_lang = detect_language(user_message) or "en"
        messages = [ChatMessage(role=m["role"], content=m["content"]) for m in req.messages]

        await event_bus.publish_status(job_id, "Analyzing your question")
        plan = await plan_canon_query(client, messages)

        await event_bus.publish_status(job_id, f"Searching canon law: {plan.topic_summary}")
        chunks = retrieve_context(plan)

        await event_bus.publish_status(job_id, "Drafting the answer")
        final = await answer_from_context(client, user_message, user_lang, chunks)

        await event_bus.publish(job_id, Event(kind="content_delta", data={"text": final.answer}))
        await event_bus.publish(
            job_id,
            Event(kind="citations", data={"citations": [_citation_payload(c) for c in chunks]}),
        )
        await event_bus.publish_done(job_id)
    except Exception as exc:  # noqa: BLE001
        await event_bus.publish_error(job_id, str(exc))


@router.post("/canon-chat")
async def canon_chat(req: CanonChatRequest) -> dict:
    job_id = uuid.uuid4().hex[:12]
    event_bus.create(job_id)

    asyncio.create_task(_run_canon_turn(job_id, req))
    return {"job_id": job_id}


@router.get("/canon-events/{job_id}")
async def canon_events(job_id: str) -> EventSourceResponse:
    return EventSourceResponse(event_bus.stream(job_id))
