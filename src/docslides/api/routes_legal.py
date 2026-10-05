"""Legal tab endpoints.

  * POST /api/legal-chat + GET /api/legal-events/{job_id}: runs one turn of
    the grounded Israeli-law pipeline (legal/pipeline.py) and streams it:
    "status" per stage, Pass 0's chain-of-thought as a single "reasoning_delta"
    (the pipeline runs each pass as one call rather than token-streamed, so
    this arrives as one chunk rather than incrementally -- the general chat
    tab's reasoning panel displays it the same way either way), the answer as
    "content_delta" (citation tokens rendered as [n] footnote markers), a
    "citations" event with the footnotes (law, section, effective range,
    relation, verified or not), and a "legal_report" event with the spec
    section 9 output (research memorandum, escalation, coverage gaps).
  * POST /api/legal-case: the same, for a case instead of a question
    (legal/pipeline.run_case_turn) -- the answer is a paralegal work file
    ending in a recommended next step, streamed over the same events, so
    GET /api/legal-events/{job_id} serves both. A case can come with a whole
    case folder (`documents`, each uploaded through /api/upload first): every
    document is read, and analyzed one by one when they don't fit the case
    file together, before the work file is written from them.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Header
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from docslides.api.events import SESSION_HEADER, Event, event_bus
from docslides.legal.pipeline import LegalTurnResult, run_case_turn, run_legal_turn
from docslides.llm.client import model_size

router = APIRouter(prefix="/api", tags=["legal"])


class CaseDocument(BaseModel):
    path: str  # server-side path from a prior /api/upload call
    name: str  # the file's original name, as the case folder had it


class LegalChatRequest(BaseModel):
    messages: list[dict]  # [{"role": "user"|"assistant"|"system", "content": str}]
    attachment_path: str | None = None  # server-side path from a prior /api/upload call
    documents: list[CaseDocument] | None = None  # a case folder's files (/api/legal-case only)
    model_size: str | None = None  # the UI's model-size choice (config.model_sizes); None = the configured models


TurnFn = Callable[..., Awaitable[LegalTurnResult]]


async def _run_turn(job_id: str, req: LegalChatRequest, turn: TurnFn) -> None:
    try:
        text = req.messages[-1]["content"] if req.messages else ""

        async def status(message: str) -> None:
            await event_bus.publish_status(job_id, message)

        kwargs: dict = {"attachment_path": req.attachment_path}
        if req.documents:
            kwargs["documents"] = [(d.path, d.name) for d in req.documents]
        result = await turn(text, job_id, status, **kwargs)

        if result.analysis_notes:
            await event_bus.publish(job_id, Event(kind="reasoning_delta", data={"text": result.analysis_notes}))
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


def _start(req: LegalChatRequest, turn: TurnFn, session: str | None) -> dict:
    job_id = uuid.uuid4().hex[:12]
    with model_size(req.model_size):  # the job's task inherits it
        event_bus.start(job_id, _run_turn(job_id, req, turn), session)
    return {"job_id": job_id}


@router.post("/legal-chat")
async def legal_chat(req: LegalChatRequest, session: str | None = Header(default=None, alias=SESSION_HEADER)) -> dict:
    return _start(req, run_legal_turn, session)


@router.post("/legal-case")
async def legal_case(req: LegalChatRequest, session: str | None = Header(default=None, alias=SESSION_HEADER)) -> dict:
    return _start(req, run_case_turn, session)


@router.get("/legal-events/{job_id}")
async def legal_events(job_id: str) -> EventSourceResponse:
    return EventSourceResponse(event_bus.stream(job_id))
