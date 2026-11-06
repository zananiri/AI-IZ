"""Browser-session lifecycle: POST /api/sessions/{session}/cancel stops every
job a page started (chat turns, rewrites, slide/document generation, Legal
questions and cases). The UI calls it when the page is refreshed or closed
(ui/gradio_app.py's unload handler), so a reloaded page starts fresh rather
than leaving its old requests running on the model server.

POST /api/jobs/{job_id}/cancel stops a single job: the chats' Stop button."""

from __future__ import annotations

from fastapi import APIRouter

from docslides.api.events import event_bus
from docslides.logging_setup import get_logger

router = APIRouter(prefix="/api", tags=["session"])
logger = get_logger(__name__)


@router.post("/sessions/{session}/cancel")
async def cancel_session(session: str) -> dict:
    cancelled = event_bus.cancel_session(session)
    if cancelled:
        logger.info("session_jobs_cancelled", session=session, jobs=cancelled)
    return {"cancelled": cancelled}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    cancelled = event_bus.cancel_job(job_id)
    if cancelled:
        logger.info("job_cancelled", job_id=job_id)
    return {"cancelled": cancelled}
