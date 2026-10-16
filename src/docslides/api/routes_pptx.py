"""PPTX generation job kickoff, live status stream, and download of any generated file."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from docslides.api.events import SESSION_HEADER, event_bus, job_outputs
from docslides.llm.client import model_size
from docslides.pipeline.orchestrator import run_pipeline

router = APIRouter(prefix="/api", tags=["pptx"])

# /api/download serves every generated file: slide-pipeline decks and the chat's
# documents/generator.py output alike.
_MEDIA_TYPES = {
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pdf": "application/pdf",
}


class GenerateRequest(BaseModel):
    file_path: str
    target_lang: str
    model_size: str | None = None  # the UI's model-size choice (config.model_sizes); None = the configured models


@router.post("/generate")
async def generate(req: GenerateRequest, session: str | None = Header(default=None, alias=SESSION_HEADER)) -> dict:
    job_id = uuid.uuid4().hex[:12]

    async def _run() -> None:
        try:
            output_path = await run_pipeline(job_id, req.file_path, req.target_lang)
        except Exception:  # noqa: BLE001 -- run_pipeline already published its own error event
            return
        job_outputs[job_id] = str(output_path)

    with model_size(req.model_size):
        event_bus.start(job_id, _run(), session)
    return {"job_id": job_id}


@router.get("/events/{job_id}")
async def stream_events(job_id: str) -> EventSourceResponse:
    return EventSourceResponse(event_bus.stream(job_id))


@router.get("/download/{job_id}")
async def download(job_id: str) -> FileResponse:
    output_path = job_outputs.get(job_id)
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output not found; job may still be running.")
    return FileResponse(
        path=output_path,
        filename=Path(output_path).name,
        media_type=_MEDIA_TYPES.get(Path(output_path).suffix.lower(), "application/octet-stream"),
    )
