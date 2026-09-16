"""PPTX generation job kickoff, live status stream, and download."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from docslides.api.events import event_bus, job_outputs
from docslides.pipeline.orchestrator import run_pipeline

router = APIRouter(prefix="/api", tags=["pptx"])


class GenerateRequest(BaseModel):
    file_path: str
    target_lang: str


@router.post("/generate")
async def generate(req: GenerateRequest, background_tasks: BackgroundTasks) -> dict:
    job_id = uuid.uuid4().hex[:12]
    event_bus.create(job_id)

    async def _run() -> None:
        output_path = await run_pipeline(job_id, req.file_path, req.target_lang)
        job_outputs[job_id] = str(output_path)

    background_tasks.add_task(_run)
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
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )
