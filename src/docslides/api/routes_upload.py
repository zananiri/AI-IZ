"""File upload endpoint."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile

from docslides.config import get_config

router = APIRouter(prefix="/api", tags=["upload"])

ALLOWED_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".tiff"}


@router.post("/upload")
async def upload_file(file: UploadFile) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return {"error": f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}"}

    upload_dir = Path(get_config().paths.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    doc_id = uuid.uuid4().hex[:12]
    dest = upload_dir / f"{doc_id}{suffix}"
    contents = await file.read()
    dest.write_bytes(contents)

    return {"document_id": doc_id, "file_path": str(dest), "original_filename": file.filename}
