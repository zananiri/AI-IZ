"""GET /api/lpj-file?path=... -- opens a source document cited by the Agentic RAG tab.

The LPJ service (RAG_LPJ/pipeline, `python -m lpj_rag serve`) listens on loopback only, so a browser
on another PC can't reach it; this route relays the file through the app. The service serves only
files in its own inventory, by the path the sources panel shows.
"""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse

router = APIRouter(prefix="/api", tags=["lpj"])

LPJ_RAG_URL = os.environ.get("LPJ_RAG_URL", "http://127.0.0.1:8765")
_PASS_HEADERS = ("content-type", "content-length", "content-disposition")


@router.get("/lpj-file")
async def lpj_file(path: str = Query(..., min_length=1)):
    client = httpx.AsyncClient(timeout=httpx.Timeout(10, read=120))
    try:
        req = client.build_request("GET", f"{LPJ_RAG_URL}/file", params={"path": path})
        resp = await client.send(req, stream=True)
    except httpx.HTTPError:
        await client.aclose()
        return JSONResponse({"error": "the Agentic RAG service is not running"}, status_code=503)
    if resp.status_code != 200:
        body = await resp.aread()
        await resp.aclose()
        await client.aclose()
        return JSONResponse({"error": body.decode(errors="replace")[:200]}, status_code=resp.status_code)

    async def stream():
        try:
            async for block in resp.aiter_bytes():
                yield block
        finally:
            await resp.aclose()
            await client.aclose()

    headers = {k: v for k, v in resp.headers.items() if k.lower() in _PASS_HEADERS}
    return StreamingResponse(stream(), headers=headers, media_type=resp.headers.get("content-type"))
