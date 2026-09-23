"""FastAPI application entrypoint."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from docslides.api.routes_canon import router as canon_router
from docslides.api.routes_chat import router as chat_router
from docslides.api.routes_legal import router as legal_router
from docslides.api.routes_pptx import router as pptx_router
from docslides.api.routes_upload import router as upload_router
from docslides.config import get_config
from docslides.llm.client import aclose_all_clients
from docslides.logging_setup import configure_logging


def _silence_benign_proactor_errors(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    # On Windows, ProactorEventLoop logs a ConnectionResetError when a peer
    # drops the connection while a pipe transport is being torn down. The
    # request has already completed by then, so it's noise, not a failure.
    exception = context.get("exception")
    if isinstance(exception, ConnectionResetError):
        return
    loop.default_exception_handler(context)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    get_config()
    if os.name == "nt":
        asyncio.get_running_loop().set_exception_handler(_silence_benign_proactor_errors)
    yield
    await aclose_all_clients()


app = FastAPI(title="docslides", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(pptx_router)
app.include_router(chat_router)
app.include_router(legal_router)
app.include_router(canon_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# Gradio UI is mounted as a sub-application at /ui so a single "app" service
# (per docker-compose.yml) serves both the API and the chat interface.
def _mount_ui() -> None:
    import gradio as gr

    from docslides.ui.gradio_app import APP_CSS, build_app

    demo = build_app()
    gr.mount_gradio_app(app, demo, path="/ui", css=APP_CSS)


_mount_ui()


def run() -> None:
    port = int(os.environ.get("DOCSLIDES_APP_PORT", "8456"))
    uvicorn.run("docslides.api.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    run()
