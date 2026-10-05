"""The model-size selector (12B/27B, config.model_sizes): the size a request names reaches the job
it starts, and the selector sits at the top of the page for both chat tabs."""

import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from docslides.api import routes_legal
from docslides.llm import client as llm_client


def test_the_requested_size_reaches_the_legal_job(monkeypatch):
    seen = []

    async def fake_turn(text, job_id, status, attachment_path=None):
        seen.append(llm_client._model_size.get())
        raise RuntimeError("stop here")  # ends the job with an "error" event

    monkeypatch.setattr(routes_legal, "run_legal_turn", fake_turn)
    app = FastAPI()
    app.include_router(routes_legal.router)
    with TestClient(app) as client:
        for size in ("27B", None):
            client.post("/api/legal-chat", json={"messages": [{"role": "user", "content": "q"}], "model_size": size})
        for _ in range(50):
            if len(seen) == 2:
                break
            time.sleep(0.05)

    assert seen == ["27B", None]


def test_the_selector_is_on_top_of_both_tabs_and_defaults_to_12b(monkeypatch):
    from docslides.config import get_config
    from docslides.ui import gradio_app

    monkeypatch.setattr(get_config().llm, "backend", "ollama")
    demo = gradio_app.build_app()
    selector = next(b for b in demo.blocks.values() if getattr(b, "elem_id", None) == "model-size")
    assert (selector.choices, selector.value, selector.interactive) == ([("12B", "12B"), ("27B", "27B")], "12B", True)
    # An input to every send: the General chat, the rewrite, the Legal question and the case folder.
    sends = [dep for dep in demo.fns.values() if selector._id in [i._id for i in dep.inputs] and len(dep.inputs) > 1]
    assert len(sends) == 4
    assert gradio_app._cutoff_markdown("27B") == "**LLM cutoff date:** August 2024"

    monkeypatch.setattr(get_config().llm, "backend", "vllm")
    selector = next(b for b in gradio_app.build_app().blocks.values() if getattr(b, "elem_id", None) == "model-size")
    assert selector.interactive is False  # vLLM serves one model
