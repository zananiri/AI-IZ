"""System status tab: model rows are merged per deployment, Ollama tags match like `ollama run`,
and one failing check never takes the report down."""

from __future__ import annotations

from docslides import system_status
from docslides.config import get_config
from docslides.system_status import (
    FAIL,
    OK,
    Check,
    _deployments,
    _ollama_has,
    render_markdown,
    run_checks,
)


def test_ollama_tag_matching():
    assert _ollama_has(["gemma4:12b", "llama3:latest"], "gemma4:12b") == "gemma4:12b"
    assert _ollama_has(["gemma4:12b", "llama3:latest"], "llama3") == "llama3:latest"
    assert _ollama_has(["gemma4:12b"], "gemma4:31b") is None


def test_shared_model_is_one_row():
    cfg = get_config()
    deployments = _deployments()
    keys = [(d.cfg.backend, d.cfg.base_url.rstrip("/"), d.cfg.model) for d in deployments]
    assert len(keys) == len(set(keys))
    roles = [role for d in deployments for role in d.roles]
    assert "General chat" in roles and "Legal GPT" in roles
    if (cfg.llm.backend, cfg.llm.base_url, cfg.llm.model) == (
        cfg.legal.orchestrator.backend, cfg.legal.orchestrator.base_url, cfg.legal.orchestrator.model
    ):
        assert any(d.roles[:2] == ["General chat", "Legal GPT"] for d in deployments)


def test_crashing_check_is_reported_not_raised(monkeypatch):
    def boom():
        raise RuntimeError("kaboom")

    quiet = lambda *a, **k: []
    for name in dir(system_status):
        if name.startswith("_check_") and name != "_check_model":
            monkeypatch.setattr(system_status, name, quiet)
    monkeypatch.setattr(system_status, "_servers", quiet)
    monkeypatch.setattr(system_status, "_deployments", list)
    monkeypatch.setattr(system_status, "_check_gpu", boom)

    checks = run_checks()
    assert [(c.state, "kaboom" in c.detail) for c in checks] == [(FAIL, True)]


def test_render_groups_and_escapes_pipes():
    md = render_markdown(
        [Check("LLMs", "m", OK, "a | b"), Check("OCR", "t", FAIL, "missing")], elapsed_s=1.0
    )
    assert "### LLMs" in md and "### OCR" in md
    assert "a \\| b" in md
    assert "1 OK · 0 warnings · 1 down" in md
