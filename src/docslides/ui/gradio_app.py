"""Gradio chat interface: one chat tab handles everything -- ask questions,
request translations/rewrites, or ask for a PowerPoint deck, optionally with
an attached document (PDF/DOCX/PPTX/XLSX/image). The backend
(api/routes_chat.py) classifies whether a turn with an attachment wants
slides generated (handed off to the full pipeline) or something else
(answered inline with the document as context) -- there's no separate
"Document -> Slides" tab; ask for it in the chat instead.

Talks to the FastAPI backend (docslides.api.main) over HTTP/SSE rather than
calling pipeline internals directly, so the UI and API can scale/deploy
independently (see docker-compose.yml).
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import gradio as gr
import httpx

from docslides.config import get_config
from docslides.ingestion.language_detect import detect_language

API_BASE_URL = os.environ.get("DOCSLIDES_API_URL", "http://localhost:8456")


def _is_rtl_lang(lang: str | None) -> bool:
    return bool(lang) and get_config().languages.is_rtl(lang)


# While a tab's request runs, its message box pulses with the theme's accent
# border -- the same look as Gradio's own "generating" border, which only
# plain Textbox outputs get, only once the first update arrives, and never the
# General tab's MultimodalTextbox (its progress tracker is switched off).
# Toggled from Python through elem_classes, so every tab glows the same way
# from Send until the answer is done. Passed as `css=` wherever the app is
# served (run() below, and api/main.py's mount).
APP_CSS = """
.ai-working { position: relative; }
.ai-working::after {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  border: 2px solid var(--color-accent); border-radius: inherit;
  z-index: var(--layer-1, 1);
  animation: ai-working-pulse 2s cubic-bezier(.4, 0, .6, 1) infinite;
}
@keyframes ai-working-pulse { 0%, 100% { opacity: 1; } 50% { opacity: .5; } }
@media (prefers-reduced-motion: reduce) { .ai-working::after { animation: none; } }
"""

_WORKING_CLASSES = ["ai-working"]


def _glow_while_running(handler, outputs: list, box):
    """`handler`, a streaming send handler for `outputs`, with the message
    box `box` (one of those outputs) glowing from the moment Send is pressed
    until the handler finishes -- or fails, so a failed upload or request
    never leaves it glowing."""
    index = next(i for i, component in enumerate(outputs) if component is box)

    def with_box(values, working: bool) -> tuple:
        values = list(values)
        update = values[index] if isinstance(values[index], dict) else gr.update(value=values[index])
        values[index] = {**update, "elem_classes": _WORKING_CLASSES if working else []}
        return tuple(values)

    idle = tuple(gr.update() for _ in outputs)

    @functools.wraps(handler)
    def run(*args):
        yield with_box(idle, working=True)
        try:
            for values in handler(*args):
                yield with_box(values, working=True)
        except Exception:
            yield with_box(idle, working=False)
            raise
        yield with_box(idle, working=False)

    return run


# ---------------------------------------------------------------------------
# Chat tab (documents, translation, rewriting, and slide generation all go
# through here -- see module docstring)
# ---------------------------------------------------------------------------


def _upload_file(file_path: str) -> dict:
    """Uploads an attachment (picked via the message box's attach button) to
    the backend and returns its server-side path/name. Raising gr.Error here
    aborts the send and shows the failure to the user instead of silently
    dropping the attachment."""
    with httpx.Client(timeout=60) as client:
        with open(file_path, "rb") as f:
            resp = client.post(f"{API_BASE_URL}/api/upload", files={"file": f})
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise gr.Error(data["error"])

    return {"path": data["file_path"], "name": data["original_filename"]}


def _stream_job(job_id: str, history: list, rtl_hint: bool):
    """`history` must already include the user's turn as the last entry, in
    Gradio's "messages" format (`gr.Chatbot` in Gradio 6.x only accepts
    `[{"role": "user"|"assistant", "content": str}, ...]` -- the old
    `[[user, bot], ...]` "tuples" format this UI used before Gradio 6 is no
    longer supported and silently breaks every send). The assistant's turn
    is appended fresh each yield rather than mutated in place.

    Handles four event kinds on the shared job SSE stream: "status" (pipeline
    progress, e.g. during slide generation -- shown as a single evolving
    line until real content starts), "reasoning_delta"/"content_delta" (the
    streamed chat answer), and "done" (which carries `output_path` when the
    turn was a slide-generation request, turned into a download link). Also
    derives a one-line `llm_status` ("Waiting for model" / "Thinking" /
    "Writing response" / ...) so the UI always shows what the model is
    currently doing, not just the final text."""
    reasoning_text = ""
    content_text = ""
    status_text = "Connecting"
    llm_status = f"🔌 {status_text}..."
    started_streaming = False

    with httpx.Client(timeout=None) as client:
        with client.stream("GET", f"{API_BASE_URL}/api/chat-events/{job_id}") as resp:
            event_kind = None
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    event_kind = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip())
                    if event_kind == "status":
                        status_text = data["message"]
                        llm_status = f"⏳ {status_text}..."
                    elif event_kind == "reasoning_delta":
                        started_streaming = True
                        reasoning_text += data["text"]
                        llm_status = "🧠 Thinking..."
                    elif event_kind == "content_delta":
                        started_streaming = True
                        content_text += data["text"]
                        detected = detect_language(content_text[:200])
                        rtl_hint = _is_rtl_lang(detected)
                        llm_status = "✍️ Writing response..."
                    elif event_kind == "done":
                        output_path = data.get("output_path")
                        if output_path:
                            download_url = f"{API_BASE_URL}/api/download/{job_id}"
                            content_text = f"Your presentation is ready: [Download {Path(output_path).name}]({download_url})"
                            started_streaming = True
                        llm_status = "✅ Done"
                    elif event_kind == "error":
                        content_text += f"\n\n⚠️ {data.get('message')}"
                        started_streaming = True
                        llm_status = "❌ Error"

                    display_text = content_text if started_streaming else f"_{status_text}..._"
                    new_history = history + [{"role": "assistant", "content": display_text}]
                    yield (
                        gr.update(value=new_history),
                        gr.update(value=reasoning_text, visible=bool(reasoning_text), rtl=rtl_hint),
                        gr.update(value=None),
                        gr.update(value=llm_status),
                    )
                    if event_kind in ("done", "error"):
                        break  # after the yield, so the final status or error still shows


def send_chat_message(message: dict, history: list):
    """`message` is a `gr.MultimodalTextbox` payload: `{"text": str, "files":
    [local_path, ...]}`. Attachments are picked via that box's own attach
    button rather than a separate upload widget, and are per-message (no
    cross-turn "stays attached" state) -- upload happens here, right before
    the turn is sent."""
    text = (message.get("text") or "").strip()
    files = message.get("files") or []
    if not text and not files:
        yield history, gr.update(), gr.update(), gr.update()
        return

    attachment = _upload_file(files[0]) if files else None

    display_message = text
    if attachment:
        display_message = f"📎 {attachment['name']}\n\n{text}" if text else f"📎 {attachment['name']}"
    history = history + [{"role": "user", "content": display_message}]

    payload: dict = {
        "messages": [
            {
                "role": "user",
                "content": text or f"(No message -- I just attached {attachment['name'] if attachment else 'a file'}.)",
            }
        ]
    }
    if attachment:
        payload["attachment_path"] = attachment["path"]

    with httpx.Client(timeout=60) as client:
        resp = client.post(f"{API_BASE_URL}/api/chat", json=payload)
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

    yield from _stream_job(job_id, history, rtl_hint=False)


def send_tone_rewrite(message: dict, professionalism: int, creativity: int, history: list):
    """Rewrites whatever is in the same message box used for regular chat --
    typed/pasted text, or a file attached via its 📎 button (extracted
    server-side, same as a chat attachment). No separate input for the
    content or for describing what's being rewritten -- the tone sliders are
    the only thing this adds on top of the normal message box."""
    text = (message.get("text") or "").strip()
    files = message.get("files") or []
    if not text and not files:
        yield history, gr.update(), gr.update(), gr.update()
        return

    attachment = _upload_file(files[0]) if files else None

    display_message = f"[Rewrite request] 📎 {attachment['name']}" if attachment else f"[Rewrite request] {text}"
    if attachment and text:
        display_message += f"\n\n{text}"
    history = history + [{"role": "user", "content": display_message}]

    payload: dict = {
        "text": text,
        "professionalism": professionalism,
        "creativity": creativity,
    }
    if attachment:
        payload["attachment_path"] = attachment["path"]

    with httpx.Client(timeout=60) as client:
        resp = client.post(f"{API_BASE_URL}/api/tone-rewrite", json=payload)
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

    yield from _stream_job(job_id, history, rtl_hint=False)


def build_chat_tab() -> None:
    cfg = get_config()
    gr.Markdown(f"_Model: **{cfg.llm.model}** via **{cfg.llm.backend}**_")

    chatbot = gr.Chatbot(label="Chat")
    llm_status = gr.Markdown(value="_Idle_", label="LLM status", show_label=True, container=True)
    reasoning_panel = gr.Textbox(label="Reasoning (model's thinking)", lines=6, visible=False)

    with gr.Row():
        msg_box = gr.MultimodalTextbox(
            label="Message",
            scale=4,
            lines=4,
            max_lines=24,
            placeholder='Ask a question, paste a large block of text to rewrite/translate/summarize, or say '
            '"make this into a presentation" -- attach a document (PDF, DOCX, PPTX, XLSX, image, or .txt) instead with the 📎 button',
            # .txt matters here beyond ordinary file attachments: pasting a
            # large enough block of text makes the browser/Gradio turn the
            # paste itself into a text/plain file attachment instead of
            # inline text (see ingestion/parser.py's plain-text fast path) --
            # without .txt allowed, every large paste was rejected outright
            # with "Invalid file type: text/plain".
            file_types=[".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".tiff", ".txt"],
            file_count="single",
            sources=["upload"],
        )
        send_btn = gr.Button("Send", scale=1)

    with gr.Accordion("Tone control", open=False):
        gr.Markdown(
            "Rewrites whatever is in the message box above -- typed/pasted text, or "
            "an attached document (📎) -- with the tone below, instead of answering it "
            "normally. Professionalism drives wording/register (system prompt); "
            "Creativity drives sampling randomness only. Use this button instead of Send."
        )
        professionalism_slider = gr.Slider(
            1, 5, value=3, step=1,
            label="Professionalism: 1=Casual, 2=Conversational, 3=Standard business, 4=Formal, 5=Executive/Legal",
        )
        creativity_slider = gr.Slider(
            1, 5, value=3, step=1,
            label="Creativity: 1=Minimal, 2=Low, 3=Balanced, 4=High, 5=Maximum",
        )
        rewrite_btn = gr.Button("Rewrite with tone controls")

    chat_outputs = [chatbot, reasoning_panel, msg_box, llm_status]
    send = _glow_while_running(send_chat_message, chat_outputs, msg_box)
    send_btn.click(fn=send, inputs=[msg_box, chatbot], outputs=chat_outputs)
    msg_box.submit(fn=send, inputs=[msg_box, chatbot], outputs=chat_outputs)
    rewrite_btn.click(
        fn=_glow_while_running(send_tone_rewrite, chat_outputs, msg_box),
        inputs=[msg_box, professionalism_slider, creativity_slider, chatbot],
        outputs=chat_outputs,
    )


# ---------------------------------------------------------------------------
# Legal tab -- grounded RAG over Israeli law (see api/routes_legal.py,
# legal/pipeline.py): Qwen researches, drafts and verifies against sources
# approved into the signed index; DictaLM (user-selected tier) only normalizes
# Hebrew questions and polishes Hebrew answers. Citations appear as [n]
# markers in the answer and as footnotes (with verification status) in the
# side panel; the Pass A research memorandum is viewable below them.
# ---------------------------------------------------------------------------

_LEGAL_DISCLAIMER = "_Draft pending review by a licensed attorney -- not legal advice._"


def _format_legal_footnotes(footnotes: list[dict]) -> str:
    if not footnotes:
        return "_No citations yet._"
    entries = []
    for note in footnotes:
        law_section = f"{note.get('law', '')} — {note.get('section', '')}"
        details = " · ".join(
            part
            for part in (
                note.get("source_type"),
                note.get("source_origin"),
                note.get("effective"),
                None if note.get("status") == "current" else f"**{note.get('status')}**",
                "/".join(note.get("relations", [])),
            )
            if part
        )
        verified = (
            "✅ verified against source"
            if note.get("verified")
            else "⚠️ **not verified**: " + "; ".join(note.get("problems", []))
        )
        amended = "".join(f"  \n⚠️ amended: {a}" for a in note.get("amended_by", []))
        entries.append(f"**[{note['number']}]** {law_section}  \n_{details}_  \n{verified}{amended}")
    return "\n\n".join(entries)


def _legal_bubble(content_text: str, report: dict | None) -> str:
    if report is None:
        return content_text
    parts = []
    if report.get("escalation_flag"):
        reasons = report.get("escalation_reasons") or [report.get("escalation_reason") or ""]
        parts.append("⚠️ **Escalation -- attorney review needed:**\n" + "\n".join(f"- {r}" for r in reasons if r))
    parts.append(content_text)
    if report.get("coverage_gaps"):
        parts.append(f"**Coverage gaps:** {report['coverage_gaps']}")
    parts.extend(f"_{note}_" for note in report.get("notes", []))
    parts.append(_LEGAL_DISCLAIMER)
    return "\n\n".join(parts)


def _stream_legal_job(job_id: str, history: list):
    """Same SSE contract as `_stream_job`, plus "citations" (footnotes, to
    the side panel) and "legal_report" (memorandum, escalation, DictaLM
    tier used), which arrives after the answer text and wraps it with the
    escalation banner, coverage gaps and disclaimer."""
    content_text = ""
    status_text = "Connecting"
    llm_status = f"🔌 {status_text}..."
    started_streaming = False
    citations_md = "_No citations yet._"
    report: dict | None = None
    memo = None

    with httpx.Client(timeout=None) as client:
        with client.stream("GET", f"{API_BASE_URL}/api/legal-events/{job_id}") as resp:
            event_kind = None
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    event_kind = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip())
                    if event_kind == "status":
                        status_text = data["message"]
                        llm_status = f"⏳ {status_text}..."
                    elif event_kind == "content_delta":
                        started_streaming = True
                        content_text += data["text"]
                        llm_status = "✍️ Writing response..."
                    elif event_kind == "citations":
                        citations_md = _format_legal_footnotes(data.get("citations", []))
                    elif event_kind == "legal_report":
                        report = data
                        memo = data.get("research_memorandum")
                    elif event_kind == "done":
                        llm_status = "✅ Done"
                        if report and report.get("dicta_used"):
                            llm_status += f" · DictaLM: {report.get('dicta_tier')} ({report.get('dicta_model')})"
                    elif event_kind == "error":
                        content_text += f"\n\n⚠️ {data.get('message')}"
                        started_streaming = True
                        llm_status = "❌ Error"

                    lang = (report or {}).get("reply_language") or (
                        detect_language(content_text[:200]) if started_streaming else None
                    )
                    display_text = _legal_bubble(content_text, report) if started_streaming else f"_{status_text}..._"
                    new_history = history + [{"role": "assistant", "content": display_text}]
                    yield (
                        gr.update(value=new_history, rtl=_is_rtl_lang(lang)),
                        gr.update(value=None),
                        gr.update(value=citations_md),
                        gr.update(value=llm_status),
                        gr.update(value=memo),
                    )
                    if event_kind in ("done", "error"):
                        break


def send_legal_message(message: str, history: list, dicta_tier: str):
    message = (message or "").strip()
    if not message:
        yield history, gr.update(), gr.update(), gr.update(), gr.update()
        return

    history = history + [{"role": "user", "content": message}]
    payload = {"messages": [{"role": "user", "content": message}], "dicta_tier": dicta_tier}

    with httpx.Client(timeout=60) as client:
        resp = client.post(f"{API_BASE_URL}/api/legal-chat", json=payload)
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

    yield from _stream_legal_job(job_id, history)


def check_dicta_tier(selected: str):
    """Non-blocking RAM suggestion: shows the banner + a one-click switch when
    the selected DictaLM tier doesn't fit, and never changes the tier itself."""
    hidden = (gr.update(visible=False), gr.update(visible=False), None)
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(f"{API_BASE_URL}/api/legal/dicta-tiers", params={"selected": selected})
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError:
        return hidden
    if not data.get("message"):
        return hidden
    suggest = data.get("suggest")
    labels = {t["key"]: t["label"] for t in data.get("tiers", [])}
    return (
        gr.update(value=f"⚠️ {data['message']}", visible=True),
        gr.update(value=f"Switch to {labels.get(suggest, suggest)}", visible=bool(suggest)),
        suggest,
    )


def build_legal_tab(demo: gr.Blocks, legal_tab: gr.Tab) -> None:
    legal = get_config().legal
    gr.Markdown(
        f"_Research & verification: **{legal.orchestrator.model}** via **{legal.orchestrator.backend}** "
        "· Hebrew normalization & polish: DictaLM (tier below)_"
    )
    dicta_tier = gr.Radio(
        choices=[(tier.label, key) for key, tier in legal.dicta_tiers.items()],
        value=legal.default_dicta_tier,
        label="DictaLM tier (Hebrew questions / answers only)",
    )
    with gr.Row():
        tier_banner = gr.Markdown(visible=False)
        switch_tier_btn = gr.Button(visible=False, size="sm", scale=0)
    suggested_tier = gr.State(None)

    with gr.Row():
        with gr.Column(scale=3):
            legal_chatbot = gr.Chatbot(label="Legal Assistant")
            legal_llm_status = gr.Markdown(value="_Idle_", label="LLM status", show_label=True, container=True)
            with gr.Row():
                legal_msg_box = gr.Textbox(
                    label="Legal question",
                    scale=4,
                    placeholder="Ask a question about Israeli law in any language -- answered only from the "
                    "reviewed statutes, regulations and rulings in the local index, in your language",
                )
                legal_send_btn = gr.Button("Send", scale=1)
        with gr.Column(scale=1):
            gr.Markdown("### Citations")
            citations_panel = gr.Markdown(value="_No citations yet._")
            with gr.Accordion("Research memorandum (Pass A)", open=False):
                memo_view = gr.JSON(value=None, label="Claims → evidence")

    send_inputs = [legal_msg_box, legal_chatbot, dicta_tier]
    send_outputs = [legal_chatbot, legal_msg_box, citations_panel, legal_llm_status, memo_view]
    send = _glow_while_running(send_legal_message, send_outputs, legal_msg_box)
    legal_send_btn.click(fn=send, inputs=send_inputs, outputs=send_outputs)
    legal_msg_box.submit(fn=send, inputs=send_inputs, outputs=send_outputs)

    # Checked on app start, whenever the tab is opened, and on every tier change.
    check_outputs = [tier_banner, switch_tier_btn, suggested_tier]
    dicta_tier.change(fn=check_dicta_tier, inputs=[dicta_tier], outputs=check_outputs)
    legal_tab.select(fn=check_dicta_tier, inputs=[dicta_tier], outputs=check_outputs)
    demo.load(fn=check_dicta_tier, inputs=[dicta_tier], outputs=check_outputs)
    switch_tier_btn.click(fn=lambda suggest: gr.update(value=suggest), inputs=[suggested_tier], outputs=[dicta_tier])


# ---------------------------------------------------------------------------
# Canon GPT tab -- a RAG pipeline (see api/routes_canon.py, canon/pipeline.py,
# canon/retrieval.py): an orchestrator model reformulates the question and
# guesses which code(s) it's about, retrieval fetches matching provisions
# from a local vector store built offline from vatican.va
# (scripts/ingest_canon_law.py), and the orchestrator answers grounded in
# that retrieved text. Unlike the Legal tab, citations here are real source
# links pulled from retrieval metadata rather than model-generated text.
# ---------------------------------------------------------------------------


def _format_canon_citations(citations: list[dict]) -> str:
    if not citations:
        return "_No citations yet._"
    lines = []
    for c in citations:
        label = c.get("label", "")
        url = c.get("url")
        breadcrumb = c.get("breadcrumb", "")
        entry = f"[{label}]({url})" if url else label
        if breadcrumb:
            entry += f" — {breadcrumb}"
        lines.append(f"- {entry}")
    return "\n".join(lines)


def _stream_canon_job(job_id: str, history: list):
    """Same SSE event contract as `_stream_legal_job`; the "citations" event
    here carries structured {label, url, breadcrumb} entries built straight
    from retrieval metadata (see routes_canon.py), rendered as linked
    citations instead of plain text."""
    content_text = ""
    status_text = "Connecting"
    llm_status = f"\U0001f50c {status_text}..."
    started_streaming = False
    citations_md = "_No citations yet._"

    with httpx.Client(timeout=None) as client:
        with client.stream("GET", f"{API_BASE_URL}/api/canon-events/{job_id}") as resp:
            event_kind = None
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    event_kind = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip())
                    if event_kind == "status":
                        status_text = data["message"]
                        llm_status = f"⏳ {status_text}..."
                    elif event_kind == "content_delta":
                        started_streaming = True
                        content_text += data["text"]
                        llm_status = "✍️ Writing response..."
                    elif event_kind == "citations":
                        citations_md = _format_canon_citations(data.get("citations", []))
                    elif event_kind == "done":
                        llm_status = "✅ Done"
                    elif event_kind == "error":
                        content_text += f"\n\n⚠️ {data.get('message')}"
                        started_streaming = True
                        llm_status = "❌ Error"

                    detected = detect_language(content_text[:200]) if started_streaming else None
                    display_text = content_text if started_streaming else f"_{status_text}..._"
                    new_history = history + [{"role": "assistant", "content": display_text}]
                    yield (
                        gr.update(value=new_history, rtl=_is_rtl_lang(detected)),
                        gr.update(value=None),
                        gr.update(value=citations_md),
                        gr.update(value=llm_status),
                    )
                    if event_kind in ("done", "error"):
                        break  # after the yield, so the final status or error still shows


def send_canon_message(message: str, history: list):
    message = (message or "").strip()
    if not message:
        yield history, gr.update(), gr.update(), gr.update()
        return

    history = history + [{"role": "user", "content": message}]
    payload = {"messages": [{"role": "user", "content": message}]}

    with httpx.Client(timeout=60) as client:
        resp = client.post(f"{API_BASE_URL}/api/canon-chat", json=payload)
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

    yield from _stream_canon_job(job_id, history)


def build_canon_tab() -> None:
    cfg = get_config()
    gr.Markdown(
        f"_Model: **{cfg.canon.generation.model}** via **{cfg.canon.generation.backend}** "
        f"· retrieval: **{cfg.canon.embedding_model}** over CIC 1983 and CCEO 1990 (Latin)_"
    )
    with gr.Row():
        with gr.Column(scale=3):
            canon_chatbot = gr.Chatbot(label="Canon GPT")
            canon_llm_status = gr.Markdown(value="_Idle_", label="LLM status", show_label=True, container=True)
            with gr.Row():
                canon_msg_box = gr.Textbox(
                    label="Question",
                    scale=4,
                    placeholder="Ask about canon law, in any language -- researched against the "
                    "Code of Canon Law (CIC) and the Code of Canons of the Eastern Churches (CCEO)",
                )
                canon_send_btn = gr.Button("Send", scale=1)
        with gr.Column(scale=1):
            gr.Markdown("### Sources")
            canon_citations_panel = gr.Markdown(value="_No citations yet._")

    canon_outputs = [canon_chatbot, canon_msg_box, canon_citations_panel, canon_llm_status]
    send = _glow_while_running(send_canon_message, canon_outputs, canon_msg_box)
    canon_send_btn.click(fn=send, inputs=[canon_msg_box, canon_chatbot], outputs=canon_outputs)
    canon_msg_box.submit(fn=send, inputs=[canon_msg_box, canon_chatbot], outputs=canon_outputs)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AI Workbench - Ibrahim Z.") as demo:
        gr.Markdown("# AI Workbench - Ibrahim Z.")
        with gr.Tabs():
            with gr.Tab("General GPT"):
                build_chat_tab()
            with gr.Tab("Legal GPT") as legal_tab:
                build_legal_tab(demo, legal_tab)
            with gr.Tab("Canon GPT"):
                build_canon_tab()
    return demo


def run() -> None:
    demo = build_app()
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, css=APP_CSS)


if __name__ == "__main__":
    run()
