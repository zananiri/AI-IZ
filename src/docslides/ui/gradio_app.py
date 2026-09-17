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
                        break
                    elif event_kind == "error":
                        content_text += f"\n\n⚠️ {data.get('message')}"
                        started_streaming = True
                        llm_status = "❌ Error"
                        break

                    display_text = content_text if started_streaming else f"_{status_text}..._"
                    new_history = history + [{"role": "assistant", "content": display_text}]
                    yield (
                        gr.update(value=new_history),
                        gr.update(value=reasoning_text, visible=bool(reasoning_text), rtl=rtl_hint),
                        gr.update(value=None),
                        gr.update(value=llm_status),
                    )


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
            placeholder='Ask a question, request a translation or rewrite, or say "make this into a presentation" -- attach a document (PDF, DOCX, PPTX, XLSX, image) with the 📎 button',
            file_types=[".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".tiff"],
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

    send_btn.click(
        fn=send_chat_message,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, reasoning_panel, msg_box, llm_status],
    )
    msg_box.submit(
        fn=send_chat_message,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, reasoning_panel, msg_box, llm_status],
    )
    rewrite_btn.click(
        fn=send_tone_rewrite,
        inputs=[msg_box, professionalism_slider, creativity_slider, chatbot],
        outputs=[chatbot, reasoning_panel, msg_box, llm_status],
    )


# ---------------------------------------------------------------------------
# Legal tab -- a 3-step pipeline (see api/routes_legal.py, legal/pipeline.py):
# an orchestrator model plans the research and reformulates the question in
# Hebrew, a Hebrew legal-domain model (DictaLM) analyzes it in Hebrew, and
# the orchestrator verifies the result and translates the final answer back
# into whatever language the user asked in. Citations/relevant laws are kept
# out of the chat bubble and shown in a side panel instead.
# ---------------------------------------------------------------------------


def _format_citations(citations: list[str], relevant_laws: list[str]) -> str:
    if not citations and not relevant_laws:
        return "_No citations yet._"
    sections = []
    if relevant_laws:
        sections.append("**Relevant laws**\n" + "\n".join(f"- {law}" for law in relevant_laws))
    if citations:
        sections.append("**Citations**\n" + "\n".join(f"- {c}" for c in citations))
    return "\n\n".join(sections)


def _stream_legal_job(job_id: str, history: list):
    """Same SSE event contract as `_stream_job`, plus a "citations" event
    kind (the verification stage's structured citations/relevant-laws
    output) routed to the side panel instead of the chat bubble. Also
    derives a persistent `llm_status` line -- the 3-step pipeline (plan ->
    Hebrew analysis -> verify) already publishes a status event per stage
    (see api/routes_legal.py), this just keeps that visible instead of
    discarding it once the answer starts rendering."""
    content_text = ""
    status_text = "Connecting"
    llm_status = f"🔌 {status_text}..."
    started_streaming = False
    citations_md = "_No citations yet._"

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
                        citations_md = _format_citations(data.get("citations", []), data.get("relevant_laws", []))
                    elif event_kind == "done":
                        llm_status = "✅ Done"
                        break
                    elif event_kind == "error":
                        content_text += f"\n\n⚠️ {data.get('message')}"
                        started_streaming = True
                        llm_status = "❌ Error"
                        break

                    detected = detect_language(content_text[:200]) if started_streaming else None
                    display_text = content_text if started_streaming else f"_{status_text}..._"
                    new_history = history + [{"role": "assistant", "content": display_text}]
                    yield (
                        gr.update(value=new_history, rtl=_is_rtl_lang(detected)),
                        gr.update(value=None),
                        gr.update(value=citations_md),
                        gr.update(value=llm_status),
                    )


def send_legal_message(message: str, history: list):
    message = (message or "").strip()
    if not message:
        yield history, gr.update(), gr.update(), gr.update()
        return

    history = history + [{"role": "user", "content": message}]
    payload = {"messages": [{"role": "user", "content": message}]}

    with httpx.Client(timeout=60) as client:
        resp = client.post(f"{API_BASE_URL}/api/legal-chat", json=payload)
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

    yield from _stream_legal_job(job_id, history)


def build_legal_tab() -> None:
    cfg = get_config()
    gr.Markdown(
        f"_Orchestrator: **{cfg.legal.orchestrator.model}** via **{cfg.legal.orchestrator.backend}** "
        f"· Hebrew analyst: **{cfg.legal.hebrew_analyst.model}** via **{cfg.legal.hebrew_analyst.backend}**_"
    )
    with gr.Row():
        with gr.Column(scale=3):
            legal_chatbot = gr.Chatbot(label="Legal Assistant")
            legal_llm_status = gr.Markdown(value="_Idle_", label="LLM status", show_label=True, container=True)
            with gr.Row():
                legal_msg_box = gr.Textbox(
                    label="Legal question",
                    scale=4,
                    placeholder="Ask a legal question in any language -- researched against Israeli law via a Hebrew legal-analysis model",
                )
                legal_send_btn = gr.Button("Send", scale=1)
        with gr.Column(scale=1):
            gr.Markdown("### Citations & Relevant Laws")
            citations_panel = gr.Markdown(value="_No citations yet._")

    legal_send_btn.click(
        fn=send_legal_message,
        inputs=[legal_msg_box, legal_chatbot],
        outputs=[legal_chatbot, legal_msg_box, citations_panel, legal_llm_status],
    )
    legal_msg_box.submit(
        fn=send_legal_message,
        inputs=[legal_msg_box, legal_chatbot],
        outputs=[legal_chatbot, legal_msg_box, citations_panel, legal_llm_status],
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AI Workbench - Ibrahim Z.") as demo:
        gr.Markdown("# AI Workbench - Ibrahim Z.")
        with gr.Tabs():
            with gr.Tab("General GPT"):
                build_chat_tab()
            with gr.Tab("Legal GPT"):
                build_legal_tab()
    return demo


def run() -> None:
    demo = build_app()
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    run()
