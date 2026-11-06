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
import re
import tempfile
from datetime import datetime
from html import escape
from pathlib import Path
from urllib.parse import quote

import gradio as gr
import httpx

from docslides.api.events import SESSION_HEADER
from docslides.config import get_config
from docslides.ingestion.language_detect import detect_language
from docslides.legal.caselaw import caselaw_stats
from docslides.legal.corpus_retrieval import corpus_stats
from docslides.system_status import status_report
from docslides.ui import brand

API_BASE_URL = os.environ.get("DOCSLIDES_API_URL", "http://localhost:8456")


def _is_rtl_lang(lang: str | None) -> bool:
    return bool(lang) and get_config().languages.is_rtl(lang)


# While a tab's request runs, its chat box takes the theme's accent
# border -- the same look as Gradio's own "generating" border, which only
# plain Textbox outputs get, only once the first update arrives, and never the
# General tab's MultimodalTextbox (its progress tracker is switched off).
# Toggled from Python through elem_classes, so every tab glows the same way
# from Send until the answer is done. Passed as `css=` wherever the app is
# served (run() below, and api/main.py's mount), together with APP_THEME.
APP_THEME = brand.build_theme()

APP_CSS = """
/* The Codex Sentinel header: the single-ink logo at the left edge (ltr so an rtl page doesn't
   flip it to the right), over a plain rule. The --cs-* variables colour the logo per mode. */
.brand-header {
  --cs-ink: #1F3A5F; --cs-accent: #A8874A; --cs-page: var(--body-background-fill);
  display: flex; flex-direction: column; align-items: flex-start; gap: 4px; direction: ltr;
  padding: 16px 4px 12px; margin-bottom: 4px;
  border-bottom: 1px solid var(--border-color-primary);
}
.dark .brand-header { --cs-ink: #E6EBF2; --cs-accent: #C9A86A; }
.brand-header .brand-logo { width: min(400px, 100%); }
.brand-header .brand-logo svg { display: block; width: 100%; height: auto; }
.brand-header .brand-byline { color: var(--body-text-color-subdued); font-size: var(--text-sm, 12px); }
.brand-header h1 { color: var(--color-accent); margin: 0; }
/* Headings in answers: a serif with Hebrew, for a document's voice. */
.chat-log .message-content :is(h1, h2, h3) {
  font-family: "Frank Ruhl Libre", Georgia, "Times New Roman", serif; font-weight: 600;
}
/* Selected tab: navy text over a thin gold rule. */
.tab-wrapper button[role="tab"].selected, button[role="tab"][aria-selected="true"] {
  color: var(--color-accent) !important; font-weight: 600;
}
.tab-wrapper button[role="tab"].selected::after, button[role="tab"][aria-selected="true"]::after {
  background: #A8874A !important;
}
.dark .tab-wrapper button[role="tab"].selected, .dark button[role="tab"][aria-selected="true"] { color: #9DB8DC !important; }
.dark .tab-wrapper button[role="tab"].selected::after, .dark button[role="tab"][aria-selected="true"]::after {
  background: #C9A86A !important;
}
/* While a tab's request runs, its chat box takes a steady accent border (the status shows in the
   chat itself). */
.block.llm-status-general, .block.llm-status-legal { position: relative; }
.gradio-container:has(#general-msg.ai-working) .block.llm-status-general::after,
.gradio-container:has(#legal-msg.ai-working) .block.llm-status-legal::after {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  border: 2px solid var(--border-color-accent); border-radius: inherit;
  z-index: var(--layer-1, 1);
}
/* The General tab's LLM cutoff line above the chat: one point bigger. */
.llm-cutoff, .llm-cutoff * { font-size: calc(var(--text-md, 14px) + 1pt) !important; }
/* The Legal tab's sources summary above the chat history: small and muted. */
.legal-sources, .legal-sources * {
  color: var(--body-text-color-subdued) !important; font-size: var(--text-sm, 12px) !important;
}
/* Both chat tabs' send arrow: the primary navy, and two text rows taller (growing
   downward from the top of the box). A fixed size in every state -- idle,
   disabled while a request runs, after the box is cleared -- so it never
   changes size once a message is sent. */
.chat-input button.submit-button {
  background: var(--button-primary-background-fill) !important;
  color: var(--button-primary-text-color) !important;
  box-sizing: border-box !important; flex: 0 0 auto !important;
  width: var(--size-9, 36px) !important; min-width: var(--size-9, 36px) !important;
  max-width: var(--size-9, 36px) !important; padding: 0 !important;
  height: calc(var(--size-9, 36px) + 42px) !important;
  min-height: calc(var(--size-9, 36px) + 42px) !important;
  max-height: calc(var(--size-9, 36px) + 42px) !important;
  align-self: flex-start;
}
.chat-input button.submit-button:hover { background: var(--button-primary-background-fill-hover) !important; }
/* Right-to-left text (Hebrew, Arabic) in the chats: each paragraph, list
   item and the message box take their direction from their own text, so a
   Hebrew line reads right-to-left (punctuation and numbers in place) next to
   an English one. APP_HEAD sets dir="auto" on the chat messages' blocks. */
.chat-log .message-content [dir="auto"] { text-align: start !important; }
/* The Stop button under each chat's message box, at its right edge. */
.chat-stop-row { justify-content: flex-end; }
.chat-input textarea {
  unicode-bidi: plaintext; text-align: start !important;
}
"""

# Enter sends the prompt in both chat tabs (Shift+Enter still adds a line):
# multi-line MultimodalTextboxes otherwise only insert a newline on Enter.
# Passed as `head=` next to APP_CSS; it also carries the SVG favicon.
APP_HEAD = brand.favicon_head() + """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Frank+Ruhl+Libre:wght@500;600&display=swap">
<script>
document.addEventListener("keydown", (e) => {
  if (e.key !== "Enter" || e.shiftKey || e.isComposing) return;
  const box = e.target.closest && e.target.closest(".chat-input");
  if (!box || e.target.tagName !== "TEXTAREA") return;
  const btn = box.querySelector("button.submit-button");
  if (!btn || btn.disabled) return;
  e.preventDefault(); e.stopPropagation();
  btn.click();
}, true);
// Chat messages: every block takes its direction from its own first strong
// character, so Hebrew/Arabic lines (and lists) read right-to-left while
// English ones stay left-to-right. Re-applied as answers stream in.
(() => {
  const BLOCKS = "p, li, ul, ol, h1, h2, h3, h4, h5, h6, blockquote, td, th, pre";
  const mark = (root) => {
    root.querySelectorAll(".chat-log .message-content").forEach((msg) => {
      if (msg.getAttribute("dir") !== "auto") msg.setAttribute("dir", "auto");
      msg.querySelectorAll(BLOCKS).forEach((el) => {
        if (el.getAttribute("dir") !== "auto") el.setAttribute("dir", "auto");
      });
    });
  };
  let queued = false;
  new MutationObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; mark(document); });
  }).observe(document.documentElement, { childList: true, subtree: true, characterData: true });
})();
</script>
"""

# The message boxes' own class stays on while they glow: dropping it would lose the branded send
# arrow and Enter-to-send after the first message.
_IDLE_CLASSES = ["chat-input"]
_WORKING_CLASSES = ["chat-input", "ai-working"]


def _glow_while_running(handler, outputs: list, box):
    """`handler`, a streaming send handler for `outputs`, with the message
    box `box` (one of those outputs) glowing from the moment Send is pressed
    until the handler finishes -- or fails, so a failed upload or request
    never leaves it glowing."""
    index = next(i for i, component in enumerate(outputs) if component is box)

    def with_box(values, working: bool) -> tuple:
        values = list(values)
        update = values[index] if isinstance(values[index], dict) else gr.update(value=values[index])
        values[index] = {**update, "elem_classes": _WORKING_CLASSES if working else _IDLE_CLASSES}
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


def _session_headers(request: gr.Request | None) -> dict:
    """Tags a job request with the page's session, so refreshing or closing the page cancels it
    (_reset_session below, api/routes_session.py)."""
    session = getattr(request, "session_hash", None)
    return {SESSION_HEADER: session} if session else {}


def _reset_session(request: gr.Request) -> None:
    """The page was refreshed or closed (Gradio's unload event): cancel every job it left running
    on the server, so the reloaded page starts fresh. Best effort -- a backend that's down has
    nothing running anyway."""
    session = getattr(request, "session_hash", None)
    if not session:
        return
    try:
        with httpx.Client(timeout=10) as client:
            client.post(f"{API_BASE_URL}/api/sessions/{session}/cancel")
    except httpx.HTTPError:
        pass


# The request each chat is running right now, per page (Gradio session) and tab: a backend job id,
# or the open stream to the Agentic RAG service. The tab's Stop button stops it (_stop_chat).
_RUNNING: dict[tuple[str, str], str | httpx.Response] = {}


def _running_key(request: gr.Request | None, tab: str) -> tuple[str, str] | None:
    session = getattr(request, "session_hash", None)
    return (session, tab) if session else None


def _start_job(endpoint: str, payload: dict, request: gr.Request | None, tab: str) -> str:
    """POSTs `payload` to /api/`endpoint`, which starts a job, and remembers its id as `tab`'s
    running request so Stop can cancel it."""
    with httpx.Client(timeout=60) as client:
        resp = client.post(f"{API_BASE_URL}/api/{endpoint}", json=payload, headers=_session_headers(request))
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
    if key := _running_key(request, tab):
        _RUNNING[key] = job_id
    return job_id


def _stop_chat(tab: str, request: gr.Request | None) -> bool:
    """The tab's running request, stopped: its backend job cancelled (the model stops generating),
    or the Agentic RAG stream closed (the service stops at the disconnect). Best effort; returns
    whether something was still running."""
    key = _running_key(request, tab)
    running = _RUNNING.pop(key, None) if key else None
    if isinstance(running, httpx.Response):
        was_open = not running.is_closed
        try:
            running.close()
        except Exception:  # noqa: BLE001 -- already closed
            pass
        return was_open
    if running:
        try:
            with httpx.Client(timeout=10) as client:
                return bool(client.post(f"{API_BASE_URL}/api/jobs/{running}/cancel").json().get("cancelled"))
        except (httpx.HTTPError, ValueError):
            pass
    return False


_STOPPED_NOTE = "_⏹ Stopped._"


def _mark_stopped(history: list) -> list:
    """The chat with the interrupted answer marked as stopped (or a stopped reply added when the
    model hadn't started answering)."""
    history = list(history or [])
    if history and history[-1].get("role") == "assistant":
        text = _message_text(history[-1].get("content")).rstrip()
        if text.startswith("_") and text.endswith("..._"):  # just the "_Working..._" placeholder
            text = ""
        history[-1] = {"role": "assistant", "content": f"{text}\n\n{_STOPPED_NOTE}".strip()}
    elif history:
        history.append({"role": "assistant", "content": _STOPPED_NOTE})
    return history


def _add_stop_button(tab: str, events: list, chatbot, box, status) -> gr.Button:
    """A Stop button for a chat tab: it stops `events` (the tab's send handlers) at once, cancels
    what they started on the server, marks the interrupted answer in `chatbot`, and returns the
    message box `box` to idle with `status` (the LLM status line) saying it was stopped."""
    stop = gr.Button("⏹ Stop", variant="stop", size="sm", scale=0, min_width=90, elem_classes=["chat-stop"])

    def on_stop(history: list, request: gr.Request):
        stopped = _stop_chat(tab, request)
        return (
            _mark_stopped(history) if stopped else gr.update(),
            gr.update(elem_classes=_IDLE_CLASSES),
            gr.update(value="⏹ Stopped" if stopped else "_Idle_"),
        )

    stop.click(fn=on_stop, inputs=chatbot, outputs=[chatbot, box, status], cancels=events, queue=False)
    return stop


def _sizes_selectable() -> bool:
    """The model-size selector switches models per request only under Ollama; a vLLM server serves
    a single model (llm/client.py's _sized)."""
    return get_config().llm.backend == "ollama" and len(get_config().model_sizes) > 1


def _cutoff_markdown(size: str | None) -> str:
    """The knowledge cutoff above the General tab's chat box: the picked size's model's, or Gemma 4's
    (the model vLLM serves)."""
    choice = get_config().model_sizes.get(size or "") if _sizes_selectable() else None
    return f"**LLM cutoff date:** {(choice and choice.knowledge_cutoff) or 'January 2025'}"


def build_model_size_selector() -> gr.Radio:
    """The LLM 12 Billion / LLM 31 Billion choice at the top of the page, shared by the General and
    Legal tabs: every request either tab sends runs on the picked size's models (config.model_sizes).
    Shows each size's label, never a model name; the value sent with requests is its key ("12B")."""
    cfg = get_config()
    sizes = list(cfg.model_sizes)
    default = cfg.default_model_size if cfg.default_model_size in sizes else (sizes[0] if sizes else None)
    selectable = _sizes_selectable()
    return gr.Radio(
        [(choice.label or key, key) for key, choice in cfg.model_sizes.items()],
        value=default,
        label="Model size",
        info="Used by General GPT, Legal GPT and Agentic RAG. The smaller model answers faster and needs less memory; "
        "the larger one is more capable." if selectable else "Fixed: the server runs a single model.",
        interactive=selectable,
        elem_id="model-size",
    )


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


_FILE_KINDS = {".pptx": "presentation", ".xlsx": "Excel workbook", ".pdf": "PDF"}


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
                            kind = _FILE_KINDS.get(Path(output_path).suffix.lower(), "file")
                            content_text = f"Your {kind} is ready: [Download {Path(output_path).name}]({download_url})"
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


def send_chat_message(message: dict, history: list, size: str | None = None, request: gr.Request = None):
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
    payload["model_size"] = size

    job_id = _start_job("chat", payload, request, "general")

    yield from _stream_job(job_id, history, rtl_hint=False)


def send_tone_rewrite(
    message: dict,
    professionalism: int,
    creativity: int,
    history: list,
    size: str | None = None,
    request: gr.Request = None,
):
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
        "model_size": size,
    }
    if attachment:
        payload["attachment_path"] = attachment["path"]

    job_id = _start_job("tone-rewrite", payload, request, "general")

    yield from _stream_job(job_id, history, rtl_hint=False)


def build_chat_tab(model_size_selector: gr.Radio) -> None:
    # Gemma 4's training-data cutoff, so users know how current its knowledge is.
    cutoff = gr.Markdown(_cutoff_markdown(model_size_selector.value), elem_classes=["llm-cutoff"])
    model_size_selector.change(fn=_cutoff_markdown, inputs=model_size_selector, outputs=cutoff)
    chatbot = gr.Chatbot(label="Chat", height="75vh", min_height=560,
                         elem_classes=["chat-log", "llm-status-general"])
    llm_status = gr.Markdown(
        value="_Idle_", label="LLM status", show_label=True, container=True, visible=False
    )
    reasoning_panel = gr.Textbox(label="Reasoning (model's thinking)", lines=6, visible=False)

    msg_box = gr.MultimodalTextbox(
        label="Message",
        # Same size as the Legal tab's box, so the send arrow looks the same; it grows as text comes in.
        max_lines=24,
        placeholder='Ask a question, paste a large block of text to rewrite/translate/summarize, or ask for '
        'a file -- "make this into a presentation", "create an Excel budget for...", "write a PDF report on..." '
        '-- attach a document (PDF, DOCX, PPTX, XLSX, image, or .txt) instead with the 📎 button',
        # .txt matters here beyond ordinary file attachments: pasting a
        # large enough block of text makes the browser/Gradio turn the
        # paste itself into a text/plain file attachment instead of
        # inline text (see ingestion/parser.py's plain-text fast path) --
        # without .txt allowed, every large paste was rejected outright
        # with "Invalid file type: text/plain".
        file_types=[".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".tiff", ".txt"],
        file_count="single",
        sources=["upload"],
        elem_id="general-msg",
        elem_classes=_IDLE_CLASSES,
    )
    stop_row = gr.Row(elem_classes=["chat-stop-row"])  # its Stop button is added with the send events below

    with gr.Accordion("Tone control", open=False):
        gr.Markdown(
            "Rewrites whatever is in the message box above -- typed/pasted text, or "
            "an attached document (📎) -- with the tone below, instead of answering it "
            "normally. Professionalism drives wording/register (system prompt); "
            "Creativity drives sampling randomness only. Use this button instead of the send arrow."
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
    send_event = msg_box.submit(fn=send, inputs=[msg_box, chatbot, model_size_selector], outputs=chat_outputs)
    rewrite_event = rewrite_btn.click(
        fn=_glow_while_running(send_tone_rewrite, chat_outputs, msg_box),
        inputs=[msg_box, professionalism_slider, creativity_slider, chatbot, model_size_selector],
        outputs=chat_outputs,
    )
    with stop_row:
        _add_stop_button("general", [send_event, rewrite_event], chatbot, msg_box, llm_status)


# ---------------------------------------------------------------------------
# Legal tab -- grounded RAG over Israeli law (see api/routes_legal.py,
# legal/pipeline.py): Gemma 4 plans the search, retrieves from the bulk legal
# corpus (legal/corpus_retrieval.py; legal.retrieval.source), researches,
# drafts and verifies against what it retrieved, and answers in the
# question's language.
# Citations appear as [n]
# markers in the answer and as footnotes (with verification status) in the
# side panel; the Pass A research memorandum is viewable below them.
# ---------------------------------------------------------------------------

_LEGAL_DISCLAIMER = "_Draft pending review by a licensed attorney -- not legal advice._"


def _format_legal_footnotes(footnotes: list[dict]) -> str:
    if not footnotes:
        return "_No citations yet._"
    entries = []
    for note in footnotes:
        if note.get("kind") == "case_file":  # Case files chat: a passage of the attorney's own files
            details = " · ".join(p for p in (note.get("doc_type"), note.get("matter"), note.get("doc_date")) if p)
            quote = " ".join((note.get("excerpt") or "").split())
            entries.append(f"**[{note['number']}]** {note.get('where', '')}  \n_{details}_  \n> {quote}")
            continue
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
    the side panel) and "legal_report" (memorandum, escalation), which
    arrives after the answer text and wraps it with the escalation banner,
    coverage gaps and disclaimer. "reasoning_delta" carries Pass 0's
    chain-of-thought as one chunk (the pipeline runs each pass as a single
    call rather than token-streaming it), shown the same reasoning panel as
    the general chat tab."""
    content_text = ""
    reasoning_text = ""
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
                    elif event_kind == "reasoning_delta":
                        reasoning_text += data["text"]
                        llm_status = "🧠 Thinking..."
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
                    elif event_kind == "error":
                        content_text += f"\n\n⚠️ {data.get('message')}"
                        started_streaming = True
                        llm_status = "❌ Error"

                    lang = (report or {}).get("reply_language") or (
                        detect_language(content_text[:200]) if started_streaming else None
                    )
                    display_text = _legal_bubble(content_text, report) if started_streaming else f"_{status_text}..._"
                    new_history = history + [{"role": "assistant", "content": display_text}]
                    # The notes are written in the question's language: a Hebrew one laid out left to
                    # right puts every line's punctuation and numbering at the wrong end.
                    reasoning_lang = lang or (detect_language(reasoning_text[:200]) if reasoning_text else None)
                    yield (
                        gr.update(value=new_history, rtl=_is_rtl_lang(lang)),
                        gr.update(value=reasoning_text, visible=bool(reasoning_text),
                                  rtl=_is_rtl_lang(reasoning_lang)),
                        gr.update(value=None),
                        gr.update(value=citations_md),
                        gr.update(value=llm_status),
                        gr.update(value=memo),
                    )
                    if event_kind in ("done", "error"):
                        break


LEGAL_MODE_QUESTION = "Question"
LEGAL_MODE_CASE = "Case analysis"
LEGAL_MODE_FILES = "Case files chat"
_FILES_MARKER = f"💬 **{LEGAL_MODE_FILES}**"
_ALL_MATTERS = "All matters"


_QUESTION_PLACEHOLDER = (
    "Ask a question about Israeli law in any language -- answered only from the Israeli laws and procedural "
    "regulations in the legal corpus, in your language -- or attach a document (PDF, DOCX, PPTX, XLSX, image, "
    "or .txt) with the 📎 button to have it reviewed against the law"
)
_FILES_PLACEHOLDER = (
    "Ask about the case files in legal_data/ -- facts, dates, who said what, a chronology -- in any language. "
    "Pick a matter to search only its files; follow-up questions keep the conversation."
)


def _case_file_matters() -> list[str]:
    """The matters indexed from legal_data/ (api/routes_legal.py), or [] when the API can't say."""
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{API_BASE_URL}/api/case-files/matters")
            resp.raise_for_status()
            return resp.json().get("matters", [])
    except Exception:  # noqa: BLE001 -- the dropdown still offers "All matters"
        return []


def _case_files_history(history: list) -> list[dict]:
    """The earlier Case files chat turns in the Legal chat, as user/assistant messages: the question
    without its mode line, the answer without the banner, notes and disclaimer the bubble adds."""
    messages: list[dict] = []
    for question, reply in zip(history, history[1:]):
        q_text = _message_text(question.get("content"))
        if question.get("role") != "user" or reply.get("role") != "assistant" or not q_text.startswith(_FILES_MARKER):
            continue
        q_text = q_text.split("\n\n", 1)[1] if "\n\n" in q_text else ""
        parts = [part for part in _message_text(reply.get("content")).split("\n\n")
                 if part.strip() and not part.startswith(("_", "⚠️"))]
        if q_text.strip() and parts:
            messages += [{"role": "user", "content": q_text.strip()},
                         {"role": "assistant", "content": "\n\n".join(parts)}]
    return messages[-6:]


def index_case_files(matter: str | None):
    """The 🔄 button in Case files chat mode: indexes new and changed files in legal_data/."""
    gr.Info("Indexing legal_data/ -- new and changed files only. This can take a few minutes.")
    try:
        with httpx.Client(timeout=None) as client:
            resp = client.post(f"{API_BASE_URL}/api/case-files/index")
            if resp.status_code == 404:
                raise gr.Error(resp.json().get("detail", "legal_data/ not found"))
            resp.raise_for_status()
            summary = resp.json()
    except httpx.HTTPError as exc:
        raise gr.Error(f"Indexing failed: {exc}") from exc
    counts = summary.get("counts", {})
    gr.Info("Case files: " + (", ".join(f"{n} {action}" for action, n in sorted(counts.items())) or "no files"))
    for failed in [f for f in summary.get("files", []) if f["action"] in ("failed", "skipped")]:
        gr.Warning(f"{failed['file']}: {failed['action']} -- {failed['message']}")
    matters = summary.get("matters", [])
    return gr.update(choices=[_ALL_MATTERS, *matters], value=matter if matter in matters else _ALL_MATTERS)


def send_legal_message(
    message: dict, mode: str, history: list, size: str | None = None, matter: str | None = None,
    request: gr.Request = None,
):
    """`message` is a `gr.MultimodalTextbox` payload: `{"text": str, "files":
    [local_path, ...]}` -- an attached document (contract, filing, any file
    the chat tab accepts) is uploaded the same way as there. In question mode
    its extracted text is read alongside the question, applying the retrieved
    law to it rather than treating it as a citable source; in case mode the
    typed text and the document together are the case file, and the answer is
    a work file ending in a recommended next step."""
    text = (message.get("text") or "").strip()
    files = message.get("files") or []
    if not text and not files:
        yield history, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return

    if mode == LEGAL_MODE_FILES:
        yield from _send_case_files_question(text, history, size, matter, request)
        return

    attachment = _upload_file(files[0]) if files else None
    case_mode = mode == LEGAL_MODE_CASE

    display_message = text
    if attachment:
        display_message = f"📎 {attachment['name']}\n\n{text}" if text else f"📎 {attachment['name']}"
    if case_mode:
        display_message = f"🗂️ **{LEGAL_MODE_CASE}**\n\n{display_message}"
    history = history + [{"role": "user", "content": display_message}]

    if case_mode:
        content = text  # empty is fine: the attached documents are then the whole case file
    else:
        content = text or f"(No question -- I just attached {attachment['name'] if attachment else 'a file'}. Review it against the applicable Israeli law.)"
    payload: dict = {"messages": [{"role": "user", "content": content}], "model_size": size}
    if attachment:
        payload["attachment_path"] = attachment["path"]

    endpoint = "legal-case" if case_mode else "legal-chat"
    job_id = _start_job(endpoint, payload, request, "legal")

    yield from _stream_legal_job(job_id, history)


def _send_case_files_question(text: str, history: list, size: str | None, matter: str | None,
                              request: gr.Request | None):
    """Case files chat mode: the question is answered from the attorney's indexed case files
    (legal_data/, legal/case_files.py), within the picked matter, following up on this mode's
    earlier turns in the chat. Attachments aren't used in this mode."""
    if not text:
        raise gr.Error("Type a question about the case files.")
    matter = None if matter in (None, "", _ALL_MATTERS) else matter
    earlier = _case_files_history(history)
    scope = f" -- {matter}" if matter else ""
    history = history + [{"role": "user", "content": f"{_FILES_MARKER}{scope}\n\n{text}"}]
    payload = {"messages": [*earlier, {"role": "user", "content": text}], "model_size": size, "matter": matter}
    job_id = _start_job("legal-case-files", payload, request, "legal")
    yield from _stream_legal_job(job_id, history)


def _message_text(content) -> str:
    """A chat message's text, whether Gradio holds it as a string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or "")
    if isinstance(content, (list, tuple)):
        return "\n".join(_message_text(part) for part in content)
    return ""


def _add_markdown_runs(paragraph, text: str) -> None:
    """Adds `text` to `paragraph`, turning Markdown **bold** into bold runs."""
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if part:
            paragraph.add_run(part).bold = i % 2 == 1


def _add_markdown_to_docx(doc, text: str) -> None:
    """Appends Markdown `text` to the Word document `doc`: headings, bullet/numbered lists and bold
    are kept; right-to-left text (Hebrew, Arabic) is laid out right-to-left."""
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn

    rtl = _is_rtl_lang(detect_language(text[:500]))
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if heading := re.match(r"^(#{1,6})\s+(.*)$", stripped):
            paragraph = doc.add_heading(level=min(len(heading.group(1)), 4))
            _add_markdown_runs(paragraph, heading.group(2).strip())
        elif bullet := re.match(r"^[-*•]\s+(.*)$", stripped):
            paragraph = doc.add_paragraph(style="List Bullet")
            _add_markdown_runs(paragraph, bullet.group(1))
        elif numbered := re.match(r"^\d+[.)]\s+(.*)$", stripped):
            paragraph = doc.add_paragraph(style="List Number")
            _add_markdown_runs(paragraph, numbered.group(1))
        else:
            paragraph = doc.add_paragraph()
            _add_markdown_runs(paragraph, stripped)
        if rtl:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            paragraph._p.get_or_add_pPr().append(paragraph._p.makeelement(qn("w:bidi"), {}))
            for run in paragraph.runs:
                run._r.get_or_add_rPr().append(run._r.makeelement(qn("w:rtl"), {}))


def save_case_analysis_docx(history: list):
    """Saves the latest assistant reply in the Legal tab (the case analysis work file) as a Word
    document and returns it for download."""
    from docx import Document

    reply = next(
        (_message_text(m.get("content")) for m in reversed(history or []) if m.get("role") == "assistant"), ""
    ).strip()
    if not reply:
        raise gr.Error("There is no analysis to save yet.")

    doc = Document()
    _add_markdown_to_docx(doc, reply)
    path = Path(tempfile.mkdtemp()) / f"case_analysis_{datetime.now():%Y%m%d_%H%M%S}.docx"
    doc.save(path)
    return gr.update(value=str(path), visible=True)


# The file types a case folder's documents can be -- the same ones the 📎 button accepts (api/routes_upload.py).
_CASE_FILE_TYPES = (".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".tiff", ".txt")
# Each document gets a share of the case file's token budget (legal/pipeline._case_documents_material):
# past this many, the shares get too small to say anything.
_CASE_FOLDER_MAX_DOCUMENTS = 25


def _case_folder_documents(files: list) -> tuple[list[str], list[str]]:
    """The case documents among a picked folder's files, sorted by name, and the names of the files
    skipped (unsupported types; hidden and system files like .DS_Store aren't even mentioned)."""
    paths = sorted((str(getattr(f, "name", f)) for f in files or []), key=lambda p: Path(p).name.lower())
    visible = [p for p in paths if not Path(p).name.startswith((".", "~$")) and Path(p).name != "Thumbs.db"]
    documents = [p for p in visible if Path(p).suffix.lower() in _CASE_FILE_TYPES]
    skipped = [Path(p).name for p in visible if Path(p).suffix.lower() not in _CASE_FILE_TYPES]
    return documents, skipped


def send_case_folder(
    files: list, message: dict | None, history: list, size: str | None = None, request: gr.Request = None
):
    """A case folder picked with the 📂 button next to the Mode selector (case mode only): every
    supported document in it is uploaded and the case is analyzed at once -- the backend reads each
    document, digests them one by one when they don't fit the case file together, and writes the
    work file from them. Whatever is typed in the message box goes along as the case description."""
    documents, skipped = _case_folder_documents(files)
    if not documents:
        raise gr.Error("That folder has no case documents (PDF, DOCX, PPTX, XLSX, image or .txt files).")
    if len(documents) > _CASE_FOLDER_MAX_DOCUMENTS:
        gr.Warning(
            f"The folder has {len(documents)} documents; only the first {_CASE_FOLDER_MAX_DOCUMENTS} "
            "(by name) are analyzed."
        )
        skipped += [Path(p).name for p in documents[_CASE_FOLDER_MAX_DOCUMENTS:]]
        documents = documents[:_CASE_FOLDER_MAX_DOCUMENTS]

    text = ((message or {}).get("text") or "").strip()
    listing = "\n".join(f"- {Path(p).name}" for p in documents)
    display_message = f"🗂️ **{LEGAL_MODE_CASE}** -- 📂 case folder, {len(documents)} document(s):\n{listing}"
    if skipped:
        display_message += f"\n\n_Not analyzed: {', '.join(skipped)}_"
    if text:
        display_message += f"\n\n{text}"
    history = history + [{"role": "user", "content": display_message}]

    uploaded = [_upload_file(p) for p in documents]
    payload = {
        "messages": [{"role": "user", "content": text}],
        "documents": [{"path": u["path"], "name": u["name"] or Path(p).name} for u, p in zip(uploaded, documents)],
        "model_size": size,
    }
    job_id = _start_job("legal-case", payload, request, "legal")

    yield from _stream_legal_job(job_id, history)


def _legal_sources_summary() -> str:
    """The one-line summary above the Legal tab's chat history: when the sources were last updated and
    how many laws and judgments the answers can draw on."""
    corpus = corpus_stats()
    caselaw = caselaw_stats()
    records = (corpus or {}).get("records_by_category", {})
    dates = [d[:10] for d in ((corpus or {}).get("built_at"), (caselaw or {}).get("built_at")) if d]
    # Judgments indexed into the corpus itself (vectorize.py's supreme_court category) count too
    # when the separate case-law index isn't installed.
    judgments = caselaw["judgments"] if caselaw else records.get("supreme_court")

    def count(value) -> str:
        return f"{value:,}" if value else "not installed"

    return (
        f"Legal sources -- last update: {max(dates) if dates else 'unknown'}"
        f" · laws: {count(records.get('laws'))}"
        f" · procedural regulations: {count(records.get('procedural_rules'))}"
        f" · case law (Supreme Court): {count(judgments)}"
        f" · corpus: {_corpus_summary(corpus)}"
    )


def _corpus_summary(stats: dict | None) -> str:
    if not stats:
        return "not installed -- see scripts/legal_data/install_corpus.py"
    per_category = " · ".join(f"{name} {count:,}" for name, count in stats["categories"].items())
    return (
        f"{stats['chunks']:,} chunks ({per_category}) from {stats['records']:,} source documents"
        f" · last pulled {(stats['built_at'] or 'unknown')[:10]}"
    )


def build_legal_tab(model_size_selector: gr.Radio) -> None:
    legal = get_config().legal
    source = (
        f"legal corpus ({', '.join(legal.corpus.categories)})"
        if legal.retrieval.source == "corpus"
        else "signed index"
    )
    gr.Markdown(
        "_Research, drafting & verification: **LLM**"
        f" · sources: **{source}**_"
    )

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(_legal_sources_summary(), elem_classes=["legal-sources"])
            legal_chatbot = gr.Chatbot(label="Legal Assistant", height="75vh", min_height=560,
                                     elem_classes=["chat-log", "llm-status-legal"])
            legal_llm_status = gr.Markdown(
                value="_Idle_", label="LLM status", show_label=True, container=True, visible=False
            )
            legal_reasoning_panel = gr.Textbox(label="Reasoning (model's thinking)", lines=6, visible=False)
            with gr.Row(equal_height=True):
                legal_mode = gr.Radio(
                    [LEGAL_MODE_QUESTION, LEGAL_MODE_CASE, LEGAL_MODE_FILES],
                    value=LEGAL_MODE_QUESTION,
                    label="Mode",
                    info="Question: a grounded answer with citations. Case analysis: describe the case and/or "
                    "pick its folder with 📂 (or attach a document) -- you get a work file (facts, chronology, "
                    "legal issues, deadlines, red flags, missing information, a draft document and the "
                    "recommended next step). Case files chat: ask about the case files indexed from "
                    "legal_data/ -- answers cite file, page and paragraph.",
                    scale=4,
                )
                # Case mode only: picks the folder holding the case's files; the analysis starts as soon
                # as it's picked.
                case_folder_button = gr.UploadButton(
                    "📂 Browse case folder",
                    file_count="directory",
                    visible=False,
                    scale=1,
                    min_width=160,
                    elem_id="case-folder",
                )
                # Case files chat only: which matter (legal_data/ subfolder) to ask about, and a button
                # that indexes files added to legal_data/ since the last run.
                matter_dropdown = gr.Dropdown(
                    [_ALL_MATTERS], value=_ALL_MATTERS, label="Matter", visible=False, scale=2, min_width=180,
                    allow_custom_value=False,
                )
                index_button = gr.Button("🔄 Index legal_data", visible=False, scale=1, min_width=150)
            legal_msg_box = gr.MultimodalTextbox(
                label="Legal question",
                placeholder=_QUESTION_PLACEHOLDER,
                file_types=[".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".tiff", ".txt"],
                file_count="single",
                sources=["upload"],
                elem_id="legal-msg",
                elem_classes=_IDLE_CLASSES,
            )
            legal_stop_row = gr.Row(elem_classes=["chat-stop-row"])
        with gr.Column(scale=1):
            save_docx_button = gr.Button("💾 Save analysis as Word", variant="secondary")
            docx_file = gr.File(label="Word document", visible=False, interactive=False)
            gr.Markdown("### Citations")
            citations_panel = gr.Markdown(value="_No citations yet._")
            with gr.Accordion("Research memorandum (Pass A)", open=False):
                memo_view = gr.JSON(value=None, label="Claims → evidence")

    send_inputs = [legal_msg_box, legal_mode, legal_chatbot, model_size_selector, matter_dropdown]
    send_outputs = [legal_chatbot, legal_reasoning_panel, legal_msg_box, citations_panel, legal_llm_status, memo_view]
    send = _glow_while_running(send_legal_message, send_outputs, legal_msg_box)
    send_event = legal_msg_box.submit(fn=send, inputs=send_inputs, outputs=send_outputs)
    def on_mode(mode: str):
        files = mode == LEGAL_MODE_FILES
        matters = gr.update(visible=True, choices=[_ALL_MATTERS, *_case_file_matters()]) if files else gr.update(visible=False)
        box = gr.update(placeholder=_FILES_PLACEHOLDER if files else _QUESTION_PLACEHOLDER)
        return gr.update(visible=mode == LEGAL_MODE_CASE), matters, gr.update(visible=files), box

    legal_mode.change(fn=on_mode, inputs=legal_mode,
                      outputs=[case_folder_button, matter_dropdown, index_button, legal_msg_box])
    index_button.click(fn=index_case_files, inputs=matter_dropdown, outputs=matter_dropdown)
    folder_event = case_folder_button.upload(
        fn=_glow_while_running(send_case_folder, send_outputs, legal_msg_box),
        inputs=[case_folder_button, legal_msg_box, legal_chatbot, model_size_selector],
        outputs=send_outputs,
    )
    with legal_stop_row:
        _add_stop_button("legal", [send_event, folder_event], legal_chatbot, legal_msg_box, legal_llm_status)
    save_docx_button.click(fn=save_case_analysis_docx, inputs=legal_chatbot, outputs=docx_file)


# ---------------------------------------------------------------------------
# Agentic RAG tab -- chat with the LPJ projects archive (ACN pastoral projects).
# The index and its retrieval live in a separate project (RAG_LPJ/pipeline,
# package lpj_rag, its own venv) served on this machine by
# `python -m lpj_rag serve`; this tab only talks to that service over
# loopback. Answers come from the local Ollama model picked in the size selector.
# ---------------------------------------------------------------------------

LPJ_RAG_URL = os.environ.get("LPJ_RAG_URL", "http://127.0.0.1:8765")
_LPJ_START_HINT = (
    "Start it in a terminal: `cd C:\\Users\\iB\\Projects\\RAG_LPJ\\pipeline` then "
    "`.venv\\Scripts\\python -m lpj_rag serve`"
)


def _lpj_status() -> str:
    try:
        with httpx.Client(timeout=5) as client:
            health = client.get(f"{LPJ_RAG_URL}/health").json()
    except (httpx.HTTPError, ValueError):
        return f"⚠️ The Agentic RAG service is not running at {LPJ_RAG_URL}. {_LPJ_START_HINT}"
    if not health.get("ok"):
        return f"⚠️ {health.get('error', 'The LPJ index is not ready.')}"
    vectors = "" if health.get("vectors") else " · ⚠️ keyword search only (vectors not built)"
    return (f"LPJ project files -- index built {health['built_at'][:16].replace('T', ' ')} UTC"
            f" · projects: {health['projects']:,} · documents: {health['documents']:,}"
            f" · chunks: {health['chunks']:,}{vectors}")


_LPJ_NO_SOURCES = "<p><em>No sources yet.</em></p>"


def _lpj_file_url(source: dict) -> str:
    """The app's relay for a cited file (api/routes_lpj.py); PDFs open at the cited page."""
    url = "/api/lpj-file?path=" + quote(source["path"], safe="")
    return url + (f"#page={source['page']}" if source["path"].lower().endswith(".pdf") and source.get("page") else "")


def _format_lpj_sources(sources: list[dict]) -> str:
    """Each source as a link that opens the original file in a new browser tab. PDFs and images
    open in the browser; Word and Excel files download."""
    if not sources:
        return _LPJ_NO_SOURCES
    items = []
    for s in sources:
        name = s["path"].rsplit("/", 1)[-1]
        folder = s["path"].rsplit("/", 1)[0] if "/" in s["path"] else ""
        items.append(
            f'<li><b>[{escape(s["tag"])}]</b> <a href="{escape(_lpj_file_url(s))}" target="_blank" '
            f'rel="noopener">{escape(name)}</a> -- p.{s["page"]} · {escape(s["doc_type"].replace("_", " "))}'
            f'<br><small>{escape(folder)}</small></li>'
        )
    return "<ul class='lpj-sources'>" + "".join(items) + "</ul>"


def _link_lpj_citations(answer: str, sources: list[dict]) -> str:
    """[S3] in the answer -> a link to that source's file."""
    urls = {s["tag"]: _lpj_file_url(s) for s in sources}
    return re.sub(r"\[(S\d+)\]", lambda m: f"[[{m[1]}]]({urls[m[1]]})" if m[1] in urls else m[0], answer)


def send_lpj_message(text: str, history: list, size: str | None = None, request: gr.Request = None):
    """Streams the service's events (lpj_rag/serve.py: status, sources, delta, done, error) into the
    chat. Outputs: chatbot, message box, sources panel, status line."""
    text = (text or "").strip()
    if not text:
        yield history, gr.update(), gr.update(), gr.update()
        return
    choice = get_config().model_sizes.get(size or "") if _sizes_selectable() else None
    history = history + [{"role": "user", "content": text}, {"role": "assistant", "content": ""}]
    answer, sources_md, sources = "", _LPJ_NO_SOURCES, []
    yield history, "", sources_md, "🔌 Connecting..."
    try:
        with httpx.Client(timeout=httpx.Timeout(10, read=None)) as client:
            with client.stream("POST", f"{LPJ_RAG_URL}/ask",
                               json={"question": text, "model": choice.model if choice else None}) as resp:
                resp.raise_for_status()
                if key := _running_key(request, "lpj"):
                    _RUNNING[key] = resp  # Stop closes it
                for line in resp.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    kind = event.get("event")
                    if kind == "status":
                        yield history, gr.update(), sources_md, f"⏳ {event['message']}..."
                    elif kind == "sources":
                        sources = event["sources"]
                        sources_md = _format_lpj_sources(sources)
                        yield history, gr.update(), sources_md, gr.update()
                    elif kind == "delta":
                        answer += event["text"]
                        history[-1] = {"role": "assistant", "content": answer}
                        yield history, gr.update(), sources_md, "✍️ Writing the answer..."
                    elif kind == "error":
                        history[-1] = {"role": "assistant", "content": f"{answer}\n\n⚠️ {event['message']}".strip()}
                        yield history, gr.update(), sources_md, "⚠️ Failed"
                        return
    except httpx.ConnectError:
        history[-1] = {"role": "assistant",
                       "content": f"⚠️ The Agentic RAG service is not running at {LPJ_RAG_URL}. {_LPJ_START_HINT}"}
        yield history, gr.update(), sources_md, "⚠️ Service not running"
        return
    if answer:
        history[-1] = {"role": "assistant", "content": _link_lpj_citations(answer, sources)}
    yield history, gr.update(), sources_md, "_Idle_"


def _chat_turns(history: list) -> list[tuple[str, str]]:
    """(role, text) for each non-empty message, with citation links ([[S3]](/api/...)) back to [S3]."""
    turns = []
    for m in history or []:
        text = re.sub(r"\[\[(S\d+)\]\]\([^)]*\)", r"[\1]", _message_text(m.get("content"))).strip()
        if text and m.get("role") in ("user", "assistant"):
            turns.append((m["role"], text))
    return turns


def save_rag_chat_docx(history: list):
    """The whole Agentic RAG conversation as a Word document: each question as a heading, each
    answer under it (Markdown kept, right-to-left text laid out right-to-left)."""
    from docx import Document

    turns = _chat_turns(history)
    if not turns:
        raise gr.Error("There is no conversation to save yet.")
    doc = Document()
    doc.add_heading("Agentic RAG conversation", level=0)
    doc.add_paragraph(f"{datetime.now():%Y-%m-%d %H:%M}")
    for role, text in turns:
        if role == "user":
            _add_markdown_to_docx(doc, "## " + " ".join(text.split()))
        else:
            _add_markdown_to_docx(doc, text)
    path = Path(tempfile.mkdtemp()) / f"agentic_rag_chat_{datetime.now():%Y%m%d_%H%M%S}.docx"
    doc.save(path)
    return gr.update(value=str(path), visible=True)


def _plain(text: str) -> str:
    """Markdown inline marks and list/heading prefixes stripped, for a slide bullet."""
    text = re.sub(r"^\s*(#{1,6}|[-*•]|\d+[.)])\s+", "", text)
    return re.sub(r"\*\*(.+?)\*\*|__(.+?)__|`([^`]*)`", lambda m: next(g for g in m.groups() if g is not None), text).strip()


def _fallback_deck(turns: list[tuple[str, str]]):
    """One slide per question, its answer's lines as bullets -- used when the model can't plan a deck."""
    from docslides.llm.schemas import SlideContent

    slides, question = [], ""
    for role, text in turns:
        if role == "user":
            question = " ".join(text.split())
            continue
        bullets = [_plain(line) for line in text.splitlines() if _plain(line)][:8]
        slides.append(SlideContent(title=_plain(question)[:90] or "Answer", bullets=bullets,
                                   notes=text, layout_type="title_bullets"))
    return slides


async def make_rag_chat_pptx(history: list, size: str | None = None):
    """A PowerPoint deck built from the Agentic RAG conversation: the model (the size picked in the
    selector) turns the questions and answers into a titled deck in the chat's language; if it
    can't, each answer becomes a slide of its own. Built with the app's slide template and RTL
    handling (slides/pptx_builder.py)."""
    from pydantic import BaseModel

    from docslides.llm.client import ChatMessage, LLMCallSite, get_client, model_size
    from docslides.llm.schemas import SlideContent
    from docslides.slides.pptx_builder import build_pptx

    class ChatDeck(BaseModel):
        title: str
        slides: list[SlideContent]

    turns = _chat_turns(history)
    if not any(role == "assistant" for role, _ in turns):
        raise gr.Error("There is no answer to turn into slides yet.")
    transcript = "\n\n".join(f"{'QUESTION' if r == 'user' else 'ANSWER'}:\n{t}" for r, t in turns)
    lang = detect_language(" ".join(t for r, t in turns if r == "assistant")[:1000]) or "en"
    system = (
        "Turn this question-and-answer conversation about project files into a concise presentation. "
        "Return a deck title and 3 to 12 slides. Each slide: a short title, 2 to 6 short bullets that "
        "state facts from the answers only (keep numbers, amounts, dates, project and ACN numbers and "
        "source tags like [S3] exactly as written; invent nothing), speaker notes with any detail that "
        "did not fit, and layout_type 'title_bullets' (or 'section_header' for a section divider with "
        "no bullets). Write in the same language as the answers."
    )
    deck = None
    try:
        with model_size(size if _sizes_selectable() else None):
            deck = await get_client().complete_json(
                messages=[ChatMessage(role="system", content=system), ChatMessage(role="user", content=transcript)],
                call_site=LLMCallSite("slide_fill"),
                schema=ChatDeck,
            )
    except Exception as exc:  # the deck still gets made, from the answers as written
        gr.Warning(f"The model couldn't plan the slides ({type(exc).__name__}); one slide per answer instead.")
    if deck and deck.slides:
        title, slides = deck.title, deck.slides
    else:
        title, slides = "Agentic RAG", _fallback_deck(turns)
    cover = SlideContent(title=title, bullets=[f"{datetime.now():%Y-%m-%d}"], layout_type="section_header")
    path = Path(tempfile.mkdtemp()) / f"agentic_rag_{datetime.now():%Y%m%d_%H%M%S}.pptx"
    # Right-to-left is decided per slide, from its own text (a chat can mix English and Arabic).
    build_pptx([(s, detect_language(" ".join([s.title, *s.bullets])[:500]) or lang) for s in [cover, *slides]], path)
    return gr.update(value=str(path), visible=True)


def build_lpj_tab(tab: gr.Tab, model_size_selector: gr.Radio) -> None:
    gr.Markdown(
        "_Ask about the LPJ project files: approvals, budgets, invoices, reports -- by project, ACN number "
        "(YY/NNNNN) or GA number (GA NNN YY). Answers cite the file and page; everything runs on this machine._"
    )
    with gr.Row():
        with gr.Column(scale=3):
            status_line = gr.Markdown("_Checking the LPJ index..._", elem_classes=["legal-sources"])
            chatbot = gr.Chatbot(label="Agentic RAG", height="75vh", min_height=560,
                                 elem_classes=["chat-log", "llm-status-general"])
            llm_status = gr.Markdown(value="_Idle_", label="LLM status", show_label=True, container=True)
            msg_box = gr.Textbox(
                label="Question",
                placeholder="e.g. How much did ACN approve for project 23/00123? · "
                "Which projects in Jordan are still awaiting a final report? · كم بلغت ميزانية المخيمات الصيفية؟",
                lines=2,
                max_lines=12,
                submit_btn=True,
                elem_id="lpj-msg",
                elem_classes=_IDLE_CLASSES,
            )
            lpj_stop_row = gr.Row(elem_classes=["chat-stop-row"])
        with gr.Column(scale=1):
            refresh = gr.Button("🔄 Check index", variant="secondary")
            save_docx_button = gr.Button("💾 Save chat as Word", variant="secondary")
            docx_file = gr.File(label="Word document", visible=False, interactive=False)
            pptx_button = gr.Button("📊 Make PowerPoint from chat", variant="secondary")
            pptx_file = gr.File(label="PowerPoint deck", visible=False, interactive=False)
            gr.Markdown("### Sources\n_Click a file to open it (PDFs open at the cited page)._")
            sources_panel = gr.HTML(value=_LPJ_NO_SOURCES)

    save_docx_button.click(fn=save_rag_chat_docx, inputs=chatbot, outputs=docx_file)
    pptx_button.click(fn=make_rag_chat_pptx, inputs=[chatbot, model_size_selector], outputs=pptx_file)
    outputs = [chatbot, msg_box, sources_panel, llm_status]
    send_event = msg_box.submit(fn=_glow_while_running(send_lpj_message, outputs, msg_box),
                                inputs=[msg_box, chatbot, model_size_selector], outputs=outputs)
    with lpj_stop_row:
        _add_stop_button("lpj", [send_event], chatbot, msg_box, llm_status)
    tab.select(fn=_lpj_status, outputs=status_line)
    refresh.click(fn=_lpj_status, outputs=status_line)


def build_status_tab(tab: gr.Tab) -> None:
    """Health of every framework and model the app uses (system_status.py). Nothing runs until the
    tab is opened: each visit re-checks, and the buttons re-check on demand."""
    gr.Markdown(
        "_LLM servers and models, LibreOffice, MinerU, PDF and OCR engines, GPU, retrieval and language "
        "models, legal data. Checked each time this tab is opened. A model that's downloaded but not in "
        "memory is only asked to answer by **Test every model** (that loads it -- it can take minutes, "
        "and may push another model out of memory)._"
    )
    with gr.Row():
        refresh = gr.Button("Refresh", variant="secondary", scale=0)
        test_all = gr.Button("Test every model", variant="secondary", scale=0)
    report = gr.Markdown("_Open this tab to run the checks._")

    tab.select(fn=lambda: status_report(False), outputs=report, show_progress="full")
    refresh.click(fn=lambda: status_report(False), outputs=report, show_progress="full")
    test_all.click(fn=lambda: status_report(True), outputs=report, show_progress="full")


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Codex Sentinel - By Ibrahim Zananiri") as demo:
        gr.HTML(brand.header_html("Codex Sentinel - By Ibrahim Zananiri"))
        model_size_selector = build_model_size_selector()
        with gr.Tabs():
            with gr.Tab("General GPT"):
                build_chat_tab(model_size_selector)
            with gr.Tab("Legal GPT"):
                build_legal_tab(model_size_selector)
            with gr.Tab("Agentic RAG") as lpj_tab:
                build_lpj_tab(lpj_tab, model_size_selector)
            with gr.Tab("System status") as status_tab:
                build_status_tab(status_tab)
        # Refreshing or closing the page stops whatever it left running on the server.
        demo.unload(_reset_session)
    return demo


def run() -> None:
    demo = build_app()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=APP_THEME,
        css=APP_CSS,
        head=APP_HEAD,
        favicon_path=brand.favicon_path(),
    )


if __name__ == "__main__":
    run()
