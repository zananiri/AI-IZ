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

from docslides.api.events import SESSION_HEADER
from docslides.config import get_config
from docslides.ingestion.language_detect import detect_language
from docslides.legal.caselaw import caselaw_stats
from docslides.legal.corpus_retrieval import corpus_stats
from docslides.system_status import status_report

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
/* While a tab's request runs, only the LLM status box above its input glows
   (General: above the message box; Legal: above the Mode box). */
.block.llm-status-general, .block.llm-status-legal { position: relative; }
.gradio-container:has(#general-msg.ai-working) .block.llm-status-general::after,
.gradio-container:has(#legal-msg.ai-working) .block.llm-status-legal::after {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  border: 2px solid var(--color-accent); border-radius: inherit;
  box-shadow: 0 0 12px 2px rgba(249, 115, 22, .5);
  z-index: var(--layer-1, 1);
  animation: ai-working-pulse 2s cubic-bezier(.4, 0, .6, 1) infinite;
}
@keyframes ai-working-pulse { 0%, 100% { opacity: 1; } 50% { opacity: .5; } }
@media (prefers-reduced-motion: reduce) {
  .block.llm-status-general::after, .block.llm-status-legal::after { animation: none !important; }
}
/* The Legal tab's sources summary above the chat history: orange italics. */
.legal-sources, .legal-sources * { color: #f97316 !important; font-style: italic !important; }
/* Both chat tabs' send arrow: orange, and two text rows taller (growing
   downward from the top of the box). A fixed size in every state -- idle,
   disabled while a request runs, after the box is cleared -- so it never
   changes size once a message is sent. */
.chat-input button.submit-button {
  background: #f97316 !important; color: #fff !important;
  box-sizing: border-box !important; flex: 0 0 auto !important;
  width: var(--size-9, 36px) !important; min-width: var(--size-9, 36px) !important;
  max-width: var(--size-9, 36px) !important; padding: 0 !important;
  height: calc(var(--size-9, 36px) + 42px) !important;
  min-height: calc(var(--size-9, 36px) + 42px) !important;
  max-height: calc(var(--size-9, 36px) + 42px) !important;
  align-self: flex-start;
}
.chat-input button.submit-button:hover { background: #ea580c !important; }
/* Right-to-left text (Hebrew, Arabic) in the chats: each paragraph, list
   item and the message box take their direction from their own text, so a
   Hebrew line reads right-to-left (punctuation and numbers in place) next to
   an English one. APP_HEAD sets dir="auto" on the chat messages' blocks. */
.chat-log .message-content [dir="auto"] { text-align: start !important; }
.chat-input textarea {
  unicode-bidi: plaintext; text-align: start !important;
}
"""

# Enter sends the prompt in both chat tabs (Shift+Enter still adds a line):
# multi-line MultimodalTextboxes otherwise only insert a newline on Enter.
# Passed as `head=` next to APP_CSS.
APP_HEAD = """
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

# The message boxes' own class stays on while they glow: dropping it would lose the orange send
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
        info="Used by General GPT and Legal GPT. The smaller model answers faster and needs less memory; "
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

    with httpx.Client(timeout=60) as client:
        resp = client.post(f"{API_BASE_URL}/api/chat", json=payload, headers=_session_headers(request))
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

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

    with httpx.Client(timeout=60) as client:
        resp = client.post(f"{API_BASE_URL}/api/tone-rewrite", json=payload, headers=_session_headers(request))
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

    yield from _stream_job(job_id, history, rtl_hint=False)


def build_chat_tab(model_size_selector: gr.Radio) -> None:
    chatbot = gr.Chatbot(label="Chat", elem_classes=["chat-log"])
    llm_status = gr.Markdown(
        value="_Idle_", label="LLM status", show_label=True, container=True, elem_classes=["llm-status-general"]
    )
    reasoning_panel = gr.Textbox(label="Reasoning (model's thinking)", lines=6, visible=False)
    # Gemma 4's training-data cutoff, so users know how current its knowledge is.
    cutoff = gr.Markdown(_cutoff_markdown(model_size_selector.value))
    model_size_selector.change(fn=_cutoff_markdown, inputs=model_size_selector, outputs=cutoff)

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
    msg_box.submit(fn=send, inputs=[msg_box, chatbot, model_size_selector], outputs=chat_outputs)
    rewrite_btn.click(
        fn=_glow_while_running(send_tone_rewrite, chat_outputs, msg_box),
        inputs=[msg_box, professionalism_slider, creativity_slider, chatbot, model_size_selector],
        outputs=chat_outputs,
    )


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


def send_legal_message(
    message: dict, mode: str, history: list, size: str | None = None, request: gr.Request = None
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
    with httpx.Client(timeout=60) as client:
        resp = client.post(f"{API_BASE_URL}/api/{endpoint}", json=payload, headers=_session_headers(request))
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

    yield from _stream_legal_job(job_id, history)


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
    with httpx.Client(timeout=60) as client:
        resp = client.post(f"{API_BASE_URL}/api/legal-case", json=payload, headers=_session_headers(request))
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

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
            legal_chatbot = gr.Chatbot(label="Legal Assistant", elem_classes=["chat-log"])
            legal_llm_status = gr.Markdown(
                value="_Idle_", label="LLM status", show_label=True, container=True, elem_classes=["llm-status-legal"]
            )
            legal_reasoning_panel = gr.Textbox(label="Reasoning (model's thinking)", lines=6, visible=False)
            with gr.Row(equal_height=True):
                legal_mode = gr.Radio(
                    [LEGAL_MODE_QUESTION, LEGAL_MODE_CASE],
                    value=LEGAL_MODE_QUESTION,
                    label="Mode",
                    info="Question: a grounded answer with citations. Case analysis: describe the case and/or "
                    "pick its folder with 📂 (or attach a document) -- you get a work file (facts, chronology, "
                    "legal issues, deadlines, red flags, missing information, a draft document and the "
                    "recommended next step).",
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
            legal_msg_box = gr.MultimodalTextbox(
                label="Legal question",
                placeholder="Ask a question about Israeli law in any language -- answered only from the "
                "Israeli laws and procedural regulations in the legal corpus, in your language -- or attach "
                "a document (PDF, DOCX, PPTX, XLSX, image, or .txt) with the 📎 button to have it reviewed "
                "against the law",
                file_types=[".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".tiff", ".txt"],
                file_count="single",
                sources=["upload"],
                elem_id="legal-msg",
                elem_classes=_IDLE_CLASSES,
            )
        with gr.Column(scale=1):
            gr.Markdown("### Citations")
            citations_panel = gr.Markdown(value="_No citations yet._")
            with gr.Accordion("Research memorandum (Pass A)", open=False):
                memo_view = gr.JSON(value=None, label="Claims → evidence")

    send_inputs = [legal_msg_box, legal_mode, legal_chatbot, model_size_selector]
    send_outputs = [legal_chatbot, legal_reasoning_panel, legal_msg_box, citations_panel, legal_llm_status, memo_view]
    send = _glow_while_running(send_legal_message, send_outputs, legal_msg_box)
    legal_msg_box.submit(fn=send, inputs=send_inputs, outputs=send_outputs)
    legal_mode.change(
        fn=lambda mode: gr.update(visible=mode == LEGAL_MODE_CASE), inputs=legal_mode, outputs=case_folder_button
    )
    case_folder_button.upload(
        fn=_glow_while_running(send_case_folder, send_outputs, legal_msg_box),
        inputs=[case_folder_button, legal_msg_box, legal_chatbot, model_size_selector],
        outputs=send_outputs,
    )


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
    with gr.Blocks(title="AI Workbench - Ibrahim Z.") as demo:
        with gr.Row(equal_height=True):
            gr.Markdown("# AI Workbench - Ibrahim Z.")
            model_size_selector = build_model_size_selector()
        with gr.Tabs():
            with gr.Tab("General GPT"):
                build_chat_tab(model_size_selector)
            with gr.Tab("Legal GPT"):
                build_legal_tab(model_size_selector)
            with gr.Tab("System status") as status_tab:
                build_status_tab(status_tab)
        # Refreshing or closing the page stops whatever it left running on the server.
        demo.unload(_reset_session)
    return demo


def run() -> None:
    demo = build_app()
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, css=APP_CSS, head=APP_HEAD)


if __name__ == "__main__":
    run()
