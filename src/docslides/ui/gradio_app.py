"""Gradio chat interface: document upload -> slide generation with a
persistent pipeline-status line, and a chat/tone-rewrite tab with a
collapsible reasoning panel streamed separately from the final answer.

Talks to the FastAPI backend (docslides.api.main) over HTTP/SSE rather than
calling pipeline internals directly, so the UI and API can scale/deploy
independently (see docker-compose.yml).
"""

from __future__ import annotations

import json
import os

import gradio as gr
import httpx

from docslides.config import get_config
from docslides.ingestion.language_detect import detect_language

API_BASE_URL = os.environ.get("DOCSLIDES_API_URL", "http://localhost:8080")


def _language_choices() -> list[str]:
    return get_config().languages.supported


def _is_rtl_lang(lang: str | None) -> bool:
    return bool(lang) and get_config().languages.is_rtl(lang)


# ---------------------------------------------------------------------------
# Document -> Slides tab
# ---------------------------------------------------------------------------


def upload_and_generate(file_path: str, target_lang: str):
    if not file_path:
        yield "Please upload a document first.", None, gr.update()
        return

    with httpx.Client(timeout=60) as client:
        with open(file_path, "rb") as f:
            upload_resp = client.post(f"{API_BASE_URL}/api/upload", files={"file": f})
        upload_resp.raise_for_status()
        server_path = upload_resp.json()["file_path"]

        gen_resp = client.post(
            f"{API_BASE_URL}/api/generate",
            json={"file_path": server_path, "target_lang": target_lang},
        )
        gen_resp.raise_for_status()
        job_id = gen_resp.json()["job_id"]

    status_lines: list[str] = []
    with httpx.Client(timeout=None) as client:
        with client.stream("GET", f"{API_BASE_URL}/api/events/{job_id}") as resp:
            event_kind = None
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    event_kind = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip())
                    if event_kind == "status":
                        status_lines.append(data["message"])
                        yield "\n".join(status_lines[-8:]), None, gr.update(interactive=False)
                    elif event_kind == "done":
                        output_path = data.get("output_path")
                        download_url = f"{API_BASE_URL}/api/download/{job_id}"
                        status_lines.append("Done. Fetching file...")
                        with httpx.Client(timeout=60) as dl_client:
                            file_resp = dl_client.get(download_url)
                            local_path = f"/tmp/{job_id}.pptx" if os.name != "nt" else f"{job_id}.pptx"
                            with open(local_path, "wb") as f:
                                f.write(file_resp.content)
                        yield "\n".join(status_lines), local_path, gr.update(interactive=True)
                    elif event_kind == "error":
                        status_lines.append(f"ERROR: {data['message']}")
                        yield "\n".join(status_lines), None, gr.update(interactive=True)


def build_generate_tab() -> None:
    with gr.Tab("Document -> Slides"):
        gr.Markdown("Upload a document and generate a PowerPoint deck in the target language.")
        with gr.Row():
            file_input = gr.File(label="Document", file_types=[".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg"])
            target_lang = gr.Dropdown(choices=_language_choices(), value="en", label="Target language")
        generate_btn = gr.Button("Generate slides", variant="primary")
        status_box = gr.Textbox(label="Pipeline status", lines=10, interactive=False)
        output_file = gr.File(label="Generated .pptx")

        generate_btn.click(
            fn=upload_and_generate,
            inputs=[file_input, target_lang],
            outputs=[status_box, output_file, generate_btn],
        )


# ---------------------------------------------------------------------------
# Chat / tone-rewrite tab
# ---------------------------------------------------------------------------


def _stream_job(job_id: str, history: list, rtl_hint: bool):
    """`history` must already include the user's turn as the last row, e.g.
    `[..., [user_message, None]]` -- the assistant's response is streamed
    into that row's second element in place."""
    reasoning_text = ""
    content_text = ""
    base_history = history[:-1]
    user_turn = history[-1][0]

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
                    if event_kind == "reasoning_delta":
                        reasoning_text += data["text"]
                    elif event_kind == "content_delta":
                        content_text += data["text"]
                        detected = detect_language(content_text[:200])
                        rtl_hint = _is_rtl_lang(detected)
                    elif event_kind in ("done", "error"):
                        if event_kind == "error":
                            content_text += f"\n\n[error: {data.get('message')}]"
                        break

                    new_history = base_history + [[user_turn, content_text]]
                    yield (
                        gr.update(value=new_history, rtl=rtl_hint),
                        gr.update(value=reasoning_text, visible=bool(reasoning_text), rtl=rtl_hint),
                        gr.update(value=""),
                    )


def send_chat_message(message: str, history: list):
    history = history + [[message, None]]
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{API_BASE_URL}/api/chat",
            json={"messages": [{"role": "user", "content": message}]},
        )
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

    yield from _stream_job(job_id, history, rtl_hint=False)


def send_tone_rewrite(text: str, task_description: str, professionalism: int, creativity: int, history: list):
    history = history + [[f"[Rewrite request] {text}", None]]
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{API_BASE_URL}/api/tone-rewrite",
            json={
                "text": text,
                "task_description": task_description or "Rewrite the following text.",
                "professionalism": professionalism,
                "creativity": creativity,
            },
        )
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

    yield from _stream_job(job_id, history, rtl_hint=False)


def build_chat_tab() -> None:
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(label="Chat")
        reasoning_panel = gr.Textbox(label="Reasoning (model's thinking)", lines=6, visible=False)

        with gr.Row():
            msg_box = gr.Textbox(label="Message", scale=4)
            send_btn = gr.Button("Send", scale=1)

        with gr.Accordion("Tone control (for rewrite requests)", open=False):
            gr.Markdown(
                "Professionalism drives wording/register (system prompt); "
                "Creativity drives sampling randomness only."
            )
            professionalism_slider = gr.Slider(
                1, 5, value=3, step=1,
                label="Professionalism: 1=Casual, 2=Conversational, 3=Standard business, 4=Formal, 5=Executive/Legal",
            )
            creativity_slider = gr.Slider(
                1, 5, value=3, step=1,
                label="Creativity: 1=Minimal, 2=Low, 3=Balanced, 4=High, 5=Maximum",
            )
            task_description_box = gr.Textbox(
                label="What are you rewriting?", placeholder="e.g. Rewrite this email"
            )
            rewrite_btn = gr.Button("Rewrite with tone controls")

        send_btn.click(
            fn=send_chat_message,
            inputs=[msg_box, chatbot],
            outputs=[chatbot, reasoning_panel, msg_box],
        )
        msg_box.submit(
            fn=send_chat_message,
            inputs=[msg_box, chatbot],
            outputs=[chatbot, reasoning_panel, msg_box],
        )
        rewrite_btn.click(
            fn=send_tone_rewrite,
            inputs=[msg_box, task_description_box, professionalism_slider, creativity_slider, chatbot],
            outputs=[chatbot, reasoning_panel, msg_box],
        )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="docslides") as demo:
        gr.Markdown("# docslides -- offline multilingual document-to-PowerPoint")
        build_generate_tab()
        build_chat_tab()
    return demo


def run() -> None:
    demo = build_app()
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    run()
