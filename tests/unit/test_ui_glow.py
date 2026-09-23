"""The message-box glow every tab shows while its request runs (ui/gradio_app.py)."""

import gradio as gr
import pytest

from docslides.ui.gradio_app import _glow_while_running


def test_message_box_glows_from_send_until_the_handler_finishes():
    status, box = gr.Markdown(), gr.Textbox()

    def handler(message):
        yield gr.update(value="working"), gr.update(value=None)
        yield gr.update(value="done"), "plain value"

    updates = list(_glow_while_running(handler, [status, box], box)("question"))

    assert [u[1]["elem_classes"] for u in updates] == [["ai-working"]] * 3 + [[]]
    assert updates[1][0]["value"] == "working"  # the other outputs pass through untouched
    assert updates[2][1]["value"] == "plain value"  # a raw value for the box is kept


def test_message_box_stops_glowing_when_the_handler_fails():
    status, box = gr.Markdown(), gr.Textbox()

    def handler(message):
        yield gr.update(value="working"), gr.update(value=None)
        raise gr.Error("upload failed")

    seen = []
    with pytest.raises(gr.Error):
        for values in _glow_while_running(handler, [status, box], box)("question"):
            seen.append(values[1]["elem_classes"])
    assert seen == [["ai-working"], ["ai-working"], []]
