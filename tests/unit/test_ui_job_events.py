"""The chat tabs' SSE reader (ui/gradio_app._job_events): a stream that drops or ends without
"done"/"error" -- the server stopped mid-turn -- shows as an error, not a status left up as if
the model were still working."""

import httpx

from docslides.ui import gradio_app


def _serve(monkeypatch, handler):
    real_client = httpx.Client
    monkeypatch.setattr(
        gradio_app.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)
    )


def _sse(*events: tuple[str, str]) -> bytes:
    return "".join(f"event: {kind}\r\ndata: {data}\r\n\r\n" for kind, data in events).encode()


def test_events_stop_at_done(monkeypatch):
    body = (_sse(("status", '{"message": "Pass A"}')) + b": ping\r\n\r\n"  # sse-starlette's keep-alive
            + _sse(("content_delta", '{"text": "answer"}'), ("done", "{}"), ("status", '{"message": "never read"}')))
    _serve(monkeypatch, lambda request: httpx.Response(200, content=body))

    events = list(gradio_app._job_events("/api/legal-events/job"))

    assert [kind for kind, _ in events] == ["status", "content_delta", "done"]
    assert events[1][1] == {"text": "answer"}


def test_a_stream_that_ends_without_done_is_an_error(monkeypatch):
    _serve(monkeypatch, lambda request: httpx.Response(200, content=_sse(("status", '{"message": "Pass A"}'))))

    events = list(gradio_app._job_events("/api/legal-events/job"))

    assert events[-1] == ("error", {"message": gradio_app._STREAM_LOST})


def test_a_dropped_connection_is_an_error(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    _serve(monkeypatch, refuse)

    (kind, data), = gradio_app._job_events("/api/legal-events/job")

    assert kind == "error" and data["message"].startswith(gradio_app._STREAM_LOST)
    assert "ConnectError" in data["message"]


def test_the_legal_bubble_shows_the_lost_stream(monkeypatch):
    _serve(monkeypatch, lambda request: httpx.Response(200, content=_sse(("status", '{"message": "Pass A"}'))))

    *_, last = gradio_app._stream_legal_job("job", [{"role": "user", "content": "question"}])

    assert gradio_app._STREAM_LOST in last[0]["value"][-1]["content"]
    assert last[4]["value"] == "❌ Error"
