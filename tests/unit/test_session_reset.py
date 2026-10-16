"""Refreshing or closing the page cancels every job it left running on the server
(api/events.EventBus.cancel_session, api/routes_session.py, ui/gradio_app._reset_session)."""

import asyncio

from docslides.api.events import CANCELLED_MESSAGE, EventBus


async def _events(bus, job_id):
    return [event async for event in bus.stream(job_id)]


def test_cancelling_a_session_stops_its_jobs_and_ends_their_streams():
    async def scenario():
        bus = EventBus()
        started = asyncio.Event()

        async def slow_job():
            started.set()
            await asyncio.sleep(3600)

        async def quick_job(job_id):
            await bus.publish_done(job_id)

        bus.start("mine", slow_job(), session="page-1")
        bus.start("other", quick_job("other"), session="page-2")
        await started.wait()
        listener = asyncio.create_task(_events(bus, "mine"))

        assert bus.cancel_session("page-1") == 1
        events = await asyncio.wait_for(listener, 1)
        other = await asyncio.wait_for(_events(bus, "other"), 1)
        return bus, events, other

    bus, events, other = asyncio.run(scenario())

    assert [(e["event"], CANCELLED_MESSAGE in e["data"]) for e in events] == [("error", True)]
    assert [e["event"] for e in other] == ["done"]  # another page's job is untouched
    assert not bus._tasks and not bus._session_jobs


def test_a_job_a_closed_page_was_still_starting_is_refused():
    async def scenario():
        bus = EventBus()
        ran = []

        async def job():
            ran.append(True)

        assert bus.cancel_session("page-1") == 0
        bus.start("late", job(), session="page-1")
        events = await asyncio.wait_for(_events(bus, "late"), 1)
        await asyncio.sleep(0)
        return events, ran

    events, ran = asyncio.run(scenario())

    assert [e["event"] for e in events] == ["error"] and not ran


def test_the_page_registers_the_reset_on_unload(monkeypatch):
    import gradio as gr

    from docslides.ui import gradio_app

    posted = []

    class FakeClient:
        def __init__(self, **_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def post(self, url, **_):
            posted.append(url)

    monkeypatch.setattr(gradio_app.httpx, "Client", FakeClient)
    gradio_app._reset_session(type("Request", (), {"session_hash": "abc123"})())

    assert posted == [f"{gradio_app.API_BASE_URL}/api/sessions/abc123/cancel"]
    assert gradio_app._session_headers(type("Request", (), {"session_hash": "abc123"})()) == {
        "X-Client-Session": "abc123"
    }
    demo = gradio_app.build_app()
    assert any(t[1] == "unload" for dep in demo.fns.values() for t in dep.targets)
    assert isinstance(demo, gr.Blocks)
