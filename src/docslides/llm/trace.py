"""A record of every LLM call: the call site, the model's reasoning ("thinking"),
its output, why it stopped, token counts and time -- written by llm/client.py.

Two sinks:
  * the collector of the current turn (`collect`): legal/pipeline.py writes
    it into the turn's audit entry as `llm_calls`;
  * one JSON line per call appended to <logging.llm_trace_dir>/<YYYY-MM-DD>.jsonl,
    with the full prompt when logging.llm_trace_prompts is on.

The reasoning is what this exists for: it shows how the model read the evidence
before it wrote anything, which is where most wrong answers start. Records hold
questions and evidence verbatim -- the directory is as sensitive as the audit log.
"""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import orjson

from docslides.config import get_config

# A list, not a copy per task: asyncio tasks started inside `collect` (the parallel
# citation checks) inherit the variable and append to the same list.
_calls: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar("llm_trace_calls", default=None)
_job_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("llm_trace_job_id", default=None)
_lock = threading.Lock()


@contextmanager
def collect(job_id: str | None = None) -> Iterator[list[dict]]:
    """Every call made inside the block is appended to the list it yields."""
    calls: list[dict] = []
    calls_token, job_token = _calls.set(calls), _job_id.set(job_id)
    try:
        yield calls
    finally:
        _calls.reset(calls_token)
        _job_id.reset(job_token)


def current_calls() -> list[dict] | None:
    return _calls.get()


def record(call: dict, messages: list[dict] | None = None) -> None:
    now = datetime.now(timezone.utc)
    call = {"at": now.isoformat(timespec="seconds"), "job_id": _job_id.get(), **call}
    calls = _calls.get()
    if calls is not None:
        calls.append(call)
    cfg = get_config().logging
    if not cfg.llm_trace_dir:
        return
    line = {**call, "messages": messages} if messages is not None and cfg.llm_trace_prompts else call
    try:
        directory = Path(cfg.llm_trace_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with _lock, open(directory / f"{now:%Y-%m-%d}.jsonl", "ab") as f:
            f.write(orjson.dumps(line, default=str) + b"\n")
    except OSError:
        pass  # a trace that can't be written must never fail the call it describes
