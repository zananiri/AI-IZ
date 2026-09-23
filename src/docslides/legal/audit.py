"""Legal tab audit log: one JSON line per answered question, appended to
<config.legal.audit_dir>/<YYYY-MM-DD>.jsonl.

Each entry records the full claim -> evidence trail: the query and its
normalized form, the reply language, which models handled it (including
the DictaLM tier, for reproducibility -- Hebrew phrasing differs subtly
between tiers), the retrieved chunk ids with their bundle hashes and
verification level, every memorandum/draft/polish attempt with the check
results that accepted or rejected it, and the final answer. Entries contain
the user's question verbatim; the directory is as sensitive as the questions.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

import orjson

from docslides.config import get_config

_lock = threading.Lock()


def write_entry(entry: dict) -> Path:
    now = datetime.now(timezone.utc)
    path = Path(get_config().legal.audit_dir) / f"{now:%Y-%m-%d}.jsonl"
    line = orjson.dumps({"logged_at": now.isoformat(), **entry}, default=str) + b"\n"
    with _lock, open(path, "ab") as f:
        f.write(line)
    return path
