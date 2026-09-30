"""Done-markers and atomic writes, so every stage can be re-run after a Colab disconnect."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def quiet_hf() -> None:
    """No per-tensor "Loading weights" progress bars (a thousand log lines per model load on Kaggle)."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except Exception:  # noqa: BLE001 -- transformers missing or moved: only cosmetic
        pass


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, data) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=1, default=str))


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_parquet(path: Path, table: pa.Table, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd", **kwargs)
    os.replace(tmp, path)


def stage_done(state_dir: Path, stage: str) -> dict | None:
    return read_json(state_dir / f"{stage}.done.json")


def mark_done(state_dir: Path, stage: str, **info) -> None:
    atomic_write_json(state_dir / f"{stage}.done.json", {"stage": stage, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **info})


def clear_done(state_dir: Path, stage: str) -> None:
    (state_dir / f"{stage}.done.json").unlink(missing_ok=True)
