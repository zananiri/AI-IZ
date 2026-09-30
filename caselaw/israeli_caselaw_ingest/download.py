"""Stage 1: download cases_all.parquet from the Hugging Face dataset (the only network access
besides model weights). Skips the download when the file is there and its size matches."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from .config import cache_dir
from .state import log


def dataset_path(cfg: dict) -> Path:
    return cache_dir(cfg) / cfg["source"]["filename"]


def remote_size(cfg: dict) -> int | None:
    """The file's size on the Hub, or None when the Hub can't be reached."""
    src = cfg["source"]
    try:
        from huggingface_hub import HfApi

        info = HfApi().get_paths_info(src["repo_id"], [src["filename"]], repo_type=src["repo_type"],
                                      revision=src.get("revision"))
        return int(info[0].size) if info else None
    except Exception as exc:  # noqa: BLE001 -- offline, or the API changed: fall back to "present is enough"
        log(f"could not read the remote file size ({type(exc).__name__}: {exc})")
        return None


def download(cfg: dict, force: bool = False) -> Path:
    src = cfg["source"]
    target = dataset_path(cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = remote_size(cfg)
    if target.exists() and not force:
        size = target.stat().st_size
        if expected is None or size == expected:
            log(f"{target} present ({size / 1e9:.2f} GB){'' if expected else ', remote size unknown'}: skipping download")
            return target
        log(f"{target} is {size} bytes, the Hub has {expected}: downloading again")
    from huggingface_hub import hf_hub_download

    log(f"downloading {src['repo_id']}/{src['filename']} ({(expected or 0) / 1e9:.2f} GB) to {target.parent}")
    got = Path(hf_hub_download(repo_id=src["repo_id"], repo_type=src["repo_type"], filename=src["filename"],
                               revision=src.get("revision"), local_dir=str(target.parent)))
    if got.resolve() != target.resolve():
        got.replace(target)
    if expected is not None and target.stat().st_size != expected:
        raise SystemExit(f"downloaded size {target.stat().st_size} != expected {expected}; run download again")
    meta = pq.ParquetFile(target).metadata
    log(f"downloaded: {meta.num_rows:,} rows, {meta.num_row_groups} row groups")
    return target
