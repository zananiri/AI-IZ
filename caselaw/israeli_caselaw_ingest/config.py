"""Configuration: config.yaml deep-merged over built-in defaults, plus the resolved paths."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR.parent / "config.yaml"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, root: str | None = None,
                sample_n: int | None = None, overrides: dict | None = None) -> dict:
    """The shipped config.yaml, then `path` over it, then --root / $CASELAW_ROOT and --sample-n."""
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if path and Path(path).resolve() != DEFAULT_CONFIG:
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    root = root or os.environ.get("CASELAW_ROOT")
    if root:
        cfg["paths"]["root"] = root
    if sample_n is not None:
        cfg["sample_n"] = sample_n or None
    return cfg


@dataclass(frozen=True)
class Paths:
    root: Path        # the Drive root (shared by sample and full runs: holds the dataset cache)
    work: Path        # outputs of this run: root, or root/sample_<n>

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def filtered(self) -> Path:
        return self.work / "filtered"

    @property
    def documents_parts(self) -> Path:
        return self.work / "documents_parts"

    @property
    def documents(self) -> Path:
        return self.work / "documents.parquet"

    @property
    def chunks(self) -> Path:
        return self.work / "chunks"

    @property
    def embeddings(self) -> Path:
        return self.work / "embeddings"

    @property
    def lancedb(self) -> Path:
        return self.work / "lancedb"

    @property
    def faiss(self) -> Path:
        return self.work / "faiss"

    @property
    def bm25(self) -> Path:
        return self.work / "bm25"

    @property
    def reports(self) -> Path:
        return self.work / "reports"

    @property
    def state(self) -> Path:
        return self.work / "state"

    @property
    def models(self) -> Path:
        return self.root / "models"


def paths_for(cfg: dict) -> Paths:
    root = Path(cfg["paths"]["root"]).expanduser()
    work = root / f"sample_{cfg['sample_n']}" if cfg.get("sample_n") else root
    paths = Paths(root=root, work=work)
    return paths


def cache_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["root"]).expanduser() / cfg["paths"].get("cache_dir", "cache")


def config_hash(section: Any) -> str:
    """A short hash of the settings a stage depends on, stored in its done-marker."""
    return hashlib.sha1(json.dumps(section, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]
