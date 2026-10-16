"""Per-run manifest and log for the corpus fetchers: <output>/_manifests/<run_id>_<script>.json
(+ .log). Records what was contacted and why it was allowed (robots.txt decisions, requests per
host), every source with its URL and license, every file with its hash and whether it was
downloaded or skipped, record counts, filter counts, warnings and errors."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from docslides.legal_data.http import PoliteClient


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunManifest:
    def __init__(self, script: str, root: Path, args: dict, *, dry_run: bool, sample: int | None) -> None:
        self.root = root
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = root / "_manifests"
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{self.run_id}_{script}.json"
        self.log_path = directory / f"{self.run_id}_{script}.log"
        self._log = open(self.log_path, "a", encoding="utf-8")
        self.data: dict = {
            "run_id": self.run_id,
            "script": script,
            "args": args,
            "dry_run": dry_run,
            "sample": sample,
            "started_at": _now(),
            "finished_at": None,
            "status": "running",
            "user_agent": None,
            "sources": [],
            "robots": [],
            "requests_per_host": {},
            "files": [],
            "counts": {},
            "filters": {},
            "warnings": [],
            "errors": [],
        }

    def log(self, message: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {message}"
        print(line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    def source(self, name: str, url: str, license: str, attribution: str | None = None, **extra) -> None:
        self.data["sources"].append({"name": name, "url": url, "license": license, "attribution": attribution,
                                     "retrieved_at": _now(), **extra})

    def file(self, url: str, path: Path, size: int, sha256: str, status: str) -> None:
        try:
            shown = str(path.relative_to(self.root))
        except ValueError:
            shown = str(path)
        self.data["files"].append({"url": url, "path": shown.replace("\\", "/"), "bytes": size,
                                   "sha256": sha256, "status": status})

    def count(self, key: str, n: int = 1) -> None:
        self.data["counts"][key] = self.data["counts"].get(key, 0) + n

    def set(self, section: str, key: str, value) -> None:
        self.data.setdefault(section, {})[key] = value

    def warn(self, message: str) -> None:
        self.data["warnings"].append(message)
        self.log(f"WARNING: {message}")

    def error(self, message: str) -> None:
        self.data["errors"].append(message)
        self.log(f"ERROR: {message}")

    def finish(self, status: str, client: PoliteClient | None = None) -> Path:
        self.data.update(status=status, finished_at=_now())
        if client is not None:
            self.data.update(user_agent=client.user_agent, robots=client.robots_log,
                             requests_per_host=dict(client.requests_per_host))
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tmp.replace(self.path)
        self.log(f"manifest: {self.path}")
        self._log.close()
        return self.path
