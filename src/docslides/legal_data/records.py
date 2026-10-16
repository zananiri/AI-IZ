"""The normalized corpus record (one JSON object per line) and its JSONL writer/reader.

One schema for every category; fields that don't apply stay null/empty. `record_hash` covers
everything but retrieved_at, so re-fetching an unchanged document doesn't re-index it
(scripts/legal_data/vectorize.py)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal["laws", "procedural_rules", "supreme_court"]
AuthorityLevel = Literal["basic_law", "law", "ordinance", "regulation", "judgment", "decision"]
RecordStatus = Literal["in_force", "repealed", "unknown"]


class Quality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoding: Literal["ok", "repaired", "flagged"] = "ok"
    extraction: Literal["ok", "low", "n/a"] = "n/a"  # PDF text extraction; n/a for text sources
    notes: list[str] = Field(default_factory=list)


class SubsectionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str  # "א", "1" -- without parentheses, as legal/structure.py stores them
    text: str  # starts with its own "(א) " line, nested items follow


class SectionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["section", "preamble", "schedule"] = "section"
    number: str  # "12", "5א"; "preamble"; a schedule's heading anchor
    title: str | None = None
    division: str | None = None  # חלק
    chapter: str | None = None  # פרק
    subchapter: str | None = None  # סימן (and deeper headings)
    note: str | None = None  # e.g. "תיקון: תשנ״ה" from the section heading
    intro: str = ""
    subsections: list[SubsectionRecord] = Field(default_factory=list)
    text: str


class CorpusRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    source_url: str
    license: str
    attribution: str | None = None
    category: Category
    title: str
    authority_level: AuthorityLevel
    status: RecordStatus = "unknown"
    status_source: str | None = None  # where status came from: "odata", "wikisource_title", ...
    effective_date: str | None = None  # ISO date
    retrieved_at: str
    language: str = "he"
    text: str
    text_sha256: str = ""
    quality: Quality = Field(default_factory=Quality)

    # laws / procedural_rules
    law_id: int | None = None  # KNS_IsraelLaw.Id (laws) or KNS_SecondaryLaw.Id (regulations)
    law_id_source: str | None = None  # "wikisource_registry_id" | "name_match"
    knesset_name: str | None = None
    is_basic_law: bool | None = None
    authorizing_law_ids: list[int] = Field(default_factory=list)
    sections: list[SectionRecord] = Field(default_factory=list)
    section_numbers: list[str] = Field(default_factory=list)
    publication_date: str | None = None
    latest_amendment_date: str | None = None
    gazette_citations: list[dict] = Field(default_factory=list)  # {ref, name, url} -- cited, not fetched
    wikisource_title: str | None = None
    wikisource_pageid: int | None = None
    wikisource_revid: int | None = None
    wikisource_timestamp: str | None = None
    pdf_files: list[dict] = Field(default_factory=list)  # {url, path, sha256, group_type}

    # supreme_court
    case_number: str | None = None
    court: str | None = None
    judges: list[str] = Field(default_factory=list)
    decision_date: str | None = None
    doc_type: str | None = None
    technical: bool | None = None
    case_name: str | None = None
    year: int | None = None
    pages: int | None = None
    source_case_id: str | None = None
    anonymized: bool = False

    def finalize(self) -> CorpusRecord:
        self.text_sha256 = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.sections and not self.section_numbers:
            self.section_numbers = [s.number for s in self.sections if s.kind == "section"]
        return self


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_hash(record: dict) -> str:
    """Content hash of a record: everything except when it was retrieved."""
    payload = {k: v for k, v in record.items() if k != "retrieved_at"}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


class JsonlWriter:
    """<dir>/<stem>.jsonl, or <stem>-00001.jsonl, -00002 ... with `shard_size`. Written to .tmp
    files and renamed into place by close(), so a failed run leaves the previous output intact;
    shards of the same stem that the new run didn't produce are removed."""

    def __init__(self, directory: Path, stem: str, shard_size: int | None = None) -> None:
        self.directory, self.stem, self.shard_size = directory, stem, shard_size
        directory.mkdir(parents=True, exist_ok=True)
        self._paths: list[Path] = []
        self._file = None
        self._in_shard = 0
        self.count = 0

    def _open_next(self) -> None:
        if self._file:
            self._file.close()
        name = f"{self.stem}-{len(self._paths) + 1:05d}.jsonl" if self.shard_size else f"{self.stem}.jsonl"
        path = self.directory / name
        self._paths.append(path)
        self._file = open(path.with_name(path.name + ".tmp"), "w", encoding="utf-8", newline="\n")
        self._in_shard = 0

    def write(self, record: CorpusRecord | dict) -> None:
        if self._file is None or (self.shard_size and self._in_shard >= self.shard_size):
            self._open_next()
        data = record.model_dump(mode="json") if isinstance(record, CorpusRecord) else record
        self._file.write(json.dumps(data, ensure_ascii=False) + "\n")
        self._in_shard += 1
        self.count += 1

    def close(self, commit: bool = True) -> list[Path]:
        if self._file:
            self._file.close()
            self._file = None
        temps = [p.with_name(p.name + ".tmp") for p in self._paths]
        if not commit:
            for tmp in temps:
                tmp.unlink(missing_ok=True)
            return []
        if not self._paths:  # nothing written: an empty output file, not a stale one
            self._open_next()
            self._file.close()
            self._file = None
            temps = [p.with_name(p.name + ".tmp") for p in self._paths]
        pattern = f"{self.stem}-*.jsonl" if self.shard_size else f"{self.stem}.jsonl"
        for stale in self.directory.glob(pattern):
            if stale not in self._paths:
                stale.unlink()
        for tmp, path in zip(temps, self._paths):
            tmp.replace(path)
        return self._paths


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
