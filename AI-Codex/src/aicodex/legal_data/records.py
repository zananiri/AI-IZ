"""The normalized corpus record (one JSON object per line) and its JSONL writer/reader.

One schema for every category (Spanish legislation from the BOE, EU law from EUR-Lex, case
law from the Tribunal Constitucional); fields that don't apply stay null/empty. `record_hash` covers
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

Category = Literal["laws", "regulations", "regional_laws", "constitutional_court", "eu_law"]
CATEGORIES: tuple[str, ...] = ("laws", "regulations", "regional_laws", "constitutional_court", "eu_law")
LEGISLATION_CATEGORIES: tuple[str, ...] = ("laws", "regulations", "regional_laws", "eu_law")
JUDGMENT_CATEGORIES: tuple[str, ...] = ("constitutional_court",)
# The Spanish hierarchy of norms (art. 9.3 CE), top first; EU law has primacy in its own sphere.
AuthorityLevel = Literal[
    "constitution", "organic_law", "law", "legislative_decree", "decree_law", "international_treaty",
    "regulation", "order", "resolution", "circular", "regional_law", "regional_decree_law",
    "regional_regulation", "eu_treaty", "eu_regulation", "eu_directive", "eu_decision", "judgment", "decision",
]
RecordStatus = Literal["in_force", "repealed", "unknown"]


class Quality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoding: Literal["ok", "repaired", "flagged"] = "ok"
    extraction: Literal["ok", "low", "n/a"] = "n/a"  # PDF text extraction; n/a for text sources
    notes: list[str] = Field(default_factory=list)


class SubsectionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str  # apartado "1", "2"; or letra "a" -- without the trailing "." / ")"
    text: str


class SectionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # section: an artículo; disposition: a disposición adicional/transitoria/derogatoria/final;
    # schedule: an anexo; preamble: the preámbulo / exposición de motivos.
    kind: Literal["section", "preamble", "schedule", "disposition"] = "section"
    number: str  # "14", "14 bis"; "preamble"; "Disposición adicional primera"; "Anexo I"
    title: str | None = None
    division: str | None = None  # Libro / Título
    chapter: str | None = None  # Capítulo
    subchapter: str | None = None  # Sección / Subsección
    note: str | None = None  # the BOE's footnote on the current wording ("Se modifica por ...")
    intro: str = ""
    subsections: list[SubsectionRecord] = Field(default_factory=list)
    text: str
    version_date: str | None = None  # ISO date this wording took effect (BOE <version fecha_vigencia>)


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
    status_source: str | None = None  # where status came from: "boe_vigencia", "eurlex", ...
    effective_date: str | None = None  # ISO date
    retrieved_at: str
    language: str = "es"
    text: str
    text_sha256: str = ""
    quality: Quality = Field(default_factory=Quality)

    # legislation (laws / regulations / regional_laws / eu_law)
    law_id: str | None = None  # BOE identifier ("BOE-A-2000-323") or CELEX number ("32016R0679")
    rank: str | None = None  # "Ley Orgánica", "Real Decreto", "Reglamento (UE)" ...
    official_number: str | None = None  # "1/2000"
    department: str | None = None  # BOE departamento (ministry, or the autonomous community)
    scope: str | None = None  # "estatal" | "autonomico" | "ue"
    region: str | None = None  # autonomous community, for regional_laws
    eli_url: str | None = None
    journal: str | None = None  # "Boletín Oficial del Estado", "Diario Oficial de la Unión Europea"
    journal_number: str | None = None
    disposition_date: str | None = None
    publication_date: str | None = None
    latest_amendment_date: str | None = None  # the newest version date among the sections
    source_updated_at: str | None = None  # the source's own last-update stamp (incremental runs)
    consolidated: bool | None = None
    subjects: list[str] = Field(default_factory=list)
    sections: list[SectionRecord] = Field(default_factory=list)
    section_numbers: list[str] = Field(default_factory=list)

    # constitutional_court
    ecli: str | None = None
    case_number: str | None = None  # "Recurso de amparo 6194-2020"
    court: str | None = None  # "Tribunal Constitucional"
    chamber: str | None = None  # "Pleno", "Sala Primera", "Sección Tercera"
    judges: list[str] = Field(default_factory=list)
    decision_date: str | None = None
    doc_type: str | None = None  # "SENTENCIA", "AUTO", "DECLARACIÓN"
    resolution_number: str | None = None  # "84/2025"
    procedure_type: str | None = None  # "Recurso de amparo", "Cuestión de inconstitucionalidad" ...
    boe_publication: str | None = None  # "BOE núm. 117, de 15 de mayo de 2025"
    summary: str | None = None  # the court's own síntesis
    year: int | None = None
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
