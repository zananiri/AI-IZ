"""Chunk + metadata types for the Legal tab's Israeli-law index.

`ChunkMetadata` is the per-chunk schema from the ingestion spec (source_id,
law_name, chapter, section/subsection, effective-date range, status,
source_type/origin, ingestion date, reviewer, cross-references), plus the
bookkeeping fields retrieval needs: `chunk_id` (unique per stored chunk --
a long provision split into sibling parts shares one `source_id` but each
part has its own `chunk_id`), `section_key` (what cross-references point
at), `part`/`breadcrumb`, and `part_index`/`part_count`.

Chroma metadata values must be scalars, so `to_chroma`/`from_chroma` flatten
None to "" and the cross-reference list to a JSON string. `content_hash` is
computed over that flattened form, so the signed bundle (legal/bundle.py)
can re-verify a chunk exactly as it comes back out of the vector store.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

SourceType = Literal["statute", "regulation", "ruling", "uploaded_document"]
SourceOrigin = Literal["knesset", "reshumot", "court_gov_il", "manual_upload"]
Status = Literal["current", "amended", "repealed"]

# Fields covered by the integrity hash: everything a citation depends on.
_HASHED_FIELDS = (
    "source_id",
    "law_name",
    "section_number",
    "subsection_number",
    "effective_date_start",
    "effective_date_end",
    "status",
    "source_type",
    "source_origin",
)


@dataclass
class ChunkMetadata:
    chunk_id: str
    source_id: str
    section_key: str
    law_id: str
    law_name: str
    chapter: str | None
    part: str | None
    section_number: str
    subsection_number: str | None
    breadcrumb: str
    effective_date_start: str
    effective_date_end: str | None
    status: Status
    source_type: SourceType
    source_origin: SourceOrigin
    ingestion_date: str
    language: str
    part_index: int = 1
    part_count: int = 1
    reviewed_by: str | None = None
    cross_references: list[str] = field(default_factory=list)
    gazette: str | None = None  # e.g. "ספר החוקים 3546" -- lets "what did amendment X change" match
    law_key: str = ""  # normalized law identity, matches amendments to this law (legal/amendments.py)
    amends: list[dict] = field(default_factory=list)  # what this chunk amends in other laws
    # A provision this (amending) chunk inserts into another law, by that law's number:
    # "116יז10(ד)" for a chunk of section 6(4) that adds it (legal/insertions.py).
    inserted_section: str = ""

    @property
    def display_section(self) -> str:
        """How a citation names this provision: "6(4) › 116יז10(ד)" for an inserted one."""
        own = self.section_number + (f"({self.subsection_number})" if self.subsection_number else "")
        return f"{own} › {self.inserted_section}" if self.inserted_section else own

    def to_chroma(self) -> dict[str, str | int]:
        flat: dict[str, str | int] = {}
        for key, value in asdict(self).items():
            if key in ("cross_references", "amends"):
                flat[key] = json.dumps(value, ensure_ascii=False)
            elif value is None:
                flat[key] = ""
            else:
                flat[key] = value
        return flat

    @classmethod
    def from_chroma(cls, meta: dict) -> ChunkMetadata:
        values = dict(meta)
        values["cross_references"] = json.loads(values.get("cross_references") or "[]")
        values["amends"] = json.loads(values.get("amends") or "[]")
        for key in ("chapter", "part", "subsection_number", "effective_date_end", "reviewed_by", "gazette"):
            values[key] = values.get(key) or None
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in values.items() if k in known})


@dataclass
class LegalChunk:
    text: str  # breadcrumb header + provision text -- what gets embedded
    metadata: ChunkMetadata

    @property
    def content_hash(self) -> str:
        return content_hash(self.text, self.metadata.to_chroma())

    def to_dict(self) -> dict:
        return {"text": self.text, "metadata": asdict(self.metadata)}

    @classmethod
    def from_dict(cls, data: dict) -> LegalChunk:
        return cls(text=data["text"], metadata=ChunkMetadata(**data["metadata"]))


def content_hash(text: str, flat_metadata: dict) -> str:
    payload = {"text": text, **{k: str(flat_metadata.get(k, "")) for k in _HASHED_FIELDS}}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
