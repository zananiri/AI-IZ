"""Stage 0 staging area for the Legal tab's index (ingestion spec 0.1-0.3).

Nothing reaches the vector store directly. Every source file -- official or
uploaded -- is parsed and chunked into a pending *batch* (one JSON file
under config.legal.ingestion.staging_dir), reviewed, and only then approved:
embedded, upserted, and recorded in the signed bundle (legal/bundle.py).

  * Official sources (statute / regulation / ruling) may be approved with or
    without a named reviewer; `reviewed_by` stays null without one.
  * Uploads (uploads/ folder) require a named reviewer every time --
    official sources carry inherent authority, uploads don't.

Re-approving a new batch for the same law version (same law_id + effective
start date) replaces that version's chunks, deleting ones that no longer
exist, so a corrected re-ingest can't leave stale provisions behind.

Automatic superseding: versions of one law are kept side by side (temporal
questions need the old text), and every approval or retraction re-chains
them by effective start date. Each version except the latest gets
effective_date_end = the day before the next version starts, and status
"amended" (unless it was itself declared "repealed"). The latest version
keeps whatever its own source declared. Declared values are stored in the
signed manifest entries, so removing the newest version restores the
previous one to its declared state. Hashes of re-chained chunks are updated
in the manifest as part of the same signed save.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from itertools import pairwise
from pathlib import Path
from typing import Literal

from docslides.config import get_config
from docslides.legal import bundle, retrieval
from docslides.legal.chunking import chunk_sections
from docslides.legal.models import LegalChunk, content_hash
from docslides.legal.sources import LoadedSource, is_source_file, load_upload
from docslides.legal.structure import parse_sections

BatchStatus = Literal["pending", "approved", "rejected"]


class StagingError(Exception):
    pass


@dataclass
class Batch:
    batch_id: str
    created_at: str
    source_path: str
    source_sha256: str
    source_type: str
    source_origin: str
    law_id: str
    law_name: str
    effective_date_start: str
    requires_signoff: bool
    status: BatchStatus = "pending"
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    rejection_reason: str | None = None
    superseded: list[str] = field(default_factory=list)  # versions re-chained by this approval
    chunks: list[LegalChunk] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["chunks"] = [c.to_dict() for c in self.chunks]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Batch:
        return cls(**{**data, "chunks": [LegalChunk.from_dict(c) for c in data.get("chunks", [])]})


def _staging_dir() -> Path:
    return Path(get_config().legal.ingestion.staging_dir)


def _batch_path(batch_id: str) -> Path:
    return _staging_dir() / f"{batch_id}.json"


def _save(batch: Batch) -> None:
    path = _batch_path(batch.batch_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(batch.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_chunks(source: LoadedSource) -> list[LegalChunk]:
    sections = parse_sections(source.text)
    if not sections:
        raise StagingError(f"{source.path.name}: no text found to chunk")
    return chunk_sections(sections, source.meta, ingestion_date=date.today().isoformat())


def load_batch(batch_id: str) -> Batch:
    path = _batch_path(batch_id)
    if not path.exists():
        raise StagingError(f"No staged batch '{batch_id}' in {_staging_dir()}")
    return Batch.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_batches(status: BatchStatus | None = None) -> list[Batch]:
    batches = [
        Batch.from_dict(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(_staging_dir().glob("*.json"))
        if not p.name.startswith("_")  # state files (_uploads_state.json, _legal_txt_state.json)
    ]
    return [b for b in batches if status is None or b.status == status]


def stage(source: LoadedSource) -> Batch:
    """Chunks `source` into a new pending batch -- or returns the existing
    pending batch if this exact file content is already awaiting review."""
    for existing in list_batches("pending"):
        if existing.source_sha256 == source.sha256:
            return existing
    is_upload = source.meta.source_type == "uploaded_document"
    batch = Batch(
        batch_id=f"{datetime.now():%Y%m%d-%H%M%S}-{source.sha256[:8]}",
        created_at=_now(),
        source_path=str(source.path),
        source_sha256=source.sha256,
        source_type=source.meta.source_type,
        source_origin=source.meta.source_origin,
        law_id=source.meta.law_id,
        law_name=source.meta.law_name,
        effective_date_start=source.meta.effective_date_start,
        requires_signoff=is_upload,
        chunks=build_chunks(source),
    )
    _save(batch)
    return batch


def approve(batch_id: str, reviewer: str | None) -> Batch:
    batch = load_batch(batch_id)
    if batch.status != "pending":
        raise StagingError(f"Batch {batch_id} is already {batch.status}")
    reviewer = (reviewer or "").strip() or None
    if batch.requires_signoff and reviewer is None:
        raise StagingError(
            f"Batch {batch_id} is an uploaded document -- it needs a named reviewer's sign-off (--reviewer)"
        )
    bundle.require_key()  # fail before touching the vector store, not after

    approved_at = _now()
    for chunk in batch.chunks:
        chunk.metadata.reviewed_by = reviewer

    new_ids = {c.metadata.chunk_id for c in batch.chunks}
    stale = [cid for cid in retrieval.chunk_ids_for_version(batch.law_id, batch.effective_date_start) if cid not in new_ids]
    retrieval.delete_chunks(stale)
    retrieval.upsert_chunks(batch.chunks)

    entries = bundle.load_entries()
    for chunk_id in stale:
        entries.pop(chunk_id, None)
    for chunk in batch.chunks:
        entries[chunk.metadata.chunk_id] = {
            "sha256": chunk.content_hash,
            "batch_id": batch.batch_id,
            "source_sha256": batch.source_sha256,
            "reviewed_by": reviewer,
            "approved_at": approved_at,
            "declared_status": chunk.metadata.status,
            "declared_effective_date_end": chunk.metadata.effective_date_end,
        }
    changes = _rechain(batch.law_id, entries, actor=reviewer, when=approved_at)
    bundle.save_entries(entries)

    batch.superseded = changes
    batch.status = "approved"
    batch.reviewed_by = reviewer
    batch.reviewed_at = approved_at
    _save(batch)
    return batch


def _rechain(law_id: str, entries: dict[str, dict], actor: str | None, when: str) -> list[str]:
    """Re-derives effective_date_end/status for every stored version of
    `law_id` from the version ordering (see module docstring). Updates the
    vector store's metadata and `entries` (hashes) in place; the caller
    saves `entries`. Returns one human-readable line per version changed."""
    rows = retrieval.chunks_for_law(law_id)
    starts = sorted({meta["effective_date_start"] for _, _, meta in rows})
    next_start = dict(pairwise(starts))

    ids, metadatas, changed_versions = [], [], {}
    for chunk_id, text, meta in rows:
        entry = entries.get(chunk_id)
        if entry is None:
            continue  # not in the bundle; retrieval already refuses it
        declared_status = entry.get("declared_status", meta["status"])
        declared_end = entry.get("declared_effective_date_end") or ""
        start = meta["effective_date_start"]
        if start in next_start:
            closes = (date.fromisoformat(next_start[start]) - timedelta(days=1)).isoformat()
            end = min(declared_end, closes) if declared_end else closes
            status = "repealed" if declared_status == "repealed" else "amended"
        else:
            end, status = declared_end, declared_status
        if (meta["effective_date_end"], meta["status"]) == (end, status):
            continue
        new_meta = {**meta, "effective_date_end": end, "status": status}
        ids.append(chunk_id)
        metadatas.append(new_meta)
        entry.update(sha256=content_hash(text, new_meta), rechained_at=when, rechained_by=actor)
        changed_versions[start] = (
            f"{meta['law_name']} @ {start}: {meta['status']} -> {status}, "
            f"effective until {end or 'open-ended'}"
        )

    retrieval.update_metadatas(ids, metadatas)
    return [changed_versions[s] for s in sorted(changed_versions)]


def retract_version(law_id: str, effective_date_start: str, actor: str | None = None) -> list[str]:
    """Removes one version of a law from the index and the signed bundle,
    then re-chains the remaining versions. Returns the re-chain changes."""
    bundle.require_key()
    ids = retrieval.chunk_ids_for_version(law_id, effective_date_start)
    retrieval.delete_chunks(ids)
    entries = bundle.load_entries()
    for chunk_id in ids:
        entries.pop(chunk_id, None)
    changes = _rechain(law_id, entries, actor=actor, when=_now())
    bundle.save_entries(entries)
    return changes


def discard_pending(source_sha256: str, reason: str) -> None:
    """Rejects pending batches of this exact file content, so the next
    stage() builds a fresh batch (e.g. after its metadata sidecar changed)."""
    for batch in list_batches("pending"):
        if batch.source_sha256 == source_sha256:
            reject(batch.batch_id, reviewer="system", reason=reason)


def reject(batch_id: str, reviewer: str, reason: str) -> Batch:
    batch = load_batch(batch_id)
    if batch.status != "pending":
        raise StagingError(f"Batch {batch_id} is already {batch.status}")
    batch.status = "rejected"
    batch.reviewed_by = reviewer
    batch.reviewed_at = _now()
    batch.rejection_reason = reason
    _save(batch)
    return batch


_UPLOADS_STATE_FILE = "_uploads_state.json"


def scan_uploads() -> tuple[list[Batch], list[tuple[Path, str]]]:
    """Stages every new or changed file in the uploads folder. Returns
    (batches staged this scan, [(file, error), ...] for files that failed)."""
    uploads_dir = Path(get_config().legal.ingestion.uploads_dir)
    state_path = _staging_dir() / _UPLOADS_STATE_FILE
    state: dict[str, str] = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}

    staged: list[Batch] = []
    failed: list[tuple[Path, str]] = []
    for path in sorted(p for p in uploads_dir.rglob("*") if is_source_file(p)):
        try:
            source = load_upload(path)
            if state.get(str(path)) == source.sha256:
                continue
            staged.append(stage(source))
            state[str(path)] = source.sha256
        except Exception as exc:  # noqa: BLE001 -- one bad file mustn't block the rest
            failed.append((path, str(exc)))

    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    return staged, failed
