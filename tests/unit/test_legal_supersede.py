from pathlib import Path

import pytest

from docslides.legal import staging
from docslides.legal.folder_ingest import derive_sidecar
from docslides.legal.models import content_hash
from docslides.legal.sources import looks_visual_order


def _row(start, status="current", end=""):
    meta = {
        "law_id": "law", "law_name": "חוק", "source_id": f"law@{start}:1", "section_number": "1",
        "subsection_number": "", "effective_date_start": start, "effective_date_end": end, "status": status,
        "source_type": "statute", "source_origin": "knesset",
    }
    return (f"law@{start}:1", f"text {start}", meta)


@pytest.fixture
def store(monkeypatch):
    rows: list = []
    updates: dict = {}
    monkeypatch.setattr(staging.retrieval, "chunks_for_law", lambda law_id: rows)
    monkeypatch.setattr(
        staging.retrieval, "update_metadatas", lambda ids, metas: updates.update(zip(ids, metas))
    )
    return rows, updates


def _entries(rows, **declared):
    return {
        cid: {"sha256": content_hash(text, meta), "declared_status": declared.get(meta["effective_date_start"], "current"),
              "declared_effective_date_end": None}
        for cid, text, meta in rows
    }


def test_newer_version_closes_the_previous_one(store):
    rows, updates = store
    rows += [_row("2020-01-01"), _row("2024-07-15")]
    entries = _entries(rows)

    changes = staging._rechain("law", entries, actor=None, when="now")

    old = updates["law@2020-01-01:1"]
    assert (old["status"], old["effective_date_end"]) == ("amended", "2024-07-14")
    assert "law@2024-07-15:1" not in updates  # latest keeps its declared state
    assert entries["law@2020-01-01:1"]["sha256"] == content_hash("text 2020-01-01", old)
    assert len(changes) == 1 and "current -> amended" in changes[0]


def test_older_version_ingested_late_is_closed_against_the_newer_one(store):
    rows, updates = store
    rows += [_row("2024-07-15"), _row("2010-03-01")]
    staging._rechain("law", _entries(rows), actor=None, when="now")
    assert updates["law@2010-03-01:1"]["effective_date_end"] == "2024-07-14"


def test_removing_newest_version_restores_the_previous_declared_state(store):
    rows, updates = store
    rows.append(_row("2020-01-01", status="amended", end="2024-07-14"))  # newer version already gone
    staging._rechain("law", _entries(rows), actor=None, when="now")
    assert (updates["law@2020-01-01:1"]["status"], updates["law@2020-01-01:1"]["effective_date_end"]) == ("current", "")


def test_declared_repeal_is_kept(store):
    rows, updates = store
    rows += [_row("2020-01-01"), _row("2024-07-15")]
    staging._rechain("law", _entries(rows, **{"2020-01-01": "repealed"}), actor=None, when="now")
    assert updates["law@2020-01-01:1"]["status"] == "repealed"


def test_derive_sidecar_from_title(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")
    statute = derive_sidecar(pdf, 'ספר החוקים\nחוק החוזים (חלק כללי), התשל"ג-1973\n1. חוזה נכרת...')
    assert statute["law_name"] == 'חוק החוזים (חלק כללי), התשל"ג-1973'
    assert statute["effective_date_start"] == "1973-01-01"
    assert (statute["source_type"], statute["source_origin"], statute["language"]) == ("statute", "knesset", "he")
    regulation = derive_sidecar(pdf, 'תקנות הדוגמה, התשפ"ה-2025\n1. ...')
    assert (regulation["source_type"], regulation["source_origin"]) == ("regulation", "reshumot")


def test_visual_order_detection():
    logical = "הם אמרו שלום לכל החברים שלהם בגן " * 10
    visual = " ".join(word[::-1] for word in logical.split())
    assert not looks_visual_order(logical)
    assert looks_visual_order(visual)


def test_state_files_are_not_read_as_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(staging, "_staging_dir", lambda: Path(tmp_path))
    (tmp_path / "_legal_txt_state.json").write_text('{"C:/x.pdf": {}}', encoding="utf-8")
    assert staging.list_batches() == []
