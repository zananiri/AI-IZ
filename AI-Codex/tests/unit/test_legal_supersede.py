from pathlib import Path

import pytest

from aicodex.legal import staging
from aicodex.legal.folder_ingest import derive_sidecar
from aicodex.legal.models import content_hash
from aicodex.legal.sources import looks_visual_order


def _row(start, status="current", end=""):
    meta = {
        "law_id": "law", "law_name": "Ley", "source_id": f"law@{start}:1", "section_number": "1",
        "subsection_number": "", "effective_date_start": start, "effective_date_end": end, "status": status,
        "source_type": "statute", "source_origin": "boe",
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
    statute = derive_sidecar(pdf, "BOLETÍN OFICIAL DEL ESTADO\nNúm. 7 Sábado 8 de enero de 2000 Sec. I. Pág. 575\n"
                                  "Ley 1/2000, de 7 de enero,\nde Enjuiciamiento Civil.\nArtículo 1. ...")
    assert statute["law_name"] == "Ley 1/2000, de 7 de enero, de Enjuiciamiento Civil"
    assert statute["effective_date_start"] == "2000-01-08"  # the bulletin's date
    assert statute["gazette"] == "BOE núm. 7"
    assert (statute["source_type"], statute["source_origin"], statute["language"]) == ("statute", "boe", "es")
    regulation = derive_sidecar(pdf, "Real Decreto 35/2023, de 24 de enero, por el que se aprueba algo.\n1. ...")
    assert (regulation["source_type"], regulation["source_origin"]) == ("regulation", "boe")
    assert regulation["effective_date_start"] == "2023-01-24"
    regional = derive_sidecar(pdf, "DOGC Núm. 9000\nLey 5/2025, de 3 de marzo, de prueba.\n")
    assert regional["source_origin"] == "diario_autonomico" and regional["gazette"] == "DOGC núm. 9000"


def test_visual_order_detection():
    logical = "הם אמרו שלום לכל החברים שלהם בגן " * 10
    visual = " ".join(word[::-1] for word in logical.split())
    assert not looks_visual_order(logical)
    assert looks_visual_order(visual)


def test_state_files_are_not_read_as_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(staging, "_staging_dir", lambda: Path(tmp_path))
    (tmp_path / "_legal_txt_state.json").write_text('{"C:/x.pdf": {}}', encoding="utf-8")
    assert staging.list_batches() == []
