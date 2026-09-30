"""End to end on the synthetic dataset: filter -> clean -> chunk -> embed (hashing encoder) ->
BM25 -> validate, then a resumed run that must change nothing."""

import datetime as dt

import pyarrow.parquet as pq
import pytest
from synthetic import make_dataset

from israeli_caselaw_ingest.chunk import chunk_files, run_chunk
from israeli_caselaw_ingest.clean import run_clean
from israeli_caselaw_ingest.config import paths_for
from israeli_caselaw_ingest.download import dataset_path
from israeli_caselaw_ingest.embed import run_embed
from israeli_caselaw_ingest.filter import run_filter
from israeli_caselaw_ingest.search import Searcher, run_bm25
from israeli_caselaw_ingest.validate import run_validate


@pytest.fixture
def built(cfg):
    make_dataset(dataset_path(cfg), n=300)
    run_filter(cfg)
    run_clean(cfg)
    run_chunk(cfg)
    run_embed(cfg, yes=True)
    run_bm25(cfg)
    return cfg


def _chunks(cfg):
    return pq.read_table([str(f) for f in chunk_files(paths_for(cfg).chunks)]).to_pylist() \
        if len(chunk_files(paths_for(cfg).chunks)) == 1 else \
        [r for f in chunk_files(paths_for(cfg).chunks) for r in pq.read_table(f).to_pylist()]


def test_filter_rules(built):
    paths = paths_for(built)
    docs = {d["doc_id"]: d for d in pq.read_table(paths.documents).to_pylist()}
    assert "doc000005" not in docs          # 2022
    assert "doc000004" not in docs          # VerdictDt 2021-12-31T21:00 is 1 Jan 2022
    assert "doc000006" not in docs          # empty
    assert "doc000007" not in docs          # duplicate of doc000000
    assert "doc000008" not in docs          # no date
    assert docs["doc000003"]["decision_date"] == dt.date(2003, 7, 9)   # bogus 1920 -> VerdictDt
    assert all(d["decision_date"] < dt.date(2022, 1, 1) for d in docs.values())
    assert not any(d["doc_type"] in ("צו ביניים",) for d in docs.values())
    report = (paths.reports / "filter_report.md").read_text(encoding="utf-8")
    assert "duplicate_text" in report and "on_or_after_cutoff" in report and "decision_technical_or_unknown" in report


def test_mojibake_rows_are_repaired_and_headers_parsed(built):
    docs = {d["doc_id"]: d for d in pq.read_table(paths_for(built).documents).to_pylist()}
    assert docs["doc000001"]["encoding_repair"] in ("latin1", "cp1252")
    assert docs["doc000002"]["encoding_repair"] == "cp1253"
    for i in ("doc000001", "doc000002"):
        assert docs[i]["text"].startswith("בבית המשפט העליון") and docs[i]["header_parsed"]
        assert "העותק כפוף" not in docs[i]["text"] and "www." not in docs[i]["text"]
    parsed = sum(d["header_parsed"] for d in docs.values()) / len(docs)
    assert parsed > 0.9
    assert docs["doc000000"]["source_url"].startswith("https://supremedecisions.court.gov.il/Home/Download?path=HebrewVerdicts")


def test_chunks(built):
    rows = _chunks(built)
    ids = [r["chunk_id"] for r in rows]
    assert len(ids) == len(set(ids))
    by_doc = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], []).append(r)
    for doc_rows in by_doc.values():
        doc_rows.sort(key=lambda r: r["chunk_index"])
        assert doc_rows[0]["section"] == "header" and doc_rows[0]["chunk_index"] == 0
        assert [r["chunk_index"] for r in doc_rows] == list(range(doc_rows[0]["n_chunks"]))
        assert all(r["chunk_id"] == f"{r['chunk_id'].split(':')[0]}:{r['chunk_index']}" for r in doc_rows)
    body = [r for r in rows if r["section"] != "header"]
    assert max(r["token_count"] for r in body) <= 180
    assert any(r["is_holding"] and "אשר על כן" in r["text"] for r in rows)
    assert any('ע"א 6821/93' in r["citations"] for r in rows)
    assert all(r["context_prefix"].startswith("[") for r in rows)


def test_chunk_ids_are_stable_across_runs(built):
    before = [r["chunk_id"] for r in _chunks(built)]
    run_chunk(built, force=True)
    assert [r["chunk_id"] for r in _chunks(built)] == before


def test_search_and_validate(built):
    s = Searcher(built)
    hits = s.bm25('ע"א 6821/93', 5)
    assert hits and any('ע"א 6821/93' in r["citations"] for r in s.rows([h[0] for h in hits], ["chunk_id", "citations"]).values())
    assert len(s.dense("עקרון המידתיות", 5)) == 5
    assert len(s.hybrid("חובת ההנמקה של רשות מנהלית", 5)) == 5
    result = run_validate(built)
    assert result["max_date"] < "2022-01-01"
    report = (paths_for(built).reports / "validation_report.md").read_text(encoding="utf-8")
    assert "Sanity queries" in report and "hybrid" in report


def test_rerun_resumes_without_redoing(built, capsys):
    run_filter(built)
    run_clean(built)
    run_chunk(built)
    run_embed(built)
    out = capsys.readouterr().out
    assert "done already" in out and "embed: shard" not in out


def test_sample_n_keeps_the_first_n(cfg):
    make_dataset(dataset_path(cfg), n=300)
    cfg["sample_n"] = 40
    result = run_filter(cfg)
    assert result["kept"] == 40
    assert paths_for(cfg).work.name == "sample_40"
