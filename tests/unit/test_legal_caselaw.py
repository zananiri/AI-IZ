"""The case-law evidence source (legal/caselaw.py) over an index built by caselaw/israeli_caselaw_ingest
from its synthetic dataset, with the hashing encoder standing in for bge-m3."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("lancedb")
pytest.importorskip("bm25s")

CASELAW = Path(__file__).resolve().parents[2] / "caselaw"
sys.path.insert(0, str(CASELAW))
sys.path.insert(0, str(CASELAW / "tests"))

from docslides.config import get_config  # noqa: E402
from docslides.legal import caselaw  # noqa: E402


@pytest.fixture(scope="module")
def index_root(tmp_path_factory):
    from israeli_caselaw_ingest.chunk import run_chunk
    from israeli_caselaw_ingest.clean import run_clean
    from israeli_caselaw_ingest.config import load_config
    from israeli_caselaw_ingest.download import dataset_path
    from israeli_caselaw_ingest.embed import run_embed
    from israeli_caselaw_ingest.filter import run_filter
    from israeli_caselaw_ingest.search import run_bm25
    from synthetic import make_dataset

    root = tmp_path_factory.mktemp("caselaw")
    cfg = load_config(root=str(root), overrides={
        "chunk": {"tokenizer": "regex", "min_tokens": 120, "max_tokens": 180, "overlap_tokens": 30},
        "embed": {"model": "hashing-test:64", "shard_size": 200, "save_model": False},
        "read": {"batch_size": 64}})
    make_dataset(dataset_path(cfg), n=200)
    for stage in (run_filter, run_clean, run_chunk, run_bm25):
        stage(cfg)
    run_embed(cfg, yes=True)
    return root


@pytest.fixture
def configured(index_root, monkeypatch):
    from israeli_caselaw_ingest.encoders import HashEncoder

    legal = get_config().legal
    monkeypatch.setattr(legal.corpus, "caselaw_dir", str(index_root))
    monkeypatch.setattr(legal.retrieval, "embedding_model", "hashing-test:64")
    monkeypatch.setattr(legal.retrieval, "reranker_model", None)
    encoder = HashEncoder(64)
    monkeypatch.setattr("docslides.rag.embedding.embed_texts",
                        lambda model, texts, **kw: encoder.encode(list(texts)).astype("float32").tolist())
    caselaw.open_index.cache_clear()
    yield legal
    caselaw.open_index.cache_clear()


def test_off_by_default():
    assert get_config().legal.corpus.caselaw_dir is None
    assert caselaw.search_caselaw("שאלה", []) == []


def test_search_finds_the_cited_judgment_one_excerpt_per_judgment(configured):
    hits = caselaw.search_caselaw('ע"א 6821/93 בנק המזרחי', ["עקרון המידתיות"])
    assert 0 < len(hits) <= configured.corpus.caselaw_top_k
    assert len({h["doc_id"] for h in hits}) == len(hits)
    assert all(h["section"] != "header" for h in hits)
    assert any('6821/93' in h["text"] for h in hits)
    assert sum(caselaw.count_tokens(h["text"]) for h in hits) <= configured.corpus.caselaw_max_tokens


def test_render_and_record(configured):
    hits = caselaw.search_caselaw("חובת ההנמקה של רשות מנהלית", [])
    block = caselaw.render_caselaw(hits)
    assert block.startswith("<case_law>") and block.endswith("</case_law>") and "[C1]" in block
    record = caselaw.caselaw_record(hits)
    assert record[0]["source_url"].startswith("https://supremedecisions.court.gov.il/")
    assert caselaw.render_caselaw([]) == ""


def test_missing_index_answers_without_case_law(monkeypatch, tmp_path):
    monkeypatch.setattr(get_config().legal.corpus, "caselaw_dir", str(tmp_path / "nowhere"))
    caselaw.open_index.cache_clear()
    assert caselaw.search_caselaw("שאלה", []) == []
    caselaw.open_index.cache_clear()
