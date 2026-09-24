"""What legal/retrieval.retrieve() lets through to the model -- the relevance
margin, the sibling margin and the token budget -- against an in-memory
stand-in for the Chroma collection, plus the Hebrew quote normalization in
prompts.format_evidence."""

import math

import numpy as np
import pytest

from docslides.config import get_config
from docslides.legal import prompts, retrieval
from docslides.legal.models import ChunkMetadata, content_hash
from docslides.legal.retrieval import RetrievedLegalChunk


def _vector(distance: float) -> np.ndarray:
    """A unit vector at cosine distance `distance` from the query vector [1, 0]."""
    angle = math.acos(1.0 - distance)
    return np.array([math.cos(angle), math.sin(angle)])


def _meta(chunk_id: str, source_id: str, section: str, part: int = 1, parts: int = 1, refs=()) -> dict:
    return ChunkMetadata(
        chunk_id=chunk_id, source_id=source_id, section_key=f"v:{section}", law_id="law", law_name="חוק",
        chapter=None, part=None, section_number=section, subsection_number=None, breadcrumb="חוק",
        effective_date_start="2026-01-01", effective_date_end=None, status="current", source_type="statute",
        source_origin="knesset", ingestion_date="2026-09-23", language="he", part_index=part, part_count=parts,
        cross_references=list(refs),
    ).to_chroma()


class FakeCollection:
    """The slice of the Chroma collection API that retrieve() uses."""

    def __init__(self, rows):  # (chunk_id, text, flat metadata, distance from the query)
        self.rows = [(cid, text, meta, _vector(distance)) for cid, text, meta, distance in rows]

    def count(self):
        return len(self.rows)

    def query(self, query_embeddings, n_results, include):
        query = np.asarray(query_embeddings[0])
        ranked = sorted(self.rows, key=lambda r: 1 - float(query @ r[3]))[:n_results]
        return {
            "ids": [[r[0] for r in ranked]],
            "documents": [[r[1] for r in ranked]],
            "metadatas": [[r[2] for r in ranked]],
            "distances": [[1 - float(query @ r[3]) for r in ranked]],
            "embeddings": [np.array([r[3] for r in ranked])],
        }

    def get(self, where=None, ids=None, include=None):
        hits = self.rows
        if where:
            ((field, condition),) = where.items()
            hits = [r for r in hits if r[2][field] in condition["$in"]]
        if ids is not None:
            hits = [r for r in hits if r[0] in ids]
        return {
            "ids": [r[0] for r in hits],
            "documents": [r[1] for r in hits],
            "metadatas": [r[2] for r in hits],
            "embeddings": np.array([r[3] for r in hits]),
        }


@pytest.fixture
def load_index(monkeypatch):
    cfg = get_config().legal.retrieval
    settings = {
        "top_k": 6, "fetch_k": 24, "mmr_lambda": 1.0, "low_relevance_distance": 0.55, "min_relevant_chunks": 2,
        "max_cross_refs": 4, "relevance_margin": 0.08, "sibling_margin": 0.12, "max_evidence_tokens": None,
        # the distance path by default; tests below switch each of these on
        "keyword_search": False, "section_lookup_max": 0, "reranker_model": None,
        "rerank_margin": 0.6, "rerank_floor": 0.1, "min_rerank_score": 0.35, "rerank_candidates": 16,
    }
    for name, value in settings.items():
        monkeypatch.setattr(cfg, name, value)
    monkeypatch.setattr(retrieval, "_embed", lambda texts: [[1.0, 0.0] for _ in texts])
    monkeypatch.setattr(retrieval, "count_tokens", lambda text: len(text.split()))

    def load(rows):
        collection = FakeCollection(rows)
        entries = {cid: {"sha256": content_hash(text, meta)} for cid, text, meta, _ in rows}
        monkeypatch.setattr(retrieval, "_get_collection", lambda: collection)
        monkeypatch.setattr(retrieval.bundle, "verified_entries", lambda: (entries, "signed"))
        retrieval._derived_cache.clear()
        return cfg

    return load


def test_hits_far_from_the_best_one_are_cut_but_still_count_toward_coverage(load_index, monkeypatch):
    cfg = load_index([
        ("v:1", "סעיף אחד קובע מועד", _meta("v:1", "v:1", "1"), 0.30),
        ("v:2", "סעיף שני קובע סכום", _meta("v:2", "v:2", "2"), 0.35),
        ("v:3", "סעיף שלישי עוסק בעניין אחר", _meta("v:3", "v:3", "3"), 0.45),
    ])
    result = retrieval.retrieve("שאלה")

    assert [c.chunk_id for c in result.chunks] == ["v:1", "v:2"]
    assert result.trimmed_chunk_ids == ["v:3"]
    assert result.best_distance == pytest.approx(0.30)
    assert result.low_relevance is False  # judged on all three hits, before the cut

    monkeypatch.setattr(cfg, "relevance_margin", None)
    assert [c.chunk_id for c in retrieval.retrieve("שאלה").chunks] == ["v:1", "v:2", "v:3"]


def test_other_parts_of_a_split_provision_come_in_only_when_close(load_index, monkeypatch):
    cfg = load_index([
        ("v:5#p1", "חלק ראשון של הסעיף", _meta("v:5#p1", "v:5", "5", 1, 3), 0.30),
        ("v:5#p2", "חלק שני של הסעיף", _meta("v:5#p2", "v:5", "5", 2, 3), 0.38),
        ("v:5#p3", "חלק שלישי של הסעיף", _meta("v:5#p3", "v:5", "5", 3, 3), 0.50),
    ])
    monkeypatch.setattr(cfg, "top_k", 1)
    result = retrieval.retrieve("שאלה")

    assert [(c.chunk_id, c.via) for c in result.chunks] == [("v:5#p1", "search"), ("v:5#p2", "sibling_part")]
    assert result.chunks[1].distance == pytest.approx(0.38)
    assert result.trimmed_chunk_ids == ["v:5#p3"]

    monkeypatch.setattr(cfg, "sibling_margin", None)
    assert [c.chunk_id for c in retrieval.retrieve("שאלה").chunks] == ["v:5#p1", "v:5#p2", "v:5#p3"]


def _referencing_index(load_index):
    nine_words = "מילה " * 9
    return load_index([
        ("v:1", "סעיף אחד בכפוף לסעיף תשע", _meta("v:1", "v:1", "1", refs=["v:9"]), 0.30),
        ("v:9(א)", nine_words + "א", _meta("v:9(א)", "v:9(א)", "9"), 0.70),
        ("v:9(ב)", nine_words + "ב", _meta("v:9(ב)", "v:9(ב)", "9"), 0.50),
        ("v:9(ג)", nine_words + "ג", _meta("v:9(ג)", "v:9(ג)", "9"), 0.60),
    ])


def test_cross_references_fill_the_token_budget_closest_first(load_index, monkeypatch):
    cfg = _referencing_index(load_index)
    monkeypatch.setattr(cfg, "max_evidence_tokens", 5 + 12)  # the hit (5 words) + one 10-word chunk
    result = retrieval.retrieve("שאלה")

    assert [(c.chunk_id, c.via) for c in result.chunks] == [("v:1", "search"), ("v:9(ב)", "cross_reference")]
    assert result.trimmed_chunk_ids == ["v:9(ג)", "v:9(א)"]


def test_the_best_hit_is_kept_even_over_budget(load_index, monkeypatch):
    cfg = _referencing_index(load_index)
    monkeypatch.setattr(cfg, "max_evidence_tokens", 1)
    result = retrieval.retrieve("שאלה")

    assert [c.chunk_id for c in result.chunks] == ["v:1"]
    assert set(result.trimmed_chunk_ids) == {"v:9(א)", "v:9(ב)", "v:9(ג)"}


def test_evidence_text_uses_gershayim_for_hebrew_sources():
    meta = ChunkMetadata.from_chroma(_meta("v:1", "v:1", "1"))
    text = "חוק > סעיף 1\n\nמאז יום כ\"ב בתשרי התשפ\"ד"
    chunk = RetrievedLegalChunk("v:1", text, meta, 0.3, "search")
    assert "מאז יום כ״ב בתשרי התשפ״ד" in prompts.format_evidence({"v:1": [chunk]})


# --- section lookup, keyword fusion, reranker ------------------------------------------


class FakeReranker:
    """Scores a (question, chunk) pair by the number after "score=" in the chunk text."""

    def predict(self, pairs):
        return [float(text.split("score=")[1].split()[0]) for _, text in pairs]


def test_a_section_the_question_names_is_fetched_first(load_index, monkeypatch):
    cfg = load_index([
        ("v:1", "סעיף אחד קרוב מאוד לשאלה", _meta("v:1", "v:1", "1"), 0.20),
        ("v:6(6)>132א(ה)", "הוראות סעיפים קטנים לא יחולו על אות אזעקה",
         {**_meta("v:6(6)>132א(ה)", "v:6(6)>132א(ה)", "6"), "inserted_section": "132א(ה)"}, 0.60),
    ])
    monkeypatch.setattr(cfg, "section_lookup_max", 2)
    monkeypatch.setattr(cfg, "relevance_margin", 0.05)
    result = retrieval.retrieve("האם הסמכויות לפי סעיף 132א חלות על אזעקות?")

    assert [(c.chunk_id, c.via) for c in result.chunks] == [("v:6(6)>132א(ה)", "section_lookup"), ("v:1", "search")]
    assert retrieval.named_sections("בסעיף 62(ג) ובסעיף 116 יז 10") == [("62", "ג"), ("116יז10", None)]


def test_keyword_search_brings_in_what_embeddings_rank_low(load_index, monkeypatch):
    cfg = load_index([("v:1", "סעיף כללי על הבחירות", _meta("v:1", "v:1", "1"), 0.30)]
                     + [(f"v:{n}", f"הוראה אחרת מספר {n}", _meta(f"v:{n}", f"v:{n}", str(n)), 0.31) for n in range(2, 9)]
                     + [("v:12", "המפרסם נחזות עמוקה יצרף גילוי", _meta("v:12", "v:12", "12"), 0.90)])
    monkeypatch.setattr(cfg, "fetch_k", 3)
    monkeypatch.setattr(cfg, "relevance_margin", None)
    assert "v:12" not in [c.chunk_id for c in retrieval.retrieve("מה דין נחזות עמוקה?").chunks]

    monkeypatch.setattr(cfg, "keyword_search", True)
    assert "v:12" in [c.chunk_id for c in retrieval.retrieve("מה דין בנחזות עמוקה?").chunks]


def test_reranker_orders_hits_and_sets_the_thin_coverage_flag(load_index, monkeypatch):
    cfg = load_index([
        ("v:1", "קרוב אבל לא עונה score=0.05", _meta("v:1", "v:1", "1"), 0.20),
        ("v:2", "עונה ישירות score=0.95", _meta("v:2", "v:2", "2"), 0.40),
        ("v:3", "עונה חלקית score=0.50", _meta("v:3", "v:3", "3"), 0.45),
    ])
    monkeypatch.setattr(cfg, "reranker_model", "fake")
    monkeypatch.setattr(retrieval, "_reranker", lambda name: FakeReranker())
    result = retrieval.retrieve("שאלה")

    assert [(c.chunk_id, c.score) for c in result.chunks] == [("v:2", 0.95), ("v:3", 0.50)]  # v:1 under the floor
    assert result.best_rerank_score == 0.95 and result.low_relevance is False

    cfg = load_index([("v:1", "לא עונה score=0.10", _meta("v:1", "v:1", "1"), 0.20)])
    assert retrieval.retrieve("שאלה").low_relevance is True  # best score under min_rerank_score
