"""The Legal tab's corpus path (legal/corpus_retrieval.py): bulk-corpus hits become the
RetrievalResult the pipeline runs on."""

from docslides.config import get_config
from docslides.legal import corpus_retrieval

LAW = 'חוק החוזים (חלק כללי), תשל"ג-1973'


def _hit(chunk_id, text, distance, score=None, sources=("q0",), section="14", category="laws"):
    meta = {"record_id": "law:contracts", "title": LAW, "section_number": section, "status": "in_force",
            "effective_ymd": 19730601, "breadcrumb": f"{LAW} > סעיף {section}", "part_index": 1, "part_count": 1}
    return {"id": chunk_id, "category": category, "distance": distance, "meta": meta, "text": text,
            "sources": set(sources), **({"score": score} if score is not None else {})}


def test_parts_of_a_provision_share_a_source_and_corpus_metadata_maps_over():
    hits = [_hit("law:contracts@1973:14#p1", "חלק א", 0.2, 0.9, sources=("l0",)),
            _hit("law:contracts@1973:14#p2", "חלק ב", 0.3, 0.8)]

    result = corpus_retrieval.to_retrieval_result(hits)

    grouped = result.by_source_id()
    assert list(grouped) == ["law:contracts@1973:14"]
    meta = grouped["law:contracts@1973:14"][0].metadata
    assert (meta.law_name, meta.section_number, meta.effective_date_start) == (LAW, "14", "1973-06-01")
    assert (meta.status, meta.source_type, meta.source_origin) == ("current", "statute", "knesset")
    assert [c.via for c in result.chunks] == ["section_lookup", "search"]
    assert result.best_rerank_score == 0.9 and not result.low_relevance
    assert result.bundle_verification == "unsigned_corpus"


def test_evidence_is_cut_to_the_token_budget_best_first_keeping_the_best(monkeypatch):
    monkeypatch.setattr(get_config().legal.retrieval, "max_evidence_tokens", 5)
    hits = [_hit("a:1", "מילה " * 40, 0.2, 0.2, section="1"), _hit("a:2", "מילה", 0.3, 0.1, section="2")]

    result = corpus_retrieval.to_retrieval_result(hits)

    assert [c.chunk_id for c in result.chunks] == ["a:1"]  # over budget alone, but the best always stays
    assert result.trimmed_chunk_ids == ["a:2"]
    assert result.low_relevance  # best reranker score under min_rerank_score


def test_no_hits_is_thin_coverage_and_rulings_map_to_the_court():
    assert corpus_retrieval.to_retrieval_result([]).low_relevance
    meta = corpus_retrieval.chunk_metadata("sc:1:judgment#p3", "supreme_court",
                                           {"title": "ע\"א 1/20", "case_number": "1/20", "decision_ymd": 20200102})
    assert (meta.source_id, meta.source_type, meta.source_origin) == ("sc:1:judgment", "ruling", "court_gov_il")
    assert meta.section_number == "1/20" and meta.effective_date_start == "2020-01-02"


def test_a_plan_that_fails_leaves_retrieval_on_the_question_alone():
    import asyncio

    class Broken:
        async def complete_json(self, *_, **__):
            raise RuntimeError("server down")

    assert asyncio.run(corpus_retrieval.plan_issues(Broken(), "שאלה")) == []
