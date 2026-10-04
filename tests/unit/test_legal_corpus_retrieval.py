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


def test_out_of_scope_records_are_dropped_and_repealed_law_needs_its_name_and_year():
    def meta(title, status="in_force"):
        return {"title": title, "status": status}

    assert not corpus_retrieval.in_scope(meta("צו בדבר הוראות ביטחון [נוסח משולב] (יהודה והשומרון) (מס׳ 1651), התש״ע–2009"), [])
    assert not corpus_retrieval.in_scope(meta("צו בדבר הגנה על עדים (יהודה ושומרון) (מס׳ 2025), התשפ״א–2021"), [])
    assert not corpus_retrieval.in_scope(meta("החוק הפלילי הירדני, חוק מס׳ 16 לשנת 1960"), [])
    assert not corpus_retrieval.in_scope(meta("הצעת תנועת החירות לחוקת יסוד למדינת ישראל"), [])
    assert corpus_retrieval.in_scope(meta("חוק להסדרת ההתיישבות ביהודה והשומרון, התשע״ז–2017"), [])  # a Knesset law

    old = meta("תקנות סדר הדין האזרחי, התשמ״ד–1984", status="repealed")
    assert not corpus_retrieval.in_scope(old, [])
    assert not corpus_retrieval.in_scope(old, ["תקנות סדר הדין האזרחי"])  # the 2018 regulations share the name
    assert not corpus_retrieval.in_scope(old, ["תקנות סדר הדין האזרחי, התשע״ט-2018"])
    assert corpus_retrieval.in_scope(old, ['תקנות סדר הדין האזרחי, התשמ"ד-1984'])


def test_a_plan_that_fails_leaves_retrieval_on_the_question_alone():
    import asyncio

    class Broken:
        async def complete_json(self, *_, **__):
            raise RuntimeError("server down")

    assert asyncio.run(corpus_retrieval.plan_issues(Broken(), "שאלה")) == []


def test_a_named_law_matches_itself_before_laws_that_share_its_name():
    def hit(title):
        return {"meta": {"title": title}}

    sale, apartments = hit("חוק המכר, תשכ״ח–1968"), hit("חוק המכר (דירות), תשל״ג–1973")
    assert corpus_retrieval.named_law_hits("חוק המכר, 1973", [apartments, sale]) == [sale]
    # with no exact match, a law sharing the name still counts
    assert corpus_retrieval.named_law_hits("חוק המכר (דירות)", [apartments]) == [apartments]
    assert corpus_retrieval.named_law_hits("חוק הירושה", [sale]) == []


def test_looked_up_sections_get_no_reserved_slot_but_each_issue_can():
    def hit(name, *sources):
        return {"id": name, "sources": set(sources)}

    # ranked best-first by the reranker; "guess" came only from the plan's section lookup
    ranked = [hit("q0-best", "q0"), hit("q0-next", "q0"), hit("issue2", "q2"), hit("guess", "l0"), hit("issue1", "k1")]
    assert [h["id"] for h in corpus_retrieval.pick_hits(ranked, 2, 2)] == ["q0-best", "q0-next"]
    assert [h["id"] for h in corpus_retrieval.pick_hits(ranked, 2, 3, per_issue_slot=True)] == ["issue1", "issue2", "q0-best"]


class _Reranker:
    def __init__(self, fail_on=()):
        self.fail_on, self.loaded = fail_on, []

    def __call__(self, model_name, device=None):
        self.loaded.append(device)
        return None if device in self.fail_on else object()

    def cache_clear(self):
        pass


def _warm_up(monkeypatch, free_gpus, short=(), reranker=None):
    from docslides.legal import retrieval as legal_retrieval

    cfg = get_config().legal.retrieval
    monkeypatch.setattr(cfg, "device", "cuda")
    monkeypatch.setattr(cfg, "reranker_device", None)
    monkeypatch.setattr(cfg, "corpus_lexical", False)
    embedded = []
    monkeypatch.setattr(corpus_retrieval, "embed_texts", lambda model, texts, device=None: embedded.append(device))
    gpus = iter(free_gpus)
    monkeypatch.setattr(corpus_retrieval, "_roomiest_gpu", lambda: next(gpus))
    monkeypatch.setattr(corpus_retrieval, "_gpu_short_of_headroom", lambda d: "full" if d in short else None)
    reranker = reranker or _Reranker()
    monkeypatch.setattr(legal_retrieval, "_reranker", reranker)
    return corpus_retrieval.warm_up_retrieval(), cfg, embedded, reranker


def test_the_embedder_and_the_reranker_each_take_the_gpu_with_room(monkeypatch):
    # gemma4:31b on two T4s left 3.4-4.2 GB on each (1 Oct): room for both models in fp16, one per GPU.
    placed, cfg, embedded, reranker = _warm_up(monkeypatch, ["cuda:1", "cuda:0"])
    assert placed == "cuda:1 + cuda:0" and (cfg.device, cfg.reranker_device) == ("cuda:1", "cuda:0")
    assert embedded == ["cuda:1"] and reranker.loaded == ["cuda:0"]


def test_a_gpu_without_headroom_sends_only_that_model_to_the_cpu(monkeypatch):
    _, cfg, embedded, reranker = _warm_up(monkeypatch, ["cuda:1", "cuda:0"], short={"cuda:0"})
    assert (cfg.device, cfg.reranker_device) == ("cuda:1", "cpu") and reranker.loaded == ["cuda:0", "cpu"]

    _, cfg, embedded, reranker = _warm_up(monkeypatch, ["cuda:1", "cuda:1"], reranker=_Reranker({"cuda:1"}))
    assert (cfg.device, cfg.reranker_device) == ("cuda:1", "cpu")

    _, cfg, embedded, _ = _warm_up(monkeypatch, ["cuda:1", "cuda:0"], short={"cuda:1"})
    assert (cfg.device, cfg.reranker_device) == ("cpu", "cuda:0") and embedded == ["cuda:1", "cpu"]


def test_an_oom_mid_run_moves_both_models_to_the_cpu(monkeypatch):
    from docslides.legal.retrieval import cpu_on_gpu_oom

    cfg = get_config().legal.retrieval
    monkeypatch.setattr(cfg, "device", "cuda:1")
    monkeypatch.setattr(cfg, "reranker_device", "cuda:0")
    calls = []

    @cpu_on_gpu_oom
    def search():
        calls.append((cfg.device, cfg.reranker_device))
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return "ok"

    assert search() == "ok" and calls == [("cuda:1", "cuda:0"), ("cpu", "cpu")]
