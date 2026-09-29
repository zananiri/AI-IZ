"""Retrieval for the Legal tab over the bulk corpus (legal.corpus.vectordb_dir -- the Chroma
store scripts/legal_data/vectorize.py builds from the legal_txt/ JSONL, one collection per
category) instead of the signed index of drop-in law PDFs (legal/retrieval.py). Selected by
legal.retrieval.source: "corpus".

Planned retrieval, as the bulk eval runs it (scripts/legal_data/eval_run.py):

  1. the orchestrator lists the issues the question raises and the law + sections it believes
     govern each (`plan_issues`, thinking off);
  2. `retrieve_planned` searches on the question and on each "law + issue", looks every named
     section up directly, keeps at most MAX_PER_SECTION chunks per section and reranks the pool
     with legal.retrieval.reranker_model;
  3. `to_retrieval_result` turns the hits into the RetrievalResult the pipeline runs on:
     chunks grouped into provisions by source_id, trimmed best-first to
     legal.retrieval.max_evidence_tokens, and the thin-coverage flag set from the reranker score
     (or, without a reranker, the embedding distance).

The corpus isn't signed, so nothing is checked against the bundle, and it holds no amendment
index: those parts of the pipeline stay empty on this path.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from docslides.cleaning.tokens import count_tokens
from docslides.config import get_config
from docslides.legal import amendments
from docslides.legal.models import ChunkMetadata
from docslides.legal.retrieval import RetrievalResult, RetrievedLegalChunk
from docslides.legal_data.hebrew import normalize_for_embedding
from docslides.llm.client import ChatMessage, LLMCallSite, SamplingParams
from docslides.llm.schemas import EvalIssue, EvalRetrievalPlan
from docslides.logging_setup import get_logger
from docslides.rag.embedding import embed_texts

logger = get_logger(__name__)

PLAN_PROMPT = """List the distinct legal issues this question about ISRAELI law raises (at most \
{max_issues}) and, for each, the Israeli statute or regulation that governs it -- its full official \
Hebrew name -- with the section numbers you believe apply. Prefer the primary statute (a חוק or \
פקודה) over regulations, unless the question is specifically about a regulation. If you are not \
sure of a section number, leave sections empty rather than guess. Write each issue as a short \
Hebrew phrase in the statute's own terms (at most ten words) -- it is searched for in the law's text.

<question>{question}</question>"""

# 512 cut the JSON off mid-plan for 1 of 100 questions with Qwen and 4 with Gemma 12B (29 Sept
# eval review) -- each then retrieved on the question alone.
PLAN_MAX_TOKENS = 1024
FETCH_K = 24  # dense candidates per query per category, before dedupe and reranking
RERANK_POOL = 40  # candidates the cross-encoder scores
LOOKUP_K = 5  # hits per direct section lookup (filtered to the section number, then to the law)
MAX_PER_SECTION = 2  # chunks one section may contribute (a long section splits into parts)
MAX_LOOKUP_SLOTS = 3  # context slots reserved for sections the plan named
LEXICAL_K = 24  # BM25 candidates for the question, per category (legal/corpus_lexical.py)
LEXICAL_ISSUE_K = 8  # ... and for each "law + issue" of the plan
OUT_OF_SCOPE_MARGIN = 2  # over-fetch factor, so hits dropped by in_scope don't thin the pool

_SECTION_RE = re.compile(r"\d{1,4}[א-ת]{0,3}\d{0,3}")
_NIQQUD_RE = re.compile(r"[֑-ׇ]")
_YEAR_RE = re.compile(r"(?:^|\s)ה?תש[א-ת]{1,3}(?=\s|$)|\d{4}")
_GREGORIAN_YEAR_RE = re.compile(r"(?<!\d)(?:18|19|20)\d\d(?!\d)")
_PART_SUFFIX_RE = re.compile(r"#p\d+")

_SOURCE_TYPES = {"laws": "statute", "procedural_rules": "regulation", "supreme_court": "ruling"}
_SOURCE_ORIGINS = {"laws": "knesset", "procedural_rules": "reshumot", "supreme_court": "court_gov_il"}


def law_key(name: str) -> str:
    """A law's name reduced for matching: text before the first comma (the year), no niqqud,
    quotes, dashes or brackets, and no leading ה on words -- so "חוק הכַּשרוּת המשפטית, תשכ״ב–1962"
    and "חוק הכשרות המשפטית והאפוטרופסות, התשכ"ב-1962" compare equal on their shared part."""
    name = _NIQQUD_RE.sub("", (name or "").replace("־", " ").split(",")[0])
    name = re.sub(r"[\"'״׳`()\-–—\s]+", " ", name)
    name = _YEAR_RE.sub(" ", name)
    return " ".join(w[1:] if w.startswith("ה") and len(w) > 3 else w for w in name.split())


def same_law(named: str, title: str) -> bool:
    a, b = law_key(named), law_key(title)
    return bool(a) and bool(b) and (a == b or a in b or b in a)


@lru_cache(maxsize=4)
def _exclusion(patterns: tuple[str, ...]) -> re.Pattern | None:
    return re.compile("|".join(f"(?:{p})" for p in patterns)) if patterns else None


def in_scope(meta: dict, named_laws: list[str]) -> bool:
    """Whether a corpus record may be retrieved at all: not a title legal.corpus.exclude_title_patterns
    rules out (West Bank military orders, the Jordanian criminal law, drafts), and -- with
    repealed_only_when_named -- not repealed, unless the plan names that very law: its name and
    its year ("תקנות סדר הדין האזרחי, התשמ"ד-1984"). The name alone doesn't do: the 1984 Civil
    Procedure Regulations share it with the 2018 ones in force."""
    corpus_cfg = get_config().legal.corpus
    title = meta.get("title") or ""
    exclusion = _exclusion(tuple(corpus_cfg.exclude_title_patterns))
    if exclusion is not None and exclusion.search(title):
        return False
    if corpus_cfg.repealed_only_when_named and meta.get("status") == "repealed":
        years = set(_GREGORIAN_YEAR_RE.findall(title))
        return any(same_law(name, title) and years & set(_GREGORIAN_YEAR_RE.findall(name)) for name in named_laws)
    return True


def corpus_stats() -> dict | None:
    """Total chunks/records and the build date of the installed bulk corpus (legal.corpus.vectordb_dir),
    read from _build_info.json (written by scripts/legal_data/vectorize.py) -- no embedding model or
    Chroma connection needed. None if the corpus isn't installed."""
    corpus_cfg = get_config().legal.corpus
    info_path = Path(corpus_cfg.vectordb_dir) / "_build_info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    categories = info.get("categories", {})
    return {
        "chunks": sum(c.get("chunks", 0) for c in categories.values()),
        "records": sum(c.get("records", 0) for c in categories.values()),
        "categories": {name: c.get("chunks", 0) for name, c in categories.items()},
        "built_at": info.get("built_at"),
    }


def section_numbers(sections: list[str]) -> list[str]:
    """"סעיף 14(ד)" -> "14", "25א" -> "25א": the base numbers a plan names, as the corpus
    stores them in section_number."""
    out: list[str] = []
    for raw in sections:
        match = _SECTION_RE.search(str(raw))
        if match and match.group(0) not in out:
            out.append(match.group(0))
    return out


async def plan_issues(qwen, question: str, max_issues: int = 3) -> list[EvalIssue]:
    """The orchestrator's own list of issues and governing laws (thinking off: a short,
    structured call). An empty list on failure, which leaves retrieval on the question alone."""
    try:
        plan = await qwen.complete_json(
            [ChatMessage("user", PLAN_PROMPT.format(max_issues=max_issues, question=question))],
            LLMCallSite("legal_retrieval_plan"), schema=EvalRetrievalPlan,
            sampling=SamplingParams(temperature=0.0, max_tokens=PLAN_MAX_TOKENS), enable_thinking=False,
        )
    except Exception as exc:  # noqa: BLE001 -- retrieval still works without a plan
        logger.warning("legal_retrieval_plan_failed", error=f"{type(exc).__name__}: {exc}")
        return []
    return [i for i in plan.issues if i.law.strip()][:max_issues]


def _hits_from(result: dict, category: str, source: str) -> list[dict]:
    return [
        {"id": chunk_id, "category": category, "distance": distance, "meta": meta, "text": document, "sources": {source}}
        for chunk_id, distance, meta, document in zip(
            (result.get("ids") or [[]])[0], (result.get("distances") or [[]])[0],
            (result.get("metadatas") or [[]])[0], (result.get("documents") or [[]])[0])
    ]


def _lexical_hits(collection, category: str, ranked: list[tuple[str, float]], source: str) -> list[dict]:
    """BM25 results as hits, in BM25 order: text and metadata from the Chroma collection (the
    lexical copy is folded, and the model must read the text as embedded). Distance 1.0 -- no
    embedding distance -- so a chunk dense search also found keeps its own."""
    if not ranked:
        return []
    found = collection.collection.get(ids=[chunk_id for chunk_id, _ in ranked], include=["metadatas", "documents"])
    by_id = {chunk_id: (meta or {}, document or "")
             for chunk_id, meta, document in zip(found.get("ids") or [], found.get("metadatas") or [],
                                                 found.get("documents") or [])}
    return [{"id": chunk_id, "category": category, "distance": 1.0, "meta": by_id[chunk_id][0],
             "text": by_id[chunk_id][1], "sources": {source}}
            for chunk_id, _ in ranked if chunk_id in by_id]


def retrieve_planned(query_text: str, issues: list[EvalIssue], categories: list[str], top_k: int,
                     per_issue_slot: bool = False) -> list[dict]:
    """Retrieval steered by the answering model's own reading of the question:

    1. dense search on the question itself and on each "law + issue" the plan names, and (with
       legal.retrieval.corpus_lexical) BM25 on the same texts -- exact terms of art;
    2. a direct lookup of every section the plan names (filtered to that section number,
       then to the named law) -- the step that surfaces, say, section 15 of the Contracts Law
       for a fact pattern about misrepresentation, which the question's wording alone doesn't;
    3. records `in_scope` rules out are dropped (West Bank orders, drafts, repealed law the plan
       doesn't name), the rest fused by reciprocal rank;
    4. at most MAX_PER_SECTION chunks per section, then the cross-encoder
       (legal.retrieval.reranker_model) reranks the pool against the question and the issues;
    5. up to MAX_LOOKUP_SLOTS slots go to the looked-up sections (and, for issue spotting, one
       to each issue's best hit) whatever their rerank score; the rest go by rerank score.

    Falls back to plain embedding order when the reranker can't be loaded."""
    from docslides.legal.corpus_lexical import lexical_index
    from docslides.legal.retrieval import _reranker
    from docslides.legal_data.corpus_index import CorpusCollection

    legal_cfg = get_config().legal
    corpus_cfg = legal_cfg.corpus
    device = legal_cfg.retrieval.device
    named_laws = [i.law for i in issues]
    queries = [query_text] + [f"{i.law} {i.issue}" for i in issues]
    lookups = [(i.law, number) for i in issues for number in section_numbers(i.sections)[:3]]
    texts = queries + [f"{law} סעיף {number}" for law, number in lookups]
    vectors = embed_texts(
        legal_cfg.retrieval.embedding_model,
        [normalize_for_embedding(t, corpus_cfg.fold_final_letters_for_embedding) for t in texts],
        device=device,
    )

    pool: dict[str, dict] = {}

    def add_all(hits: list[dict], limit: int) -> None:
        kept_hits = [h for h in hits if in_scope(h["meta"], named_laws)][:limit]
        for rank, hit in enumerate(kept_hits):
            kept = pool.setdefault(hit["id"], {**hit, "sources": set(), "fused": 0.0})
            kept["sources"] |= hit["sources"]
            kept["fused"] += 1.0 / (60 + rank)
            kept["distance"] = min(kept["distance"], hit["distance"])

    for category in categories:
        path = Path(corpus_cfg.vectordb_dir) / category
        if not path.exists():
            continue
        collection = CorpusCollection(path, f"{corpus_cfg.collection_prefix}_{category}")
        if collection.count() == 0:
            continue
        for qi, vector in enumerate(vectors[: len(queries)]):
            result = collection.query(vector, FETCH_K * OUT_OF_SCOPE_MARGIN)
            add_all(_hits_from(result, category, f"q{qi}"), FETCH_K)
        for li, (law, number) in enumerate(lookups):
            vector = vectors[len(queries) + li]
            result = collection.query(vector, LOOKUP_K, where={"section_number": number})
            add_all([h for h in _hits_from(result, category, f"l{li}") if same_law(law, h["meta"].get("title", ""))],
                    LOOKUP_K)
        lexical = lexical_index(corpus_cfg.vectordb_dir, category) if legal_cfg.retrieval.corpus_lexical else None
        if lexical is not None:
            for qi, text in enumerate(queries):
                k = LEXICAL_K if qi == 0 else LEXICAL_ISSUE_K
                ranked = lexical.search(text, k * OUT_OF_SCOPE_MARGIN)
                add_all(_lexical_hits(collection, category, ranked, f"k{qi}"), k)

    # One section may split into several chunks; keep its best MAX_PER_SECTION.
    per_section: dict[tuple, int] = {}
    candidates: list[dict] = []
    for hit in sorted(pool.values(), key=lambda h: (-h["fused"], h["distance"])):
        key = (hit["meta"].get("title"), hit["meta"].get("section_number") or hit["id"])
        if per_section.get(key, 0) >= MAX_PER_SECTION:
            continue
        per_section[key] = per_section.get(key, 0) + 1
        candidates.append(hit)

    lookup_hits = [h for h in candidates if any(s.startswith("l") for s in h["sources"])]
    head = candidates[:RERANK_POOL]
    head += [h for h in lookup_hits if h not in head]
    reranker = _reranker(legal_cfg.retrieval.reranker_model, device) if legal_cfg.retrieval.reranker_model else None
    if reranker is not None and head:
        rerank_query = query_text + "\n" + "; ".join(f"{i.issue} – {i.law}" for i in issues)
        for hit, score in zip(head, reranker.predict([(rerank_query, h["text"]) for h in head])):
            hit["score"] = float(score)
        ranked = sorted(head, key=lambda h: -h["score"])
    else:
        ranked = head

    picked: list[dict] = []
    for hit in sorted(lookup_hits, key=lambda h: -h.get("score", h["fused"]))[:MAX_LOOKUP_SLOTS]:
        picked.append(hit)
    if per_issue_slot:
        for qi in range(1, len(queries)):
            best = next((h for h in ranked if {f"q{qi}", f"k{qi}"} & h["sources"]), None)
            if best is not None and best not in picked and len(picked) < top_k:
                picked.append(best)
    for hit in ranked:
        if len(picked) >= top_k:
            break
        if hit not in picked:
            picked.append(hit)
    return picked[:top_k]


def _ymd_to_iso(value) -> str:
    digits = str(value or "")
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}" if re.fullmatch(r"\d{8}", digits) else ""


def chunk_metadata(chunk_id: str, category: str, meta: dict) -> ChunkMetadata:
    """A corpus chunk's scalar metadata (legal_data/corpus_chunking.py) as the pipeline's
    ChunkMetadata. The parts of one split provision ("<source_id>#p2") share a source_id."""
    title = meta.get("title") or ""
    section = meta.get("section_number") or meta.get("case_number") or ""
    source_id = _PART_SUFFIX_RE.sub("", chunk_id)
    return ChunkMetadata(
        chunk_id=chunk_id,
        source_id=source_id,
        section_key=f"{meta.get('record_id') or ''}:{section}",
        law_id=str(meta.get("record_id") or meta.get("law_id") or ""),
        law_name=title,
        chapter=None,
        part=None,
        section_number=str(section),
        subsection_number=meta.get("subsection") or None,
        breadcrumb=meta.get("breadcrumb") or title,
        effective_date_start=_ymd_to_iso(meta.get("effective_ymd") or meta.get("decision_ymd")),
        effective_date_end=None,
        status="repealed" if meta.get("status") == "repealed" else "current",
        source_type=_SOURCE_TYPES.get(category, "statute"),
        source_origin=_SOURCE_ORIGINS.get(category, "knesset"),
        ingestion_date="",
        language="he",
        part_index=int(meta.get("part_index") or 1),
        part_count=int(meta.get("part_count") or 1),
        law_key=amendments.law_key(title),
        inserted_section=meta.get("inserted_section") or "",
    )


def to_retrieval_result(hits: list[dict]) -> RetrievalResult:
    """Planned-retrieval hits (best first) as the pipeline's RetrievalResult: every part of a
    provision stays with it, and whole provisions go in best-first until
    legal.retrieval.max_evidence_tokens is spent (the best one is always kept)."""
    cfg = get_config().legal.retrieval
    by_source: dict[str, list[RetrievedLegalChunk]] = {}
    for hit in hits:
        metadata = chunk_metadata(hit["id"], hit["category"], hit["meta"])
        via = "section_lookup" if any(s.startswith("l") for s in hit.get("sources", ())) else "search"
        by_source.setdefault(metadata.source_id, []).append(
            RetrievedLegalChunk(hit["id"], hit["text"], metadata, hit["distance"], via, hit.get("score"))
        )

    chunks: list[RetrievedLegalChunk] = []
    trimmed: list[str] = []
    spent = 0
    for parts in by_source.values():
        size = sum(count_tokens(p.text) for p in parts)
        if chunks and cfg.max_evidence_tokens is not None and spent + size > cfg.max_evidence_tokens:
            trimmed += [p.chunk_id for p in parts]
            continue
        chunks += parts
        spent += size

    distances = [c.distance for c in chunks if c.distance is not None]
    scores = [c.score for c in chunks if c.score is not None]
    best_distance = min(distances) if distances else None
    best_score = max(scores) if scores else None
    if not chunks:
        low_relevance = True
    elif best_score is not None:
        low_relevance = best_score < cfg.min_rerank_score
    else:
        low_relevance = best_distance is None or best_distance > cfg.low_relevance_distance
    return RetrievalResult(
        chunks=chunks,
        low_relevance=low_relevance,
        best_distance=best_distance,
        bundle_verification="unsigned_corpus",
        trimmed_chunk_ids=trimmed,
        best_rerank_score=best_score,
    )


def _roomiest_gpu() -> str:
    """"cuda:<i>" for the GPU with the most free memory -- with the LLM split over two GPUs, the
    one it left more room on -- or plain "cuda"."""
    try:
        import torch

        count = torch.cuda.device_count()
        if count > 1:
            free = [torch.cuda.mem_get_info(i)[0] for i in range(count)]
            return f"cuda:{max(range(count), key=free.__getitem__)}"
    except Exception:  # noqa: BLE001, S110 -- let the library pick
        pass
    return "cuda"


def warm_up_retrieval() -> str:
    """Loads the embedder, the reranker and the BM25 indexes before the first question, so it isn't
    charged for them -- and moves retrieval to the CPU (legal.retrieval.device) when the GPU hasn't
    room for it beside the LLM. Raises if the reranker can't load even there: an eval must not run
    on embedding distance alone. Returns the device retrieval now runs on ("auto" = library default)."""
    from docslides.legal.corpus_lexical import lexical_index
    from docslides.legal.retrieval import _reranker

    legal_cfg = get_config().legal
    retrieval = legal_cfg.retrieval
    if retrieval.device == "cuda":
        retrieval.device = _roomiest_gpu()

    def load(device: str | None) -> None:
        embed_texts(retrieval.embedding_model, ["warm up"], device=device)
        if retrieval.reranker_model and _reranker(retrieval.reranker_model, device) is None:
            _reranker.cache_clear()  # don't keep the failed load
            raise RuntimeError(f"reranker did not load on {device or 'the default device'}")

    try:
        load(retrieval.device)
    except Exception as exc:
        if retrieval.device == "cpu":
            raise
        logger.warning("legal_retrieval_moved_to_cpu", device=retrieval.device, error=f"{type(exc).__name__}: {exc}")
        retrieval.device = "cpu"
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001, S110 -- nothing to free without torch/CUDA
            pass
        load("cpu")
    if retrieval.corpus_lexical:
        for category in legal_cfg.corpus.categories:
            lexical_index(legal_cfg.corpus.vectordb_dir, category)
    return retrieval.device or "auto"


def retrieve_corpus(query: str, issues: list[EvalIssue]) -> RetrievalResult:
    corpus_cfg = get_config().legal.corpus
    hits = retrieve_planned(query, issues, corpus_cfg.categories, corpus_cfg.top_k)
    return to_retrieval_result(hits)
