#!/usr/bin/env python3
"""Answer, then judge, legal_txt/Evals/israeli_legal_eval/questions.jsonl (the 500-question
set) against the bulk legal corpus (data/legal_corpus_vectordb, scripts/legal_data/vectorize.py)
-- a separate, much broader index from the Legal tab's own signed bundle (src/docslides/legal/),
which this eval set is too broad to run against (five curated laws vs. the eval's eleven areas
of law).

    # 1) answer -- no_context (the model alone, no retrieval) and rag (retrieval over the corpus).
    #    Skips ids already in --out, so re-running after an interruption resumes.
    python scripts/legal_data/eval_run.py answer --mode no_context \
        --questions legal_txt/Evals/israeli_legal_eval/questions.jsonl \
        --out data/legal/eval/bulk500/answers_no_context.jsonl
    python scripts/legal_data/eval_run.py answer --mode rag \
        --questions legal_txt/Evals/israeli_legal_eval/questions.jsonl \
        --out data/legal/eval/bulk500/answers_rag.jsonl

    # 2) build judge requests (unmodified israeli_legal_eval/score.py):
    python legal_txt/Evals/israeli_legal_eval/score.py prepare \
        --questions legal_txt/Evals/israeli_legal_eval/questions.jsonl \
        --gold legal_txt/Evals/israeli_legal_eval/gold.jsonl \
        --answers data/legal/eval/bulk500/answers_rag.jsonl \
        --out data/legal/eval/bulk500/judge_requests_rag.jsonl

    # 3) judge (fills judge_requests.jsonl's prompts in, writes judged.jsonl):
    python scripts/legal_data/eval_run.py judge \
        --requests data/legal/eval/bulk500/judge_requests_rag.jsonl \
        --out data/legal/eval/bulk500/judged_rag.jsonl

    # 4) report (unmodified score.py):
    python legal_txt/Evals/israeli_legal_eval/score.py report \
        --gold legal_txt/Evals/israeli_legal_eval/gold.jsonl \
        --answers data/legal/eval/bulk500/answers_rag.jsonl \
        --judged data/legal/eval/bulk500/judged_rag.jsonl \
        --json-out data/legal/eval/bulk500/report_rag.json

Repeat all four for --mode no_context (skip retrieval questions there -- --mode no_context still
answers every item, including mcq/yesno ones score.py grades without a judge, for comparability).

Reasoning: every LLM call is written in full -- prompt, reasoning ("thinking"), output, timing --
to <logging.llm_trace_dir>/<date>.jsonl (llm/trace.py), with job_id set to the question id, as
long as DOCSLIDES_CONFIG's logging.llm_trace_dir is set (see notebooks/kaggle_legal_eval_bulk500.ipynb).
scripts/llm_trace_report.py renders that file to Markdown, one section per question.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from docslides.config import get_config
from docslides.legal.evaluation import get_judge_client
from docslides.legal_data.hebrew import normalize_for_embedding
from docslides.llm import trace
from docslides.llm.client import (
    ChatMessage,
    LLMCallSite,
    SamplingParams,
    get_legal_orchestrator_client,
)
from docslides.llm.schemas import BulkEvalJudgement
from docslides.rag.embedding import embed_texts

NO_CONTEXT_SYSTEM = "You are a legal assistant. Answer the user's question in the language it was asked in."

RAG_SYSTEM = """You are a legal assistant answering questions about ISRAELI law.

Answer using the excerpts under <context> below, which come from an index of Israeli legislation \
and regulations. Follow the question's own <instructions>. Cite the law and section your answer \
relies on. If <context> is empty, thin, or about something else, or if the question assumes a law, \
section, case or fact that does not exist, say so plainly rather than inventing an answer -- do not \
state a specific rule, number or date you cannot support from <context> or from law you are certain \
of. Answer in the language the question is written in."""

ANSWER_MAX_TOKENS = 1024
JUDGE_MAX_TOKENS = 1024


def _log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_done(out: Path) -> dict[str, dict]:
    return {r["id"]: r for r in _load_jsonl(out)} if out.exists() else {}


def _append(out: Path, row: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _question_text(q: dict) -> str:
    text = q["question"]
    if q.get("options"):
        text += "\n\n" + "\n".join(f"{letter}. {body}" for letter, body in q["options"].items())
    return text


def retrieve(query_text: str, categories: list[str], top_k: int) -> list[dict]:
    """Top `top_k` chunks (by distance, merged across `categories`) from the bulk corpus
    at config.legal.corpus.vectordb_dir. Each category is queried for `top_k` first, so a
    category with the single best hits isn't starved by one that returns weaker ones."""
    from docslides.legal_data.corpus_index import CorpusCollection

    legal_cfg = get_config().legal
    corpus_cfg = legal_cfg.corpus
    vector = embed_texts(
        legal_cfg.retrieval.embedding_model,  # same embedding model the corpus was built with -- see vectorize.py
        [normalize_for_embedding(query_text, corpus_cfg.fold_final_letters_for_embedding)],
    )[0]
    hits: list[dict] = []
    for category in categories:
        path = Path(corpus_cfg.vectordb_dir) / category
        if not path.exists():
            continue
        collection = CorpusCollection(path, f"{corpus_cfg.collection_prefix}_{category}")
        if collection.count() == 0:
            continue
        result = collection.query(vector, top_k)
        for distance, meta, document in zip(result["distances"][0], result["metadatas"][0], result["documents"][0]):
            hits.append({"category": category, "distance": distance, "meta": meta, "text": document})
    hits.sort(key=lambda h: h["distance"])
    return hits[:top_k]


def render_context(hits: list[dict]) -> str:
    if not hits:
        return "(no matching context found)"
    blocks = []
    for i, h in enumerate(hits, 1):
        meta = h["meta"]
        section = meta.get("section_number") or meta.get("case_number") or ""
        status = meta.get("status", "")
        blocks.append(f"[{i}] {meta.get('title', '')} {section} ({status})\n{h['text']}")
    return "\n\n".join(blocks)


async def answer_one(qwen, q: dict, mode: str, categories: list[str], top_k: int) -> dict:
    qtext = _question_text(q)
    if mode == "no_context":
        messages = [ChatMessage("system", NO_CONTEXT_SYSTEM), ChatMessage("user", f"{q['instructions']}\n\n{qtext}")]
        hits: list[dict] = []
    else:
        hits = retrieve(qtext, categories, top_k)
        user = f"<instructions>{q['instructions']}</instructions>\n\n<question>{qtext}</question>\n\n" \
               f"<context>\n{render_context(hits)}\n</context>"
        messages = [ChatMessage("system", RAG_SYSTEM), ChatMessage("user", user)]

    with trace.collect(job_id=q["id"]):
        text = await qwen.complete_text(
            messages, LLMCallSite("legal_eval_baseline"),
            sampling=SamplingParams(temperature=0.0, max_tokens=ANSWER_MAX_TOKENS),
            enable_thinking=True,  # the point of this run: capture how the model reasons, for fine-tuning
        )
    return {
        "id": q["id"], "answer": text.strip(),
        "retrieved": [{"category": h["category"], "distance": round(h["distance"], 4),
                        "title": h["meta"].get("title"), "section": h["meta"].get("section_number")} for h in hits],
    }


async def cmd_answer(a) -> None:
    questions = _load_jsonl(Path(a.questions))
    if a.limit:
        questions = questions[: a.limit]
    out = Path(a.out)
    done = _load_done(out)
    categories = a.categories.split(",")
    qwen = get_legal_orchestrator_client()
    _log(f"{len(questions)} questions, {len(done)} already answered, mode={a.mode}")
    for q in questions:
        if q["id"] in done:
            continue
        started = time.monotonic()
        try:
            row = await answer_one(qwen, q, a.mode, categories, a.top_k)
        except Exception as exc:  # noqa: BLE001 -- record and move on; --out is resumable
            row = {"id": q["id"], "answer": "", "error": f"{type(exc).__name__}: {exc}"}
        _append(out, row)
        _log(f"{q['id']} ({q['category']}): {round(time.monotonic() - started)}s"
             f"{' -- ' + row['error'] if row.get('error') else ''}")


async def judge_one(qwen, item: dict) -> dict:
    with trace.collect(job_id=item["id"]):
        try:
            verdict = await qwen.complete_json(
                [ChatMessage("user", item["prompt"])], LLMCallSite("legal_eval_judge"),
                schema=BulkEvalJudgement,
                sampling=SamplingParams(temperature=0.0, max_tokens=JUDGE_MAX_TOKENS),
                enable_thinking=True,  # the judge's own reasoning matters just as much for calibration
            )
        except Exception as exc:  # noqa: BLE001 -- record and move on; --out is resumable
            return {"id": item["id"], "correctness": 0, "grounding": "n/a", "hallucination": False,
                    "key_points_hit": [], "needs_review": True, "note": f"judge failed: {exc}"}
    return {"id": item["id"], **verdict.model_dump()}


async def cmd_judge(a) -> None:
    requests = _load_jsonl(Path(a.requests))
    out = Path(a.out)
    done = _load_done(out)
    qwen = get_judge_client()
    _log(f"{len(requests)} judge requests, {len(done)} already judged")
    for item in requests:
        if item["id"] in done:
            continue
        started = time.monotonic()
        row = await judge_one(qwen, item)
        _append(out, row)
        _log(f"{item['id']}: correctness={row['correctness']} ({round(time.monotonic() - started)}s)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("answer")
    pa.add_argument("--questions", default="legal_txt/Evals/israeli_legal_eval/questions.jsonl")
    pa.add_argument("--mode", choices=["no_context", "rag"], required=True)
    pa.add_argument("--categories", default="laws,procedural_rules")
    pa.add_argument("--top-k", type=int, default=8)
    pa.add_argument("--out", required=True)
    pa.add_argument("--limit", type=int, help="answer only the first N questions (smoke test)")

    pj = sub.add_parser("judge")
    pj.add_argument("--requests", required=True, help="judge_requests.jsonl from score.py prepare")
    pj.add_argument("--out", required=True)

    a = p.parse_args()
    from docslides.llm.client import aclose_all_clients

    async def run() -> None:
        try:
            await {"answer": cmd_answer, "judge": cmd_judge}[a.cmd](a)
        finally:
            await aclose_all_clients()

    asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
