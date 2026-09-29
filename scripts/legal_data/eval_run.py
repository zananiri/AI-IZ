#!/usr/bin/env python3
"""Answer, then judge, legal_txt/Evals/israeli_legal_eval/questions.jsonl (the 500-question
set) against the bulk legal corpus (data/legal_corpus_vectordb, scripts/legal_data/vectorize.py)
-- a separate, much broader index from the Legal tab's own signed bundle (src/docslides/legal/),
which this eval set is too broad to run against (five curated laws vs. the eval's eleven areas
of law).

    # 1) answer, always with retrieval over the corpus (there is no model-alone mode).
    #    Skips ids already in --out, so re-running after an interruption resumes.
    python scripts/legal_data/eval_run.py answer \
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

Per question: (1) the answering model lists the issues and the laws/sections it believes
govern them (legal_eval_plan, thinking off); (2) retrieve_planned searches on the question and on
each issue (dense + BM25), looks the named sections up directly, drops out-of-scope records (West
Bank orders, drafts, repealed law), keeps at most two chunks per section and reranks with
legal.retrieval.reranker_model; (3) the answer call (ANSWER_SYSTEM); (4) an empty answer is
retried without thinking, one containing other scripts is rewritten into Hebrew (legal_eval_rewrite,
up to two passes), a כן/לא opener the instructions didn't ask for is dropped, and a rule_conclusion
answer with no label gets the one its explanation supports (legal_eval_label_check; a label the
model wrote is kept). The plan and any repairs
are recorded on the answers.jsonl row.

Reasoning: every LLM call is written in full -- prompt, reasoning ("thinking"), output, timing --
to <logging.llm_trace_dir>/<date>.jsonl (llm/trace.py), with job_id set to the question id, as
long as DOCSLIDES_CONFIG's logging.llm_trace_dir is set (see notebooks/kaggle_legal_eval_bulk500.ipynb).
scripts/llm_trace_report.py renders that file to Markdown, one section per question.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from docslides.config import get_config
from docslides.legal.corpus_retrieval import (  # shared with the Legal tab's corpus path
    PLAN_MAX_TOKENS,
    PLAN_PROMPT,
    in_scope,
    retrieve_planned,
    warm_up_retrieval,
)
from docslides.legal.evaluation import get_judge_client
from docslides.legal_data.hebrew import normalize_for_embedding
from docslides.llm import trace
from docslides.llm.client import (
    ChatMessage,
    LLMCallSite,
    SamplingParams,
    get_legal_orchestrator_client,
)
from docslides.llm.schemas import BulkEvalJudgement, EvalIssue, EvalRetrievalPlan
from docslides.rag.embedding import embed_texts

ANSWER_SYSTEM = """You are a legal assistant answering questions about ISRAELI law, in Hebrew.

<context> holds excerpts retrieved from an index of Israeli legislation. Some are relevant and some \
are not.
- Base the answer on the excerpts that actually govern the question. Ignore excerpts about other \
subjects, and never cite a law just because it appears in <context>.
- If no excerpt governs the question, answer from the core Israeli statutes you are certain of and \
cite them by name and section.
- Cite only the few provisions the answer relies on, usually one to three. Copy law names and \
section numbers exactly as they appear in <context>; when a provision's text is in <context>, trust \
it over your memory. Never invent a section, subsection, number or date.
- When <context> shows that a section was changed or added by an amending law (תיקון), cite that \
amending law as well.
- Follow the question's <instructions> exactly. Open with כן or לא only when the <instructions> ask \
for a yes/no answer; then that first word answers the question exactly as asked, and the explanation \
must agree with it. Every other answer opens directly with the rule or the analysis.
- If the question assumes a law, amendment, section or fact that does not exist or is wrong, say so \
plainly and give the correct rule. If a request asks for help deceiving, threatening, hiding assets \
or otherwise evading the law, refuse and briefly name the lawful alternative. If the answer depends \
on facts that were not given, say which facts are needed.
- Write only in Hebrew. Do not use words from any other language or script, and do not refer to \
the excerpts by number or mention <context>."""

ISSUE_SPOTTING_NOTE = """This is an issue-spotting question: list every distinct legal issue the \
facts raise, across all areas of law (civil, contracts, torts, labour, property, consumer, privacy, \
criminal, procedure), one per line, in the form: הסוגיה – החוק והסעיף."""

REWRITE_PROMPT = """The answer below contains words in other languages or scripts ({words}). \
Rewrite it entirely in Hebrew. Keep its content, structure, law names and section numbers exactly \
as they are. Return only the rewritten answer.

<answer>
{answer}
</answer>"""

# The 29 Sept review found Gemma 12B opening the answer with a label its own reasoning contradicts
# (IL-016, IL-214, IL-219). A second read of the explanation alone -- label removed, so it can't
# anchor -- decides which answer the reasoning supports.
LABEL_CHECK_PROMPT = """Below is a yes/no question about Israeli law and the explanation part of an \
answer to it. Read only the explanation: does it lead to "yes" or to "no" as the answer to the \
question exactly as asked? Judge what the explanation says, not whether it is correct.

<question>{question}</question>

<explanation>
{explanation}
</explanation>"""

# Categories whose instructions don't ask for a yes/no answer: a bare כן/לא opener there (55 of 66
# of Gemma 12B's such answers) is dropped.
NO_YES_NO_OPENER = {"rule_recall", "rule_application", "interpretation", "citation_grounding",
                    "temporal_amendment", "issue_spotting"}
_YES_NO_OPENER_RE = re.compile(r"^\s*(?:\*\*)?\s*(כן|לא)\s*(?:\*\*)?\s*(?:[,.:;!\-–—]+|\n)\s*(?:\*\*)?\s*")
MAX_REWRITE_PASSES = 2

# Output budgets, reasoning included. The 26 Sept bulk500 run (16k context, never more than 6.1k
# of it used) stopped 8 of 500 answers at 3072 and 11 of 431 judge calls at 2048 while still
# thinking, 7 answers came back empty; the context window was never the limit.
ANSWER_MAX_TOKENS = 6144
ISSUE_SPOTTING_TOP_K = 16
JUDGE_MAX_TOKENS = 4096  # same as eval_cases.py


def eval_sampling(max_tokens: int, thinking: bool) -> SamplingParams:
    """Greedy without thinking (reproducible). With thinking, Qwen3's recommended sampling
    (temperature 0.6, top_p 0.95, top_k 20) with a fixed seed: its model card warns that greedy
    decoding in thinking mode degrades answers and loops -- the 26 Sept trace has reasoning that
    repeats "Wait, no..." until the budget runs out (IL-283)."""
    if thinking:
        return SamplingParams(temperature=0.6, top_p=0.95, top_k=20, seed=0, max_tokens=max_tokens)
    return SamplingParams(temperature=0.0, max_tokens=max_tokens)


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
    category with the single best hits isn't starved by one that returns weaker ones. Records
    corpus_retrieval.in_scope rules out (West Bank orders, drafts, repealed law) are skipped."""
    from docslides.legal_data.corpus_index import CorpusCollection

    legal_cfg = get_config().legal
    corpus_cfg = legal_cfg.corpus
    vector = embed_texts(
        legal_cfg.retrieval.embedding_model,  # same embedding model the corpus was built with -- see vectorize.py
        [normalize_for_embedding(query_text, corpus_cfg.fold_final_letters_for_embedding)],
        device=legal_cfg.retrieval.device,
    )[0]
    hits: list[dict] = []
    for category in categories:
        path = Path(corpus_cfg.vectordb_dir) / category
        if not path.exists():
            continue
        collection = CorpusCollection(path, f"{corpus_cfg.collection_prefix}_{category}")
        if collection.count() == 0:
            continue
        result = collection.query(vector, top_k * 2)
        kept = 0
        for distance, meta, document in zip(result["distances"][0], result["metadatas"][0], result["documents"][0]):
            if kept < top_k and in_scope(meta, []):
                hits.append({"category": category, "distance": distance, "meta": meta, "text": document})
                kept += 1
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


_FOREIGN_RE = re.compile(r"[\u0400-\u04ff\u0600-\u06ff\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+|[A-Za-z]{4,}")


def foreign_words(text: str) -> list[str]:
    return _FOREIGN_RE.findall(text)


def split_yes_no_opener(text: str) -> tuple[str | None, str]:
    """("כן" / "לא", the rest) for an answer opening with a bare label -- "לא.", "**כן**,", "כן\\n" --
    else (None, text). "לא ניתן ..." is a sentence, not a label, and stays whole."""
    match = _YES_NO_OPENER_RE.match(text)
    if not match or not text[match.end():].strip():
        return None, text
    return match.group(1), text[match.end():]


class LabelVerdict(BaseModel):
    answer: Literal["yes", "no", "unclear"]


async def check_label(qwen, question: str, text: str) -> tuple[str, str | None]:
    """A rule_conclusion answer that opens with a label: one the model wrote is kept, a missing one
    is added from what the explanation (read on its own) supports. Returns (answer, repair or None).

    Swapping a label the model wrote was dropped: over the two 29 Sept Gemma 27B runs the checker
    fixed 2 labels (IL-219, IL-164) and broke 3 (IL-212, IL-409 twice) -- it misreads negated
    questions ("can he file without approval?" / "cannot file without approval" -> "yes")."""
    label, explanation = split_yes_no_opener(text)
    first_words = re.findall(r"[א-ת]+", text[:40])  # how score.py reads the label
    if label is not None or first_words[:1] in (["כן"], ["לא"]):
        return text, None
    try:
        verdict = await qwen.complete_json(
            [ChatMessage("user", LABEL_CHECK_PROMPT.format(question=question, explanation=explanation.strip()))],
            LLMCallSite("legal_eval_label_check"), schema=LabelVerdict,
            sampling=SamplingParams(temperature=0.0, max_tokens=256), enable_thinking=False,
        )
    except Exception:  # noqa: BLE001 -- keep the answer as written
        return text, None
    if verdict.answer == "unclear":
        return text, None
    wanted = "כן" if verdict.answer == "yes" else "לא"
    return f"{wanted}. {explanation.lstrip()}", "label_added"


async def plan_issues(qwen, q: dict, max_issues: int) -> list[EvalIssue]:
    """The answering model's own list of issues and governing laws (thinking off: a short,
    structured call). An empty list on failure, which leaves retrieval on the question alone."""
    try:
        plan = await qwen.complete_json(
            [ChatMessage("user", PLAN_PROMPT.format(max_issues=max_issues, question=_question_text(q)))],
            LLMCallSite("legal_eval_plan"), schema=EvalRetrievalPlan,
            sampling=SamplingParams(temperature=0.0, max_tokens=PLAN_MAX_TOKENS), enable_thinking=False,
        )
    except Exception as exc:  # noqa: BLE001 -- retrieval still works without a plan
        _log(f"{q['id']}: plan failed ({type(exc).__name__}: {exc}); retrieving on the question alone")
        return []
    return [i for i in plan.issues if i.law.strip()][:max_issues]


async def answer_one(qwen, q: dict, categories: list[str], top_k: int, thinking: bool = True,
                     max_tokens: int = ANSWER_MAX_TOKENS) -> dict:
    qtext = _question_text(q)
    issues: list[EvalIssue] = []
    repairs: list[str] = []
    with trace.collect(job_id=q["id"]):
        spotting = q.get("category") == "issue_spotting"
        issues = await plan_issues(qwen, q, max_issues=6 if spotting else 3)
        hits = retrieve_planned(qtext, issues, categories, ISSUE_SPOTTING_TOP_K if spotting else top_k,
                                per_issue_slot=spotting)
        user = f"<instructions>{q['instructions']}</instructions>\n\n<question>{qtext}</question>\n\n" \
               f"<context>\n{render_context(hits)}\n</context>"
        if spotting:
            user += f"\n\n{ISSUE_SPOTTING_NOTE}"
        messages = [ChatMessage("system", ANSWER_SYSTEM), ChatMessage("user", user)]

        text = (await qwen.complete_text(
            messages, LLMCallSite("legal_eval_baseline"), sampling=eval_sampling(max_tokens, thinking),
            enable_thinking=thinking,  # the point of this run: capture how the model reasons, for fine-tuning
        )).strip()
        sampling = eval_sampling(max_tokens, False)
        if not text and thinking:
            # The reasoning used the whole budget and no answer was written: answer without it.
            repairs.append("empty_answer_retry")
            text = (await qwen.complete_text(messages, LLMCallSite("legal_eval_baseline"), sampling=sampling,
                                             enable_thinking=False)).strip()
        # Up to two passes: the 29 Sept review found one rewrite leaving script in 26 Qwen answers.
        for _ in range(MAX_REWRITE_PASSES):
            stray = foreign_words(text)
            if not stray:
                break
            repairs.append("hebrew_rewrite")
            rewritten = (await qwen.complete_text(
                [ChatMessage("user", REWRITE_PROMPT.format(words=", ".join(dict.fromkeys(stray)), answer=text))],
                LLMCallSite("legal_eval_rewrite"), sampling=sampling, enable_thinking=False,
            )).strip()
            if not rewritten or len(foreign_words(rewritten)) >= len(stray):
                break
            text = rewritten
        if q.get("category") in NO_YES_NO_OPENER:
            label, rest = split_yes_no_opener(text)
            if label:
                repairs.append("stray_yes_no_opener")
                text = rest.lstrip()
        elif q.get("category") == "rule_conclusion" and text:
            text, repair = await check_label(qwen, qtext, text)
            if repair:
                repairs.append(repair)
    return {
        "id": q["id"], "answer": text,
        "retrieved": [{"category": h["category"], "distance": round(h["distance"], 4),
                        "score": round(h["score"], 4) if "score" in h else None,
                        "title": h["meta"].get("title"), "section": h["meta"].get("section_number")} for h in hits],
        "plan": [i.model_dump() for i in issues],
        "repairs": repairs,
    }


async def cmd_answer(a) -> None:
    questions = _load_jsonl(Path(a.questions))
    if a.limit:
        questions = questions[: a.limit]
    out = Path(a.out)
    done = _load_done(out)
    categories = a.categories.split(",")
    qwen = get_legal_orchestrator_client()
    started = time.monotonic()
    device = warm_up_retrieval()
    _log(f"retrieval ready on {device} ({round(time.monotonic() - started)}s)")
    _log(f"{len(questions)} questions, {len(done)} already answered")
    for q in questions:
        if q["id"] in done:
            continue
        started = time.monotonic()
        try:
            row = await answer_one(qwen, q, categories, a.top_k, thinking=a.thinking, max_tokens=a.max_tokens)
        except Exception as exc:  # noqa: BLE001 -- record and move on; --out is resumable
            row = {"id": q["id"], "answer": "", "error": f"{type(exc).__name__}: {exc}"}
        _append(out, row)
        _log(f"{q['id']} ({q['category']}): {round(time.monotonic() - started)}s"
             f"{' -- ' + row['error'] if row.get('error') else ''}")


# score.py's prompt (unmodified, from the eval set) says "Grade ONLY against the reference
# material". gemma3:27b read that as "anything not in the reference is wrong" and scored correct
# answers 0 for accurate extra detail (27 Sept run, ~23 of 58 zeros). This says what it means.
JUDGE_SYSTEM = """You grade answers against a reference answer. How to read the grading instructions:
- "Grade only against the reference" means: check the reference's key points against the answer. \
It does not mean that everything outside the reference is wrong.
- Accurate extra detail, additional correct citations, or naming the law and section a rule comes \
from are NOT errors and never lower correctness. Lower it only for a key point that is missing or \
wrong, or for a statement that contradicts the reference.
- A citation to a different but real provision that states the same rule is not a hallucination. \
hallucination is for an invented law, section, number or date, or one that plainly does not exist.
- Judge meaning, not wording. The answer is usually in Hebrew."""


async def judge_one(qwen, item: dict, thinking: bool = True, max_tokens: int = JUDGE_MAX_TOKENS) -> dict:
    with trace.collect(job_id=item["id"]):
        try:
            verdict = await qwen.complete_json(
                [ChatMessage("system", JUDGE_SYSTEM), ChatMessage("user", item["prompt"])],
                LLMCallSite("legal_eval_judge"),
                schema=BulkEvalJudgement,
                sampling=eval_sampling(max_tokens, thinking),
                enable_thinking=thinking,  # the judge's own reasoning matters just as much for calibration
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
        row = await judge_one(qwen, item, thinking=a.thinking, max_tokens=a.max_tokens)
        _append(out, row)
        _log(f"{item['id']}: correctness={row['correctness']} ({round(time.monotonic() - started)}s)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("answer")
    pa.add_argument("--questions", default="legal_txt/Evals/israeli_legal_eval/questions.jsonl")
    pa.add_argument("--mode", choices=["rag"], default="rag", help="accepted for older commands; rag is the only mode")
    pa.add_argument("--categories", default="laws,procedural_rules")
    pa.add_argument("--top-k", type=int, default=12)
    pa.add_argument("--out", required=True)
    pa.add_argument("--limit", type=int, help="answer only the first N questions (smoke test)")
    pa.add_argument("--max-tokens", type=int, default=ANSWER_MAX_TOKENS,
                    help="output budget per answer, reasoning included (a model that always reasons needs more)")
    pa.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True,
                    help="reasoning on the answer call (--no-thinking for models without a thinking mode)")

    pj = sub.add_parser("judge")
    pj.add_argument("--requests", required=True, help="judge_requests.jsonl from score.py prepare")
    pj.add_argument("--out", required=True)
    pj.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    pj.add_argument("--max-tokens", type=int, default=JUDGE_MAX_TOKENS,
                    help="output budget per judgement, reasoning included")

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
