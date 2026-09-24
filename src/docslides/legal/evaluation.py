"""Before/after-RAG knowledge evaluation for the Legal tab
(scripts/eval_legal.py, eval sets under evals/legal/).

BEFORE RAG -- the orchestrator model alone, no retrieved context. The right
behaviour for almost every question is "I don't know"; each answer is
classified as
  * abstained               -- declined / didn't know (the correct baseline);
  * correct_without_context -- matched the gold answer anyway: either the
                               document leaked into training, or a lucky guess;
  * partial_without_context -- partly right;
  * hallucination           -- a confident wrong answer (or, in group C, any
                               specific value).
The baseline score is the share of questions answered WITHOUT hallucinating.

AFTER RAG -- the full Legal pipeline over the index. Groups A/B score three
parts, each 0-1, averaged per question:
  * retrieval -- share of the question's evidence groups found in one retrieved chunk;
  * answer    -- judge verdict vs. gold: correct 1, partially_correct 0.5, else 0;
  * citation  -- share of evidence groups found in the chunks the answer cited.
Group C (unanswerable) must still say the law doesn't state it: 1 if the
judge rates it correct/abstained, it fabricated no specifics, and no trap
pattern matched; else 0.

Answers are graded by an LLM judge against the gold answer. Deterministic
key-fact / trap regex matches are recorded next to each verdict so a
reviewer can spot judge errors.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from docslides.legal.chunking import normalize_hebrew_quotes
from docslides.llm.client import ChatMessage, LLMCallSite, QwenClient, SamplingParams
from docslides.llm.schemas import EvalContradiction, EvalJudgement

BASELINE_SYSTEM_PROMPT = "You are a legal assistant. Answer the user's question in the language it was asked in."

JUDGE_PROMPT = """\
You grade answers to questions about one specific Israeli law against a gold (reference) answer.

verdict:
- "correct": states all essential facts of the gold answer and contradicts none of them. Wording may differ; additional accurate context is fine.
- "partially_correct": some essential facts right, others missing or wrong.
- "incorrect": contradicts the gold answer, gets its key fact wrong (e.g. yes instead of no, a wrong number, values swapped), or answers a different question.
- "abstained": declines, says it does not know, or says the information is unavailable / not stated.
If the gold answer says the law does NOT state something, then saying it is not stated / not in the document is "correct", and giving a specific value is "incorrect".
If the gold answer DOES give an answer, a refusal or "could not be answered / not found" reply is "abstained" -- never "correct", whatever the reason given.
Section numbers: an amending law changes another law's sections, so the same provision can be cited either by the amending law's section (e.g. "section 6(4)") or by the section it inserts or amends (e.g. "section 116יז10(ה)"). Treat both as the same provision; never mark an answer wrong only for citing the other number.

fabricated_specifics: true if the answer asserts a specific number, date, amount or rule as the answer that the gold answer does not support.

Judge meaning, not wording. The answer may be in Hebrew. Explain briefly."""


@dataclass
class EvalQuestion:
    id: str
    group: str
    question: str
    gold: str
    key_facts: list[str] = field(default_factory=list)
    evidence: list[list[str]] = field(default_factory=list)
    traps: list[str] = field(default_factory=list)
    note: str = ""


def load_eval_set(path: Path) -> tuple[dict, list[EvalQuestion]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    known = EvalQuestion.__dataclass_fields__
    return data, [EvalQuestion(**{k: v for k, v in q.items() if k in known}) for q in data["questions"]]


def _matches(pattern: str, text: str) -> bool:
    return re.search(pattern, text) is not None


def fact_coverage(answer: str, key_facts: list[str]) -> float | None:
    if not key_facts:
        return None
    return sum(_matches(p, answer) for p in key_facts) / len(key_facts)


def trap_hits(answer: str, traps: list[str]) -> list[str]:
    return [m.group(0) for p in traps for m in [re.search(p, answer)] if m]


def evidence_coverage(texts: list[str], groups: list[list[str]]) -> float | None:
    """Share of evidence groups whose patterns all co-occur in at least one text."""
    if not groups:
        return None
    hit = sum(any(all(_matches(p, t) for p in group) for t in texts) for group in groups)
    return hit / len(groups)


def _judge_input(q: EvalQuestion, answer: str) -> str:
    # ״ for ASCII quotes: the judge quotes the texts in its JSON explanation, and an unescaped
    # 'ש"ח' there cuts it off (the same failure legal/chunking.normalize_hebrew_quotes prevents).
    return normalize_hebrew_quotes(
        f"Question:\n{q.question}\n\nGold answer:\n{q.gold}\n\nAnswer to grade:\n{answer}"
    )


CONTRADICTION_PROMPT = """\
You check one thing: does the answer CONTRADICT the gold answer? Contradict means it states something that conflicts
with a fact in the gold answer -- a different number, date or amount, yes instead of no, a condition reversed, a rule
attributed to the wrong body. Extra accurate detail, longer wording, quoting the law, or citing a different section
number for the same provision is NOT a contradiction. Omitting something is not a contradiction either.
The texts may be in Hebrew."""


async def contradiction(qwen: QwenClient, q: EvalQuestion, answer: str) -> EvalContradiction:
    return await qwen.complete_json(
        [ChatMessage("system", CONTRADICTION_PROMPT), ChatMessage("user", _judge_input(q, answer))],
        LLMCallSite("legal_eval_judge"),
        schema=EvalContradiction,
        sampling=SamplingParams(temperature=0.0, max_tokens=512),
    )


def get_judge_client() -> QwenClient:
    """The grading model: DOCSLIDES_LEGAL_JUDGE_MODEL (e.g. qwen3:14b, same server
    as the orchestrator) when set, else the orchestrator model itself."""
    import os

    from docslides.config import get_config
    from docslides.llm.client import get_legal_orchestrator_client

    model = os.environ.get("DOCSLIDES_LEGAL_JUDGE_MODEL")
    if not model:
        return get_legal_orchestrator_client()
    orchestrator = get_config().legal.orchestrator
    return QwenClient(orchestrator.model_copy(update={"model": model}))


def needs_contradiction_check(q: EvalQuestion, verdict: str, facts: float | None, traps: list[str]) -> bool:
    """An 8B judge grades long, accurate answers down for extra detail. When an
    answer holds every key fact of an answerable question and trips no trap, a
    "wrong" or "partly right" verdict must survive a narrower question: does it
    actually contradict the gold answer?"""
    return q.group != "C" and verdict in ("incorrect", "partially_correct") and facts == 1.0 and not traps


async def judge(qwen: QwenClient, q: EvalQuestion, answer: str) -> EvalJudgement:
    return await qwen.complete_json(
        [
            ChatMessage("system", JUDGE_PROMPT),
            ChatMessage("user", _judge_input(q, answer)),
        ],
        LLMCallSite("legal_eval_judge"),
        schema=EvalJudgement,
        sampling=SamplingParams(temperature=0.0, max_tokens=512),
    )


async def ask_baseline(qwen: QwenClient, q: EvalQuestion) -> str:
    return await qwen.complete_text(
        [ChatMessage("system", BASELINE_SYSTEM_PROMPT), ChatMessage("user", q.question)],
        LLMCallSite("legal_eval_baseline"),
        sampling=SamplingParams(temperature=0.0, max_tokens=1024),
    )


def classify_baseline(q: EvalQuestion, verdict: str, fabricated: bool, traps: list[str]) -> str:
    if q.group == "C":
        if traps or fabricated or verdict in ("incorrect", "partially_correct"):
            return "hallucination"
        return "abstained"
    return {
        "abstained": "abstained",
        "correct": "correct_without_context",
        "partially_correct": "partial_without_context",
    }.get(verdict, "hallucination")


_ANSWER_POINTS = {"correct": 1.0, "partially_correct": 0.5}


def score_after(q: EvalQuestion, verdict: str, fabricated: bool, traps: list[str],
                retrieved_texts: list[str], cited_texts: list[str]) -> dict:
    if q.group == "C":
        passed = verdict in ("correct", "abstained") and not fabricated and not traps
        return {"score": 1.0 if passed else 0.0, "passed": passed, "retrieval": None, "answer": None, "citation": None}
    retrieval = evidence_coverage(retrieved_texts, q.evidence)
    citation = evidence_coverage(cited_texts, q.evidence)
    answer = _ANSWER_POINTS.get(verdict, 0.0)
    parts = [p for p in (retrieval, answer, citation) if p is not None]
    return {"score": sum(parts) / len(parts), "retrieval": retrieval, "answer": answer, "citation": citation}


def rescore_after(q: EvalQuestion, record: dict, verdict: str, fabricated: bool, traps: list[str]) -> dict:
    """score_after for a saved record: reuses its stored retrieval/citation
    coverage (the retrieved texts aren't saved), with a new verdict."""
    if q.group == "C":
        return score_after(q, verdict, fabricated, traps, [], [])
    answer = _ANSWER_POINTS.get(verdict, 0.0)
    parts = [p for p in (record.get("retrieval"), answer, record.get("citation")) if p is not None]
    return {"score": sum(parts) / len(parts), "retrieval": record.get("retrieval"), "answer": answer,
            "citation": record.get("citation")}


def summarize(questions: list[EvalQuestion], before: dict, after: dict) -> dict:
    groups = sorted({q.group for q in questions})
    summary: dict = {"before": {}, "after": {}}

    done_before = [q for q in questions if q.id in before]
    if done_before:
        labels = [before[q.id]["classification"] for q in done_before]
        summary["before"] = {
            "answered": len(done_before),
            "abstained": labels.count("abstained"),
            "correct_without_context": labels.count("correct_without_context"),
            "partial_without_context": labels.count("partial_without_context"),
            "hallucinations": labels.count("hallucination"),
            "score": 100 * (1 - labels.count("hallucination") / len(labels)),
        }

    done_after = [q for q in questions if q.id in after and "score" in after[q.id]]
    if done_after:
        by_group = {}
        for group in groups:
            items = [after[q.id] for q in done_after if q.group == group]
            if not items:
                continue
            row = {"questions": len(items), "score": 100 * sum(i["score"] for i in items) / len(items)}
            for part in ("retrieval", "answer", "citation"):
                values = [i[part] for i in items if i.get(part) is not None]
                if values:
                    row[part] = 100 * sum(values) / len(values)
            by_group[group] = row
        hallucinations = sum(
            1 for q in done_after
            if (q.group == "C" and not after[q.id].get("passed"))
            or (q.group != "C" and after[q.id]["verdict"] == "incorrect")
        )
        summary["after"] = {
            "answered": len(done_after),
            "by_group": by_group,
            "hallucinations": hallucinations,
            "score": 100 * sum(after[q.id]["score"] for q in done_after) / len(done_after),
        }
    return summary


def backfill_answer_texts(before: dict, after: dict) -> None:
    """Runs recorded before the fix saved answer text under "answer", which the
    after-RAG per-part score (also "answer") then overwrote. Moves surviving
    text to "answer_text", and recovers lost after-RAG text from the
    pipeline's audit log entry for that question."""
    for run in (before, after):
        for record in run.values():
            if "answer_text" not in record and isinstance(record.get("answer"), str):
                record["answer_text"] = record["answer"]
    for qid, record in after.items():
        if record.get("answer_text") or not record.get("audit_path"):
            continue
        path = Path(record["audit_path"])
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry.get("job_id") == f"eval-{qid}":
                from docslides.legal.citations import strip_citations

                record["answer_text"] = strip_citations(entry["output"]["answer_draft"])


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.0f}%"


def render_report(meta: dict, questions: list[EvalQuestion], before: dict, after: dict, summary: dict) -> str:
    lines = [f"# Legal knowledge eval: {meta['eval_set']}", ""]
    lines += [f"- {k}: {v}" for k, v in meta.items() if k != "eval_set"]
    lines.append("")

    b, a = summary.get("before") or {}, summary.get("after") or {}
    lines += ["## Scores", "", "| | Before RAG | After RAG |", "|---|---|---|"]
    lines.append(f"| Overall score | {_pct(b.get('score'))} (non-hallucination rate) | {_pct(a.get('score'))} |")
    lines.append(f"| Hallucinations | {b.get('hallucinations', '–')} / {b.get('answered', '–')} | "
                 f"{a.get('hallucinations', '–')} / {a.get('answered', '–')} |")
    if b:
        lines.append(f"| Abstained (\"I don't know\") | {b['abstained']} | |")
        lines.append(f"| Correct without context (leak or luck) | {b['correct_without_context']} | |")
    for group, row in (a.get("by_group") or {}).items():
        parts = ", ".join(f"{p} {_pct(row.get(p))}" for p in ("retrieval", "answer", "citation") if p in row)
        lines.append(f"| Group {group} | | {_pct(row['score'])}{' (' + parts + ')' if parts else ''} |")
    lines.append("")

    lines += ["## Per question", "",
              "| ID | Before RAG | After RAG: retrieved | answer | cited | score |",
              "|---|---|---|---|---|---|"]
    for q in questions:
        bq, aq = before.get(q.id, {}), after.get(q.id, {})
        if q.group == "C" and aq:
            after_cells = f"– | {aq.get('verdict', 'error')} | – | {'PASS' if aq.get('passed') else 'FAIL'}"
        elif aq:
            after_cells = (f"{_pct(_x100(aq.get('retrieval')))} | {aq.get('verdict', 'error')} | "
                           f"{_pct(_x100(aq.get('citation')))} | {_pct(_x100(aq.get('score')))}")
        else:
            after_cells = "– | – | – | –"
        lines.append(f"| {q.id} | {bq.get('classification', '–')} | {after_cells} |")
    lines.append("")

    lines.append("## Answers")
    for q in questions:
        lines += ["", f"### {q.id} ({q.group}) {q.question}", "", f"**Gold:** {q.gold}"]
        for label, run in (("Before RAG", before.get(q.id)), ("After RAG", after.get(q.id))):
            if not run:
                continue
            lines += ["", f"**{label}** ({run.get('classification') or run.get('verdict')}): {run.get('answer_text', '')}"]
            if run.get("judge_explanation"):
                lines.append(f"  - judge: {run['judge_explanation']}")
            if run.get("fact_coverage") is not None:
                lines.append(f"  - key facts matched: {_pct(_x100(run['fact_coverage']))}")
            if run.get("trap_hits"):
                lines.append(f"  - trap matches: {run['trap_hits']}")
            if run.get("error"):
                lines.append(f"  - error: {run['error']}")
    return "\n".join(lines) + "\n"


def _x100(value: float | None) -> float | None:
    return None if value is None else 100 * value
