#!/usr/bin/env python3
"""Multi-law knowledge test for the Legal tab, graded against a separate gold file.

    python scripts/eval_legal_gold.py --run-dir data/legal/eval/multi1                    # answer + grade
    python scripts/eval_legal_gold.py --run-dir data/legal/eval/multi1 --phase grade     # re-grade saved answers
    python scripts/eval_legal_gold.py --retrieval-only                                    # seconds, no model

Two phases, kept apart on purpose:
  * answer -- reads ONLY the questions file (evals/legal/multi_law_24_questions.json) and runs
    each question through the Legal pipeline. The gold file is never opened in this phase, so
    nothing from it can reach the model under test. Answers, citations and what retrieval
    sent are saved to answers.json.
  * grade  -- opens the gold file (evals/legal/multi_law_24_gold.json) and scores each answer:
      retrieval -- share of the gold's expected sections (per law) present in what was retrieved;
      citation  -- share of them the answer actually cited;
      answer    -- an LLM judge against the gold answer, with rules per question type
                   (answerable, multi_law, ambiguous, not_in_law, unanswerable) and the item's
                   must_not list. correct 1, partially_correct 0.5, otherwise 0.
    The score of a question is the mean of the parts that apply. When the judge grades an
    answer down, a narrower contradiction check is recorded next to it for review; an
    "incorrect" that check doesn't back up (nothing contradicts the gold) counts as
    partially_correct, with the judge's verdict kept as judge_verdict.

Sections are matched by law (gazette number) and number: "6(2) › 24א(ט)" is section 6(2) of
the amending law, inserting 24א(ט); a chunk of a whole section or subsection matches anything
inside it.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

EVALS = Path(__file__).resolve().parent.parent / "evals" / "legal"
QUESTIONS = EVALS / "multi_law_24_questions.json"
GOLD = EVALS / "multi_law_24_gold.json"
POINTS = {"correct": 1.0, "partially_correct": 0.5}

TYPE_RULES = {
    "answerable": "The question has one correct answer in one law.",
    "multi_law": "The same provision appears in several laws. The answer must attribute it to EACH law the gold "
                 "answer names; naming only one of them is at best partially_correct.",
    "ambiguous": "The question matches provisions in several laws. It is correct only if it says the question is "
                 "ambiguous / asks which law is meant, or answers separately for each law consistently with the "
                 "gold answer. Answering for a single law without flagging the ambiguity is incorrect.",
    "not_in_law": "The law does not give the answer; it defers it to someone else. Correct means saying the law does "
                  "not state it (ideally who decides). Any specific answer to the question is incorrect.",
    "unanswerable": "The answer is not in the indexed laws. Correct means saying it is not stated / cannot be "
                    "answered from them. Any invented specific is incorrect.",
}

JUDGE_PROMPT = """\
You grade an answer to a question about Israeli law against a gold (reference) answer written by a lawyer.

Grade only against the gold answer and the rule of the question type. The laws in this test are real and in force,
even if you have never heard of them: never grade an answer down because you believe a law, section or date does
not exist, or because it cites a section the gold answer doesn't name.

Question type: {type} -- {rule}

verdict:
- "correct": gives the gold answer's essential facts (or, for the types above, does what the type requires) and
  contradicts none of them. Wording may differ. Detail beyond the gold answer that doesn't contradict it -- quoted
  law, a cross-reference, a related rule -- is fine and never lowers the verdict by itself.
- "partially_correct": some essential facts right, others missing.
- "incorrect": contradicts the gold answer, gets a key fact wrong (yes instead of no, a wrong number), answers a
  different question, or breaks a rule of its question type.
- "abstained": declines or says it cannot answer -- unless the question type makes that the right answer, in
  which case it is "correct".
{must_not}
The answer may end with the provisions it cites. They are its attribution: use them to judge which law it says
what about.
Section numbers: an amending law's section and the section it inserts ("6(2)" and "24א") are the same provision.
Words that are garbled or in another language count only where they make an essential fact unreadable -- that fact
is then missing.
fabricated_specifics: true if the answer asserts a specific number, date, amount or rule the gold answer does not
support. The laws and sections it cites are not specifics. Naming the law or section a rule comes from -- in the text
or in the cited provisions -- is never fabricated and never lowers the verdict, even when the gold answer names no law. Judge meaning, not wording. The texts may be in Hebrew.
First explain briefly (at most three sentences), comparing the answer's essential facts with the gold answer's;
then give the verdict."""


def _log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


# --- section matching ------------------------------------------------------------------

_NUMBER_RE = re.compile(r"^(?P<base>\d+[א-ת]*\d*)(?P<subs>(?:\([^)]+\))*)$")


def _split(section: str) -> tuple[str, list[str]]:
    match = _NUMBER_RE.match(section.strip())
    if not match:
        return section.strip(), []
    return match.group("base"), re.findall(r"\(([^)]+)\)", match.group("subs"))


def _covers(chunk_part: str | None, expected_part: str | None) -> bool:
    """Does a chunk's section (or inserted section) contain the expected one?"""
    if not expected_part:
        return not chunk_part
    if not chunk_part:
        return False
    chunk_base, chunk_subs = _split(chunk_part)
    expected_base, expected_subs = _split(expected_part)
    if chunk_base != expected_base:
        return False
    # "6" covers "6(א)"; "6(א)" covers "6(א)(2)"; "6(א)" doesn't cover "6(ב)"
    return all(a == b for a, b in zip(chunk_subs, expected_subs))


def parse_source_id(source_id: str) -> tuple[str, str, str | None]:
    """(law_id, own section, inserted section) from 'law-x@2026-07-16:6(2)>24א(ט)#p1'."""
    law_version, _, rest = source_id.partition(":")
    rest = rest.split("#p")[0]
    own, _, inserted = rest.partition(">")
    return law_version.split("@")[0], own, inserted or None


def matches(source_id: str, gazette_by_law_id: dict[str, str], expected: dict, gold_gazettes: dict[str, str]) -> bool:
    law_id, own, inserted = parse_source_id(source_id)
    if gazette_by_law_id.get(law_id) != gold_gazettes.get(expected["law_id"]):
        return False
    expected_own, _, expected_inserted = (p.strip() for p in expected["section"].partition("›"))
    if own == "preamble" or not _covers(own, expected_own):
        return False
    if expected_inserted:
        return inserted is None or _covers(inserted, expected_inserted)
    return True


def coverage(source_ids: list[str], expected: list[dict], gazettes: dict[str, str], gold_gazettes: dict[str, str]):
    if not expected:
        return None, []
    hits = [e for e in expected if any(matches(s, gazettes, e, gold_gazettes) for s in source_ids)]
    return len(hits) / len(expected), [f"{e['law_id']} {e['section']}" for e in hits]


def _gazette_map() -> dict[str, str]:
    """law_id in the index -> gazette number ("3546")."""
    from docslides.legal import retrieval

    out = {}
    for _, _, meta in retrieval.all_chunks():
        number = re.search(r"\d{3,5}", meta.get("gazette") or "")
        out[meta["law_id"]] = number.group(0) if number else ""
    return out


def _gold_gazettes(gold: dict) -> dict[str, str]:
    return {law_id: re.search(r"\d{3,5}", label).group(0) for law_id, label in gold["meta"]["law_ids"].items()}


# --- phases ----------------------------------------------------------------------------------

async def answer_phase(questions: list[dict], run_dir: Path) -> None:
    from docslides.legal import retrieval
    from docslides.legal.citations import strip_citations
    from docslides.legal.evaluation import reasoning_record
    from docslides.legal.pipeline import run_legal_turn
    from docslides.llm.client import aclose_all_clients

    if retrieval.collection_count() == 0:
        raise SystemExit("The Legal index is empty.")
    out = run_dir / "answers.json"
    answers = _load(out)
    gazettes = _gazette_map()
    try:
        for q in questions:
            if q["id"] in answers:
                continue
            _log(f"ANSWER {q['id']}: running the Legal pipeline")
            started = time.monotonic()

            async def status(message: str, qid: str = q["id"]) -> None:
                _log(f"  {qid}: {message}")

            try:
                turn = await run_legal_turn(q["question"], f"gold-{q['id']}", status)
            except Exception as exc:  # noqa: BLE001 -- recorded, graded as an error
                answers[q["id"]] = {"error": f"{type(exc).__name__}: {exc}", "seconds": round(time.monotonic() - started)}
                _save(out, answers)
                continue
            retrieved = [c["source_id"] for c in turn.retrieved_chunks]
            answers[q["id"]] = {
                "answer_text": strip_citations(turn.output["answer_draft"]),
                "cited": [{"source_id": n["source_id"], "law": n["law"], "section": n["section"],
                           "verified": n["verified"]} for n in turn.footnotes],
                "retrieved": retrieved,
                # So a later re-grade doesn't need the same index.
                "law_gazettes": {law_id: gazettes.get(law_id, "")
                                 for law_id in {parse_source_id(s)[0] for s in retrieved}},
                "escalation_reasons": turn.escalation_reasons,
                "seconds": round(time.monotonic() - started),
                "audit_path": turn.audit_path,
                **reasoning_record(turn),
            }
            _save(out, answers)
            _log(f"ANSWER {q['id']}: done in {answers[q['id']]['seconds']}s")
    finally:
        await aclose_all_clients()


async def grade_phase(questions: list[dict], run_dir: Path) -> None:
    from docslides.legal.chunking import normalize_hebrew_quotes
    from docslides.legal.evaluation import (
        JUDGE_MAX_TOKENS,
        contradiction,
        get_judge_client,
        with_citations,
    )
    from docslides.llm.client import ChatMessage, LLMCallSite, SamplingParams, aclose_all_clients
    from docslides.llm.schemas import EvalJudgement

    gold = _load(GOLD)  # opened only here, never in the answer phase
    items = {item["id"]: item for item in gold["items"]}
    gold_gazettes = _gold_gazettes(gold)
    local_gazettes = _gazette_map()
    answers = _load(run_dir / "answers.json")
    graded = _load(run_dir / "graded.json")
    qwen = get_judge_client()
    try:
        for q in questions:
            a, g = answers.get(q["id"]), items[q["id"]]
            if not a:
                continue
            record = {"id": q["id"], "type": g["type"], "question": q["question"], "gold_answer": g["gold_answer"]}
            gazettes = {**local_gazettes, **(a.get("law_gazettes") or {})}
            retrieval_share, retrieved_hits = coverage(a.get("retrieved", []), g["expected_sections"], gazettes, gold_gazettes)
            cited_ids = [c["source_id"] for c in a.get("cited", [])]
            citation_share, cited_hits = coverage(cited_ids, g["expected_sections"], gazettes, gold_gazettes)
            cited_laws = sorted({lid for lid, gz in gold_gazettes.items()
                                 for s in cited_ids if gazettes.get(parse_source_id(s)[0]) == gz})
            record.update(retrieval=retrieval_share, retrieved_hits=retrieved_hits, citation=citation_share,
                          cited_hits=cited_hits, cited_laws=cited_laws, expected_laws=g["expected_law_ids"],
                          answer_text=a.get("answer_text", ""), cited=a.get("cited", []),
                          escalation_reasons=a.get("escalation_reasons", []), seconds=a.get("seconds"),
                          analysis_notes=a.get("analysis_notes", ""))  # reasoning: answers.json
            if a.get("error"):
                record.update(verdict="error", error=a["error"], answer=0.0,
                              retrieval=None, citation=None, score=0.0)
                graded[q["id"]] = record
                _save(run_dir / "graded.json", graded)
                continue
            must_not = ("The answer is incorrect if it does any of these: " + "; ".join(g["must_not"]) + "\n") \
                if g["must_not"] else ""
            system = JUDGE_PROMPT.format(type=g["type"], rule=TYPE_RULES[g["type"]], must_not=must_not)
            shown = with_citations(record["answer_text"], [f"{c['law']}, section {c['section']}" for c in record["cited"]])
            user = normalize_hebrew_quotes(
                f"Question:\n{q['question']}\n\nGold answer:\n{g['gold_answer']}\n\nAnswer to grade:\n{shown}"
            )
            try:
                judgement = await qwen.complete_json(
                    [ChatMessage("system", system), ChatMessage("user", user)], LLMCallSite("legal_eval_judge"),
                    schema=EvalJudgement, sampling=SamplingParams(temperature=0.0, max_tokens=JUDGE_MAX_TOKENS),
                )
                verdict, explanation, fabricated = judgement.verdict, judgement.explanation, judgement.fabricated_specifics
            except Exception as exc:  # noqa: BLE001
                verdict, explanation, fabricated = "incorrect", f"judge failed: {exc}", False
            record.update(verdict=verdict, judge_explanation=explanation, fabricated_specifics=fabricated)
            if g["type"] in ("answerable", "multi_law") and verdict in ("incorrect", "partially_correct"):
                shim = type("Q", (), {"question": q["question"], "gold": g["gold_answer"]})()
                try:
                    check = await contradiction(qwen, shim, shown)
                    record["contradiction_check"] = {"contradicts_gold": check.contradicts_gold, "conflict": check.conflict}
                    record["needs_review"] = not check.contradicts_gold  # graded down, yet nothing contradicts
                    if verdict == "incorrect" and not check.contradicts_gold:
                        # "Incorrect" means it gets the answer wrong; a verdict the narrower check can't back
                        # up (the 14B judge has called an answer identical to the gold wrong) is at most a miss.
                        verdict = "partially_correct"
                        record.update(verdict=verdict, judge_verdict="incorrect")
                except Exception as exc:  # noqa: BLE001
                    record["contradiction_check"] = f"failed: {exc}"
            record["answer"] = POINTS.get(verdict, 0.0)
            parts = [p for p in (record["retrieval"], record["answer"], record["citation"]) if p is not None]
            record["score"] = sum(parts) / len(parts)
            graded[q["id"]] = record
            _save(run_dir / "graded.json", graded)
            _log(f"GRADE {q['id']} ({g['type']}): {verdict} score={record['score']:.2f} "
                 f"retrieval={retrieval_share} citation={citation_share}")
    finally:
        await aclose_all_clients()
    _save(run_dir / "summary.json", summarize(list(graded.values())))


def summarize(records: list[dict]) -> dict:
    def mean(values):
        values = [v for v in values if v is not None]
        return 100 * sum(values) / len(values) if values else None

    by_type = {}
    for kind in ("answerable", "multi_law", "ambiguous", "not_in_law", "unanswerable"):
        rows = [r for r in records if r["type"] == kind]
        if rows:
            by_type[kind] = {"questions": len(rows), "score": mean(r["score"] for r in rows),
                             "answer": mean(r.get("answer") for r in rows),
                             "retrieval": mean(r.get("retrieval") for r in rows),
                             "citation": mean(r.get("citation") for r in rows)}
    return {
        "questions": len(records),
        "score": mean(r["score"] for r in records),
        "answer": mean(r.get("answer") for r in records),
        "retrieval": mean(r.get("retrieval") for r in records),
        "citation": mean(r.get("citation") for r in records),
        "verdicts": {v: sum(r["verdict"] == v for r in records)
                     for v in ("correct", "partially_correct", "incorrect", "abstained", "error")},
        "fabricated_specifics": sum(bool(r.get("fabricated_specifics")) for r in records),
        "escalated": sum(bool(r.get("escalation_reasons")) for r in records),
        "by_type": by_type,
    }


def retrieval_only(questions: list[dict]) -> None:
    from docslides.legal import retrieval

    gold = _load(GOLD)
    items = {item["id"]: item for item in gold["items"]}
    gold_gazettes, gazettes = _gold_gazettes(gold), _gazette_map()
    found = total = 0
    for q in questions:
        g = items[q["id"]]
        result = retrieval.retrieve(q["question"])
        ids = [c.metadata.source_id for c in result.chunks]
        share, hits = coverage(ids, g["expected_sections"], gazettes, gold_gazettes)
        if share is not None:
            found += share
            total += 1
        missing = [f"{e['law_id'].split('-')[0]} {e['section']}" for e in g["expected_sections"]
                   if f"{e['law_id']} {e['section']}" not in hits]
        sent = ", ".join(f"{gazettes.get(parse_source_id(s)[0], '?')}:{s.split(':', 1)[-1]}" for s in ids)
        print(f"{q['id']} {g['type']:12} retrieved={'-' if share is None else f'{share:.0%}':>5} "
              f"flag={'yes' if result.low_relevance else 'no '} missing={missing}\n      sent: {sent}")
    print(f"\nexpected sections retrieved: {100 * found / max(total, 1):.0f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default=f"data/legal/eval/multi-{datetime.now():%Y%m%d-%H%M%S}")
    parser.add_argument("--phase", choices=["answer", "grade", "both"], default="both")
    parser.add_argument("--only", help="comma-separated question ids")
    # Accepted and ignored: older copies of notebooks/kaggle_legal_eval.ipynb still pass it.
    parser.add_argument("--no-dicta", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        questions = [q for q in questions if q["id"] in wanted]
    if args.retrieval_only:
        retrieval_only(questions)
        return 0

    from docslides.config import get_config

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    legal = get_config().legal
    if not (run_dir / "meta.json").exists():
        _save(run_dir / "meta.json", {
            "eval_set": "multi-law-24-he", "started": datetime.now().isoformat(timespec="seconds"),
            "model (pipeline)": legal.orchestrator.model,
            "judge": __import__("os").environ.get("DOCSLIDES_LEGAL_JUDGE_MODEL") or legal.orchestrator.model,
            "retrieval": f"{legal.retrieval.embedding_model}, reranker={legal.retrieval.reranker_model}, "
                         f"keyword={legal.retrieval.keyword_search}, evidence budget={legal.retrieval.max_evidence_tokens}",
        })
    if args.phase in ("answer", "both"):
        asyncio.run(answer_phase(questions, run_dir))
    if args.phase in ("grade", "both"):
        asyncio.run(grade_phase(questions, run_dir))
        summary = _load(run_dir / "summary.json")
        print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
