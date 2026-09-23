#!/usr/bin/env python3
"""Before/after-RAG knowledge test for the Legal tab, with scores.

    python scripts/eval_legal.py                         # both phases, full pipeline
    python scripts/eval_legal.py --ingest                # index legal_txt/ between the phases
    python scripts/eval_legal.py --phase before          # baseline only (model alone)
    python scripts/eval_legal.py --phase after --no-dicta
    python scripts/eval_legal.py --run-dir data/legal/eval/<run>   # resume / re-report a run
    python scripts/eval_legal.py --only A1,B4,C2

BEFORE: the orchestrator model answers each question with no context.
AFTER: the full Legal pipeline answers over the index. Every answer is graded
by the same model as an LLM judge against the gold answer. Scoring rules:
src/docslides/legal/evaluation.py. Results are saved after every question
(before.json / after.json in the run directory) and summarized in report.md.

--no-dicta skips the DictaLM normalization/polish stages. They don't affect
retrieval or facts, and on CPU they cost several model swaps per question;
the report records whether they ran.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from docslides.legal import evaluation as ev
from docslides.legal.citations import strip_citations
from docslides.llm.client import aclose_all_clients, get_legal_orchestrator_client

DEFAULT_SET = Path(__file__).resolve().parent.parent / "evals" / "legal" / "elections_2026_he.json"


def _save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


async def _grade(qwen, q: ev.EvalQuestion, answer: str) -> dict:
    try:
        judgement = await ev.judge(qwen, q, answer)
        verdict, fabricated, explanation = judgement.verdict, judgement.fabricated_specifics, judgement.explanation
    except Exception as exc:  # noqa: BLE001 -- an ungradable answer is recorded, not fatal
        verdict, fabricated, explanation = "incorrect", False, f"judge failed: {exc}"
    return {
        "verdict": verdict,
        "fabricated_specifics": fabricated,
        "judge_explanation": explanation,
        "fact_coverage": ev.fact_coverage(answer, q.key_facts),
        "trap_hits": ev.trap_hits(answer, q.traps),
    }


async def run_before(questions, out: Path) -> dict:
    qwen = get_legal_orchestrator_client()
    results = _load(out)
    for q in questions:
        if q.id in results:
            continue
        _log(f"BEFORE {q.id}: asking the model with no context")
        started = time.monotonic()
        try:
            answer = await ev.ask_baseline(qwen, q)
        except Exception as exc:  # noqa: BLE001
            results[q.id] = {"answer": "", "error": str(exc), "classification": "error"}
            _save(out, results)
            continue
        graded = await _grade(qwen, q, answer)
        graded["classification"] = ev.classify_baseline(
            q, graded["verdict"], graded["fabricated_specifics"], graded["trap_hits"]
        )
        results[q.id] = {"answer": answer, "seconds": round(time.monotonic() - started), **graded}
        _save(out, results)
        _log(f"BEFORE {q.id}: {graded['classification']}")
    return results


async def run_after(questions, out: Path, use_dicta: bool, dicta_tier: str | None) -> dict:
    from docslides.legal import retrieval
    from docslides.legal.pipeline import run_legal_turn

    if retrieval.collection_count() == 0:
        raise SystemExit("The Legal index is empty -- run with --ingest, or scripts/ingest_legal_txt.py first.")
    qwen = get_legal_orchestrator_client()
    results = _load(out)
    for q in questions:
        if q.id in results:
            continue
        _log(f"AFTER {q.id}: running the Legal pipeline")
        started = time.monotonic()

        async def status(message: str, qid: str = q.id) -> None:
            _log(f"  {qid}: {message}")

        try:
            turn = await run_legal_turn(q.question, dicta_tier, f"eval-{q.id}", status, use_dicta=use_dicta)
        except Exception as exc:  # noqa: BLE001
            results[q.id] = {"answer": "", "error": str(exc), "verdict": "incorrect", "score": 0.0,
                             "passed": False, "retrieval": 0.0 if q.group != "C" else None,
                             "citation": 0.0 if q.group != "C" else None}
            _save(out, results)
            continue

        answer = strip_citations(turn.output["answer_draft"])
        retrieved_texts = [c["text"] for c in turn.retrieved_chunks]
        cited_ids = {note["source_id"] for note in turn.footnotes}
        cited_texts = [c["text"] for c in turn.retrieved_chunks if c["source_id"] in cited_ids]
        graded = await _grade(qwen, q, answer)
        scores = ev.score_after(
            q, graded["verdict"], graded["fabricated_specifics"], graded["trap_hits"], retrieved_texts, cited_texts
        )
        results[q.id] = {
            "answer": answer,
            "seconds": round(time.monotonic() - started),
            "escalation_reasons": turn.escalation_reasons,
            "cited": [f"{n['law']} {n['section']} ({'verified' if n['verified'] else 'unverified'})"
                      for n in turn.footnotes],
            "retrieved": [c["source_id"] for c in turn.retrieved_chunks],
            "dicta_used": turn.dicta_used,
            "audit_path": turn.audit_path,
            **graded,
            **scores,
        }
        _save(out, results)
        _log(f"AFTER {q.id}: verdict={graded['verdict']} score={scores['score']:.2f}")
    return results


async def main_async(args) -> int:
    meta_set, questions = ev.load_eval_set(Path(args.eval_set))
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        questions = [q for q in questions if q.id in wanted]

    run_dir = Path(args.run_dir) if args.run_dir else Path("data/legal/eval") / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Run directory: {run_dir}")

    from docslides.config import get_config

    legal = get_config().legal
    tier_key, tier_cfg = legal.dicta_tier(args.dicta_tier)
    meta = _load(run_dir / "meta.json") or {
        "eval_set": meta_set["name"],
        "started": datetime.now().isoformat(timespec="seconds"),
        "model (before RAG, pipeline, judge)": legal.orchestrator.model,
        "DictaLM in after-RAG run": "skipped (--no-dicta)" if args.no_dicta else f"{tier_key}: {tier_cfg.llm.model}",
        "retrieval": f"{legal.retrieval.embedding_model}, top_k={legal.retrieval.top_k}",
    }
    _save(run_dir / "meta.json", meta)

    try:
        before = await run_before(questions, run_dir / "before.json") if args.phase in ("before", "both") else _load(run_dir / "before.json")
        if args.ingest and args.phase in ("after", "both"):
            from docslides.legal import folder_ingest

            _log("Indexing legal_txt/ (the RAG step)")
            for r in folder_ingest.run():
                _log(f"  [{r.action}] {r.path.name} {r.law_name} {r.chunk_count or ''} {r.message}")
        after = (
            await run_after(questions, run_dir / "after.json", not args.no_dicta, args.dicta_tier)
            if args.phase in ("after", "both") else _load(run_dir / "after.json")
        )
    finally:
        await aclose_all_clients()

    summary = ev.summarize(questions, before, after)
    _save(run_dir / "summary.json", summary)
    report = ev.render_report(meta, questions, before, after, summary)
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    b, a = summary.get("before") or {}, summary.get("after") or {}
    print("\n==================== SCORES ====================")
    if b:
        print(f"Before RAG: {b['score']:.0f}/100 non-hallucination  | hallucinations {b['hallucinations']}/{b['answered']}, "
              f"abstained {b['abstained']}, correct without context {b['correct_without_context']}")
    if a:
        print(f"After RAG:  {a['score']:.0f}/100 overall             | hallucinations {a['hallucinations']}/{a['answered']}")
        for group, row in a["by_group"].items():
            parts = "  ".join(f"{p} {row[p]:.0f}%" for p in ("retrieval", "answer", "citation") if p in row)
            print(f"   group {group}: {row['score']:.0f}/100  {parts}")
    print(f"Report: {run_dir / 'report.md'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["before", "after", "both"], default="both")
    parser.add_argument("--eval-set", default=str(DEFAULT_SET))
    parser.add_argument("--run-dir", help="existing run directory to resume or re-report")
    parser.add_argument("--only", help="comma-separated question ids")
    parser.add_argument("--ingest", action="store_true", help="index legal_txt/ before the after-RAG phase")
    parser.add_argument("--no-dicta", action="store_true", help="skip DictaLM stages in the after-RAG run")
    parser.add_argument("--dicta-tier", default=None)
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
