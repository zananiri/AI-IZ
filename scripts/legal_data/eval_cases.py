#!/usr/bin/env python3
"""Answer, then judge, the 10 paralegal cases in legal_txt/Evals/israeli_legal_eval/cases/
against cases_gold.json's 100-point rubric, per israeli_legal_eval/judge_prompt.md Prompt B.
Companion to eval_run.py (the 500 single questions); score.py doesn't cover the cases (its own
docstring: "prepare"/"report" are for gold.jsonl), so this script also does the aggregate report.

    python scripts/legal_data/eval_cases.py answer --mode rag --out data/legal/eval/cases/answers_rag.jsonl
    python scripts/legal_data/eval_cases.py judge --answers data/legal/eval/cases/answers_rag.jsonl \
        --out data/legal/eval/cases/judged_rag.jsonl
    python scripts/legal_data/eval_cases.py report --judged data/legal/eval/cases/judged_rag.jsonl \
        --out data/legal/eval/cases/report_rag.md --json-out data/legal/eval/cases/report_rag.json

Reasoning: every LLM call goes to <logging.llm_trace_dir>/<date>.jsonl (llm/trace.py) with the
full prompt and the model's reasoning, job_id set to the case id -- see eval_run.py's docstring.
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

from eval_run import RAG_SYSTEM, render_context, retrieve

from docslides.legal.evaluation import get_judge_client
from docslides.llm import trace
from docslides.llm.client import (
    ChatMessage,
    LLMCallSite,
    SamplingParams,
    get_legal_orchestrator_client,
)
from docslides.llm.schemas import CaseJudgement

DEFAULT_DIR = Path("legal_txt/Evals/israeli_legal_eval")
RUBRIC_SECTIONS = ["facts_summary", "chronology", "legal_issues", "deadlines", "red_flags",
                    "missing_info", "deliverable", "next_step"]

CASE_SYSTEM = """You act as a paralegal at a law firm, working under ISRAELI law. Follow the task \
instructions exactly, in the order given, using only facts that appear in the case file -- if you \
assume something, say so explicitly. Reply in Hebrew."""

WORK_FILE_MAX_TOKENS = 3072
JUDGE_MAX_TOKENS = 2048


def _log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_done(out: Path) -> dict[str, dict]:
    return {r["case_id"]: r for r in _load_jsonl(out)} if out.exists() else {}


def _append(out: Path, row: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_gold(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {c["case_id"]: c for c in data["cases"]}


async def answer_one(qwen, instructions: str, case_id: str, case_text: str, mode: str,
                      categories: list[str], top_k: int) -> dict:
    if mode == "rag":
        hits = retrieve(case_text, categories, top_k)
        reference = f"\n\n<reference_material>\n{render_context(hits)}\n</reference_material>"
        system = RAG_SYSTEM.replace("You are a legal assistant", "You act as a paralegal") + \
            "\n\n" + CASE_SYSTEM
    else:
        hits, reference, system = [], "", CASE_SYSTEM
    user = f"{instructions}\n\n---\n\n{case_text}{reference}"

    with trace.collect(job_id=case_id):
        text = await qwen.complete_text(
            [ChatMessage("system", system), ChatMessage("user", user)],
            LLMCallSite("legal_eval_baseline"),
            sampling=SamplingParams(temperature=0.0, max_tokens=WORK_FILE_MAX_TOKENS),
            enable_thinking=True,
        )
    return {"case_id": case_id, "work_file": text.strip(),
            "retrieved": [{"category": h["category"], "distance": round(h["distance"], 4),
                            "title": h["meta"].get("title"), "section": h["meta"].get("section_number")}
                           for h in hits]}


async def cmd_answer(a) -> None:
    base = Path(a.dir)
    instructions = (base / "cases" / "00_TASK_INSTRUCTIONS.md").read_text(encoding="utf-8")
    gold = load_gold(base / "cases_gold.json")
    out = Path(a.out)
    done = _load_done(out)
    categories = a.categories.split(",")
    qwen = get_legal_orchestrator_client()
    _log(f"{len(gold)} cases, {len(done)} already answered, mode={a.mode}")
    for case_id, entry in gold.items():
        if case_id in done:
            continue
        case_text = (base / "cases" / entry["file"]).read_text(encoding="utf-8")
        started = time.monotonic()
        try:
            row = await answer_one(qwen, instructions, case_id, case_text, a.mode, categories, a.top_k)
        except Exception as exc:  # noqa: BLE001 -- record and move on; --out is resumable
            row = {"case_id": case_id, "work_file": "", "error": f"{type(exc).__name__}: {exc}"}
        _append(out, row)
        _log(f"{case_id}: {round(time.monotonic() - started)}s{' -- ' + row['error'] if row.get('error') else ''}")


def judge_prompt(case_text: str, work_file: str, gold_entry: dict) -> str:
    return (
        f"You are grading a paralegal work file prepared by an AI for a lawyer, under ISRAELI law. "
        f"The reference date for all deadlines is {gold_entry['reference_date']}.\n\n"
        f"<case_file>{case_text}</case_file>\n\n<model_work_file>{work_file}</model_work_file>\n\n"
        f"<gold_rubric>{json.dumps(gold_entry, ensure_ascii=False)}</gold_rubric>\n\n"
        "Score each section from 0 to its weight in gold_rubric.rubric_weights, by how many of the "
        "matching gold_rubric list it covers correctly (proportional credit):\n"
        "- facts_summary: accurate, no invented facts, under ~200 words\n"
        "- chronology: compare against gold_rubric.chronology; dates and sources correct\n"
        "- legal_issues: compare against gold_rubric.legal_issues; law and section right\n"
        "- deadlines: compare against gold_rubric.deadlines; calculations right; most urgent flagged\n"
        "- red_flags: compare against gold_rubric.red_flags\n"
        "- missing_info: compare against gold_rubric.missing_info\n"
        "- deliverable: check every item in gold_rubric.deliverable_checks\n"
        "- next_step: consistent with gold_rubric.next_step\n\n"
        "Then apply penalties (from gold_rubric.penalties) for: each invented fact, date or amount not "
        "in the case file; each wrong or non-existent citation; each violation of an item in "
        "gold_rubric.must_not; unsafe or unethical advice (e.g. coordinating testimony, hiding assets). "
        "penalties[].points is negative. total is the sum of the eight section scores plus the "
        "(negative) penalty points -- it can go below 0."
    )


async def judge_one(qwen, case_id: str, case_text: str, work_file: str, gold_entry: dict) -> dict:
    prompt = judge_prompt(case_text, work_file, gold_entry)
    with trace.collect(job_id=case_id):
        try:
            verdict = await qwen.complete_json(
                [ChatMessage("user", prompt)], LLMCallSite("legal_eval_judge"),
                schema=CaseJudgement,
                sampling=SamplingParams(temperature=0.0, max_tokens=JUDGE_MAX_TOKENS),
                enable_thinking=True,
            )
        except Exception as exc:  # noqa: BLE001 -- record and move on; --out is resumable
            return {"case_id": case_id, "total": 0.0, "note": f"judge failed: {exc}",
                    **{s: 0.0 for s in RUBRIC_SECTIONS}, "penalties": [], "missed_items": []}
    return {"case_id": case_id, **verdict.model_dump()}


async def cmd_judge(a) -> None:
    base = Path(a.dir)
    gold = load_gold(base / "cases_gold.json")
    answers = {r["case_id"]: r for r in _load_jsonl(Path(a.answers))}
    out = Path(a.out)
    done = _load_done(out)
    qwen = get_judge_client()
    _log(f"{len(answers)} answers, {len(done)} already judged")
    for case_id, ans in answers.items():
        if case_id in done or not ans.get("work_file"):
            continue
        entry = gold[case_id]
        case_text = (base / "cases" / entry["file"]).read_text(encoding="utf-8")
        started = time.monotonic()
        row = await judge_one(qwen, case_id, case_text, ans["work_file"], entry)
        _append(out, row)
        _log(f"{case_id}: total={row['total']} ({round(time.monotonic() - started)}s)")


def cmd_report(a) -> None:
    judged = _load_jsonl(Path(a.judged))
    lines = ["# Paralegal cases eval report", "", f"cases scored: {len(judged)} / 10", "",
             "| case | " + " | ".join(RUBRIC_SECTIONS) + " | penalties | total |",
             "|---|" + "---|" * (len(RUBRIC_SECTIONS) + 2)]
    totals, section_totals = [], {s: [] for s in RUBRIC_SECTIONS}
    for row in judged:
        pen = sum(p["points"] for p in row.get("penalties", []))
        cells = " | ".join(f"{row.get(s, 0):.0f}" for s in RUBRIC_SECTIONS)
        lines.append(f"| {row['case_id']} | {cells} | {pen:.0f} | {row['total']:.0f} |")
        totals.append(row["total"])
        for s in RUBRIC_SECTIONS:
            section_totals[s].append(row.get(s, 0))
    if totals:
        avg_cells = " | ".join(f"{sum(v) / len(v):.1f}" for v in section_totals.values())
        lines.append(f"| **average** | {avg_cells} | | {sum(totals) / len(totals):.1f} |")
    lines += ["", "## Penalties and missed items", ""]
    for row in judged:
        if row.get("penalties") or row.get("missed_items"):
            lines.append(f"### {row['case_id']} (total {row['total']:.0f})")
            for p in row.get("penalties", []):
                lines.append(f"- **{p['type']}** ({p['points']}): {p['detail']}")
            for m in row.get("missed_items", []):
                lines.append(f"- missed: {m}")
            if row.get("note"):
                lines.append(f"- note: {row['note']}")
            lines.append("")
    report = "\n".join(lines) + "\n"
    Path(a.out).write_text(report, encoding="utf-8")
    print(report)
    if a.json_out:
        Path(a.json_out).write_text(
            json.dumps({"average_total": sum(totals) / len(totals) if totals else 0, "cases": judged},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("answer")
    pa.add_argument("--dir", default=str(DEFAULT_DIR), help="extracted israeli_legal_eval/ directory")
    pa.add_argument("--mode", choices=["no_context", "rag"], default="rag")
    pa.add_argument("--categories", default="laws,procedural_rules")
    pa.add_argument("--top-k", type=int, default=16)
    pa.add_argument("--out", required=True)

    pj = sub.add_parser("judge")
    pj.add_argument("--dir", default=str(DEFAULT_DIR))
    pj.add_argument("--answers", required=True)
    pj.add_argument("--out", required=True)

    pr = sub.add_parser("report")
    pr.add_argument("--judged", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--json-out")

    a = p.parse_args()
    if a.cmd == "report":
        cmd_report(a)
        return

    from docslides.llm.client import aclose_all_clients

    async def run() -> None:
        try:
            await {"answer": cmd_answer, "judge": cmd_judge}[a.cmd](a)
        finally:
            await aclose_all_clients()

    asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
