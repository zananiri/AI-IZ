#!/usr/bin/env python3
"""Answer, then judge, the 10 paralegal cases in legal_txt/Evals/israeli_legal_eval/cases/
against cases_gold.json's 100-point rubric, per israeli_legal_eval/judge_prompt.md Prompt B.
Companion to eval_run.py (the 500 single questions); score.py doesn't cover the cases (its own
docstring: "prepare"/"report" are for gold.jsonl), so this script also does the aggregate report.

    python scripts/legal_data/eval_cases.py answer --out data/legal/eval/cases/answers_rag.jsonl [--limit 5]
    python scripts/legal_data/eval_cases.py judge --answers data/legal/eval/cases/answers_rag.jsonl \
        --out data/legal/eval/cases/judged_rag.jsonl
    python scripts/legal_data/eval_cases.py report --judged data/legal/eval/cases/judged_rag.jsonl \
        --out data/legal/eval/cases/report_rag.md --json-out data/legal/eval/cases/report_rag.json

Reasoning: every LLM call goes to <logging.llm_trace_dir>/<date>.jsonl (llm/trace.py) with the
full prompt and the model's reasoning, job_id set to the case id -- see eval_run.py's docstring.

Per case: (1) the model lists the case's issues and governing laws (plan_issues, as for the single
questions) and retrieve_planned gathers the reference material for each; (2) the work file
(CASE_SYSTEM); (3) sections the work file left out are written and appended (legal_eval_repair);
(4) if it holds dates or amounts that aren't in the case file, the model revises it once -- showing
the calculation behind a deadline, dropping what it can't support. The plan and repairs are
recorded on the row.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from eval_run import eval_sampling, plan_issues, render_context

from docslides.config import get_config
from docslides.legal import caselaw
from docslides.legal.corpus_retrieval import retrieve_planned, warm_up_retrieval
from docslides.legal.evaluation import get_judge_client
from docslides.llm import trace
from docslides.llm.client import (
    ChatMessage,
    LLMCallSite,
    get_legal_orchestrator_client,
)
from docslides.llm.schemas import CaseJudgement

DEFAULT_DIR = Path("legal_txt/Evals/israeli_legal_eval")
RUBRIC_SECTIONS = ["facts_summary", "chronology", "legal_issues", "deadlines", "red_flags",
                    "missing_info", "deliverable", "next_step"]

# The 29 Sept review of the Qwen 14B / Gemma 12B work files: every case lost points for an invented
# period or deadline, a law applied outside its scope, or a fact misread from the file; the
# deliverable was often an outline or missing. The 29 Sept review of the Gemma 27B files: a bare
# "state a period only when ... otherwise write 'יש לבדוק'" rule made the model hedge even on periods
# it knew (a 14-day objection deadline); it counted from the order date, not delivery; it summed
# 4,500 + 3,000 + 1,500 as 8,000; it asserted in a demand letter a fact the client didn't know;
# and it covered only the main claim of each case.
CASE_SYSTEM = """You act as a paralegal at a law firm, working under ISRAELI law. Prepare the work \
file the task instructions describe. <reference_material> holds excerpts retrieved from an index of \
Israeli legislation; some are relevant and some are not.
- Write all eight sections, in the order given, each under its own numbered heading (1 to 8) -- \
including section 7, the full text of the document the case asks for, not an outline of it.
- Take facts, names, dates and amounts only from the case file, exactly as written there; check \
each one against its document. If you assume something, say so explicitly.
- For every legal issue cite the law and section. Before relying on a law, check that it covers \
this kind of party and transaction. Never cite a law just because it appears in <reference_material>.
- In section 3, list every claim the facts support, not only the main one -- payments owed, \
procedure, the competent court -- each with its law and section.
- In section 4, for each deadline name the event it runs from (a dismissal, delivery of the goods, \
publication of a notice -- often not the contract date) and the provision that sets the period, \
then compute it: event date + period = end date, and say whether that date has passed as of the \
date the file was received ("today"). When the case file, a provision in <reference_material> or a \
law you are certain of gives the period, compute the date; write "יש לבדוק את המועד" only for a \
period you cannot source, and say what must be checked. Never invent a period. Mark the most \
urgent step.
- Write out every sum you rely on (1,200 + 800 = 2,000) and check it.
- What the case file says is unknown stays unknown in every section, the draft included: ask \
about it or demand it, never assert it. Leave ID numbers out of documents addressed to the other side.
- Answer every question the client asks in the case file.
- Never advise anything unlawful or unethical (coordinating testimony, hiding assets, misleading \
a court or the other side).
- Write only in Hebrew, and start directly with section 1 -- no preamble."""

# A heading line for each of the eight sections: what missing_sections looks for.
SECTION_HEADINGS = {
    "1. תקציר עובדתי": r"תקציר",
    "2. ציר זמן": r"ציר\s*(?:ה)?זמן|כרונולוגי",
    "3. סוגיות משפטיות": r"סוגיות",
    "4. מועדים ודחיפות": r"מועדים|דחיפות",
    "5. סתירות וסימני אזהרה": r"סתירות|אזהרה",
    "6. מידע ומסמכים חסרים": r"חסרים|חסר",
    "7. טיוטת מסמך": r"טיוטה|טיוטת",
    "8. המלצה לצעד הבא": r"המלצה|צעד\s*הבא",
}
# "## 4. מועדים", "**4. מועדים ודחיפות**", "4) מועדים": a line that opens with a heading marker.
_HEADING_LINE_RE = re.compile(r"^[ \t]*(?=#|\*\*|\d{1,2}[ \t]*[.):])[#* \t\d.):]*(?P<title>[^\n]{0,80})$", re.MULTILINE)

COMPLETE_PROMPT = """The work file above is missing these sections: {sections}. Write only those \
sections, each under its numbered heading, following the same rules (section 7 is the full text \
of the document the case asks for). Do not repeat the sections already written."""

FACTS_PROMPT = """Below are a case file and a paralegal work file prepared from it. These dates and \
amounts in the work file do not appear in the case file: {values}.
For each one: if it is a deadline or a sum you calculated, redo the calculation from the case \
file's own figures and dates -- correct the value everywhere it appears if it was wrong -- and \
write the calculation (and the provision that sets any period) next to it; if it is a fact the case file \
doesn't give, remove it or write "לא ידוע". Correct nothing else. Return the whole work file, in \
Hebrew, with the same sections and headings.

<case_file>
{case_text}
</case_file>

<work_file>
{work_file}
</work_file>"""

WORK_FILE_MAX_TOKENS = 6144  # a work file has eight sections, plus the reasoning before them
JUDGE_MAX_TOKENS = 4096
# A case raises 6-8 issues (cases_gold.json); at 6 the plan dropped case_01's notice-period issue.
CASE_MAX_ISSUES = 8

_PROMPT_TAG_RE = re.compile(r"</?(?:work_file|case_file)>")

# A comma after a value ends it ("3.2.2026, יוסי:", "13,000,") unless a digit follows (a thousands
# separator). Treating any comma as part of the number hid every such date in the case file, so
# dates copied correctly from it were reported as unsupported (case_02, case_03 on 29 Sept).
_DATE_RE = re.compile(r"(?<![\d.,/])(\d{1,2})[./](\d{1,2})(?:[./](?:\d{4}|\d{2}))?(?!\d|,\d|[./]\d)")
_AMOUNT_RE = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*(?:₪|ש[\"״]ח|שקל)|(?<![\d.,])(\d{1,3}(?:,\d{3})+)(?!\d|,\d)")


def missing_sections(work_file: str) -> list[str]:
    """The SECTION_HEADINGS whose keyword appears on no heading-like line of the work file."""
    lines = [m.group("title") for m in _HEADING_LINE_RE.finditer(work_file) if m.group("title").strip()]
    return [name for name, pattern in SECTION_HEADINGS.items()
            if not any(re.search(pattern, line) for line in lines)]


def _dates(text: str) -> set[tuple[int, int]]:
    return {(int(d), int(m)) for d, m in _DATE_RE.findall(text) if 1 <= int(d) <= 31 and 1 <= int(m) <= 12}


def _amounts(text: str) -> set[str]:
    return {(a or b).replace(",", "") for a, b in _AMOUNT_RE.findall(text)}


def unsupported_values(work_file: str, case_text: str) -> list[str]:
    """Dates (day.month) and amounts in the work file that the case file doesn't contain -- each
    either a calculation to show or a fact the model made up."""
    case_dates, case_numbers = _dates(case_text), {n.replace(",", "") for n in re.findall(r"\d[\d,]*", case_text)}
    out = [f"{d}.{m}" for d, m in sorted(_dates(work_file) - case_dates, key=lambda x: (x[1], x[0]))]
    out += [f"{int(a):,}" for a in sorted(_amounts(work_file) - case_numbers, key=int)]
    return out


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


async def answer_one(qwen, instructions: str, case_id: str, case_text: str,
                      categories: list[str], top_k: int, thinking: bool = True,
                      max_tokens: int = WORK_FILE_MAX_TOKENS) -> dict:
    repairs: list[str] = []
    with trace.collect(job_id=case_id):
        # Always with retrieval over the corpus (there is no model-alone mode), planned as for the
        # single questions: one search per issue the model sees, not one on the whole case text.
        issues = await plan_issues(qwen, {"id": case_id, "question": case_text}, max_issues=CASE_MAX_ISSUES)
        hits = retrieve_planned(case_text, issues, categories, top_k, per_issue_slot=True)
        reference = f"\n\n<reference_material>\n{render_context(hits)}\n</reference_material>"
        case_hits = caselaw.search_caselaw(case_text, [i.issue for i in issues]) \
            if get_config().legal.corpus.caselaw_dir else []
        if case_hits:
            reference += "\n\n" + caselaw.render_caselaw(case_hits)
        messages = [ChatMessage("system", CASE_SYSTEM),
                    ChatMessage("user", f"{instructions}\n\n---\n\n{case_text}{reference}")]

        text = (await qwen.complete_text(
            messages, LLMCallSite("legal_eval_baseline"), sampling=eval_sampling(max_tokens, thinking),
            enable_thinking=thinking,
        )).strip()
        sampling = eval_sampling(max_tokens, False)
        if not text and thinking:
            repairs.append("empty_answer_retry")
            text = (await qwen.complete_text(messages, LLMCallSite("legal_eval_baseline"), sampling=sampling,
                                             enable_thinking=False)).strip()

        missing = missing_sections(text) if text else []
        if missing:
            repairs.append("missing_sections")
            addition = (await qwen.complete_text(
                [*messages, ChatMessage("assistant", text),
                 ChatMessage("user", COMPLETE_PROMPT.format(sections=", ".join(missing)))],
                LLMCallSite("legal_eval_repair"), sampling=sampling, enable_thinking=False,
            )).strip()
            if addition:
                text = f"{text}\n\n{addition}"

        unsupported = unsupported_values(text, case_text) if text else []
        if unsupported:
            revised = (await qwen.complete_text(
                [ChatMessage("system", CASE_SYSTEM),
                 ChatMessage("user", FACTS_PROMPT.format(values=", ".join(unsupported), case_text=case_text,
                                                         work_file=text))],
                LLMCallSite("legal_eval_repair"), sampling=sampling, enable_thinking=False,
            )).strip()
            # The prompt's own tags, echoed back (case_03 on 29 Sept ended with "</work_file>").
            revised = _PROMPT_TAG_RE.sub("", revised).strip()
            # A revision that dropped sections or much of the text is worse than the unchecked file.
            if revised and len(revised) >= 0.7 * len(text) and \
                    len(missing_sections(revised)) <= len(missing_sections(text)):
                repairs.append("facts_revised")
                text = revised
    return {"case_id": case_id, "work_file": text,
            "retrieved": [{"category": h["category"], "distance": round(h["distance"], 4),
                           "score": round(h["score"], 4) if "score" in h else None,
                           "title": h["meta"].get("title"), "section": h["meta"].get("section_number")}
                          for h in hits],
            "plan": [i.model_dump() for i in issues],
            "caselaw": caselaw.caselaw_record(case_hits),
            "unsupported_values": unsupported,
            "repairs": repairs}


async def cmd_answer(a) -> None:
    base = Path(a.dir)
    instructions = (base / "cases" / "00_TASK_INSTRUCTIONS.md").read_text(encoding="utf-8")
    gold = load_gold(base / "cases_gold.json")
    out = Path(a.out)
    done = _load_done(out)
    categories = a.categories.split(",")
    qwen = get_legal_orchestrator_client()
    started = time.monotonic()
    device = warm_up_retrieval()
    _log(f"retrieval ready on {device} ({round(time.monotonic() - started)}s)")
    cases = list(gold.items())[: a.limit] if a.limit else list(gold.items())
    _log(f"{len(cases)} of {len(gold)} cases, {len(done)} already answered")
    for case_id, entry in cases:
        if case_id in done:
            continue
        case_text = (base / "cases" / entry["file"]).read_text(encoding="utf-8")
        started = time.monotonic()
        try:
            row = await answer_one(qwen, instructions, case_id, case_text, categories, a.top_k, thinking=a.thinking,
                                   max_tokens=a.max_tokens)
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
        "matching gold_rubric list it covers correctly (proportional credit): a section that covers k "
        "of n gold items correctly gets about k/n of its weight, however well written it is. Put every "
        "gold item the work file does not cover in missed_items.\n"
        "- facts_summary: accurate, no invented facts, under ~200 words\n"
        "- chronology: compare against gold_rubric.chronology; dates and sources correct\n"
        "- legal_issues: compare against gold_rubric.legal_issues; law and section right\n"
        "- deadlines: compare against gold_rubric.deadlines; calculations right; most urgent flagged\n"
        "- red_flags: compare against gold_rubric.red_flags\n"
        "- missing_info: compare against gold_rubric.missing_info\n"
        "- deliverable: check every item in gold_rubric.deliverable_checks\n"
        "- next_step: consistent with gold_rubric.next_step\n\n"
        "Then apply penalties (from gold_rubric.penalties) for: each invented fact, date, amount or "
        "deadline not in the case file or the law; each wrong period or deadline calculation; each "
        "wrong or non-existent citation; each violation of an item in "
        "gold_rubric.must_not; unsafe or unethical advice (e.g. coordinating testimony, hiding assets). "
        "penalties[].points is negative. total is the sum of the eight section scores plus the "
        "(negative) penalty points -- it can go below 0."
    )


async def judge_one(qwen, case_id: str, case_text: str, work_file: str, gold_entry: dict,
                    thinking: bool = True, max_tokens: int = JUDGE_MAX_TOKENS) -> dict:
    prompt = judge_prompt(case_text, work_file, gold_entry)
    with trace.collect(job_id=case_id):
        try:
            verdict = await qwen.complete_json(
                [ChatMessage("user", prompt)], LLMCallSite("legal_eval_judge"),
                schema=CaseJudgement,
                sampling=eval_sampling(max_tokens, thinking),
                enable_thinking=thinking,
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
        row = await judge_one(qwen, case_id, case_text, ans["work_file"], entry, thinking=a.thinking,
                              max_tokens=a.max_tokens)
        _append(out, row)
        _log(f"{case_id}: total={row['total']} ({round(time.monotonic() - started)}s)")


def cmd_report(a) -> None:
    judged = _load_jsonl(Path(a.judged))
    lines = ["# Paralegal cases eval report", "", f"cases scored: {len(judged)}", "",
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
    pa.add_argument("--mode", choices=["rag"], default="rag", help="accepted for older commands; rag is the only mode")
    pa.add_argument("--categories", default="laws,procedural_rules")
    pa.add_argument("--top-k", type=int, default=16)
    pa.add_argument("--out", required=True)
    pa.add_argument("--limit", type=int, help="answer only the first N cases (cases_gold.json order)")
    pa.add_argument("--max-tokens", type=int, default=WORK_FILE_MAX_TOKENS, help="output budget per work file, reasoning included")
    pa.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)

    pj = sub.add_parser("judge")
    pj.add_argument("--dir", default=str(DEFAULT_DIR))
    pj.add_argument("--answers", required=True)
    pj.add_argument("--out", required=True)
    pj.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    pj.add_argument("--max-tokens", type=int, default=JUDGE_MAX_TOKENS, help="output budget per judgement, reasoning included")

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
