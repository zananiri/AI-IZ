#!/usr/bin/env python3
"""Readable report of the LLM trace (src/docslides/llm/trace.py): per question, every
model call in order -- its reasoning, what it returned, why it stopped, how long it took.

    python scripts/llm_trace_report.py data/llm_trace/2026-09-25.jsonl > trace.md
    python scripts/llm_trace_report.py data/multi/audit/2026-09-25.jsonl --job gold-q04
    python scripts/llm_trace_report.py data/multi/llm_trace/*.jsonl --calls legal_analysis,legal_draft --prompts

Reads trace files (one line per call) or Legal audit files (one line per turn, its calls
under llm_calls). A call that stopped on "length" ran out of tokens: a loop, or thinking
that used up its budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load(paths: list[str]) -> tuple[dict[str, list[dict]], dict[str, str]]:
    calls: dict[str, list[dict]] = defaultdict(list)
    queries: dict[str, str] = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "llm_calls" in row:  # an audit entry: one turn
                job = row.get("job_id") or "(no job)"
                queries[job] = row.get("query", "")
                calls[job].extend(row["llm_calls"])
            else:
                calls[row.get("job_id") or "(no job)"].append(row)
    return calls, queries


def _block(text: str, limit: int) -> list[str]:
    if limit and len(text) > limit:
        text = text[:limit] + f"\n... [{len(text) - limit} more characters]"
    return ["```text", text.replace("```", "ˋˋˋ"), "```"]


def render(calls: dict[str, list[dict]], queries: dict[str, str], wanted: set[str] | None, prompts: bool,
           limit: int) -> str:
    everything = [c for rows in calls.values() for c in rows]
    stops = Counter((c.get("call_site"), c.get("done_reason")) for c in everything)
    seconds: Counter = Counter()
    for c in everything:
        seconds[c.get("call_site")] += c.get("seconds") or 0
    lines = ["# LLM trace", "", "| call site | calls | seconds | stopped on length | errors |", "|---|---|---|---|---|"]
    for site in sorted({c.get("call_site") for c in everything}, key=str):
        rows = [c for c in everything if c.get("call_site") == site]
        lines.append(f"| {site} | {len(rows)} | {seconds[site]:.0f} | {stops[(site, 'length')]} | "
                     f"{sum(bool(c.get('error')) for c in rows)} |")
    for job, rows in calls.items():
        lines += ["", f"## {job}" + (f" -- {queries[job]}" if queries.get(job) else ""), ""]
        for n, c in enumerate(rows, 1):
            if wanted and c.get("call_site") not in wanted:
                continue
            head = (f"### {n}. {c.get('call_site')} (attempt {c.get('attempt')}"
                    f"{', thinking' if c.get('thinking') else ''}, {c.get('seconds')} s, "
                    f"tokens {c.get('prompt_tokens')} in / {c.get('completion_tokens')} out, "
                    f"stop: {c.get('done_reason')})")
            lines += [head, ""]
            if c.get("error"):
                lines += [f"**Error:** {c['error']}", ""]
            if prompts and c.get("messages"):
                for m in c["messages"]:
                    lines += [f"<details><summary>{m['role']} prompt ({len(m['content'])} chars)</summary>", ""]
                    lines += [*_block(m["content"], limit), "", "</details>", ""]
            if c.get("reasoning"):
                lines += [f"<details open><summary>Reasoning ({len(c['reasoning'])} chars)</summary>", ""]
                lines += [*_block(c["reasoning"], limit), "", "</details>", ""]
            lines += ["**Output**", "", *_block(c.get("content") or "", limit), ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="trace or audit .jsonl files")
    parser.add_argument("--job", help="comma-separated job ids (e.g. gold-q04,eval-B5)")
    parser.add_argument("--calls", help="comma-separated call sites to show (e.g. legal_analysis)")
    parser.add_argument("--prompts", action="store_true", help="include the prompts (trace files only)")
    parser.add_argument("--max-chars", type=int, default=0, help="truncate each text block (0 = full)")
    args = parser.parse_args()

    calls, queries = load(args.files)
    if args.job:
        jobs = {j.strip() for j in args.job.split(",")}
        calls = {job: rows for job, rows in calls.items() if job in jobs}
    wanted = {s.strip() for s in args.calls.split(",")} if args.calls else None
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(render(calls, queries, wanted, args.prompts, args.max_chars))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
