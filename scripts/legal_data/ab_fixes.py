#!/usr/bin/env python3
"""A/B test of the 30 Sept retrieval and answering fixes on ten questions the 29 Sept Gemma 27B run
got wrong, one question per root cause, run on your own computer.

Each "arm" answers the same ten questions with eval_run.py answer --variant <...>, so the only
difference between arms is the fix. The script then scores every arm without a judge (was the
governing section retrieved; does the answer state the point the 29 Sept answer left out) and, with
--judge, with the judge and score.py as the full runs do. It writes comparison.md and a zip to send back.

Needs: this repo on main with its Python environment (pip install -e .), the corpus database at
data/legal_corpus_vectordb (laws/, procedural_rules/; the same one the Kaggle runs use), and Ollama
running with the model pulled (ollama pull gemma3:12b). By default Gemma 3 12B answers with each fix
(five arms, no run without fixes -- the reference is the 29 Sept Gemma 27B score) and nothing is judged:
send the zip back for grading. About 1-3 minutes per question per arm on one GPU, 1-3 hours in all.
Re-running resumes: answers already written are kept (--fresh starts over).

    python scripts/legal_data/ab_fixes.py                       # Gemma 12B, the five fix arms, no judge
    python scripts/legal_data/ab_fixes.py --arms extract,doctrines
    python scripts/legal_data/ab_fixes.py --arms baseline,all --model gemma3:27b-it-qat --judge
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "legal_txt" / "Evals"
BASELINE_REPORT = ROOT / "evals" / "gemma27b_full_judged" / "report_gemma27b_full.json"
BASELINE_JUDGED = ROOT / "evals" / "gemma27b_full_judged" / "judged_gemma27b_full.jsonl"

# id -> (root cause, the fix aimed at it, what the 29 Sept answer got wrong, patterns the answer must
# contain -- all of them -- to count as stating the missing point; a crude check, the judge decides).
QUESTIONS = {
    "IL-063": ("omission, section retrieved", "whole_sections / completeness",
               "dropped the 'unless the blanket stipulation is unreasonable' qualifier of s.6",
               [r"בלתי סביר"]),
    "IL-093": ("omission, section retrieved", "whole_sections / completeness",
               "dropped 'unless rescission would be unjust' from s.7",
               [r"בלתי צודק|לא צודק"]),
    "IL-301": ("omission, section retrieved", "whole_sections / xref",
               "gave the 15 days per order but not the 30-day overall cap of s.17",
               [r"(?:30|שלושים)\s+(?:ה)?ימים|(?:30|שלושים)\s+יום"]),
    "IL-030": ("omission, section retrieved", "xref / completeness",
               "missed the prevention exception of s.28 and a possible extension",
               [r"סעיף 28|מנע"]),
    "IL-033": ("right law, wrong section", "toc",
               "awarded restitution under s.21; the contract is void (s.30) and restitution is discretionary (s.31)",
               [r"סעיף 31|31 ל", r"שיקול דעת|רשאי"]),
    "IL-034": ("right law, wrong section", "toc / doctrines",
               "relied on electricity-supply rules; misses good faith in exercising a right (s.39)",
               [r"סעיף 39|39 ל|תום לב"]),
    "IL-259": ("right law, wrong section", "toc",
               "answered 'no'; resignation over a material worsening of conditions counts as dismissal, s.11(a)",
               [r"^\W*(?:מסקנה\W*)?כן", r"הרעה מוחשית"]),
    "IL-180": ("case-law doctrine", "doctrines",
               "concluded the husband keeps what is registered to him; misses the case-law presumption of sharing",
               [r"הלכת השיתוף|חזקת (?:ה)?שיתוף"]),
    "IL-344": ("amendment timeline", "doctrines",
               "said manslaughter still exists; Amendment 137 abolished it in 2019",
               [r"בוטל|אינה קיימת|אינה עוד", r"137|2019"]),
    "IL-058": ("amendment timeline", "doctrines",
               "didn't say which version applies; a 2024 contract stays under the old s.25",
               [r"^\W*לא\b|הנוסח הקודם|הנוסח הישן", r"נכרת|חודש"]),
}

ARMS = {
    "baseline": "baseline",
    "whole_sections": "whole_sections",
    "toc_xref": "toc,xref",
    "extract": "completeness",
    "doctrines": "doctrines",
    "all": "whole_sections,toc,xref,reg_cap,grouped,completeness,doctrines",
}

SECTION_RE = re.compile(r"\d{1,4}[א-ת]{0,3}")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def base_section(value) -> str | None:
    found = SECTION_RE.search(str(value or ""))
    return found.group(0) if found else None


def governing_retrieved(gold: dict, row: dict) -> bool | None:
    sys.path.insert(0, str(ROOT / "src"))
    from docslides.legal.corpus_retrieval import same_law

    cites = [c for c in gold.get("citations", []) if c.get("law")]
    if not cites:
        return None
    return any(same_law(c["law"], r.get("title") or "") and base_section(r.get("section")) == base_section(c["section"])
               for c in cites for r in row.get("retrieved", []))


def states_missing_point(qid: str, answer: str) -> bool:
    return all(re.search(p, answer.strip(), re.MULTILINE) for p in QUESTIONS[qid][3])


def score_py(work: Path) -> Path:
    """score.py from the eval set's zip (the set's own, unmodified scorer)."""
    target = work / "israeli_legal_eval"
    if not (target / "score.py").exists():
        with zipfile.ZipFile(EVAL / "israeli_legal_eval.zip") as archive:
            archive.extractall(work)
    return target / "score.py"


TIMED_OUT = -9


def run(cmd: list[str], env: dict, log: Path, timeout_s: float | None = None) -> int:
    """The command's exit code, or TIMED_OUT when it was stopped after `timeout_s` (eval_run.py writes
    each answer as it goes, so only the question in progress is lost)."""
    print("  $", " ".join(cmd), flush=True)
    with log.open("a", encoding="utf-8", errors="replace") as out:
        out.write(f"\n$ {' '.join(cmd)}\n")
        out.flush()
        try:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, check=False,
                                  timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return TIMED_OUT
    return proc.returncode


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", default=",".join(a for a in ARMS if a != "baseline"),
                   help=f"comma-separated, from: {', '.join(ARMS)} (default: every arm with a fix)")
    p.add_argument("--ids", default=",".join(QUESTIONS), help="a subset of the ten question ids")
    p.add_argument("--model", default="gemma3:12b", help="the Ollama model that answers")
    p.add_argument("--base-url", default="http://localhost:11434")
    p.add_argument("--thinking", action="store_true", help="for a model with a thinking mode (Qwen3); off for Gemma")
    p.add_argument("--context-length", type=int, default=16384, help="Ollama num_ctx, as in the Kaggle runs")
    p.add_argument("--max-tokens", type=int, default=6144)
    p.add_argument("--timeout-s", type=int, default=1800, help="per model call")
    p.add_argument("--device", default=None, help="embedder/reranker device: cuda, cpu (default: library choice)")
    p.add_argument("--judge", action="store_true", help="also grade with the judge and score.py")
    p.add_argument("--judge-model", default=None, help="judge model (default: legal.judge_model, else --model)")
    p.add_argument("--out", default=None, help="default: data/legal/eval/ab_fixes_<model>")
    p.add_argument("--fresh", action="store_true", help="discard earlier answers in --out")
    p.add_argument("--stop-after-s", type=float, default=None,
                   help="time budget for answering, all arms together (Kaggle): the arm running when it "
                        "ends is stopped, later arms are skipped, and the report covers what was answered")
    a = p.parse_args()

    arms = [x for x in a.arms.split(",") if x]
    unknown = [x for x in arms if x not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; choose from {', '.join(ARMS)}")
    ids = [x for x in a.ids.split(",") if x]
    out = (ROOT / (a.out or f"data/legal/eval/ab_fixes_{re.sub(r'[^A-Za-z0-9.]+', '_', a.model)}")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "data" / "legal_corpus_vectordb" / "laws").exists():
        print("warning: data/legal_corpus_vectordb/laws not found -- answers will have no retrieval", file=sys.stderr)

    questions = {q["id"]: q for q in load_jsonl(EVAL / "questions.jsonl")}
    gold = {g["id"]: g for g in load_jsonl(EVAL / "gold.jsonl")}
    subset = out / "questions.jsonl"
    write_jsonl(subset, [questions[i] for i in ids])
    gold_subset = out / "gold.jsonl"
    write_jsonl(gold_subset, [gold[i] for i in ids])

    env = {**os.environ,
           "DOCSLIDES_LEGAL_ORCHESTRATOR_BACKEND": "ollama",
           "DOCSLIDES_LEGAL_ORCHESTRATOR_BASE_URL": a.base_url,
           "DOCSLIDES_LEGAL_ORCHESTRATOR_MODEL": a.model,
           "DOCSLIDES_LEGAL_ORCHESTRATOR_MAX_MODEL_LEN": str(a.context_length),
           "DOCSLIDES_LEGAL_ORCHESTRATOR_SUPPORTS_THINKING": str(a.thinking).lower(),
           "DOCSLIDES_LEGAL_ORCHESTRATOR_REQUEST_TIMEOUT_S": str(a.timeout_s),
           "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",  # Windows: Hebrew in logs and files
           "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    if a.judge_model:
        env["DOCSLIDES_LEGAL_JUDGE_MODEL"] = a.judge_model
    if a.device:
        # eval_run.py reads legal.retrieval.device from the config; a one-line override file does it.
        cfg = out / "device_config.yaml"
        base = (ROOT / os.environ.get("DOCSLIDES_CONFIG", "config/config.yaml")).read_text(encoding="utf-8")
        cfg.write_text(base, encoding="utf-8")
        try:
            import yaml

            data = yaml.safe_load(base)
            data.setdefault("legal", {}).setdefault("retrieval", {})["device"] = a.device
            cfg.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
            env["DOCSLIDES_CONFIG"] = str(cfg)
        except ImportError:
            print("warning: PyYAML missing, --device ignored", file=sys.stderr)

    log = out / "run.log"
    timings: dict[str, float] = {}
    deadline = time.monotonic() + a.stop_after_s if a.stop_after_s else None
    for arm in arms:
        left = deadline - time.monotonic() if deadline else None
        if left is not None and left < 120:
            print(f"\n== {arm}: skipped, the time budget is spent", flush=True)
            continue
        arm_dir = out / arm
        arm_dir.mkdir(exist_ok=True)
        answers = arm_dir / "answers.jsonl"
        if a.fresh and answers.exists():
            answers.unlink()
        print(f"\n== {arm} ({ARMS[arm] or 'no fix'})", flush=True)
        started = time.monotonic()
        cmd = [sys.executable, "scripts/legal_data/eval_run.py", "answer", "--questions", str(subset),
               "--out", str(answers), "--max-tokens", str(a.max_tokens)]
        if ARMS[arm]:
            cmd += ["--variant", ARMS[arm]]
        if not a.thinking:
            cmd.append("--no-thinking")
        code = run(cmd, env, log, timeout_s=left)
        if code == TIMED_OUT:
            print(f"  {arm}: stopped at the time budget", file=sys.stderr)
        elif code:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            print(f"  answer step failed for {arm}; last lines of {log}:\n    " + "\n    ".join(tail), file=sys.stderr)
            if not (arm_dir / "answers.jsonl").exists():
                raise SystemExit("stopping: the first arm failed before answering anything -- send the lines above")
        timings[arm] = time.monotonic() - started
        if a.judge:
            scorer = score_py(out)
            requests = arm_dir / "judge_requests.jsonl"
            judged = arm_dir / "judged.jsonl"
            run([sys.executable, str(scorer), "prepare", "--questions", str(subset), "--gold", str(gold_subset),
                 "--answers", str(answers), "--out", str(requests)], env, log)
            judge_cmd = [sys.executable, "scripts/legal_data/eval_run.py", "judge", "--requests", str(requests),
                         "--out", str(judged)]
            if not a.thinking:
                judge_cmd.append("--no-thinking")
            run(judge_cmd, env, log)
            run([sys.executable, str(scorer), "report", "--gold", str(gold_subset), "--answers", str(answers),
                 "--judged", str(judged), "--json-out", str(arm_dir / "report.json")], env, log)

    report(out, arms, ids, gold, timings)


def report(out: Path, arms: list[str], ids: list[str], gold: dict, timings: dict[str, float]) -> None:
    before = {}
    if BASELINE_REPORT.exists():
        before = {r["id"]: r["score"] for r in json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))["rows"]}
    table: dict[str, dict[str, dict]] = {}
    for arm in arms:
        rows = {r["id"]: r for r in load_jsonl(out / arm / "answers.jsonl")}
        scores = {}
        if (out / arm / "report.json").exists():
            scores = {r["id"]: r["score"] for r in json.loads((out / arm / "report.json").read_text(encoding="utf-8"))["rows"]}
        for qid in ids:
            row = rows.get(qid)
            if row is None:
                continue
            table.setdefault(qid, {})[arm] = {
                "retrieved": governing_retrieved(gold[qid], row),
                "states_point": states_missing_point(qid, row.get("answer", "")),
                "score": scores.get(qid),
                "error": row.get("error"),
                "repairs": row.get("repairs", []),
                "cards": row.get("doctrine_cards", []),
                "missing": row.get("missing", []),
                "added": sorted({r.get("via") for r in row.get("retrieved", []) if r.get("via") in ("t", "x")}),
            }

    def cell(v: dict | None) -> str:
        if v is None:
            return "–"
        if v["error"]:
            return "error"
        parts = ["S" if v["retrieved"] else ("s" if v["retrieved"] is False else "·"),
                 "✓" if v["states_point"] else "✗"]
        if v["score"] is not None:
            parts.append(f"{v['score']:.2f}")
        return " ".join(parts)

    legend = ("Each cell: **S** governing section retrieved / **s** not; **✓** the answer states the point the "
              "29 Sept answer left out / **✗** not (a pattern check -- read the answers); then the judge score "
              "when run with --judge. \"29 Sept 27B\" is the judged score of the 29 Sept Gemma 27B run, "
              "which had none of these fixes.")
    lines = ["# A/B of the 30 Sept fixes on ten failed questions", "", legend, "",
             "| id | root cause | aimed fix | 29 Sept 27B | " + " | ".join(arms) + " |",
             "|---|---|---|--:|" + "|".join("---" for _ in arms) + "|"]
    for qid in ids:
        cause, fix, _, _ = QUESTIONS[qid]
        b = before.get(qid)
        lines.append(f"| {qid} | {cause} | {fix} | {'' if b is None else f'{b:.2f}'} | "
                     + " | ".join(cell(table.get(qid, {}).get(arm)) for arm in arms) + " |")
    totals = ["| **total** | | | " + (f"{sum(before.get(i, 0) for i in ids):.2f}" if before else "") + " | "]
    cells = []
    for arm in arms:
        vals = [table.get(i, {}).get(arm) for i in ids]
        vals = [v for v in vals if v]
        point = sum(v["states_point"] for v in vals)
        retrieved = sum(bool(v["retrieved"]) for v in vals)
        judged = [v["score"] for v in vals if v["score"] is not None]
        cells.append(f"S {retrieved}/{len(vals)}, ✓ {point}/{len(vals)}" + (f", {sum(judged):.2f}" if judged else ""))
    lines.append(totals[0] + " | ".join(cells) + " |")
    lines += ["", "## Minutes per arm", ""] + [f"- {arm}: {timings.get(arm, 0) / 60:.1f}" for arm in arms if arm in timings]
    lines += ["", "## What each question tests", ""]
    lines += [f"- **{qid}** ({QUESTIONS[qid][0]}): {QUESTIONS[qid][2]}" for qid in ids]
    lines += ["", "## Details", ""]
    for qid in ids:
        for arm in arms:
            v = table.get(qid, {}).get(arm)
            if v and (v["cards"] or v["missing"] or v["added"] or v["repairs"]):
                lines.append(f"- {qid} / {arm}: cards {v['cards']}, sections added via {v['added']}, "
                             f"completeness found missing {v['missing']}, repairs {v['repairs']}")
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + "\n".join(lines[:len(ids) + 8]))

    bundle = out.parent / f"{out.name}_results.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in out.rglob("*"):
            if path.is_file() and "israeli_legal_eval" not in path.parts:
                archive.write(path, path.relative_to(out.parent))
    print(f"\ncomparison: {out / 'comparison.md'}\nsend back: {bundle}")


if __name__ == "__main__":
    main()
