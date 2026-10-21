#!/usr/bin/env python3
"""The two reports of an eval run: a detailed technical report (to find what to fix) and an attorney
review page (Hebrew, no system or component names), both self-contained HTML.

    # optional: the judge's English notes rewritten as plain Hebrew for the attorney page
    # (calls the judge model, legal.judge_model / DOCSLIDES_LEGAL_JUDGE_MODEL; resumable)
    python scripts/legal_data/review_reports.py translate --run-dir RUN --out RUN/notes_he.json

    python scripts/legal_data/review_reports.py build --eval-dir legal_txt/Evals/israeli_legal_eval_v3 \
        --run-dir RUN --out-dir RUN/reports [--meta RUN/run_meta.json]

RUN holds what notebooks/kaggle_legal_eval_attorney_review.ipynb writes: questions/answers_rag.jsonl,
questions/judged_rag.jsonl, cases/answers_rag.jsonl, cases/judged_rag.jsonl, llm_trace/*.jsonl and
(optionally) notes_he.json. Item scores follow the eval's score.py exactly: multiple choice by the
letter, yes/no as half label + quarter per correctness point, everything else correctness / 2.

The attorney page names no model, tool or pipeline part. Its strings are scrubbed (BANNED below) and
the build fails if any banned term is left in it.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import html
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

CATEGORY_EN = {
    "rule_recall": "Rule recall", "rule_conclusion": "Yes/no conclusion", "mcq_bar": "Bar-exam multiple choice",
    "rule_application": "Applying the rule to facts", "interpretation": "Interpretation",
    "citation_grounding": "Finding the section", "issue_spotting": "Issue spotting",
    "abstention": "Should refuse or correct", "temporal_amendment": "Amendments and dates",
}
CATEGORY_HE = {
    "rule_recall": "זכירת כלל", "rule_conclusion": "מסקנה (כן/לא)", "mcq_bar": "שאלות אמריקאיות (מתכונת בחינת הלשכה)",
    "rule_application": "יישום הכלל על עובדות", "interpretation": "פרשנות", "citation_grounding": "איתור החוק והסעיף",
    "issue_spotting": "זיהוי סוגיות", "abstention": "הימנעות ותיקון הנחה שגויה", "temporal_amendment": "תיקוני חקיקה ותחולה בזמן",
}
CATEGORY_HE_DESC = {
    "rule_recall": "שאלה קצרה על כלל שבחוק", "rule_conclusion": "תשובת כן/לא ונימוק", "mcq_bar": "ארבע אפשרויות, נבדק לפי האות",
    "rule_application": "ניתוח עובדות במבנה סוגיה–כלל–יישום–מסקנה", "interpretation": "פירוש מונח או הוראה",
    "citation_grounding": "באיזה חוק ובאיזה סעיף", "issue_spotting": "רשימת כל הסוגיות שעולות מהעובדות",
    "abstention": "הנחה שגויה, חוק שאינו קיים, בקשה פסולה או שאלה מחוץ לתחום", "temporal_amendment": "איזה נוסח חל ומתי",
}
SECTIONS = ["facts_summary", "chronology", "legal_issues", "deadlines", "red_flags", "missing_info", "deliverable", "next_step"]
SECTION_HE = {"facts_summary": "תקציר עובדתי", "chronology": "ציר זמן", "legal_issues": "סוגיות משפטיות",
              "deadlines": "מועדים ודחיפות", "red_flags": "סתירות וסימני אזהרה", "missing_info": "מידע חסר",
              "deliverable": "טיוטת המסמך", "next_step": "הצעד הבא"}
PENALTY_HE = {"fabricated_fact_or_date": "עובדה או תאריך שהומצאו", "fabricated_or_wrong_citation": "אסמכתה שגויה או מומצאת",
              "must_not_violation": "מסקנה אסורה לפי המחוון", "unsafe_or_unethical_advice": "ייעוץ לא בטוח או לא אתי"}

# Terms the attorney page must not contain (system, model and pipeline names). Checked case-insensitively.
BANNED = re.compile(
    r"gemma|ollama|gpt|qwen|llama|claude|openai|anthropic|kaggle|\bllm\b|\brag\b|retriev|rerank|embedding|bm25|"
    r"chroma|lancedb|\bchunks?\b|\bprompt\b|\bmodel\b|vector|pipeline|\btoken|"
    r"מודל|אחזור|צ'אנק|וקטור|פרומפט|בינה מלאכותית|מנוע חיפוש",
    re.IGNORECASE)
# Leaked chat-template or prompt tags in answers.
LEAKED_TAGS = re.compile(r"</?(?:start_of_turn|end_of_turn|context|case_law|reference_material|work_file|case_file)>|<\|[^|>]{1,40}\|>")

HEB_LETTER = re.compile(r"^[\s\*\(\[\"'״]*([אבגד])(?=[\s\.\)\]:,\-–\*]|$)")


# ---------------------------------------------------------------- loading and scoring (score.py rules)

def load_jsonl(path: Path) -> list[dict]:
    if not path or not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_letter(ans: str) -> str | None:
    first = (ans or "").strip().splitlines()[0] if (ans or "").strip() else ""
    m = HEB_LETTER.match(first)
    if m:
        return m.group(1)
    m = re.search(r"(?:תשובה|התשובה)\s*[:\-]?\s*([אבגד])(?=[\s\.\)]|$)", ans or "")
    return m.group(1) if m else None


def parse_label(ans: str) -> str | None:
    words = re.findall(r"[א-ת]+", (ans or "")[:40])
    return words[0] if words and words[0] in ("כן", "לא") else None


def item_score(g: dict, answer: str, j: dict | None) -> tuple[float | None, dict]:
    extra = {}
    if g["scoring"] == "exact_letter":
        letter = parse_letter(answer)
        extra["letter"] = letter
        return float(letter == g.get("answer_letter")), extra
    if g["scoring"] == "label_then_judge":
        label = parse_label(answer)
        extra["label"] = label
        lab = float(label == g.get("answer_label"))
        extra["labelCorrect"] = bool(lab)
        return (lab if not j else 0.5 * lab + 0.25 * j["correctness"]), extra
    return (j["correctness"] / 2 if j else None), extra


# ---------------------------------------------------------------- the governing section retrieved (as in the Kaggle notebooks)

NIQQUD = re.compile(r"[֑-ׇ]")
YEAR = re.compile(r"(?:^|\s)ה?תש[א-ת]{1,3}(?=\s|$)|\d{4}")
SECTION = re.compile(r"\d{1,4}[א-ת]{0,3}")


def law_key(name: str) -> str:
    name = NIQQUD.sub("", (name or "").replace("־", " ").split(",")[0])
    name = re.sub(r"[\"'״׳`()\[\]\-–—\s]+", " ", name)
    name = YEAR.sub(" ", name)
    return " ".join(w[1:] if w.startswith("ה") and len(w) > 3 else w for w in name.split())


def same_law(a: str, b: str) -> bool:
    a, b = law_key(a), law_key(b)
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def base_section(s) -> str | None:
    found = SECTION.search(str(s or ""))
    return found.group(0) if found else None


def governing_retrieved(gold: dict, retrieved: list[dict]) -> bool | None:
    cites = [c for c in gold.get("citations", []) if c.get("law")]
    if not cites:
        return None
    for c in cites:
        sec = base_section(c.get("section"))
        for r in retrieved or []:
            if same_law(c["law"], r.get("title") or "") and (sec is None or base_section(r.get("section")) == sec):
                return True
    return False


# ---------------------------------------------------------------- run data

def run_files(run: Path) -> dict:
    return {"answers": run / "questions" / "answers_rag.jsonl", "judged": run / "questions" / "judged_rag.jsonl",
            "case_answers": run / "cases" / "answers_rag.jsonl", "case_judged": run / "cases" / "judged_rag.jsonl",
            "trace": run / "llm_trace", "notes_he": run / "notes_he.json"}


def trace_rows(trace_dir: Path) -> list[dict]:
    if not trace_dir or not trace_dir.exists():
        return []
    rows = []
    for p in sorted(trace_dir.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                r = json.loads(line)
                r.pop("messages", None)  # the full prompts stay in the trace files
                rows.append(r)
    return rows


def key_marks(g: dict, j: dict | None, hits: set) -> list[bool] | None:
    """Which gold key points the judge found. Full marks with no list means all of them; a lower
    grade with no list says nothing about which ones, so no marks."""
    if not j:
        return None
    if not hits:
        return [True] * len(g["key_points"]) if j.get("correctness") == 2 else None
    return [i + 1 in hits for i in range(len(g["key_points"]))]


def clean_answer(text: str) -> str:
    return LEAKED_TAGS.sub("", text or "").strip()


def collect(eval_dir: Path, run: Path, meta: dict) -> dict:
    files = run_files(run)
    questions = {q["id"]: q for q in load_jsonl(eval_dir / "questions.jsonl")}
    gold = load_jsonl(eval_dir / "gold.jsonl")
    answers = {a["id"]: a for a in load_jsonl(files["answers"])}
    judged = {j["id"]: j for j in load_jsonl(files["judged"]) if not str(j.get("note", "")).startswith("judge failed")}
    failed_judge = {j["id"] for j in load_jsonl(files["judged"]) if str(j.get("note", "")).startswith("judge failed")}
    notes_he = json.loads(files["notes_he"].read_text(encoding="utf-8")) if files["notes_he"].exists() else {}
    traces = trace_rows(files["trace"])
    by_job = collections.defaultdict(list)
    for t in traces:
        by_job[t.get("job_id")].append(t)

    items = []
    for g in gold:
        q = questions.get(g["id"], {})
        a = answers.get(g["id"])
        j = judged.get(g["id"])
        answer = clean_answer((a or {}).get("answer", ""))
        score, extra = item_score(g, answer, j) if a else (None, {})
        if a and (a.get("error") or not answer):
            score = 0.0 if g["scoring"] != "judge" or j else None
        retrieved = (a or {}).get("retrieved") or []
        calls = by_job.get(g["id"], [])
        gov = governing_retrieved(g, retrieved) if a else None
        if not a:
            diag = "not_answered"
        elif a.get("error") or not answer:
            diag = "empty_or_error"
        elif score is None:
            diag = "not_judged"
        elif score >= 1:
            diag = "right"
        elif g["category"] == "abstention":
            diag = "abstention_failed"
        elif extra.get("labelCorrect") is False:
            diag = "wrong_label"
        elif gov is False:
            diag = "source_not_found"
        elif gov is True:
            diag = "source_found_answer_wrong"
        else:
            diag = "answer_wrong"
        hits = set((j or {}).get("key_points_hit") or [])
        items.append({
            "id": g["id"], "cat": g["category"], "catEn": CATEGORY_EN.get(g["category"], g["category"]),
            "catHe": CATEGORY_HE.get(g["category"], g["category"]), "area": g["area"], "conf": g["confidence"],
            "scoring": g["scoring"], "instructions": q.get("instructions", ""), "question": q.get("question", ""),
            "options": q.get("options"), "gold": g["gold_answer"], "keyPoints": g["key_points"],
            "keyHit": key_marks(g, j, hits),
            "cites": [f"{c['law']} {c['section']}".strip() for c in g["citations"]],
            "goldLetter": g.get("answer_letter"), "goldLabel": g.get("answer_label"),
            "answer": answer, "error": (a or {}).get("error"), "score": score, **extra,
            "correctness": (j or {}).get("correctness"), "grounding": (j or {}).get("grounding"),
            "halluc": bool((j or {}).get("hallucination")), "needsReview": bool((j or {}).get("needs_review")),
            "note": (j or {}).get("note", ""), "noteHe": notes_he.get(f"q:{g['id']}", ""),
            "judgeFailed": g["id"] in failed_judge, "diag": diag, "govRetrieved": gov,
            "retrieved": [f"{r.get('title', '')} §{r.get('section', '')}" + (f" [{r['via']}]" if r.get("via") else "")
                          for r in retrieved],
            "caselaw": [c.get("citation") for c in (a or {}).get("caselaw") or [] if c.get("citation")],
            "plan": [f"{p.get('issue', '')} — {p.get('law', '')} {', '.join(p.get('sections') or [])}"
                     for p in (a or {}).get("plan") or []],
            "scope": ((a or {}).get("scope") or {}).get("scope"), "repairs": (a or {}).get("repairs") or [],
            "seconds": round(sum(c.get("seconds") or 0 for c in calls), 1), "calls": len(calls),
            "lengthStops": sum(c.get("done_reason") == "length" for c in calls if c.get("call_site") != "legal_eval_judge"),
            "maxPrompt": max((c.get("prompt_tokens") or 0 for c in calls), default=0),
        })

    cases_gold = json.loads((eval_dir / "cases_gold.json").read_text(encoding="utf-8"))["cases"] \
        if (eval_dir / "cases_gold.json").exists() else []
    case_answers = {c["case_id"]: c for c in load_jsonl(files["case_answers"])}
    case_judged = {c["case_id"]: c for c in load_jsonl(files["case_judged"])}
    limit = meta.get("case_limit")
    cases = []
    for cg in cases_gold[:limit] if limit else cases_gold:
        cid = cg["case_id"]
        a, j = case_answers.get(cid), case_judged.get(cid)
        if j and str(j.get("note", "")).startswith("judge failed"):
            j = None
        calls = by_job.get(cid, [])
        case_md = eval_dir / "cases" / cg["file"]
        cases.append({
            "id": cid, "title": cg.get("title") or cg["file"], "refDate": cg.get("reference_date"),
            "caseText": case_md.read_text(encoding="utf-8") if case_md.exists() else "",
            "workFile": clean_answer((a or {}).get("work_file", "")), "error": (a or {}).get("error"),
            "weights": cg.get("rubric_weights") or {}, "gold": {k: cg.get(k) for k in (
                "chronology", "legal_issues", "deadlines", "red_flags", "missing_info", "deliverable_checks", "next_step", "must_not")},
            "sections": {s: (j or {}).get(s) for s in SECTIONS} if j else None,
            "penalties": [{**p, "typeHe": PENALTY_HE.get(p.get("type"), p.get("type")),
                           "detailHe": notes_he.get(f"c:{cid}:p{i}", "")} for i, p in enumerate((j or {}).get("penalties") or [])],
            "missed": (j or {}).get("missed_items") or [],
            "missedHe": [notes_he.get(f"c:{cid}:m{i}", "") for i in range(len((j or {}).get("missed_items") or []))],
            "total": (j or {}).get("total"), "note": (j or {}).get("note", ""), "noteHe": notes_he.get(f"c:{cid}:note", ""),
            "retrieved": [f"{r.get('title', '')} §{r.get('section', '')}" for r in (a or {}).get("retrieved") or []],
            "caselaw": [c.get("citation") for c in (a or {}).get("caselaw") or [] if c.get("citation")],
            "repairs": (a or {}).get("repairs") or [], "unsupported": (a or {}).get("unsupported_values") or [],
            "contextTrimmed": (a or {}).get("context_trimmed"),
            "seconds": round(sum(c.get("seconds") or 0 for c in calls), 1), "calls": len(calls),
            "lengthStops": sum(c.get("done_reason") == "length" for c in calls if c.get("call_site") != "legal_eval_judge"),
        })

    return {"items": items, "cases": cases, "traces": traces, "summary": summarize(items, cases, traces), "meta": meta}


def pct(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(100 * sum(values) / len(values), 1) if values else None


def summarize(items: list[dict], cases: list[dict], traces: list[dict]) -> dict:
    scored = [i for i in items if i["score"] is not None]

    def group(key, label=None):
        out = []
        agg = collections.defaultdict(list)
        for i in items:
            agg[i[key]].append(i)
        for k, rows in sorted(agg.items(), key=lambda kv: -len(kv[1])):
            sc = [r["score"] for r in rows if r["score"] is not None]
            out.append({"key": k, "label": label(k) if label else k, "n": len(rows), "scored": len(sc), "score": pct(sc),
                        "right": sum(s >= 1 for s in sc), "partial": sum(0 < s < 1 for s in sc), "wrong": sum(s == 0 for s in sc),
                        "unscored": len(rows) - len(sc)})
        return out

    judged = [i for i in items if i["correctness"] is not None]
    mcq = [i for i in items if i["scoring"] == "exact_letter" and i["score"] is not None]
    yn = [i for i in items if i["scoring"] == "label_then_judge" and "labelCorrect" in i]
    case_totals = [c["total"] for c in cases if c["total"] is not None]
    sites = collections.defaultdict(list)
    for t in traces:
        sites[t.get("call_site")].append(t)
    gov = [i["govRetrieved"] for i in items if i["govRetrieved"] is not None]
    return {
        "questions": len(items), "answered": sum(i["diag"] != "not_answered" for i in items), "scored": len(scored),
        "overall": pct(i["score"] for i in scored),
        "right": sum(i["score"] >= 1 for i in scored), "partial": sum(0 < i["score"] < 1 for i in scored),
        "wrong": sum(i["score"] == 0 for i in scored),
        "mcq": [sum(i["score"] for i in mcq), len(mcq)], "yesno": [sum(i["labelCorrect"] for i in yn), len(yn)],
        "byCategory": group("cat", lambda k: CATEGORY_EN.get(k, k)), "byArea": group("area"), "byConfidence": group("conf"),
        "grounding": dict(collections.Counter(i["grounding"] for i in judged)),
        "hallucinations": sum(i["halluc"] for i in judged), "judged": len(judged),
        "needsReview": [i["id"] for i in items if i["needsReview"]],
        "diag": dict(collections.Counter(i["diag"] for i in items)),
        "govRetrievedPct": pct(gov), "govRetrievedN": len(gov),
        "repairs": dict(collections.Counter(r for i in items for r in i["repairs"])),
        "casesAvg": round(statistics.mean(case_totals), 1) if case_totals else None, "casesScored": len(case_totals),
        "caseSectionAvg": {s: round(statistics.mean([c["sections"][s] or 0 for c in cases if c["sections"]]), 1)
                           for s in SECTIONS} if any(c["sections"] for c in cases) else {},
        "callSites": {site: {"n": len(cs), "lengthStops": sum(c.get("done_reason") == "length" for c in cs),
                             "errors": sum(bool(c.get("error")) for c in cs),
                             "reasoned": sum(bool((c.get("reasoning") or "").strip()) for c in cs),
                             "medianS": round(statistics.median([c.get("seconds") or 0 for c in cs]), 1),
                             "totalMin": round(sum(c.get("seconds") or 0 for c in cs) / 60, 1),
                             "maxPrompt": max((c.get("prompt_tokens") or 0) for c in cs),
                             "maxOut": max((c.get("completion_tokens") or 0) for c in cs)}
                      for site, cs in sorted(sites.items(), key=lambda kv: str(kv[0]))},
    }


# ---------------------------------------------------------------- findings for the technical report

def findings(data: dict) -> list[dict]:
    s, items = data["summary"], data["items"]
    out = []
    cats = [c for c in s["byCategory"] if c["score"] is not None]
    if cats:
        weak = sorted(cats, key=lambda c: c["score"])[:3]
        out.append({"sev": "bad" if weak[0]["score"] < 60 else "partial",
                    "text": "Weakest categories: " + "; ".join(f"{c['label']} {c['score']}% ({c['n']})" for c in weak) + "."})
        strong = [c for c in cats if c["score"] >= 85]
        if strong:
            out.append({"sev": "good", "text": "At or above 85%: " + "; ".join(f"{c['label']} {c['score']}%" for c in strong) + "."})
    miss = s["diag"].get("source_not_found", 0)
    wrong_found = s["diag"].get("source_found_answer_wrong", 0)
    if miss or wrong_found:
        out.append({"sev": "bad" if miss > wrong_found else "partial",
                    "text": f"Of the answers below full marks, {miss} had the governing section missing from the retrieved "
                            f"context (a retrieval fix) and {wrong_found} had it retrieved but still answered wrongly or "
                            "incompletely (a prompting / reasoning fix)."})
    if s["govRetrievedPct"] is not None:
        out.append({"sev": "good" if s["govRetrievedPct"] >= 85 else "partial",
                    "text": f"Governing section retrieved for {s['govRetrievedPct']}% of {s['govRetrievedN']} items with a gold citation."})
    if s["hallucinations"]:
        ids = [i["id"] for i in items if i["halluc"]]
        out.append({"sev": "bad", "text": f"{s['hallucinations']} judged answers invent a law, section, number or date: {', '.join(ids)}."})
    mis = [i["id"] for i in items if i["grounding"] == "misgrounded"]
    if mis:
        out.append({"sev": "partial", "text": f"{len(mis)} answers cite a wrong or non-existent source for a correct statement: {', '.join(mis)}."})
    if s["diag"].get("wrong_label"):
        ids = [i["id"] for i in items if i["diag"] == "wrong_label"]
        out.append({"sev": "partial", "text": f"Yes/no label wrong (or missing as the first word) in {len(ids)}: {', '.join(ids)}."})
    if s["diag"].get("abstention_failed"):
        ids = [i["id"] for i in items if i["diag"] == "abstention_failed"]
        out.append({"sev": "bad", "text": f"Should have refused or corrected the premise but didn't: {', '.join(ids)}."})
    bad_calls = {k: v for k, v in s["callSites"].items() if v["lengthStops"] or v["errors"]}
    if bad_calls:
        out.append({"sev": "bad", "text": "Calls cut off at the token limit or failed: " + "; ".join(
            f"{k} {v['lengthStops']} cut / {v['errors']} errors" for k, v in bad_calls.items()) + "."})
    empty = [i["id"] for i in items if i["diag"] in ("empty_or_error", "not_answered")]
    if empty:
        out.append({"sev": "bad", "text": f"{len(empty)} questions have no usable answer: {', '.join(empty[:15])}."})
    unjudged = [i["id"] for i in items if i["diag"] == "not_judged" or i["judgeFailed"]]
    if unjudged:
        out.append({"sev": "partial", "text": f"{len(unjudged)} answers have no judge result (excluded from the score): {', '.join(unjudged[:15])}."})
    if s["casesAvg"] is not None:
        sec = s["caseSectionAvg"]
        weights = (data["cases"][0]["weights"] if data["cases"] else {}) or {}
        rel = sorted(((k, v / weights[k]) for k, v in sec.items() if weights.get(k)), key=lambda kv: kv[1])[:3]
        out.append({"sev": "bad" if s["casesAvg"] < 50 else "partial",
                    "text": f"Case work files average {s['casesAvg']}/100 over {s['casesScored']}. Weakest sections relative to "
                            "their weight: " + "; ".join(f"{k} {100 * v:.0f}%" for k, v in rel) + "."})
        pens = collections.Counter(p["type"] for c in data["cases"] for p in c["penalties"])
        if pens:
            out.append({"sev": "bad", "text": "Case penalties: " + ", ".join(f"{k} ×{v}" for k, v in pens.items()) + "."})
    if s["needsReview"]:
        out.append({"sev": "partial", "text": f"Judge flagged {len(s['needsReview'])} medium-confidence items for a lawyer: "
                                              f"{', '.join(s['needsReview'])}."})
    return out


RECOMMEND = {
    "source_not_found": "Retrieval: for these items add the governing law to the planner's hints or doctrine cards, check "
                        "the law-name aliases and the table-of-contents navigation, and confirm the section is in the corpus "
                        "(corpus_coverage.md).",
    "source_found_answer_wrong": "Answering: the section was in the context. Compare the answer with the gold key points; "
                                 "missing secondary issues point at the completeness check and the extract step.",
    "wrong_label": "Yes/no: check the label repair (legal_eval_label_check) and that the explanation agrees with the label.",
    "abstention_failed": "Scope check: these should be flagged by legal_eval_scope (false premise, fictional law, harmful).",
    "empty_or_error": "Stability: see the trace for errors and calls stopped at length; raise the answer budget or trim context.",
}


# ---------------------------------------------------------------- HTML

def dump(data) -> str:
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


THEME = """
:root{--page:#f4f5f2;--surface:#fff;--surface-2:#ebede8;--ink:#151a17;--ink-2:#464e49;--muted:#6f7772;--line:#d7dbd3;
--accent:#1d4e89;--accent-soft:#e3ebf6;--good:#1f7a3a;--good-soft:#e1f1e5;--partial:#9a5f00;--partial-soft:#f8ecd4;
--bad:#b3261e;--bad-soft:#f8e1df;--sans:"IBM Plex Sans","IBM Plex Sans Hebrew",system-ui,"Segoe UI",Arial,sans-serif;
--mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;--display:"Frank Ruhl Libre",Georgia,serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#101311;--surface:#171b18;
--surface-2:#1f2420;--ink:#eef1ec;--ink-2:#bcc3bd;--muted:#8a938c;--line:#2d342f;--accent:#86aee6;--accent-soft:#1b2738;
--good:#62c47f;--good-soft:#17281c;--partial:#e2ab40;--partial-soft:#2c2413;--bad:#f07d73;--bad-soft:#321a18}}
:root[data-theme="dark"]{color-scheme:dark;--page:#101311;--surface:#171b18;--surface-2:#1f2420;--ink:#eef1ec;--ink-2:#bcc3bd;
--muted:#8a938c;--line:#2d342f;--accent:#86aee6;--accent-soft:#1b2738;--good:#62c47f;--good-soft:#17281c;--partial:#e2ab40;
--partial-soft:#2c2413;--bad:#f07d73;--bad-soft:#321a18}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.55}
.wrap{max-width:1160px;margin:0 auto;padding:36px 16px 80px;display:grid;gap:44px}
h1,h2,h3{margin:0;text-wrap:balance}h1{font-family:var(--display);font-size:clamp(1.8rem,4.5vw,2.6rem);line-height:1.1}
h2{font-family:var(--display);font-size:1.5rem}h3{font-size:1rem;font-weight:600}p{margin:0;max-width:75ch}
.eyebrow{font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}.lede{color:var(--ink-2);font-size:1.05rem}
.muted{color:var(--muted)}.mono{font-family:var(--mono);font-size:.85em}section{display:grid;gap:16px}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.fig{background:var(--surface);padding:14px 16px;display:grid;gap:3px}.fig .v{font-family:var(--display);font-size:2rem;font-weight:700;line-height:1;font-variant-numeric:tabular-nums}
.fig .l{font-weight:600;font-size:.88rem}.fig .s{font-size:.82rem;color:var(--ink-2)}
.findings{list-style:none;margin:0;padding:0;display:grid;gap:10px}.findings li{display:grid;grid-template-columns:10px 1fr;gap:10px}
.findings li::before{content:"";width:10px;height:10px;border-radius:50%;margin-top:.45em;background:var(--muted)}
.findings li.sev-bad::before{background:var(--bad)}.findings li.sev-partial::before{background:var(--partial)}.findings li.sev-good::before{background:var(--good)}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:start;padding:8px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:.78rem;font-weight:600;color:var(--muted);background:var(--surface-2)}tr:last-child td{border-bottom:0}td.num,th.num{text-align:end}
.bar{height:8px;border-radius:4px;background:var(--surface-2);overflow:hidden;min-width:80px}.bar>i{display:block;height:100%;background:var(--accent)}
.toolbar{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.toolbar.sticky{position:sticky;top:0;z-index:5;box-shadow:0 4px 14px rgba(0,0,0,.06)}
.field{display:flex;gap:6px;align-items:center;font-size:.86rem;color:var(--ink-2)}
input,select,textarea,button{font:inherit;color:inherit}input[type=text],input[type=search],select,textarea{border:1px solid var(--line);background:var(--page);border-radius:7px;padding:5px 8px;max-width:100%}
.btn{border:1px solid var(--line);background:var(--surface-2);border-radius:7px;padding:6px 12px;cursor:pointer;font-size:.88rem}
.btn:hover{border-color:var(--accent)}.btn.primary{background:var(--accent);border-color:var(--accent);color:var(--surface);font-weight:600}
.list{display:grid;gap:14px}.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.card>header{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:center;padding:12px 14px;border-bottom:1px solid var(--line)}
.card .body{padding:12px 14px;display:grid;gap:12px}.chip{font-size:.76rem;border-radius:999px;padding:2px 9px;background:var(--surface-2);color:var(--ink-2);white-space:nowrap}
.chip.good{background:var(--good-soft);color:var(--good)}.chip.partial{background:var(--partial-soft);color:var(--partial)}.chip.bad{background:var(--bad-soft);color:var(--bad)}
.chip.acc{background:var(--accent-soft);color:var(--accent)}.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.panel{border:1px solid var(--line);border-radius:9px;padding:10px 12px;background:var(--page);min-width:0}.panel h4{margin:0 0 6px;font-size:.8rem;color:var(--muted);font-weight:600}
.md{overflow-wrap:anywhere}.md p{margin:.3em 0}.md table{font-size:.86rem}.md ul,.md ol{margin:.3em 0;padding-inline-start:1.3em}
details>summary{cursor:pointer;color:var(--accent);font-size:.88rem}ul.kp{margin:0;padding-inline-start:1.2em}ul.kp li.hit{color:var(--good)}ul.kp li.miss{color:var(--bad)}
.small{font-size:.84rem}.he{direction:rtl;text-align:right;font-family:"IBM Plex Sans Hebrew",var(--sans)}
.review{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;border-top:1px dashed var(--line);padding-top:10px}
.review label{font-size:.86rem;display:flex;gap:5px;align-items:center}.review textarea{flex:1 1 100%;min-height:54px}
.done{outline:2px solid var(--good-soft)}
@media (max-width:600px){.wrap{padding-top:24px}.fig .v{font-size:1.6rem}}
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Frank+Ruhl+Libre:wght@500;700&family=IBM+Plex+Mono:wght@400'
         '&family=IBM+Plex+Sans+Hebrew:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">')
MARKED = '<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>'

COMMON_JS = r"""
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function md(s){ const safe = String(s ?? "").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  try { if (window.marked) return marked.parse(safe, {mangle:false, headerIds:false}); } catch(e) {}
  return "<p>" + safe.replace(/\n/g,"<br>") + "</p>"; }
const fmt = v => v == null ? "–" : (Math.round(v*10)/10).toString();
function store(k, v){ try { if (v === undefined) return JSON.parse(localStorage.getItem(k) || "null"); localStorage.setItem(k, JSON.stringify(v)); } catch(e) { return null; } }
function gradeOf(s){ return s == null ? ["unscored",""] : s >= 1 ? ["good","right"] : s > 0 ? ["partial","partial"] : ["bad","wrong"]; }
function bar(p){ return `<div class="bar" aria-hidden="true"><i style="width:${Math.max(0,Math.min(100,p||0))}%"></i></div>`; }
"""


def technical_html(data: dict) -> str:
    s, meta = data["summary"], data["meta"]
    data_js = {"summary": s, "items": data["items"], "cases": data["cases"], "meta": meta, "findings": findings(data),
               "recommend": RECOMMEND, "sectionWeights": data["cases"][0]["weights"] if data["cases"] else {}}
    title = "Legal Eval Technical Report"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>{FONTS}<style>{THEME}</style></head><body><div class="wrap">
<header style="display:grid;gap:12px"><div class="eyebrow">{html.escape(meta.get('run_name', ''))} · {html.escape(meta.get('date', ''))}</div>
<h1>Technical report: answers, grading and where to improve</h1>
<p class="lede">{html.escape(meta.get('description', ''))}</p></header>
<section><h2>Headline</h2><div class="figs" id="figs"></div><ul class="findings" id="findings"></ul></section>
<section><h2>By category</h2><div class="tablewrap"><table><thead><tr><th>Category</th><th class="num">n</th><th class="num">Score</th><th></th>
<th class="num">Right</th><th class="num">Partial</th><th class="num">Wrong</th><th class="num">Unscored</th><th class="num">Baseline</th></tr></thead><tbody id="cat"></tbody></table></div></section>
<section><h2>Error diagnosis</h2><p class="muted">Every item below full marks, by where it went wrong. "Source not found": the gold
governing section was not among the retrieved statute excerpts. "Source found": it was, and the answer still missed key points.</p>
<div class="tablewrap"><table><thead><tr><th>Diagnosis</th><th class="num">Items</th><th>What to look at</th></tr></thead><tbody id="diag"></tbody></table></div></section>
<section><h2>By area and confidence</h2><div class="grid2"><div class="tablewrap"><table><thead><tr><th>Area</th><th class="num">n</th><th class="num">Score</th></tr></thead><tbody id="area"></tbody></table></div>
<div class="tablewrap"><table><thead><tr><th>Gold confidence</th><th class="num">n</th><th class="num">Score</th></tr></thead><tbody id="conf"></tbody></table></div></div>
<div class="tablewrap"><table><thead><tr><th>Grounding (judged)</th><th class="num">Items</th></tr></thead><tbody id="grd"></tbody></table></div></section>
<section><h2>Paralegal cases</h2><div class="tablewrap"><table><thead id="caseHead"></thead><tbody id="caseRows"></tbody></table></div><div class="list" id="caseList"></div></section>
<section><h2>Calls and timing</h2><div class="tablewrap"><table><thead><tr><th>Call site</th><th class="num">Calls</th><th class="num">Cut at length</th>
<th class="num">Errors</th><th class="num">With reasoning</th><th class="num">Median s</th><th class="num">Total min</th><th class="num">Max prompt tok</th><th class="num">Max output tok</th></tr></thead><tbody id="calls"></tbody></table></div>
<p class="muted small" id="repairs"></p></section>
<section><h2>All questions</h2><div class="toolbar sticky">
<label class="field">Category <select id="fCat"><option value="">All</option></select></label>
<label class="field">Grade <select id="fGrade"><option value="">All</option><option value="good">Right</option><option value="partial">Partial</option><option value="bad">Wrong</option><option value="unscored">Unscored</option></select></label>
<label class="field">Diagnosis <select id="fDiag"><option value="">All</option></select></label>
<label class="field"><input type="checkbox" id="fHall"> Hallucination / misgrounded</label>
<label class="field">Search <input type="search" id="fQ" placeholder="ID, law, word"></label><span class="muted small" id="shown"></span></div>
<div class="list" id="qList"></div></section>
<section><h2>Run settings</h2><div class="panel mono small" style="white-space:pre-wrap" id="settings"></div></section>
</div>{MARKED}<script>{COMMON_JS}
const D = {dump(data_js)};
const S = D.summary, B = (D.meta.baseline || {{}});
const DIAG = {{right:"Right", source_not_found:"Source not found", source_found_answer_wrong:"Source found, answer wrong",
  answer_wrong:"Answer wrong (no gold citation)", wrong_label:"Yes/no label wrong", abstention_failed:"Should have refused / corrected",
  empty_or_error:"Empty or error", not_judged:"Not judged", not_answered:"Not answered"}};
const figs = [[fmt(S.overall)+"%","Overall score",`${{S.scored}} of ${{S.questions}} scored · score.py rules`],
  [`${{S.right}} / ${{S.partial}} / ${{S.wrong}}`,"Right / partial / wrong",""],
  [`${{S.mcq[0]}}/${{S.mcq[1]}}`,"Multiple choice","exact letter"],[`${{S.yesno[0]}}/${{S.yesno[1]}}`,"Yes/no label","first word"],
  [fmt(S.govRetrievedPct)+"%","Governing section retrieved",`${{S.govRetrievedN}} items with a citation`],
  [String(S.hallucinations),"Hallucinations",`of ${{S.judged}} judged`],
  [S.casesAvg==null?"–":fmt(S.casesAvg),"Cases, mean /100",`${{S.casesScored}} judged`]];
document.getElementById("figs").innerHTML = figs.map(f=>`<div class="fig"><div class="v">${{esc(f[0])}}</div><div class="l">${{esc(f[1])}}</div><div class="s">${{esc(f[2])}}</div></div>`).join("");
document.getElementById("findings").innerHTML = D.findings.map(f=>`<li class="sev-${{f.sev}}"><div>${{esc(f.text)}}</div></li>`).join("");
const bc = B.by_category || {{}};
document.getElementById("cat").innerHTML = S.byCategory.map(c=>`<tr><td>${{esc(c.label)}}</td><td class="num">${{c.n}}</td><td class="num">${{fmt(c.score)}}%</td><td>${{bar(c.score)}}</td><td class="num">${{c.right}}</td><td class="num">${{c.partial}}</td><td class="num">${{c.wrong}}</td><td class="num">${{c.unscored}}</td><td class="num">${{bc[c.key]==null?"–":fmt(bc[c.key])+"%"}}</td></tr>`).join("")
  + (B.label ? `<tr><td colspan="9" class="muted small">Baseline: ${{esc(B.label)}} (different questions, same criteria)</td></tr>` : "");
document.getElementById("diag").innerHTML = Object.entries(S.diag).filter(([k])=>k!=="right").sort((a,b)=>b[1]-a[1]).map(([k,v])=>`<tr><td>${{esc(DIAG[k]||k)}}</td><td class="num">${{v}}</td><td class="small">${{esc(D.recommend[k]||"")}}</td></tr>`).join("") || `<tr><td colspan="3">No errors.</td></tr>`;
const simple = (id, rows) => document.getElementById(id).innerHTML = rows.map(c=>`<tr><td>${{esc(c.label)}}</td><td class="num">${{c.n}}</td><td class="num">${{fmt(c.score)}}%</td></tr>`).join("");
simple("area", S.byArea); simple("conf", S.byConfidence);
document.getElementById("grd").innerHTML = Object.entries(S.grounding).map(([k,v])=>`<tr><td>${{esc(k)}}</td><td class="num">${{v}}</td></tr>`).join("");
const SEC = ["facts_summary","chronology","legal_issues","deadlines","red_flags","missing_info","deliverable","next_step"];
document.getElementById("caseHead").innerHTML = `<tr><th>Case</th>${{SEC.map(s=>`<th class="num">${{s.replace("_"," ")}} /${{D.sectionWeights[s]||""}}</th>`).join("")}}<th class="num">Penalties</th><th class="num">Total</th><th class="num">Baseline</th><th class="num">Min</th></tr>`;
const bcs = B.cases || {{}};
document.getElementById("caseRows").innerHTML = D.cases.map(c=>`<tr><td>${{esc(c.id)}}</td>${{SEC.map(s=>`<td class="num">${{c.sections?fmt(c.sections[s]):"–"}}</td>`).join("")}}<td class="num">${{c.penalties.reduce((a,p)=>a+(p.points||0),0)}}</td><td class="num"><b>${{fmt(c.total)}}</b></td><td class="num">${{bcs[c.id]==null?"–":bcs[c.id]}}</td><td class="num">${{fmt(c.seconds/60)}}</td></tr>`).join("");
document.getElementById("caseList").innerHTML = D.cases.map(c=>`<article class="card"><header><b>${{esc(c.id)}}</b><span class="he">${{esc(c.title)}}</span><span class="chip ${{c.total==null?"":c.total>=70?"good":c.total>=45?"partial":"bad"}}">${{fmt(c.total)}}/100</span>
 ${{c.lengthStops?`<span class="chip bad">${{c.lengthStops}} calls cut at length</span>`:""}}${{c.contextTrimmed?`<span class="chip partial">context trimmed</span>`:""}}${{c.error?`<span class="chip bad">${{esc(c.error)}}</span>`:""}}</header>
 <div class="body"><p class="small">${{esc(c.note)}}</p>
 ${{c.penalties.length?`<div class="panel"><h4>Penalties</h4><ul class="small">${{c.penalties.map(p=>`<li><b>${{esc(p.type)}}</b> (${{p.points}}): ${{esc(p.detail)}}</li>`).join("")}}</ul></div>`:""}}
 ${{c.missed.length?`<div class="panel"><h4>Missed gold items</h4><ul class="small he">${{c.missed.map(m=>`<li>${{esc(m)}}</li>`).join("")}}</ul></div>`:""}}
 <div class="grid2"><div class="panel"><h4>Retrieved (${{c.retrieved.length}}) · case law ${{c.caselaw.length}}</h4><div class="small he">${{c.retrieved.map(esc).join("<br>")}}${{c.caselaw.length?"<br><b>"+c.caselaw.map(esc).join(", ")+"</b>":""}}</div></div>
 <div class="panel"><h4>Repairs · unsupported values</h4><div class="small">${{esc(c.repairs.join(", ")||"none")}} · ${{esc(c.unsupported.join(", ")||"none")}}</div></div></div>
 <details><summary>Work file</summary><div class="md he">${{md(c.workFile)}}</div></details>
 <details><summary>Case file</summary><div class="md he">${{md(c.caseText)}}</div></details>
 <details><summary>Gold rubric</summary><pre class="mono small he" style="white-space:pre-wrap">${{esc(JSON.stringify(c.gold,null,1))}}</pre></details></div></article>`).join("");
document.getElementById("calls").innerHTML = Object.entries(S.callSites).map(([k,v])=>`<tr><td class="mono">${{esc(k)}}</td><td class="num">${{v.n}}</td><td class="num">${{v.lengthStops}}</td><td class="num">${{v.errors}}</td><td class="num">${{v.reasoned}}</td><td class="num">${{v.medianS}}</td><td class="num">${{v.totalMin}}</td><td class="num">${{v.maxPrompt}}</td><td class="num">${{v.maxOut}}</td></tr>`).join("");
document.getElementById("repairs").textContent = "Answer repairs applied: " + (Object.entries(S.repairs).map(([k,v])=>`${{k}} ×${{v}}`).join(", ") || "none");
document.getElementById("settings").textContent = JSON.stringify(D.meta, null, 1);
const fCat=document.getElementById("fCat"), fGrade=document.getElementById("fGrade"), fDiag=document.getElementById("fDiag"), fHall=document.getElementById("fHall"), fQ=document.getElementById("fQ");
S.byCategory.forEach(c=>fCat.insertAdjacentHTML("beforeend",`<option value="${{esc(c.key)}}">${{esc(c.label)}}</option>`));
Object.keys(S.diag).forEach(k=>fDiag.insertAdjacentHTML("beforeend",`<option value="${{k}}">${{esc(DIAG[k]||k)}}</option>`));
function card(i){{ const [g,gl]=gradeOf(i.score);
 const opts = i.options ? `<ul class="small he">${{Object.entries(i.options).map(([k,v])=>`<li><b>${{k}}</b> ${{esc(v)}}${{k===i.goldLetter?" ✓":""}}</li>`).join("")}}</ul>` : "";
 const kp = i.keyPoints.map((k,n)=>`<li class="${{i.keyHit?(i.keyHit[n]?"hit":"miss"):""}}">${{esc(k)}}</li>`).join("");
 return `<article class="card"><header><b class="mono">${{esc(i.id)}}</b><span class="chip">${{esc(i.catEn)}}</span><span class="chip he">${{esc(i.area)}}</span>
 <span class="chip ${{g}}">${{gl||"unscored"}} ${{i.score==null?"":fmt(i.score*100)+"%"}}</span><span class="chip">${{esc(DIAG[i.diag]||i.diag)}}</span>
 ${{i.halluc?'<span class="chip bad">hallucination</span>':""}}${{i.grounding==="misgrounded"?'<span class="chip partial">misgrounded</span>':""}}${{i.needsReview?'<span class="chip acc">needs review</span>':""}}
 ${{i.lengthStops?`<span class="chip bad">${{i.lengthStops}} cut</span>`:""}}<span class="chip">${{i.conf}} conf</span><span class="muted small">${{fmt(i.seconds)}} s · ${{i.calls}} calls</span></header>
 <div class="body"><div class="he"><b>${{esc(i.question)}}</b>${{opts}}</div>
 <div class="grid2"><div class="panel"><h4>Answer ${{i.letter?"· letter "+esc(i.letter):""}}${{i.label?"· label "+esc(i.label):""}}</h4><div class="md he">${{i.error?`<p class="chip bad">${{esc(i.error)}}</p>`:""}}${{md(i.answer)}}</div></div>
 <div class="panel"><h4>Gold ${{i.goldLetter?"· "+esc(i.goldLetter):""}}${{i.goldLabel?"· "+esc(i.goldLabel):""}}</h4><div class="he">${{esc(i.gold)}}</div><ul class="kp small he">${{kp}}</ul><div class="small muted he">${{esc(i.cites.join(" · "))}}</div></div></div>
 ${{i.note?`<p class="small"><b>Judge:</b> ${{esc(i.note)}} <span class="muted">(correctness ${{i.correctness}}, ${{esc(i.grounding||"")}})</span></p>`:""}}
 <details><summary>Retrieved ${{i.retrieved.length}} · governing ${{i.govRetrieved==null?"n/a":i.govRetrieved?"found":"missing"}} · case law ${{i.caselaw.length}} · plan ${{i.plan.length}} · repairs ${{i.repairs.length}}</summary>
 <div class="grid2"><div class="panel"><h4>Retrieved statutes</h4><div class="small he">${{i.retrieved.map(esc).join("<br>")||"none"}}</div></div>
 <div class="panel"><h4>Plan · scope · case law · repairs</h4><div class="small he">${{i.plan.map(esc).join("<br>")||"no plan"}}<br>scope: ${{esc(i.scope||"in scope")}}<br>case law: ${{esc(i.caselaw.join(", ")||"none")}}<br>repairs: ${{esc(i.repairs.join(", ")||"none")}}</div></div></div></details></div></article>`; }}
function render(){{ const q=fQ.value.trim().toLowerCase();
 const rows=D.items.filter(i=>(!fCat.value||i.cat===fCat.value)&&(!fGrade.value||gradeOf(i.score)[0]===fGrade.value)&&(!fDiag.value||i.diag===fDiag.value)
  &&(!fHall.checked||i.halluc||i.grounding==="misgrounded")&&(!q||(i.id+" "+i.question+" "+i.answer+" "+i.cites.join(" ")).toLowerCase().includes(q)));
 document.getElementById("qList").innerHTML=rows.map(card).join(""); document.getElementById("shown").textContent=`${{rows.length}} shown`; }}
[fCat,fGrade,fDiag,fHall].forEach(e=>e.addEventListener("change",render)); fQ.addEventListener("input",render); render();
</script></body></html>"""


def scrub(text: str) -> str:
    """A string for the attorney page: no system or component names left."""
    text = LEAKED_TAGS.sub("", text or "")
    text = re.sub(r"\bthe model\b|\bmodel\b", "the system", text, flags=re.I)
    text = text.replace("המודל", "המערכת").replace("מודל", "מערכת")
    return BANNED.sub("", text)


def attorney_html(data: dict) -> str:
    s, meta = data["summary"], data["meta"]
    items = []
    for i in data["items"]:
        items.append({k: i[k] for k in ("id", "cat", "catHe", "area", "conf", "scoring", "question", "options", "gold", "keyPoints",
                                        "keyHit", "cites", "goldLetter", "goldLabel", "score", "correctness", "needsReview")}
                     | {"answer": scrub(i["answer"]), "note": scrub(i["noteHe"]) if re.search(r"[א-ת]", i["noteHe"] or "") else "",
                        "halluc": i["halluc"], "misgrounded": i["grounding"] == "misgrounded", "answered": i["diag"] != "not_answered"})
    cases = []
    for c in data["cases"]:
        cases.append({"id": c["id"], "title": c["title"], "refDate": c["refDate"], "caseText": c["caseText"],
                      "workFile": scrub(c["workFile"]), "weights": c["weights"], "sections": c["sections"], "total": c["total"],
                      "note": scrub(c["noteHe"]), "gold": c["gold"],
                      "penalties": [{"typeHe": p["typeHe"], "points": p.get("points"),
                                     "detail": scrub(p["detailHe"] or (p.get("detail") if not re.search(r"[A-Za-z]{3}", p.get("detail") or "") else ""))}
                                    for p in c["penalties"]],
                      "missed": [scrub(he or (m if not re.search(r"[A-Za-z]{3}", m) else "")) for m, he in zip(c["missed"], c["missedHe"])]})
    cats = [{**c, "label": CATEGORY_HE.get(c["key"], c["key"]), "desc": CATEGORY_HE_DESC.get(c["key"], "")} for c in s["byCategory"]]
    stats = {"overall": s["overall"], "questions": s["questions"], "scored": s["scored"], "right": s["right"], "partial": s["partial"],
             "wrong": s["wrong"], "mcq": s["mcq"], "yesno": s["yesno"], "cats": cats, "areas": s["byArea"],
             "conf": s["byConfidence"], "casesAvg": s["casesAvg"], "casesScored": s["casesScored"],
             "halluc": s["hallucinations"], "judged": s["judged"]}
    payload = {"stats": stats, "items": items, "cases": cases, "sectionHe": SECTION_HE, "key": "attorney-review:" + meta.get("run_name", "run"),
               "date": meta.get("date_he", meta.get("date", ""))}
    blob = dump(payload)
    leftover = BANNED.findall(blob)
    if leftover:
        raise SystemExit(f"attorney page would name system components: {sorted(set(x.lower() for x in leftover))}")
    return f"""<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>סקירת מבחן משפטי</title>{FONTS}<style>{THEME}
body{{font-family:"IBM Plex Sans Hebrew",var(--sans)}} .findings li{{direction:rtl}}</style></head><body><div class="wrap">
<header style="display:grid;gap:12px"><div class="eyebrow">מבחן במשפט הישראלי · 100 שאלות ו-5 תיקי עבודה · {html.escape(payload['date'])}</div>
<h1>סקירת המבחן המשפטי לעורך הדין</h1>
<p class="lede">המערכת ענתה על 100 שאלות במשפט הישראלי והכינה 5 תיקי עבודה לעורך דין. כל תשובה קיבלה ציון ראשוני לפי תשובת המבחן
(תשובת הזהב) ולפי מחוון המבחן. בעמוד זה מוצגים, לכל שאלה ולכל תיק, השאלה, תשובת המערכת, תשובת המבחן והציון הראשוני. נבקש לסמן
אם הציון נכון, אם תשובת המבחן עצמה נכונה, ולהוסיף הערה כשצריך לתקן.</p></header>
<section aria-labelledby="h-stats"><h2 id="h-stats">סטטיסטיקה כללית</h2><div class="figs" id="figs"></div></section>
<section aria-labelledby="h-cat"><h2 id="h-cat">תוצאות לפי סוג השאלה</h2>
<p class="muted">לכל אחת מתשע הקטגוריות של המבחן: מספר השאלות, הציון הממוצע, ומספר התשובות הנכונות, החלקיות והשגויות.</p>
<div class="tablewrap"><table><thead><tr><th>סוג השאלה</th><th>מה נבדק</th><th class="num">שאלות</th><th class="num">ציון</th><th></th>
<th class="num">נכונות</th><th class="num">חלקיות</th><th class="num">שגויות</th></tr></thead><tbody id="cat"></tbody></table></div></section>
<section aria-labelledby="h-area"><h2 id="h-area">תוצאות לפי תחום משפט</h2><div class="grid2">
<div class="tablewrap"><table><thead><tr><th>תחום</th><th class="num">שאלות</th><th class="num">ציון</th><th></th></tr></thead><tbody id="area"></tbody></table></div>
<div class="tablewrap"><table><thead><tr><th>רמת הוודאות של תשובת המבחן</th><th class="num">שאלות</th><th class="num">ציון</th></tr></thead><tbody id="conf"></tbody></table>
<p class="small muted" style="padding:8px 12px">"בינונית": תשובת מבחן שכוללת מספר סעיף, תקופה או תיקון חקיקה שכדאי לאמת.</p></div></div></section>
<section aria-labelledby="h-how"><h2 id="h-how">איך לסקור</h2>
<p>ליד כל שאלה: (1) האם הציון הראשוני נכון – <b>מסכים</b>, <b>לא מסכים</b> או <b>לא בטוח</b>; (2) האם תשובת המבחן נכונה – <b>נכונה</b> או <b>צריכה תיקון</b>;
(3) הערה חופשית. הסימונים נשמרים בדפדפן זה בלבד. בסיום לחצו <b>הורדת הסקירה</b> (קובץ CSV) או <b>העתקה</b> ושלחו אותה בדוא"ל.</p>
<p class="muted">ציונים: <b>נכונה</b> – כל נקודות המפתח נכונות ואין טעות מהותית. <b>חלקית</b> – חסרות נקודות מפתח או יש טעות קלה. <b>שגויה</b> –
מסקנה שגויה, טעות משפטית מהותית או אין תשובה של ממש. שאלות אמריקאיות נבדקות לפי האות. בשאלות כן/לא המילה הראשונה שווה מחצית מהציון והנימוק את המחצית השנייה.
בתיקי העבודה כל חלק מקבל ניקוד לפי משקלו במחוון (100 בסך הכול), ומופחתות נקודות על עובדות או אסמכתאות שהומצאו ועל מסקנות אסורות.</p></section>
<div class="toolbar sticky" role="region" aria-label="סקירה"><label class="field">שם הסוקר <input type="text" id="reviewer" autocomplete="name"></label>
<span class="small muted" id="progress"></span><button class="btn primary" id="dl" type="button">הורדת הסקירה</button><button class="btn" id="copy" type="button">העתקה</button>
<span class="small" id="toast" aria-live="polite"></span></div>
<section aria-labelledby="h-cases"><h2 id="h-cases">תיקי העבודה</h2>
<p>לכל תיק קיבלה המערכת את כל מסמכי התיק והתבקשה להכין תיק עבודה בשמונה חלקים: תקציר, ציר זמן, סוגיות, מועדים, סימני אזהרה, מידע חסר, טיוטת מסמך וצעד הבא.</p>
<div class="tablewrap"><table><thead id="caseHead"></thead><tbody id="caseRows"></tbody></table></div><div class="list" id="caseList"></div></section>
<section aria-labelledby="h-q"><h2 id="h-q">כל השאלות</h2><div class="toolbar">
<label class="field">סוג <select id="fCat"><option value="">הכול</option></select></label>
<label class="field">ציון <select id="fGrade"><option value="">הכול</option><option value="good">נכונה</option><option value="partial">חלקית</option><option value="bad">שגויה</option></select></label>
<label class="field"><input type="checkbox" id="fTodo"> טרם נסקרו</label><label class="field"><input type="checkbox" id="fFlag"> מסומנות לבדיקה</label>
<label class="field">חיפוש <input type="search" id="fQ" placeholder="מספר, חוק או מילה"></label><span class="small muted" id="shown"></span></div>
<div class="list" id="qList"></div></section>
<section><h2>על המבחן</h2><p>המבחן כולל 100 שאלות חדשות בתשעה סוגים ובאחד-עשר תחומי משפט, ו-5 תיקי עבודה עם מלכודות מכוונות (סתירות בין מסמכים, מועדים
מטעים, ניגוד עניינים ובקשות פסולות). תשובות המבחן נכתבו לפי הדין הישראלי ולא נבדקו עדיין בידי עורך דין, ולכן הסקירה שלכם היא גם בדיקה של המבחן עצמו.
הציון הראשוני ניתן אוטומטית לפי המחוון, ומטרת הסקירה היא לאשר או לתקן אותו.</p></section>
</div>{MARKED}<script>{COMMON_JS}
const D = {blob};
const T = D.stats, KEY = D.key;
const GRADE_HE = {{good:"נכונה", partial:"חלקית", bad:"שגויה", unscored:"ללא ציון"}};
const pc = v => v==null ? "–" : fmt(v) + "%";
document.getElementById("figs").innerHTML = [[pc(T.overall),"ציון כולל",`${{T.scored}} מתוך ${{T.questions}} שאלות`],
 [`${{T.right}} / ${{T.partial}} / ${{T.wrong}}`,"נכונות / חלקיות / שגויות",""],[`${{T.mcq[0]}}/${{T.mcq[1]}}`,"שאלות אמריקאיות","אות נכונה"],
 [`${{T.yesno[0]}}/${{T.yesno[1]}}`,"כן/לא","המילה הראשונה נכונה"],[String(T.halluc),"אסמכתאות או נתונים שהומצאו",`מתוך ${{T.judged}} תשובות שנבדקו`],
 [T.casesAvg==null?"–":fmt(T.casesAvg),"תיקי עבודה, ממוצע מתוך 100",`${{T.casesScored}} תיקים`]]
 .map(f=>`<div class="fig"><div class="v">${{esc(f[0])}}</div><div class="l">${{esc(f[1])}}</div><div class="s">${{esc(f[2])}}</div></div>`).join("");
document.getElementById("cat").innerHTML = T.cats.map(c=>`<tr><td><b>${{esc(c.label)}}</b></td><td class="small muted">${{esc(c.desc)}}</td><td class="num">${{c.n}}</td><td class="num">${{pc(c.score)}}</td><td>${{bar(c.score)}}</td><td class="num">${{c.right}}</td><td class="num">${{c.partial}}</td><td class="num">${{c.wrong}}</td></tr>`).join("")
 + `<tr><td><b>סך הכול</b></td><td></td><td class="num"><b>${{T.questions}}</b></td><td class="num"><b>${{pc(T.overall)}}</b></td><td>${{bar(T.overall)}}</td><td class="num">${{T.right}}</td><td class="num">${{T.partial}}</td><td class="num">${{T.wrong}}</td></tr>`;
document.getElementById("area").innerHTML = T.areas.map(c=>`<tr><td>${{esc(c.label)}}</td><td class="num">${{c.n}}</td><td class="num">${{pc(c.score)}}</td><td>${{bar(c.score)}}</td></tr>`).join("");
document.getElementById("conf").innerHTML = T.conf.map(c=>`<tr><td>${{c.key==="high"?"גבוהה":c.key==="medium"?"בינונית":esc(c.key)}}</td><td class="num">${{c.n}}</td><td class="num">${{pc(c.score)}}</td></tr>`).join("");
const SEC = Object.keys(D.sectionHe);
const W = (D.cases[0]||{{}}).weights || {{}};
document.getElementById("caseHead").innerHTML = `<tr><th>תיק</th>${{SEC.map(s=>`<th class="num">${{esc(D.sectionHe[s])}} (${{W[s]||""}})</th>`).join("")}}<th class="num">הפחתות</th><th class="num">סך הכול</th></tr>`;
document.getElementById("caseRows").innerHTML = D.cases.map(c=>`<tr><td>${{esc(c.title)}}</td>${{SEC.map(s=>`<td class="num">${{c.sections?fmt(c.sections[s]):"–"}}</td>`).join("")}}<td class="num">${{c.penalties.reduce((a,p)=>a+(p.points||0),0)}}</td><td class="num"><b>${{fmt(c.total)}}</b></td></tr>`).join("");
let R = store(KEY) || {{reviewer:"", marks:{{}}}};
const save = () => {{ store(KEY, R); progress(); }};
const rev = document.getElementById("reviewer"); rev.value = R.reviewer || ""; rev.addEventListener("input", ()=>{{R.reviewer=rev.value; save();}});
function controls(id, kind){{ const m = R.marks[id] || {{}};
 const radio = (name, val, label) => `<label><input type="radio" name="${{name}}-${{id}}" value="${{val}}" ${{m[name]===val?"checked":""}} data-id="${{id}}" data-f="${{name}}"> ${{label}}</label>`;
 return `<div class="review" data-rid="${{id}}"><span class="small"><b>הציון${{kind==="case"?" של התיק":""}}:</b></span>${{radio("agree","yes","מסכים")}}${{radio("agree","no","לא מסכים")}}${{radio("agree","unsure","לא בטוח")}}
 <span class="small"><b>${{kind==="case"?"המחוון":"תשובת המבחן"}}:</b></span>${{radio("gold","ok","נכונה")}}${{radio("gold","fix","צריכה תיקון")}}
 <label class="small">ציון מתוקן <input type="text" size="5" data-id="${{id}}" data-f="newScore" value="${{esc(m.newScore||"")}}" inputmode="decimal"></label>
 <textarea data-id="${{id}}" data-f="comment" placeholder="הערה (למשל: הדין הנכון, הסעיף הנכון או מה חסר)">${{esc(m.comment||"")}}</textarea></div>`; }}
document.addEventListener("change", e=>{{ const el=e.target; if(!el.dataset||!el.dataset.id) return; (R.marks[el.dataset.id] ||= {{}})[el.dataset.f]=el.value; save(); mark(el.dataset.id); }});
document.addEventListener("input", e=>{{ const el=e.target; if(!el.dataset||!el.dataset.id||el.type==="radio") return; (R.marks[el.dataset.id] ||= {{}})[el.dataset.f]=el.value; save(); }});
const reviewed = id => {{ const m=R.marks[id]||{{}}; return !!(m.agree||m.gold||m.comment); }};
function mark(id){{ const c=document.getElementById("c-"+id); if(c) c.classList.toggle("done", reviewed(id)); }}
function caseCard(c){{ const g = c.gold || {{}}; const list = a => (a||[]).map(x=>`<li>${{esc(Array.isArray(x)?x.join(" · "):(typeof x==="object"?`${{x.issue}} – ${{x.law}} ${{x.section}}`:x))}}</li>`).join("");
 return `<article class="card ${{reviewed(c.id)?"done":""}}" id="c-${{c.id}}"><header><b>${{esc(c.title)}}</b><span class="chip">תאריך קבלת התיק: ${{esc(c.refDate)}}</span>
 <span class="chip ${{c.total==null?"":c.total>=70?"good":c.total>=45?"partial":"bad"}}">${{c.total==null?"ללא ציון":fmt(c.total)+" / 100"}}</span></header><div class="body">
 ${{c.note?`<p>${{esc(c.note)}}</p>`:""}}
 ${{c.sections?`<div class="tablewrap"><table><tbody>${{SEC.map(s=>`<tr><td>${{esc(D.sectionHe[s])}}</td><td class="num">${{fmt(c.sections[s])}} / ${{c.weights[s]}}</td><td>${{bar(100*(c.sections[s]||0)/(c.weights[s]||1))}}</td></tr>`).join("")}}</tbody></table></div>`:""}}
 ${{c.penalties.length?`<div class="panel"><h4>הפחתות</h4><ul class="small">${{c.penalties.map(p=>`<li><b>${{esc(p.typeHe)}}</b> (${{p.points}})${{p.detail?": "+esc(p.detail):""}}</li>`).join("")}}</ul></div>`:""}}
 ${{c.missed.filter(Boolean).length?`<div class="panel"><h4>פריטים מהמחוון שחסרים בתיק העבודה</h4><ul class="small">${{c.missed.filter(Boolean).map(m=>`<li>${{esc(m)}}</li>`).join("")}}</ul></div>`:""}}
 <details><summary>תיק העבודה שהכינה המערכת</summary><div class="md">${{md(c.workFile)||"<p>לא הוכן</p>"}}</div></details>
 <details><summary>מסמכי התיק</summary><div class="md">${{md(c.caseText)}}</div></details>
 <details><summary>המחוון (תשובת המבחן לתיק)</summary><div class="grid2 small">
 <div class="panel"><h4>ציר זמן</h4><ul>${{list(g.chronology)}}</ul></div><div class="panel"><h4>סוגיות משפטיות</h4><ul>${{list(g.legal_issues)}}</ul></div>
 <div class="panel"><h4>מועדים</h4><ul>${{list(g.deadlines)}}</ul></div><div class="panel"><h4>סתירות וסימני אזהרה</h4><ul>${{list(g.red_flags)}}</ul></div>
 <div class="panel"><h4>מידע חסר</h4><ul>${{list(g.missing_info)}}</ul></div><div class="panel"><h4>בדיקות לטיוטה</h4><ul>${{list(g.deliverable_checks)}}</ul></div>
 <div class="panel"><h4>הצעד הבא</h4><p>${{esc(g.next_step)}}</p></div><div class="panel"><h4>אסור לקבוע</h4><ul>${{list(g.must_not)}}</ul></div></div></details>
 ${{controls(c.id,"case")}}</div></article>`; }}
document.getElementById("caseList").innerHTML = D.cases.map(caseCard).join("");
const fCat=document.getElementById("fCat"), fGrade=document.getElementById("fGrade"), fTodo=document.getElementById("fTodo"), fFlag=document.getElementById("fFlag"), fQ=document.getElementById("fQ");
T.cats.forEach(c=>fCat.insertAdjacentHTML("beforeend",`<option value="${{esc(c.key)}}">${{esc(c.label)}}</option>`));
function qCard(i){{ const [g]=gradeOf(i.score);
 const opts = i.options ? `<ul class="small">${{Object.entries(i.options).map(([k,v])=>`<li><b>${{k}}</b> ${{esc(v)}}</li>`).join("")}}</ul>` : "";
 const kp = i.keyPoints.map((k,n)=>`<li class="${{i.keyHit?(i.keyHit[n]?"hit":"miss"):""}}">${{i.keyHit?(i.keyHit[n]?"✓ ":"✗ "):""}}${{esc(k)}}</li>`).join("");
 return `<article class="card ${{reviewed(i.id)?"done":""}}" id="c-${{i.id}}"><header><b class="mono">${{esc(i.id)}}</b><span class="chip acc">${{esc(i.catHe)}}</span><span class="chip">${{esc(i.area)}}</span>
 <span class="chip ${{g}}">${{GRADE_HE[g]}}${{i.score==null?"":" · "+fmt(i.score*100)+"%"}}</span>${{i.halluc?'<span class="chip bad">אסמכתה או נתון שהומצאו</span>':""}}${{i.misgrounded?'<span class="chip partial">אסמכתה שגויה</span>':""}}
 ${{i.needsReview?'<span class="chip partial">מסומנת לבדיקה</span>':""}}${{i.conf==="medium"?'<span class="chip">ודאות בינונית</span>':""}}</header><div class="body">
 <div><b>${{esc(i.question)}}</b>${{opts}}</div><div class="grid2">
 <div class="panel"><h4>תשובת המערכת</h4><div class="md">${{i.answered?md(i.answer):"<p>לא נענתה</p>"}}</div></div>
 <div class="panel"><h4>תשובת המבחן${{i.goldLetter?" · "+esc(i.goldLetter):""}}${{i.goldLabel?" · "+esc(i.goldLabel):""}}</h4><p>${{esc(i.gold)}}</p><h4 style="margin-top:8px">נקודות מפתח</h4><ul class="kp small">${{kp}}</ul>
 <p class="small muted">${{esc(i.cites.join(" · ")||"ללא אסמכתה")}}</p></div></div>${{i.note?`<p class="small"><b>הערת הבדיקה:</b> ${{esc(i.note)}}</p>`:""}}${{controls(i.id,"q")}}</div></article>`; }}
function render(){{ const q=fQ.value.trim();
 const rows=D.items.filter(i=>(!fCat.value||i.cat===fCat.value)&&(!fGrade.value||gradeOf(i.score)[0]===fGrade.value)&&(!fTodo.checked||!reviewed(i.id))
  &&(!fFlag.checked||i.needsReview||i.halluc||i.misgrounded)&&(!q||(i.id+" "+i.question+" "+i.answer+" "+i.gold+" "+i.cites.join(" ")).includes(q)));
 document.getElementById("qList").innerHTML=rows.map(qCard).join(""); document.getElementById("shown").textContent=`${{rows.length}} מוצגות`; }}
[fCat,fGrade,fTodo,fFlag].forEach(e=>e.addEventListener("change",render)); fQ.addEventListener("input",render); render();
function progress(){{ const all=[...D.items.map(i=>i.id),...D.cases.map(c=>c.id)]; document.getElementById("progress").textContent=`נסקרו ${{all.filter(reviewed).length}} מתוך ${{all.length}}`; }} progress();
function csv(){{ const head=["מספר","סוג","ציון ראשוני","מסכים עם הציון","תשובת המבחן","ציון מתוקן","הערה","סוקר"];
 const AG={{yes:"מסכים",no:"לא מסכים",unsure:"לא בטוח"}}, GO={{ok:"נכונה",fix:"צריכה תיקון"}};
 const rows=[...D.items.map(i=>[i.id,i.catHe,i.score==null?"":fmt(i.score*100)]),...D.cases.map(c=>[c.id,"תיק עבודה",c.total==null?"":fmt(c.total)])]
  .map(([id,t,s])=>{{ const m=R.marks[id]||{{}}; return [id,t,s,AG[m.agree]||"",GO[m.gold]||"",m.newScore||"",m.comment||"",R.reviewer||""]; }});
 return "\\ufeff"+[head,...rows].map(r=>r.map(v=>`"${{String(v).replace(/"/g,'""')}}"`).join(",")).join("\\r\\n"); }}
const toast = t => {{ const el=document.getElementById("toast"); el.textContent=t; setTimeout(()=>el.textContent="",3000); }};
document.getElementById("dl").addEventListener("click",()=>{{ const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob([csv()],{{type:"text/csv;charset=utf-8"}}));
 a.download="attorney_review.csv"; document.body.appendChild(a); a.click(); a.remove(); toast("הקובץ הורד"); }});
document.getElementById("copy").addEventListener("click",async()=>{{ try{{ await navigator.clipboard.writeText(csv()); toast("הסקירה הועתקה"); }}catch(e){{ toast("ההעתקה נכשלה – השתמשו בהורדה"); }} }});
</script></body></html>"""


def cmd_build(a) -> None:
    meta = json.loads(Path(a.meta).read_text(encoding="utf-8")) if a.meta and Path(a.meta).exists() else {}
    meta.setdefault("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    meta.setdefault("run_name", Path(a.run_dir).name)
    data = collect(Path(a.eval_dir), Path(a.run_dir), meta)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "technical_report.html").write_text(technical_html(data), encoding="utf-8")
    (out / "attorney_review.html").write_text(attorney_html(data), encoding="utf-8")
    (out / "report_data.json").write_text(json.dumps({k: data[k] for k in ("summary", "items", "cases", "meta")},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    s = data["summary"]
    print(f"overall {s['overall']}% ({s['scored']}/{s['questions']} scored), cases mean {s['casesAvg']} ({s['casesScored']})")
    for c in s["byCategory"]:
        print(f"  {c['label']:28s} n={c['n']:3d}  {c['score']}%  right {c['right']} / partial {c['partial']} / wrong {c['wrong']}")
    print("wrote", out / "technical_report.html", "and", out / "attorney_review.html")


# ---------------------------------------------------------------- translate: judge notes -> plain Hebrew

TRANSLATE_SYSTEM = """You rewrite grading notes on legal answers for an Israeli attorney. For each item, write the note as \
ONE short sentence of plain, professional Hebrew. Say what the answer got right or wrong in legal terms. Do not mention any \
model, system, software, search, retrieval, context, excerpt, prompt or token; call the graded text "התשובה" or "תיק העבודה". \
Keep law names and section numbers exactly. If the note is already Hebrew, tidy it the same way. Return JSON only."""


async def cmd_translate(a) -> None:
    from pydantic import BaseModel

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from docslides.legal.evaluation import get_judge_client
    from docslides.llm.client import ChatMessage, LLMCallSite, SamplingParams, aclose_all_clients

    class Note(BaseModel):
        k: str
        he: str

    class Notes(BaseModel):
        items: list[Note]

    files = run_files(Path(a.run_dir))
    texts = {}
    for j in load_jsonl(files["judged"]):
        if j.get("note") and not str(j["note"]).startswith("judge failed"):
            texts[f"q:{j['id']}"] = j["note"]
    for c in load_jsonl(files["case_judged"]):
        if c.get("note") and not str(c["note"]).startswith("judge failed"):
            texts[f"c:{c['case_id']}:note"] = c["note"]
        for i, p in enumerate(c.get("penalties") or []):
            if p.get("detail"):
                texts[f"c:{c['case_id']}:p{i}"] = p["detail"]
        for i, m in enumerate(c.get("missed_items") or []):
            texts[f"c:{c['case_id']}:m{i}"] = m
    out = Path(a.out)
    done = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    todo = [(k, v) for k, v in texts.items() if k not in done]
    print(f"{len(texts)} notes, {len(done)} already in Hebrew, {len(todo)} to rewrite")
    llm = get_judge_client()
    try:
        for start in range(0, len(todo), a.batch):
            batch = todo[start:start + a.batch]
            prompt = "Rewrite each note. Keep each k unchanged.\n\n" + json.dumps(
                [{"k": k, "note": v} for k, v in batch], ensure_ascii=False)
            try:
                result = await llm.complete_json(
                    [ChatMessage("system", TRANSLATE_SYSTEM), ChatMessage("user", prompt)], LLMCallSite("legal_eval_judge"),
                    schema=Notes, sampling=SamplingParams(temperature=0.0, max_tokens=a.max_tokens), enable_thinking=a.thinking)
            except Exception as exc:  # noqa: BLE001 -- a batch that fails is left out; the page shows no note for it
                print(f"batch {start // a.batch + 1} failed: {exc}")
                continue
            wanted = {k for k, _ in batch}
            for n in result.items:
                if n.k in wanted and re.search(r"[א-ת]", n.he):
                    done[n.k] = n.he.strip()
            out.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"batch {start // a.batch + 1}: {len(done)} / {len(texts)}")
    finally:
        await aclose_all_clients()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pb = sub.add_parser("build")
    pb.add_argument("--eval-dir", required=True)
    pb.add_argument("--run-dir", required=True)
    pb.add_argument("--out-dir", required=True)
    pb.add_argument("--meta")
    pt = sub.add_parser("translate")
    pt.add_argument("--run-dir", required=True)
    pt.add_argument("--out", required=True)
    pt.add_argument("--batch", type=int, default=12)
    pt.add_argument("--max-tokens", type=int, default=4096)
    pt.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    if a.cmd == "build":
        cmd_build(a)
    else:
        asyncio.run(cmd_translate(a))


if __name__ == "__main__":
    sys.exit(main())
