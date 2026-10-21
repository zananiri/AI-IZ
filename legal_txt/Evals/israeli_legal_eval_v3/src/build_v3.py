#!/usr/bin/env python3
"""Build israeli_legal_eval_v3 (100 new questions + 5 new paralegal cases) from items.py and
cases_gold_src.py. Deterministic: the same inputs give the same files.

    python legal_txt/Evals/israeli_legal_eval_v3/src/build_v3.py

Writes, next to src/: questions.jsonl (what the model sees), gold.jsonl (judge only), cases_gold.json
(judge only; the case files themselves are cases/*.md) and manifest.json. score.py and judge_prompt.md
are the v1 files, copied unmodified from legal_txt/Evals/israeli_legal_eval.zip, so the folder is a
drop-in --dir for scripts/legal_data/eval_cases.py and EVAL for the Kaggle notebooks.
"""

from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cases_gold_src import CASES, PENALTIES, RUBRIC_WEIGHTS  # noqa: E402
from common import INSTRUCTIONS, NEW_LAWS, SCORING  # noqa: E402
from items import ALL  # noqa: E402

EVALS = HERE.parents[1]
OUT = HERE.parent
PREFIX = "IL3"
# The 29 Sept / 1 Oct sample this set mirrors (notebooks/kaggle_legal_eval_gemma4_caselaw.ipynb).
EXPECTED = {"rule_recall": 22, "rule_conclusion": 18, "mcq_bar": 16, "rule_application": 12, "interpretation": 8,
            "citation_grounding": 7, "issue_spotting": 7, "abstention": 6, "temporal_amendment": 4}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    laws = dict(NEW_LAWS)
    for g in load(EVALS / "gold.jsonl"):
        for c in g["citations"]:
            laws.setdefault(c["law_code"], c["law"])

    counts = Counter(i["category"] for i in ALL)
    assert counts == Counter(EXPECTED), f"category counts {dict(counts)} != {EXPECTED}"
    letters = Counter(i["answer_letter"] for i in ALL if i["category"] == "mcq_bar")
    assert set(letters.values()) == {4}, f"multiple-choice letters not balanced: {dict(letters)}"
    texts = [i["question"] for i in ALL]
    assert len(set(texts)) == len(texts), "duplicate question text"

    questions, gold = [], []
    for n, it in enumerate(ALL, 1):
        qid = f"{PREFIX}-{n:03d}"
        fmt = it["format"]
        q = {"id": qid, "category": it["category"], "area": it["area"], "format": fmt,
             "instructions": INSTRUCTIONS[fmt], "question": it["question"]}
        if fmt == "mcq":
            q["options"] = it["options"]
            q["bar_part"] = it["bar_part"]
        questions.append(q)
        citations = []
        for cite in it["cites"]:
            code, _, section = cite.partition(":")
            assert code in laws, f"{qid}: unknown law code {code}"
            citations.append({"law_code": code, "law": laws[code], "section": section})
        g = {"id": qid, "category": it["category"], "area": it["area"], "format": fmt,
             "scoring": SCORING.get(fmt, "judge"), "gold_answer": it["gold_answer"], "key_points": it["key_points"],
             "citations": citations, "confidence": it["confidence"], "source": "v3_new", "verified": None,
             "reviewed_by_lawyer": False}
        for key in ("answer_label", "answer_letter", "abstain_kind"):
            if key in it:
                g[key] = it[key]
        gold.append(g)

    write_jsonl(OUT / "questions.jsonl", questions)
    write_jsonl(OUT / "gold.jsonl", gold)

    cases = []
    for case in CASES:
        assert (OUT / "cases" / case["file"]).exists(), f"missing case file {case['file']}"
        for issue in case["legal_issues"]:
            assert issue["law"] in laws or issue["law"] == "פסיקה", f"{case['case_id']}: unknown law {issue['law']}"
        cases.append({"case_id": case["case_id"], "file": case["file"], "title": case["title"],
                      "reference_date": case["reference_date"], "rubric_weights": RUBRIC_WEIGHTS, "penalties": PENALTIES,
                      **{k: case[k] for k in ("chronology", "legal_issues", "deadlines", "red_flags", "missing_info",
                                              "deliverable_checks", "next_step", "must_not")}})
    (OUT / "cases_gold.json").write_text(
        json.dumps({"rubric_weights": RUBRIC_WEIGHTS, "penalties": PENALTIES, "cases": cases}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")

    with zipfile.ZipFile(EVALS / "israeli_legal_eval.zip") as z:
        for name in ("score.py", "judge_prompt.md"):
            (OUT / name).write_bytes(z.read(f"israeli_legal_eval/{name}"))

    manifest = {
        "total": len(gold), "cases": len(cases), "id_prefix": PREFIX,
        "by_category": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "by_area": dict(Counter(g["area"] for g in gold).most_common()),
        "by_confidence": dict(Counter(g["confidence"] for g in gold)),
        "mcq_letters": dict(sorted(letters.items())),
        "yes_no_labels": dict(Counter(g["answer_label"] for g in gold if "answer_label" in g)),
        "laws_cited": len({c["law_code"] for g in gold for c in g["citations"]}),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
