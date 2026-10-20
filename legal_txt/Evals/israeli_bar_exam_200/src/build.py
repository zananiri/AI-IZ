#!/usr/bin/env python3
"""Builds questions.jsonl, gold.jsonl, cases_gold.json and manifest.json of the bar-exam set from src/ (deterministic).

    python legal_txt/Evals/israeli_bar_exam_200/src/build.py

Edit the items in src/, never the generated files."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cases_gold import cases_gold  # noqa: E402
from common import CUTOFF, LETTERS, mcq  # noqa: E402
from items_mcq import MCQ  # noqa: E402
from items_other import ABSTAIN, OPEN, SPOT, TEMPORAL, YESNO  # noqa: E402

OUT = HERE.parent
SEED = 2026
EXPECTED = {"mcq_bar": 120, "rule_conclusion": 30, "rule_application": 15, "issue_spotting": 10,
            "temporal_amendment": 10, "abstention": 15}


def build() -> list[tuple[dict, dict]]:
    positions = [i % 4 for i in range(len(MCQ))]
    random.Random(SEED).shuffle(positions)  # each letter right len(MCQ)/4 times, in no fixed pattern
    items = [mcq(*row[:6], **({"confidence": row[6]} if len(row) > 6 else {}), position=pos)
             for row, pos in zip(MCQ, positions)]
    items += YESNO + OPEN + SPOT + TEMPORAL + ABSTAIN
    for n, (q, g) in enumerate(items, 1):
        q["id"] = g["id"] = f"BAR-{n:03d}"
    return items


def main() -> None:
    items = build()
    by_category = Counter(g["category"] for _, g in items)
    assert dict(by_category) == EXPECTED, f"category counts {dict(by_category)} != {EXPECTED}"
    assert len({q["question"] for q, _ in items}) == len(items), "duplicate question text"
    letters = Counter(g["answer_letter"] for _, g in items if g["scoring"] == "exact_letter")
    labels = Counter(g["answer_label"] for _, g in items if g["scoring"] == "label_then_judge")

    def order(row: dict, first: list[str]) -> dict:
        return {k: row[k] for k in first if k in row} | {k: v for k, v in row.items() if k not in first}

    with open(OUT / "questions.jsonl", "w", encoding="utf-8") as f:
        for q, _ in items:
            f.write(json.dumps(order(q, ["id", "category", "area", "format", "instructions", "question"]),
                               ensure_ascii=False) + "\n")
    with open(OUT / "gold.jsonl", "w", encoding="utf-8") as f:
        for _, g in items:
            f.write(json.dumps(order(g, ["id", "category", "area", "format", "scoring"]), ensure_ascii=False) + "\n")
    cases = cases_gold()
    missing = [c["file"] for c in cases["cases"] if not (OUT / "cases" / c["file"]).exists()]
    assert not missing and len(cases["cases"]) == 10, f"cases: {len(cases['cases'])}, missing files {missing}"
    (OUT / "cases_gold.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    cases = cases["cases"]
    manifest = {
        "name": "israeli_bar_exam_200", "as_of": CUTOFF, "items": len(items), "cases": len(cases),
        "by_category": dict(by_category), "by_area": dict(Counter(g["area"] for _, g in items).most_common()),
        "by_confidence": dict(Counter(g["confidence"] for _, g in items)),
        "mcq_answer_letters": {k: letters[k] for k in LETTERS}, "yesno_labels": dict(labels),
        "abstain_reasons": dict(Counter(g["abstain_reason"] for _, g in items if g.get("abstain_reason"))),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
