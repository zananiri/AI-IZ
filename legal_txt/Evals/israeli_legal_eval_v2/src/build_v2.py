#!/usr/bin/env python3
"""Build israeli_legal_eval_v2 from the v1 set (legal_txt/Evals/{questions,gold}.jsonl) and the new
items in items_a/b/c.py. Deterministic: the same inputs give the same files.

    python legal_txt/Evals/israeli_legal_eval_v2/src/build_v2.py

Writes, next to src/: questions.jsonl (what the model sees), gold.jsonl (judge only) and
manifest.json (counts). Splits:
  test        -- 400 items, the only split to report
  dev         -- 200 items, for tuning prompts and retrieval
  robustness  -- 75 paraphrases of 25 test items (same gold), scored for consistency
  reserve     -- v1 items dropped to rebalance categories; kept, not scored by default
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import INSTRUCTIONS, NEW_LAWS, SCORING
from items_a import APPLICATION, ISSUES, NEGATED
from items_b import COMPARISON, COMPUTATION, MULTI_HOP
from items_c import ABSTENTION, PARAPHRASE_STYLES, PARAPHRASES, TEMPORAL

V1 = HERE.parents[1]
OUT = HERE.parent

# v1 items kept per category in the scored core (the rest go to "reserve"); None keeps all.
KEEP_V1 = {"rule_recall": 60, "rule_conclusion": 50, "mcq_bar": 60, "citation_grounding": 30}
TEST_SHARE = 2 / 3


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rank(key: str) -> str:
    """A stable pseudo-random order (sha1), so selection and splits don't depend on file order."""
    return hashlib.sha1(f"israeli_legal_eval_v2:{key}".encode()).hexdigest()


def answer_type(g: dict) -> str:
    if g["category"] == "mcq_bar":
        return "letter"
    if g.get("format") == "yesno":
        return "yes_no"
    if g["category"] == "computation":
        return "value"
    if g["category"] == "abstention":
        return "refusal_or_correction"
    if g.get("format") == "list":
        return "list"
    return "rule"


def main() -> None:
    v1_gold = load(V1 / "gold.jsonl")
    v1_q = {q["id"]: q for q in load(V1 / "questions.jsonl")}
    laws = dict(NEW_LAWS)
    for g in v1_gold:
        for c in g["citations"]:
            laws.setdefault(c["law_code"], c["law"])

    # ---- v1: rebalance, keeping every paraphrase source and preferring high-confidence gold ----
    forced = set(PARAPHRASES)
    by_cat = defaultdict(list)
    for g in v1_gold:
        by_cat[g["category"]].append(g)
    core, reserve = [], []
    for cat, items in by_cat.items():
        limit = KEEP_V1.get(cat)
        ordered = sorted(items, key=lambda g: (g["id"] not in forced, g["confidence"] != "high", rank(g["id"])))
        keep = ordered if limit is None else ordered[:limit]
        core += keep
        reserve += [g for g in items if g not in keep]

    gold_rows, q_rows = [], []

    def add(g: dict, q: dict) -> None:
        gold_rows.append(g)
        q_rows.append(q)

    def v1_row(g: dict, split: str) -> None:
        q = dict(v1_q[g["id"]])
        row = {**g, "source": "v1", "split": split, "gold_chunks": [], "as_of": None, "answer_type": answer_type(g),
               "hops": 1, "negated": False, "abstain_reason": None, "paraphrase_group": None,
               "verified": None, "reviewed_by_lawyer": False}
        add(row, q)

    # ---- new items ----
    new_items = NEGATED + APPLICATION + ISSUES + COMPARISON + MULTI_HOP + COMPUTATION + TEMPORAL + ABSTENTION
    new_rows = []
    for n, it in enumerate(new_items, 1):
        it = dict(it)
        iid = f"IL2-{n:03d}"
        fmt = it.pop("format")
        cites = []
        for c in it.pop("cites"):
            code, _, section = c.partition(":")
            if code not in laws:
                raise SystemExit(f"{iid}: unknown law code {code}")
            cites.append({"law_code": code, "law": laws[code], "section": section})
        g = {
            "id": iid, "category": it.pop("category"), "area": it.pop("area"), "format": fmt,
            "scoring": SCORING.get(fmt, "judge"), "gold_answer": it.pop("gold_answer"),
            # Unchecked against the statute text, so "medium" whatever the author's certainty
            # (kept in author_confidence) until verify_gold_v2.py and a lawyer have looked at it.
            "key_points": it.pop("key_points"), "citations": cites, "confidence": "medium",
            "author_confidence": it.pop("confidence"),
        }
        question = it.pop("question")
        g.update({"source": "v2_new", "gold_chunks": [], "as_of": it.pop("as_of", None),
                  "hops": it.pop("hops", 1), "negated": it.pop("negated", False),
                  "abstain_reason": it.pop("abstain_reason", None), "paraphrase_group": None,
                  "verified": None, "reviewed_by_lawyer": False})
        for key in ("answer_label", "answer_value"):
            if key in it:
                g[key] = it.pop(key)
        if it:
            raise SystemExit(f"{iid}: unexpected fields {sorted(it)}")
        g["answer_type"] = answer_type(g)
        q = {"id": iid, "category": g["category"], "area": g["area"], "format": fmt,
             "instructions": INSTRUCTIONS[fmt], "question": question}
        if g["as_of"]:
            q["as_of"] = g["as_of"]
        new_rows.append((g, q))

    # ---- split the core (v1 kept + new) 2:1, stratified by category ----
    pool = defaultdict(list)
    for g in core:
        pool[g["category"]].append(("v1", g, None))
    for g, q in new_rows:
        pool[g["category"]].append(("new", g, q))
    for cat, items in pool.items():
        items.sort(key=lambda t: (t[1]["id"] not in forced, rank(t[1]["id"])))
        n_test = round(len(items) * TEST_SHARE)
        for i, (kind, g, q) in enumerate(items):
            split = "test" if i < n_test else "dev"
            if kind == "v1":
                v1_row(g, split)
            else:
                g["split"] = split
                add(g, q)
    for g in reserve:
        v1_row(g, "reserve")

    # ---- paraphrases of test items: gold copied from the source ----
    by_id = {g["id"]: g for g in gold_rows}
    for src, variants in PARAPHRASES.items():
        base = by_id[src]
        if base["split"] != "test":
            raise SystemExit(f"paraphrase source {src} is not in test")
        for k, (text, style) in enumerate(zip(variants, PARAPHRASE_STYLES), 1):
            vid = f"{src}-P{k}"
            g = {**base, "id": vid, "source": "v2_paraphrase", "split": "robustness",
                 "paraphrase_group": src, "paraphrase_style": style}
            q = {**{k2: v for k2, v in v1_q[src].items() if k2 != "id"}, "id": vid, "question": text}
            if style == "english":
                q["question_language"] = "en"
            add(g, q)

    gold_rows.sort(key=lambda g: g["id"])
    q_rows.sort(key=lambda q: q["id"])
    assert [g["id"] for g in gold_rows] == [q["id"] for q in q_rows]
    assert len({g["id"] for g in gold_rows}) == len(gold_rows)

    for name, rows in (("gold.jsonl", gold_rows), ("questions.jsonl", q_rows)):
        (OUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    manifest = {
        "total": len(gold_rows),
        "by_split": dict(Counter(g["split"] for g in gold_rows)),
        "by_split_and_category": {s: dict(sorted(Counter(g["category"] for g in gold_rows if g["split"] == s).items()))
                                  for s in ("test", "dev", "robustness", "reserve")},
        "by_source": dict(Counter(g["source"] for g in gold_rows)),
        "scored_core": sum(g["split"] in ("test", "dev") for g in gold_rows),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
