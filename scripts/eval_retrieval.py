#!/usr/bin/env python3
"""Retrieval-only test for the Legal tab: what reaches the model, in seconds,
without running it.

    python scripts/eval_retrieval.py                          # the live index
    python scripts/eval_retrieval.py --sandbox legal_txt more_laws/
                                                              # a throwaway index of these folders
    python scripts/eval_retrieval.py --json data/legal/eval/retrieval.json

For each eval question (evals/legal/*.json, `evidence` patterns) it reports:
  * rank     -- position, in the evidence the model receives, of the first chunk
                holding the answer (worst over a multi-hop question's parts);
  * sent     -- chunks and tokens of evidence sent;
  * relevant -- share of those tokens that belong to answer-bearing chunks;
  * flag     -- whether retrieval flagged thin coverage, and the reranker's best score.
The summary gives recall, mean reciprocal rank, tokens sent, and how the
thin-coverage flag separates unanswerable questions (group C) from answerable ones.

--sandbox indexes the given folders (the law under test plus distractor laws,
e.g. the principal laws it amends) into data/legal/eval/retrieval-sandbox with
a throwaway signing key, so retrieval is tested against realistic competition
without touching the live index.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DEFAULT_SET = Path(__file__).resolve().parent.parent / "evals" / "legal" / "elections_2026_he.json"
SANDBOX = Path("data/legal/eval/retrieval-sandbox")


def _build_sandbox(folders: list[str]) -> None:
    from docslides.config import get_config

    os.environ.setdefault("DOCSLIDES_LEGAL_BUNDLE_KEY", "retrieval-sandbox-only")
    shutil.rmtree(SANDBOX, ignore_errors=True)
    cfg = get_config().legal
    for sub in ("vdb", "staging", "sources", "uploads", "audit"):
        (SANDBOX / sub).mkdir(parents=True)
    cfg.retrieval.vectordb_dir = str(SANDBOX / "vdb")
    cfg.ingestion.staging_dir = str(SANDBOX / "staging")
    cfg.ingestion.bundle_manifest = str(SANDBOX / "signed_bundle.json")
    cfg.ingestion.sources_dir = str(SANDBOX / "sources")
    cfg.ingestion.uploads_dir = str(SANDBOX / "uploads")
    cfg.audit_dir = str(SANDBOX / "audit")

    from docslides.legal import folder_ingest

    for folder in folders:
        cfg.ingestion.legal_txt_dir = folder
        for result in folder_ingest.run():
            print(f"[{result.action}] {result.path.name} {result.chunk_count or ''} {result.message}")


def _matches(groups: list[list[str]], text: str) -> list[bool]:
    return [all(re.search(p, text) for p in group) for group in groups]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-set", default=str(DEFAULT_SET))
    parser.add_argument("--sandbox", nargs="+", metavar="FOLDER", help="index these folders into a throwaway index")
    parser.add_argument("--json", help="also write per-question results here")
    args = parser.parse_args()

    if args.sandbox:
        _build_sandbox(args.sandbox)

    from docslides.cleaning.tokens import count_tokens
    from docslides.legal import prompts, retrieval

    questions = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))["questions"]
    rows = []
    print(f"{'id':4}{'rank':>6}{'chunks':>8}{'tokens':>8}{'relevant':>10}  flag  score  s")
    for q in questions:
        started = time.monotonic()
        result = retrieval.retrieve(q["question"])
        seconds = time.monotonic() - started
        groups = q.get("evidence") or []
        evidence = prompts.format_evidence(result.by_source_id(), {})
        tokens = count_tokens(evidence)
        ranks: list[int | None] = []
        for g in range(len(groups)):
            ranks.append(next((i + 1 for i, c in enumerate(result.chunks) if _matches(groups, c.text)[g]), None))
        relevant_tokens = sum(count_tokens(c.text) for c in result.chunks if any(_matches(groups, c.text)))
        chunk_tokens = sum(count_tokens(c.text) for c in result.chunks) or 1
        row = {
            "id": q["id"], "group": q["group"], "answerable": bool(groups),
            "rank": (max(ranks) if ranks and None not in ranks else None) if groups else None,
            "found": bool(groups) and None not in ranks,
            "chunks": len(result.chunks), "evidence_tokens": tokens,
            "relevant_share": relevant_tokens / chunk_tokens if groups else None,
            "low_relevance": result.low_relevance, "best_rerank_score": result.best_rerank_score,
            "sent": [f"{c.chunk_id.split(':', 1)[-1]} [{c.via}]" for c in result.chunks],
            "seconds": round(seconds, 1),
        }
        rows.append(row)
        score = "-" if row["best_rerank_score"] is None else f"{row['best_rerank_score']:.2f}"
        share = "-" if row["relevant_share"] is None else f"{row['relevant_share']:.0%}"
        print(f"{q['id']:4}{row['rank'] or '-'!s:>6}{row['chunks']:>8}{tokens:>8}{share:>10}  "
              f"{'yes ' if row['low_relevance'] else 'no  '}  {score:>5}  {seconds:.0f}")

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    summary = {
        "recall": sum(r["found"] for r in answerable) / max(len(answerable), 1),
        "mrr": sum(1 / r["rank"] for r in answerable if r["rank"]) / max(len(answerable), 1),
        "avg_evidence_tokens": sum(r["evidence_tokens"] for r in rows) / max(len(rows), 1),
        "avg_relevant_share": sum(r["relevant_share"] for r in answerable) / max(len(answerable), 1),
        "flagged_unanswerable": f"{sum(r['low_relevance'] for r in unanswerable)} of {len(unanswerable)}",
        "flagged_answerable": f"{sum(r['low_relevance'] for r in answerable)} of {len(answerable)}",
    }
    print("\n" + "  ".join(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in summary.items()))
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "questions": rows}, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
