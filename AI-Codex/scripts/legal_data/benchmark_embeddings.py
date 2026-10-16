#!/usr/bin/env python3
"""Compare embedding models (alone and fused with BM25) on Spanish legal retrieval, over the
articles of the fetched corpus and a question set with gold articles.

    python scripts/legal_data/benchmark_embeddings.py \
        --questions legal_txt/Evals/es_retrieval_seed.jsonl \
        --models BAAI/bge-m3 Qwen/Qwen3-Embedding-0.6B SINAI/ALIA-MrBERT-es-legal-administrative-embeddings

Documents are the articles (and dispositions) of every law the questions name, plus every other
law in --input, one document per article as the Legal tab sees it: "law > Artículo N — title" then
the text, truncated to --max-tokens. A question is answered when a gold article is in the top k.
Reports recall@1/5/10 and MRR@10 per model, for dense search alone and for dense + BM25
(legal/keyword.py's Spanish tokenizer) fused by reciprocal rank, as legal/retrieval.py does.
Query prompts come from rag/embedding.query_prompt (the model's own convention).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aicodex.legal.keyword import KeywordIndex
from aicodex.legal_data.records import LEGISLATION_CATEGORIES, read_jsonl
from aicodex.rag.embedding import encode_documents, encode_queries


def load_documents(root: Path) -> list[tuple[tuple[str, str], str]]:
    docs = []
    for category in LEGISLATION_CATEGORIES:
        for path in sorted((root / category).glob(f"{category}*.jsonl")):
            for record in read_jsonl(path):
                for section in record.get("sections") or []:
                    if section["kind"] not in ("section", "disposition"):
                        continue
                    label = f"Artículo {section['number']}" if section["kind"] == "section" else section["number"]
                    head = f"{record['title'][:120]} > {label}" + (f" — {section['title']}" if section.get("title") else "")
                    docs.append(((record.get("law_id") or "", section["number"]), f"{head}\n{section['text']}"))
    return docs


def rrf(*rankings: list[int], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking):
            scores[index] = scores.get(index, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda i: -scores[i])


def metrics(rankings: list[list[int]], golds: list[set[int]]) -> dict:
    out = {"r@1": 0.0, "r@5": 0.0, "r@10": 0.0, "mrr@10": 0.0}
    for ranking, gold in zip(rankings, golds):
        first = next((r for r, i in enumerate(ranking[:10]) if i in gold), None)
        if first is not None:
            out["mrr@10"] += 1 / (first + 1)
            out["r@10"] += 1
            out["r@5"] += first < 5
            out["r@1"] += first < 1
    return {k: round(v / max(len(golds), 1), 3) for k, v in out.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="legal_txt", help="corpus root holding laws/, regulations/ ...")
    parser.add_argument("--questions", default="legal_txt/Evals/es_retrieval_seed.jsonl")
    parser.add_argument("--models", nargs="+", default=["BAAI/bge-m3", "Qwen/Qwen3-Embedding-0.6B"])
    parser.add_argument("--max-tokens", type=int, default=384, help="truncate documents (model max_seq_length)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", help="write the results JSON here")
    args = parser.parse_args()

    docs = load_documents(Path(args.input))
    keys = [k for k, _ in docs]
    texts = [t for _, t in docs]
    questions = [json.loads(line) for line in open(args.questions, encoding="utf-8") if line.strip()]
    index_of: dict[tuple[str, str], list[int]] = {}
    for i, key in enumerate(keys):
        index_of.setdefault(key, []).append(i)
    golds = [{i for law, num in q["gold"] for i in index_of.get((law, num), [])} for q in questions]
    missing = [q["id"] for q, g in zip(questions, golds) if not g]
    print(f"{len(texts):,} documents; {len(questions)} questions; gold missing from corpus: {missing or 'none'}")

    keyword = KeywordIndex([(str(i), t) for i, t in enumerate(texts)])
    bm25 = [[int(i) for i, _ in keyword.search(q["question"], 100)] for q in questions]
    results = {"documents": len(texts), "questions": len(questions), "bm25": metrics(bm25, golds), "models": {}}
    print(f"BM25 alone: {results['bm25']}")

    import numpy as np

    for model in args.models:
        started = time.time()
        doc_vectors = np.asarray(encode_documents(model, texts, batch_size=args.batch_size, device=args.device,
                                                  max_seq_length=args.max_tokens))
        query_vectors = np.asarray(encode_queries(model, [q["question"] for q in questions], device=args.device))
        dense = [list(np.argsort(-(doc_vectors @ v))[:100]) for v in query_vectors]
        fused = [rrf(d, b) for d, b in zip(dense, bm25)]
        row = {"dense": metrics(dense, golds), "dense+bm25": metrics(fused, golds),
               "seconds": round(time.time() - started), "misses_dense+bm25": [
                   q["id"] for q, f, g in zip(questions, fused, golds) if not (set(f[:10]) & g)]}
        results["models"][model] = row
        print(f"{model}: dense {row['dense']} | dense+bm25 {row['dense+bm25']} | {row['seconds']}s", flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
