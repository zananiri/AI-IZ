#!/usr/bin/env python3
"""Tools for israeli_legal_eval_v2 (legal_txt/Evals/israeli_legal_eval_v2).

    # 1) check every gold citation against the installed corpus; fills gold_chunks and verified
    python scripts/legal_data/eval_v2.py verify \
        --gold legal_txt/Evals/israeli_legal_eval_v2/gold.jsonl \
        --out data/legal/eval_v2/gold_verified.jsonl --report data/legal/eval_v2/verification.md

    # 2) the questions and gold of one split, for eval_run.py answer and score.py
    python scripts/legal_data/eval_v2.py select --gold data/legal/eval_v2/gold_verified.jsonl \
        --questions legal_txt/Evals/israeli_legal_eval_v2/questions.jsonl --split test \
        --out-dir data/legal/eval_v2/test [--verified-only]

    # 3) after score.py report --json-out: retrieval recall against gold_chunks, and paraphrase consistency
    python scripts/legal_data/eval_v2.py extras --gold data/legal/eval_v2/gold_verified.jsonl \
        --answers answers_rag.jsonl --report-json test/report.json robustness/report.json

verify: an item is VERIFIED when every cited law is in the corpus, every cited section is found in
it, and (for items that state a rule) every number in the gold answer appears in the cited sections'
text. Other statuses: NUMBERS_MISSING, SECTION_MISSING, LAW_MISSING, LAW_ONLY (a citation without a
section), NO_CITATION. Anything but VERIFIED needs a look -- by a lawyer for NUMBERS_MISSING -- before
the item counts; select --verified-only leaves unverified new items out (v1 items are kept: they were
already reviewed against corpus_coverage).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_corpus_coverage import base_section, match_law

# Numbers are compared only where the gold states the rule itself; computed values (3 x rent, dates)
# don't appear in the statute.
NUMBER_CHECK = {"rule_recall", "interpretation", "citation_grounding", "comparison", "temporal_amendment", "mcq_bar"}
_NUMBER_RE = re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})+|(?<![\d.,])\d+(?:\.\d+)?")
_YEAR_LIKE = re.compile(r"^(1[89]\d\d|20\d\d)$")


def load(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def gold_numbers(text: str) -> set[str]:
    """Numbers a rule states (periods, amounts, ages), without section numbers or years."""
    text = re.sub(r"(?:סעיף|סעיפים|ס')\s*[\d,\s–\-ו]+[א-ת]{0,2}", " ", text)   # "סעיף 25", "סעיפים 14–18"
    text = re.sub(r"(?:תקנה|תיקון(?: מס')?)\s*\d+", " ", text)
    found = {n.replace(",", "") for n in _NUMBER_RE.findall(text)}
    return {n for n in found if not _YEAR_LIKE.match(n) and n not in {"1", "2"}}


def text_numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER_RE.findall(text or "")}


class Corpus:
    """Every chunk's (title, section) from the corpus store, and the text of the chunks asked for."""

    def __init__(self, vectordb_dir: Path, prefix: str, categories: list[str]):
        import chromadb

        self.collections = {}
        self.sections: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
        self.chunks: Counter = Counter()
        for category in categories:
            path = vectordb_dir / category
            if not path.exists():
                print(f"[warn] no {category} store at {path}", file=sys.stderr)
                continue
            coll = chromadb.PersistentClient(path=str(path)).get_collection(f"{prefix}_{category}")
            self.collections[category] = coll
            total, offset = coll.count(), 0
            while offset < total:
                got = coll.get(include=["metadatas"], limit=5000, offset=offset)
                for cid, meta in zip(got["ids"], got["metadatas"]):
                    title = str(meta.get("title") or "")
                    self.chunks[title] += 1
                    sec = base_section(meta.get("section_number"))
                    if sec:
                        self.sections[title][sec].append((category, cid))
                offset += 5000
                print(f"\r{category}: {min(offset, total):,}/{total:,}", end="", file=sys.stderr, flush=True)
            print(file=sys.stderr)
        # check_corpus_coverage.match_law reads {"chunks": n} per title
        self.index = {t: {"chunks": n} for t, n in self.chunks.items()}

    def texts(self, refs: list[tuple[str, str]]) -> str:
        out = []
        by_cat = defaultdict(list)
        for cat, cid in refs:
            by_cat[cat].append(cid)
        for cat, ids in by_cat.items():
            got = self.collections[cat].get(ids=ids, include=["documents"])
            out += got["documents"]
        return "\n".join(out)


def verify_item(g: dict, corpus: Corpus) -> dict:
    cites = g.get("citations") or []
    if not cites:
        return {"status": "NO_CITATION", "gold_chunks": [], "notes": []}
    notes, chunks, texts, statuses = [], [], [], []
    for c in cites:
        title, how = match_law(c["law"], corpus.index)
        if not title:
            statuses.append("LAW_MISSING")
            notes.append(f"law not in corpus: {c['law']}")
            continue
        section = base_section(c.get("section"))
        if not section:
            statuses.append("LAW_ONLY")
            notes.append(f"{c['law_code']}: no section cited ({how})")
            continue
        refs = corpus.sections[title].get(section, [])
        if not refs:
            statuses.append("SECTION_MISSING")
            notes.append(f"{c['law_code']} s.{section} not found in {title}")
            continue
        chunks += [cid for _, cid in refs]
        texts.append(corpus.texts(refs))
        statuses.append("OK")
    status = next((s for s in ("LAW_MISSING", "SECTION_MISSING", "LAW_ONLY") if s in statuses), "OK")
    if status == "OK" and g["category"] in NUMBER_CHECK:
        missing = sorted(gold_numbers(g["gold_answer"]) - text_numbers("\n".join(texts)))
        if missing:
            status = "NUMBERS_MISSING"
            notes.append("numbers not in the cited text: " + ", ".join(missing))
    return {"status": "VERIFIED" if status == "OK" else status, "gold_chunks": sorted(set(chunks)), "notes": notes}


def cmd_verify(a) -> int:
    from docslides.config import get_config

    cfg = get_config().legal.corpus
    gold = load(a.gold)
    corpus = Corpus(Path(a.vectordb_dir or cfg.vectordb_dir), cfg.collection_prefix,
                    (a.categories or ",".join(cfg.categories)).split(","))
    if not corpus.chunks:
        print("[fatal] no corpus found", file=sys.stderr)
        return 2
    rows, lines = [], []
    for g in gold:
        if g["source"] == "v2_paraphrase":  # same gold as its source; copied below
            rows.append(g)
            continue
        r = verify_item(g, corpus)
        g = {**g, "gold_chunks": r["gold_chunks"], "verification": r["status"],
             "verified": r["status"] == "VERIFIED" or (r["status"] == "NO_CITATION" and g["category"] == "abstention")}
        if r["notes"]:
            g["verification_notes"] = r["notes"]
        rows.append(g)
    by_id = {g["id"]: g for g in rows}
    rows = [{**g, **{k: by_id[g["paraphrase_group"]][k] for k in ("gold_chunks", "verification", "verified")}}
            if g["source"] == "v2_paraphrase" else g for g in rows]
    write_jsonl(a.out, rows)

    stats = Counter((g["source"], g["verification"]) for g in rows if g["source"] != "v2_paraphrase")
    lines = ["# Gold verification against the corpus", "",
             "| source | " + " | ".join(s for s in ("VERIFIED", "NUMBERS_MISSING", "SECTION_MISSING", "LAW_MISSING",
                                                    "LAW_ONLY", "NO_CITATION")) + " |",
             "|---|" + "---:|" * 6]
    for src in ("v1", "v2_new"):
        lines.append(f"| {src} | " + " | ".join(str(stats[(src, s)]) for s in ("VERIFIED", "NUMBERS_MISSING",
                     "SECTION_MISSING", "LAW_MISSING", "LAW_ONLY", "NO_CITATION")) + " |")
    lines += ["", "## New items that need a look", "", "| id | split | category | status | notes |", "|---|---|---|---|---|"]
    for g in rows:
        if g["source"] == "v2_new" and not g["verified"]:
            lines.append(f"| {g['id']} | {g['split']} | {g['category']} | {g['verification']} | "
                         f"{'; '.join(g.get('verification_notes', []))} |")
    report = "\n".join(lines) + "\n"
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(report, encoding="utf-8")
    print(report)
    return 0


def select(gold: list[dict], questions: list[dict], split: str, verified_only: bool) -> tuple[list, list]:
    keep = [g for g in gold if g["split"] == split
            and (not verified_only or g["source"] == "v1" or g.get("verified"))]
    ids = {g["id"] for g in keep}
    return keep, [q for q in questions if q["id"] in ids]


def cmd_select(a) -> int:
    gold, questions = select(load(a.gold), load(a.questions), a.split, a.verified_only)
    out = Path(a.out_dir)
    write_jsonl(out / "gold.jsonl", gold)
    write_jsonl(out / "questions.jsonl", questions)
    print(f"{a.split}: {len(gold)} items -> {out} ({dict(Counter(g['category'] for g in gold))})")
    return 0


def retrieval_recall(gold: list[dict], answers: list[dict]) -> dict:
    """Share of items (with verified gold chunks) whose retrieved context holds a cited section of a
    cited law. eval_run.py records each hit's title and section, not its chunk id, so the match is by
    law name and section number."""
    from docslides.legal.corpus_retrieval import same_law

    by_id = {a["id"]: a for a in answers}
    hit = total = 0
    for g in gold:
        if not g.get("gold_chunks") or g["id"] not in by_id:
            continue
        total += 1
        wanted = [(c["law"], base_section(c.get("section"))) for c in g["citations"]]
        got = [(r.get("title") or "", base_section(r.get("section"))) for r in by_id[g["id"]].get("retrieved", [])]
        hit += any(sec == rsec and same_law(law, title) for law, sec in wanted for title, rsec in got)
    return {"items": total, "recall": round(hit / total, 3) if total else None}


def consistency(gold: list[dict], rows: list[dict]) -> dict:
    """Paraphrase groups: the source's score next to its variants'. A group is consistent when every
    wording gets the same score."""
    score = {r["id"]: r["score"] for r in rows}
    groups = defaultdict(list)
    for g in gold:
        if g.get("paraphrase_group") and g["id"] in score and g["paraphrase_group"] in score:
            groups[g["paraphrase_group"]].append(score[g["id"]])
    consistent = sum(all(s == score[src] for s in ss) for src, ss in groups.items())
    by_style = defaultdict(list)
    for g in gold:
        if g.get("paraphrase_group") and g["id"] in score and g["paraphrase_group"] in score:
            by_style[g["paraphrase_style"]].append(score[g["id"]] - score[g["paraphrase_group"]])
    return {"groups": len(groups), "consistent": consistent,
            "mean_delta_by_style": {k: round(sum(v) / len(v), 3) for k, v in by_style.items()}}


def cmd_extras(a) -> int:
    gold = load(a.gold)
    out = {"retrieval": retrieval_recall(gold, load(a.answers))}
    if a.report_json:
        rows = [r for path in a.report_json for r in json.loads(Path(path).read_text(encoding="utf-8"))["rows"]]
        out["paraphrases"] = consistency(gold, rows)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pv = sub.add_parser("verify")
    pv.add_argument("--gold", default="legal_txt/Evals/israeli_legal_eval_v2/gold.jsonl")
    pv.add_argument("--out", required=True)
    pv.add_argument("--report")
    pv.add_argument("--vectordb-dir")
    pv.add_argument("--categories")
    ps = sub.add_parser("select")
    ps.add_argument("--gold", required=True)
    ps.add_argument("--questions", default="legal_txt/Evals/israeli_legal_eval_v2/questions.jsonl")
    ps.add_argument("--split", choices=["test", "dev", "robustness", "reserve"], default="test")
    ps.add_argument("--out-dir", required=True)
    ps.add_argument("--verified-only", action="store_true")
    pe = sub.add_parser("extras")
    pe.add_argument("--gold", required=True)
    pe.add_argument("--answers", required=True)
    pe.add_argument("--report-json", nargs="+",
                    help="score.py report --json-out files of the test and robustness runs, for paraphrase consistency")
    a = p.parse_args()
    return {"verify": cmd_verify, "select": cmd_select, "extras": cmd_extras}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
