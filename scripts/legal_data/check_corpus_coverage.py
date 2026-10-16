#!/usr/bin/env python3
"""Is the installed legal corpus complete for the eval? Reads every chunk's metadata from the
corpus store (legal.corpus.vectordb_dir, one Chroma collection per category -- no embedding
model needed) and checks each law and section that israeli_legal_eval/gold.jsonl cites.

    python scripts/legal_data/check_corpus_coverage.py \
        --gold legal_txt/Evals/israeli_legal_eval/gold.jsonl \
        --out data/legal/eval/corpus_coverage.md --json-out data/legal/eval/corpus_coverage.json

A law is found when a corpus title has the same name and year (the gold names laws with their
year: חוק ..., התשל"ג-1973), else by name alone ("name only" -- a different law of the same
name, or the gold's year form differs), else by most of the name's words and the same year
("similar name" -- the gold cites a law by a descriptive name, not its official one). An amending law (תיקון) is usually folded into the
law it amends on the Open Law Book, so it is reported against that law's text. Exit code 1 if any
cited law is missing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from docslides.config import get_config  # noqa: E402
from docslides.legal.corpus_retrieval import law_key, section_numbers  # noqa: E402

_YEAR_RE = re.compile(r"(1[89]\d\d|20\d\d)")
_AMENDMENT_RE = re.compile(r"\s*\(תיקון[^)]*\)")


def law_year(name: str) -> str | None:
    years = _YEAR_RE.findall(name or "")
    return years[-1] if years else None


def base_section(value) -> str | None:
    """"25א#p2" / "סעיף 25א(ב)" -> "25א"; schedules and unnumbered parts -> None."""
    text = str(value or "")
    if text.startswith("schedule:"):
        return None
    found = section_numbers([text.split("#")[0]])
    return found[0] if found else None


def load_index(vectordb_dir: Path, prefix: str, categories: list[str]) -> dict[str, dict]:
    """{title: {"category", "sections": set, "chunks": int}} over every chunk in the store."""
    try:
        import chromadb
    except ImportError:
        sys.exit('[fatal] chromadb is not installed: pip install -e ".[legal]"')

    index: dict[str, dict] = {}
    for category in categories:
        path = vectordb_dir / category
        if not path.exists():
            print(f"[warn] no {category} store at {path}", file=sys.stderr)
            continue
        collection = chromadb.PersistentClient(path=str(path)).get_collection(f"{prefix}_{category}")
        total, offset, batch = collection.count(), 0, 5000
        while offset < total:
            got = collection.get(include=["metadatas"], limit=batch, offset=offset)
            for meta in got["metadatas"]:
                title = str(meta.get("title") or "")
                entry = index.setdefault(title, {"category": category, "sections": set(), "chunks": 0})
                entry["chunks"] += 1
                section = base_section(meta.get("section_number"))
                if section:
                    entry["sections"].add(section)
            offset += batch
            print(f"\r{category}: {min(offset, total):,}/{total:,} chunks", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
    return index


def match_law(name: str, index: dict[str, dict]) -> tuple[str | None, str]:
    """(corpus title, how) for a cited law name; how is "exact", "name only", "amended law",
    "similar name" or "missing"."""
    year = law_year(name)
    by_key: dict[str, list[str]] = defaultdict(list)
    for title in index:
        by_key[law_key(title)].append(title)

    def best(titles: list[str]) -> str:
        return max(titles, key=lambda t: index[t]["chunks"])

    key = law_key(name)
    if key in by_key:
        same_year = [t for t in by_key[key] if law_year(t) == year]
        return (best(same_year), "exact") if same_year or year is None else (best(by_key[key]), "name only")
    base = _AMENDMENT_RE.sub("", name.split(",")[0])
    if base != name.split(",")[0] and law_key(base) in by_key:
        return best(by_key[law_key(base)]), "amended law"
    # The official name can differ from how the gold cites it ("החוק בעניין העמדה לדין של מבצעי
    # טבח 7 באוקטובר" vs "חוק העמדה לדין בשל אירועי טבח 7 באוקטובר 2023"): same year and most
    # of the name's words.
    words = set(key.split())
    similar = [t for t in index if law_year(t) == year and words
               and len(words & set(law_key(t).split())) / len(words | set(law_key(t).split())) >= 0.4]
    if year and similar:
        return best(similar), "similar name"
    return None, "missing"


def check(gold: list[dict], index: dict[str, dict]) -> dict:
    laws: dict[str, dict] = {}
    for item in gold:
        for citation in item.get("citations", []):
            law = laws.setdefault(citation["law"], {"code": citation.get("law_code"), "questions": set(),
                                                     "sections": Counter()})
            law["questions"].add(item["id"])
            section = base_section(citation.get("section"))
            if section:
                law["sections"][section] += 1
    rows = []
    for name, law in laws.items():
        title, how = match_law(name, index)
        present = index[title]["sections"] if title else set()
        missing = sorted(s for s in law["sections"] if s not in present) if title and how != "amended law" else []
        rows.append({"law": name, "code": law["code"], "questions": sorted(law["questions"]), "match": how,
                     "corpus_title": title, "chunks": index[title]["chunks"] if title else 0,
                     "sections_cited": sorted(law["sections"]), "sections_missing": missing})
    rows.sort(key=lambda r: (r["match"] != "missing", not r["sections_missing"], -len(r["questions"])))
    cited = sum(len(r["sections_cited"]) for r in rows if r["match"] in ("exact", "name only", "similar name"))
    absent = sum(len(r["sections_missing"]) for r in rows)
    return {"corpus_titles": len(index), "corpus_chunks": sum(e["chunks"] for e in index.values()),
            "laws_cited": len(rows), "laws_missing": sum(r["match"] == "missing" for r in rows),
            "sections_checked": cited, "sections_missing": absent, "laws": rows}


def render(result: dict) -> str:
    out = ["# Legal corpus coverage for the eval", "",
           f"corpus: {result['corpus_titles']:,} titles, {result['corpus_chunks']:,} chunks  |  "
           f"cited laws: {result['laws_cited']}, missing: **{result['laws_missing']}**  |  "
           f"cited sections checked: {result['sections_checked']}, missing: **{result['sections_missing']}**", "",
           "| law | code | questions | match | corpus title | chunks | sections missing |",
           "|---|---|---|---|---|---|---|"]
    for r in result["laws"]:
        out.append(f"| {r['law']} | {r['code'] or ''} | {len(r['questions'])} | {r['match']} | "
                   f"{r['corpus_title'] or '—'} | {r['chunks']} | {', '.join(r['sections_missing']) or ''} |")
    return "\n".join(out) + "\n"


def main() -> int:
    corpus_cfg = get_config().legal.corpus
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold", default="legal_txt/Evals/israeli_legal_eval/gold.jsonl")
    p.add_argument("--vectordb-dir", default=corpus_cfg.vectordb_dir)
    p.add_argument("--categories", default=",".join(corpus_cfg.categories))
    p.add_argument("--out", help="Markdown report (default: stdout)")
    p.add_argument("--json-out")
    a = p.parse_args()

    gold = [json.loads(line) for line in open(a.gold, encoding="utf-8") if line.strip()]
    index = load_index(Path(a.vectordb_dir), corpus_cfg.collection_prefix, a.categories.split(","))
    if not index:
        print(f"[fatal] no corpus found under {a.vectordb_dir}", file=sys.stderr)
        return 2
    result = check(gold, index)
    report = render(result)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(report, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        print(report)
    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{result['laws_missing']} of {result['laws_cited']} cited laws missing; "
          f"{result['sections_missing']} of {result['sections_checked']} cited sections missing", file=sys.stderr)
    return 1 if result["laws_missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
