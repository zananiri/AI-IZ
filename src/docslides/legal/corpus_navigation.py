"""Reading a law the way a lawyer does, on top of planned retrieval (legal/corpus_retrieval.py):
whole sections instead of fragments, the table of contents of the laws found, the sections a
retrieved section refers to, and short doctrine cards for what the statute text doesn't say.

Every step is off by default and switched on in legal.corpus (whole_sections, toc_navigation,
cross_references, regulation_cap, law_grouped_context, doctrine_cards_path), so each can be
measured alone on the eval's dev split. The 30 Sept root-cause review of the 29 Sept Gemma 27B run
(72.6%) behind them:

- 63 of the 110 points lost were on questions whose governing section *was* retrieved: half the
  half-credit answers left out a condition, exception or proviso -- often in a part of the section
  the two-chunks-per-section cap had dropped (345 hits were extra parts of a section already in);
- 29.5 points went on the right law but the wrong section, which a table of contents fixes better
  than another search;
- "subject to section 12" chains were never followed.

A hit here is the dict corpus_retrieval.retrieve_planned returns: id, category, distance, meta,
text, sources (and score after reranking). A merged hit also carries chunk_ids: every chunk it holds.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

from docslides.cleaning.tokens import count_tokens
from docslides.config import get_config
from docslides.legal.structure import extract_cross_references
from docslides.llm.client import ChatMessage, LLMCallSite, SamplingParams
from docslides.logging_setup import get_logger

logger = get_logger(__name__)

_PART_LABEL_RE = re.compile(r"\s*\((?:חלק \d+ מתוך \d+|part \d+ of \d+)\)\s*$")
_MAX_OVERLAP_CHARS = 800
TOC_MAX_ENTRIES = 300  # headings per law shown to the model
TOC_HEADING_CHARS = 80


def section_key(hit: dict) -> tuple[str, str, str]:
    meta = hit["meta"]
    return (hit.get("category", ""), str(meta.get("record_id") or meta.get("title") or ""),
            str(meta.get("section_number") or hit["id"]))


@lru_cache(maxsize=8)
def corpus_collection(category: str):
    """The Chroma collection for a corpus category, or None when it isn't installed."""
    from docslides.legal_data.corpus_index import CorpusCollection

    corpus_cfg = get_config().legal.corpus
    path = Path(corpus_cfg.vectordb_dir) / category
    if not path.exists():
        return None
    return CorpusCollection(path, f"{corpus_cfg.collection_prefix}_{category}")


# ---------------------------------------------------------------- whole sections


def collapse_sections(ranked: list[dict]) -> list[dict]:
    """One hit per section, at the rank of its best chunk; the others' sources join it."""
    out: dict[tuple, dict] = {}
    for hit in ranked:
        key = section_key(hit)
        if key in out:
            out[key]["sources"] = set(out[key]["sources"]) | set(hit.get("sources", ()))
            out[key]["distance"] = min(out[key]["distance"], hit["distance"])
        else:
            out[key] = {**hit, "sources": set(hit.get("sources", ()))}
    return list(out.values())


def _split_header(text: str) -> tuple[str, str]:
    header, sep, body = text.partition("\n\n")
    return (header, body) if sep else ("", text)


def _without_overlap(previous: str, following: str) -> str:
    """`following` less its opening that repeats the end of `previous` (the chunker repeats up to
    legal.corpus.chunk_overlap_tokens between the parts of a split section)."""
    limit = min(len(previous), len(following), _MAX_OVERLAP_CHARS)
    for size in range(limit, 19, -1):
        if previous.endswith(following[:size]):
            return following[size:].lstrip()
    return following


def merge_parts(texts: list[str]) -> str:
    """The chunks of one section, in order, as one text: the first header without its "(part 1 of
    3)" label, each later part's header dropped, and the overlap between parts removed. Subsections
    stored as their own chunks keep their header line, which names the subsection."""
    if not texts:
        return ""
    header, body = _split_header(texts[0])
    header = _PART_LABEL_RE.sub("", header)
    bodies = [body]
    last_header = header
    for text in texts[1:]:
        part_header, part_body = _split_header(text)
        part_header = _PART_LABEL_RE.sub("", part_header)
        if part_header and part_header != last_header:
            bodies.append(part_header.rsplit(" > ", 1)[-1])
            last_header = part_header
        bodies.append(_without_overlap(bodies[-1], part_body))
    merged = "\n".join(b for b in bodies if b.strip())
    return f"{header}\n\n{merged}" if header else merged


def section_chunks(collection, record_id: str, number: str) -> list[tuple[str, dict, str]]:
    """(chunk_id, metadata, text) of every chunk of one section of one record, in index order."""
    if collection is None or not record_id or not number:
        return []
    found = collection.collection.get(
        where={"$and": [{"record_id": record_id}, {"section_number": number}]},
        include=["metadatas", "documents"],
    )
    rows = list(zip(found.get("ids") or [], found.get("metadatas") or [], found.get("documents") or []))
    # Chroma returns rows in insertion order, which is the chunker's; parts of one provision by part_index.
    by_source: dict[str, list[tuple[str, dict, str]]] = {}
    for chunk_id, meta, text in rows:
        by_source.setdefault(chunk_id.split("#p")[0], []).append((chunk_id, meta or {}, text or ""))
    ordered: list[tuple[str, dict, str]] = []
    for parts in by_source.values():
        ordered += sorted(parts, key=lambda r: int(r[1].get("part_index") or 1))
    return ordered


def whole_section(hit: dict, collection, max_tokens: int) -> dict:
    """`hit` with the full text of its section (merge_parts), when the section has more than the
    one chunk and fits in `max_tokens`; otherwise the hit as it was."""
    meta = hit["meta"]
    rows = section_chunks(collection, str(meta.get("record_id") or ""), str(meta.get("section_number") or ""))
    if len(rows) <= 1:
        return hit
    text = merge_parts([t for _, _, t in rows])
    if count_tokens(text) > max_tokens:
        return hit
    return {**hit, "text": text, "chunk_ids": [c for c, _, _ in rows], "whole_section": True}


def expand_sections(hits: list[dict], collections=corpus_collection, max_tokens: int | None = None) -> list[dict]:
    """collapse_sections, then each hit's whole section."""
    limit = max_tokens or get_config().legal.corpus.whole_section_max_tokens
    return [whole_section(h, collections(h["category"]), limit) for h in collapse_sections(hits)]


def fetch_section(collection, category: str, record_id: str, number: str, source: str,
                  max_tokens: int | None = None) -> dict | None:
    """One section of one record as a hit (whole when it fits in max_tokens, else its first
    chunks up to that budget), or None when the record has no such section."""
    rows = section_chunks(collection, record_id, number)
    if not rows:
        return None
    limit = max_tokens or get_config().legal.corpus.whole_section_max_tokens
    kept: list[tuple[str, dict, str]] = []
    for row in rows:
        if kept and count_tokens(merge_parts([t for _, _, t in [*kept, row]])) > limit:
            break
        kept.append(row)
    return {"id": kept[0][0], "category": category, "distance": 1.0, "meta": kept[0][1],
            "text": merge_parts([t for _, _, t in kept]), "sources": {source},
            "chunk_ids": [c for c, _, _ in kept], "whole_section": len(kept) == len(rows)}


# ---------------------------------------------------------------- context shaping


def cap_regulations(hits: list[dict], cap: int | None, named_laws: list[str]) -> list[dict]:
    """At most `cap` regulation hits, unless the plan names a regulation (תקנות, צו)."""
    if cap is None or any(re.match(r"\s*(?:תקנות|צו)\b", law or "") for law in named_laws):
        return hits
    out, regulations = [], 0
    for hit in hits:
        if hit.get("category") == "procedural_rules":
            if regulations >= cap:
                continue
            regulations += 1
        out.append(hit)
    return out


def trim_to_budget(hits: list[dict], max_tokens: int) -> list[dict]:
    """The hits, best first, while their text fits in `max_tokens` (the first is always kept)."""
    out, spent = [], 0
    for hit in hits:
        size = count_tokens(hit["text"])
        if out and spent + size > max_tokens:
            continue
        out.append(hit)
        spent += size
    return out


def _section_sort_key(number: str) -> tuple:
    match = re.match(r"(\d+)(.*)", number or "")
    return (0, int(match.group(1)), match.group(2)) if match else (1, 0, number or "")


def group_by_law(hits: list[dict]) -> list[dict]:
    """Hits with each law's excerpts together, laws in the order of their best hit, sections in
    numeric order inside a law -- a law reads as a law, not as fragments scattered by score."""
    laws: dict[str, list[dict]] = {}
    for hit in hits:
        laws.setdefault(str(hit["meta"].get("record_id") or hit["meta"].get("title") or ""), []).append(hit)
    out: list[dict] = []
    for group in laws.values():
        out += sorted(group, key=lambda h: _section_sort_key(str(h["meta"].get("section_number") or "")))
    return out


# ---------------------------------------------------------------- cross-references


def cross_reference_hits(hits: list[dict], collections=corpus_collection, limit: int | None = None,
                         scan: int = 4) -> list[dict]:
    """Sections of the same law that the top `scan` hits refer to ("בכפוף לסעיף 12", "כאמור בסעיף
    5(ב)") and that aren't already in `hits`: at most `limit`, in reference order."""
    limit = get_config().legal.corpus.cross_reference_max if limit is None else limit
    have = {section_key(h) for h in hits}
    out: list[dict] = []
    for hit in hits[:scan]:
        meta = hit["meta"]
        record_id = str(meta.get("record_id") or "")
        own = str(meta.get("section_number") or "")
        if not record_id or own.startswith(("schedule", "preamble")):
            continue
        body = _split_header(hit["text"])[1]
        for number in extract_cross_references(body, own):
            if len(out) >= limit:
                return out
            key = (hit["category"], record_id, number)
            if key in have:
                continue
            found = fetch_section(collections(hit["category"]), hit["category"], record_id, number, "x")
            have.add(key)
            if found is not None:
                out.append(found)
    return out


# ---------------------------------------------------------------- table of contents


TOC_PROMPT = """Below is a question about Israeli law and the table of contents of the law(s) most \
likely to govern it. Name the sections whose text the answer needs: the operative rule, and any \
definition, condition, exception, deadline or remedy section it depends on. At most {max_sections} \
sections in all. Use the law number shown in brackets and the section number exactly as listed. \
Name none if no listed section governs.

<question>{question}</question>

{tocs}"""


class TocPick(BaseModel):
    law: int
    section: str


class TocPicks(BaseModel):
    sections: list[TocPick] = []


def _heading(meta: dict) -> str:
    crumb = str(meta.get("breadcrumb") or "")
    last = crumb.rsplit(" > ", 1)[-1]
    last = re.sub(r"^(?:סעיף|section)\s+\S+\s*(?:—|-)?\s*", "", last).strip()
    return last[:TOC_HEADING_CHARS]


def law_toc(collection, record_id: str) -> list[tuple[str, str]]:
    """(section number, heading) of every numbered section of one record, in order."""
    if collection is None or not record_id:
        return []
    found = collection.collection.get(where={"record_id": record_id}, include=["metadatas"])
    toc: dict[str, str] = {}
    for meta in found.get("metadatas") or []:
        number = str((meta or {}).get("section_number") or "")
        if not number or number == "preamble" or number.startswith("schedule:") or number in toc:
            continue
        toc[number] = _heading(meta)
    return sorted(toc.items(), key=lambda kv: _section_sort_key(kv[0]))[:TOC_MAX_ENTRIES]


def top_laws(hits: list[dict], n: int) -> list[tuple[str, str, str]]:
    """(category, record_id, title) of the first `n` distinct laws among the ranked hits."""
    out: list[tuple[str, str, str]] = []
    for hit in hits:
        record = (hit["category"], str(hit["meta"].get("record_id") or ""), str(hit["meta"].get("title") or ""))
        if record[1] and record not in out:
            out.append(record)
        if len(out) >= n:
            break
    return out


def render_tocs(laws: list[tuple[str, str, str]], tocs: list[list[tuple[str, str]]]) -> str:
    blocks = []
    for i, ((_, _, title), toc) in enumerate(zip(laws, tocs), 1):
        lines = "\n".join(f"{number}: {heading}" if heading else number for number, heading in toc)
        blocks.append(f"<law number=\"{i}\" name=\"{title}\">\n{lines}\n</law>")
    return "\n\n".join(blocks)


async def navigate_toc(client, question: str, hits: list[dict], collections=corpus_collection,
                       n_laws: int | None = None, max_sections: int | None = None) -> list[dict]:
    """The sections the model picks from the tables of contents of the top `n_laws` laws among
    `hits`, fetched whole and not already in `hits`. Empty on any failure."""
    corpus_cfg = get_config().legal.corpus
    n_laws = n_laws or corpus_cfg.toc_laws
    max_sections = max_sections or corpus_cfg.toc_max_sections
    laws = top_laws(hits, n_laws)
    tocs = [law_toc(collections(category), record_id) for category, record_id, _ in laws]
    laws, tocs = [law for law, toc in zip(laws, tocs) if toc], [toc for toc in tocs if toc]
    if not laws:
        return []
    try:
        picks = await client.complete_json(
            [ChatMessage("user", TOC_PROMPT.format(max_sections=max_sections, question=question,
                                                   tocs=render_tocs(laws, tocs)))],
            LLMCallSite("legal_eval_toc"), schema=TocPicks,
            sampling=SamplingParams(temperature=0.0, max_tokens=512), enable_thinking=False,
        )
    except Exception as exc:  # noqa: BLE001 -- the retrieved hits stand without it
        logger.warning("legal_toc_navigation_failed", error=f"{type(exc).__name__}: {exc}")
        return []
    have = {section_key(h) for h in hits}
    out: list[dict] = []
    for pick in picks.sections[:max_sections]:
        if not 1 <= pick.law <= len(laws):
            continue
        category, record_id, _ = laws[pick.law - 1]
        number = pick.section.strip()
        if number not in {n for n, _ in tocs[pick.law - 1]} or (category, record_id, number) in have:
            continue
        found = fetch_section(collections(category), category, record_id, number, "t")
        have.add((category, record_id, number))
        if found is not None:
            out.append(found)
    return out


# ---------------------------------------------------------------- doctrine cards


@lru_cache(maxsize=4)
def load_doctrine_cards(path: str) -> tuple[dict, ...]:
    file = Path(path)
    if not file.exists():
        logger.warning("legal_doctrine_cards_missing", path=path)
        return ()
    cards = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return tuple(cards)


def match_doctrine_cards(text: str, cards: tuple[dict, ...], limit: int) -> list[dict]:
    """Cards whose triggers appear in `text` (the question, and the plan's laws and issues), most
    triggers matched first. A trigger is a plain phrase, or several joined by "+" that must all appear."""
    scored = []
    for order, card in enumerate(cards):
        if card.get("status") == "rejected":
            continue
        hits = sum(all(part.strip() in text for part in trigger.split("+")) for trigger in card.get("triggers", []))
        if hits:
            scored.append((-hits, order, card))
    return [card for _, _, card in sorted(scored, key=lambda t: t[:2])[:limit]]


def render_doctrine_cards(cards: list[dict]) -> str:
    """The cards as a block of its own, marked as not statute text."""
    if not cards:
        return ""
    blocks = []
    for card in cards:
        sources = "; ".join(card.get("sources", []))
        blocks.append(f"[{card['title']}]\n{card['text']}" + (f"\nמקורות: {sources}" if sources else ""))
    return ("<doctrine_notes>\nNotes on case law and on how the law changed over time, which the statute text "
            "does not state. They are summaries, not statute text: rely on <context> for what the law says, use "
            "these only where the case law or the timeline matters, and cite the ruling or amending law they "
            "name, never the note itself.\n\n" + "\n\n".join(blocks) + "\n</doctrine_notes>")
