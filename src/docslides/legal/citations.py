"""[[CITE: ...]] claim-level citation tokens (spec section 3): parsing, the
citation lock around DictaLM's Hebrew polish, and rendering for the UI.

Token format:
    [[CITE: claim_id="C1" | source_id="..." | law="..." | section="..." |
            effective="..." | source_type="statute" | relation="supports"]]

Values are split on "|" rather than parsed as quoted strings: Hebrew law
names routinely contain an ASCII double quote (תשל"ג), which would break a
quote-delimited parser.

Citation lock: before polishing, each full token is swapped for a short
numbered placeholder ([[CITE:1]], [[CITE:2]], ...). The polish prompt tells
DictaLM to leave [[CITE...]] tags alone, and short placeholders give it far
less to mangle than long ASCII attribute strings. `unlock` then requires
every placeholder to come back exactly once and in the original order.
Anything else raises CitationLockError and the polish is rejected.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"\[\[CITE:(?P<body>(?:(?!\]\]).)*?=(?:(?!\]\]).)*)\]\]")
_PLACEHOLDER_RE = re.compile(r"\[\[\s*CITE\s*:\s*(\d+)\s*\]\]")
_ANY_CITE_TAG_RE = re.compile(r"\[\[\s*CITE\b[^\]]*\]\]")
_FIELDS = ("claim_id", "source_id", "law", "section", "effective", "source_type", "relation")


class CitationLockError(Exception):
    pass


@dataclass
class Citation:
    claim_id: str
    source_id: str
    law: str
    section: str
    effective: str
    source_type: str
    relation: str
    raw: str
    start: int
    end: int


def _parse_body(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in re.split(r"\s*\|\s*", body.strip()):
        key, sep, value = part.partition("=")
        if not sep:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1]
        fields[key.strip()] = value
    return fields


def parse_citations(text: str) -> list[Citation]:
    citations = []
    for match in _TOKEN_RE.finditer(text):
        fields = _parse_body(match.group("body"))
        citations.append(
            Citation(
                **{name: fields.get(name, "") for name in _FIELDS},
                raw=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )
    return citations


def format_citation(**fields: str) -> str:
    body = " | ".join(f'{name}="{fields.get(name, "")}"' for name in _FIELDS)
    return f"[[CITE: {body}]]"


def expand_citations(text: str, evidence: dict) -> str:
    """Rewrites every token into the full canonical form, taking law /
    section / effective / source_type from the cited chunk's metadata
    (`evidence`: source_id -> ChunkMetadata). The drafting model writes only
    claim_id, source_id and relation, without quotes: its draft travels
    inside a JSON string, and an unescaped quote there ends the string
    mid-answer. A value the model did give that disagrees with the metadata
    is kept, so the structural check still flags it. Unknown source_ids are
    left as written so that check can flag them too."""

    def canonical(match: re.Match[str]) -> str:
        fields = _parse_body(match.group("body"))
        meta = evidence.get(fields.get("source_id", ""))
        if meta is not None:
            section = meta.section_number + (f"({meta.subsection_number})" if meta.subsection_number else "")
            effective = (
                f"{meta.effective_date_start} to {meta.effective_date_end}" if meta.effective_date_end else "current"
            )
            defaults = {"law": meta.law_name, "section": section, "effective": effective, "source_type": meta.source_type}
            for key, value in defaults.items():
                if not fields.get(key):
                    fields[key] = value
        return format_citation(**fields)

    return _TOKEN_RE.sub(canonical, text)


def strip_citations(text: str) -> str:
    return re.sub(r"\s+([.,;:!?])", r"\1", _ANY_CITE_TAG_RE.sub("", text)).strip()


def sentence_before(text: str, token_start: int) -> str:
    """The sentence a citation token is attached to: the text between the
    previous sentence boundary / citation token and this token."""
    segment = text[:token_start].rstrip()
    cut = max(segment.rfind("\n"), segment.rfind("]]") + 1 if "]]" in segment else -1)
    for match in re.finditer(r"[.!?]\s", segment[:-1]):
        cut = max(cut, match.end() - 1)
    return strip_citations(segment[cut + 1 :]).strip()


@dataclass
class CitationLock:
    tokens: list[str]
    digest: str


def _digest(tokens: list[str]) -> str:
    return hashlib.sha256(json.dumps(tokens, ensure_ascii=False).encode("utf-8")).hexdigest()


def lock(text: str) -> tuple[str, CitationLock]:
    tokens: list[str] = []

    def swap(match: re.Match[str]) -> str:
        tokens.append(match.group(0))
        return f"[[CITE:{len(tokens)}]]"

    return _TOKEN_RE.sub(swap, text), CitationLock(tokens=tokens, digest=_digest(tokens))


def lock_problems(locked_text: str, citation_lock: CitationLock) -> list[str]:
    found = [int(n) for n in _PLACEHOLDER_RE.findall(locked_text)]
    expected = list(range(1, len(citation_lock.tokens) + 1))
    problems = []
    if found != expected:
        problems.append(f"citation tags changed: expected {expected}, found {found}")
    stray = len(_ANY_CITE_TAG_RE.findall(locked_text)) - len(found)
    if stray:
        problems.append(f"{stray} citation tag(s) altered beyond the numbered placeholder form")
    return problems


def unlock(locked_text: str, citation_lock: CitationLock) -> str:
    problems = lock_problems(locked_text, citation_lock)
    if problems:
        raise CitationLockError("; ".join(problems))
    text = _PLACEHOLDER_RE.sub(lambda m: citation_lock.tokens[int(m.group(1)) - 1], locked_text)
    if _digest([c.raw for c in parse_citations(text)]) != citation_lock.digest:
        raise CitationLockError("citation tokens differ from the locked set after unlocking")
    return text


def render_with_footnotes(text: str) -> tuple[str, list[Citation], list[int]]:
    """Replaces every token with a footnote marker "[n]", numbered by first
    appearance of each distinct source_id. Returns (display text, the
    citations in token order, each citation's footnote number)."""
    citations = parse_citations(text)
    numbers: dict[str, int] = {}
    assigned: list[int] = []
    for citation in citations:
        numbers.setdefault(citation.source_id, len(numbers) + 1)
        assigned.append(numbers[citation.source_id])

    pieces, cursor = [], 0
    for citation, number in zip(citations, assigned):
        pieces.append(text[cursor : citation.start].rstrip())
        pieces.append(f" [{number}]")
        cursor = citation.end
    pieces.append(text[cursor:])
    display = "".join(pieces)
    # Collapse "[1] [1]" left by two claims citing the same source back to back.
    display = re.sub(r"(\[\d+\])(?:\s*\1)+", r"\1", display)
    return display, citations, assigned
