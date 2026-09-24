"""[[CITE: ...]] claim-level citation tokens (spec section 3): parsing,
expansion and rendering for the UI.

Token format:
    [[CITE: claim_id="C1" | source_id="..." | law="..." | section="..." |
            effective="..." | source_type="statute" | relation="supports"]]

Values are split on "|" rather than parsed as quoted strings: Hebrew law
names routinely contain an ASCII double quote (תשל"ג), which would break a
quote-delimited parser.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"\[\[CITE:(?P<body>(?:(?!\]\]).)*?=(?:(?!\]\]).)*)\]\]")
_FIELDS = ("claim_id", "source_id", "law", "section", "effective", "source_type", "relation")
_ANY_CITE_TAG_RE = re.compile(r"\[\[\s*CITE\b[^\]]*\]\]")


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
    left as written so that check can flag them too -- except a near miss
    (see `_resolve_source_id`)."""

    def canonical(match: re.Match[str]) -> str:
        fields = _parse_body(match.group("body"))
        if fields.get("source_id", "") not in evidence:
            fields["source_id"] = _resolve_source_id(fields.get("source_id", ""), evidence)
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


def _resolve_source_id(source_id: str, evidence: dict) -> str:
    """A source_id the model garbled in its version date ('law-x@2026-11:3' for
    'law-x@2026-05-11:3') -> the one evidence id with the same law and section.
    Anything less certain is left as written."""
    law, _, rest = source_id.partition("@")
    _, _, section = rest.partition(":")
    if not law or not section:
        return source_id
    matches = [sid for sid in evidence if sid.partition("@")[0] == law and sid.partition(":")[2] == section]
    return matches[0] if len(matches) == 1 else source_id


def outside_citations(text: str, transform: Callable[[str], str]) -> str:
    """Applies `transform` to the prose between citation tokens, leaving the tokens as they are."""
    pieces, cursor = [], 0
    for match in _TOKEN_RE.finditer(text):
        pieces += [transform(text[cursor : match.start()]), match.group(0)]
        cursor = match.end()
    pieces.append(transform(text[cursor:]))
    return "".join(pieces)


def strip_citations(text: str) -> str:
    return re.sub(r"\s+([.,;:!?])", r"\1", _ANY_CITE_TAG_RE.sub("", text)).strip()


def _sentence_start(text: str, token_start: int) -> int:
    """Where the sentence a citation token is attached to begins: after the
    previous sentence boundary, line break or citation token."""
    segment = text[:token_start].rstrip()
    cut = max(segment.rfind("\n"), segment.rfind("]]") + 1 if "]]" in segment else -1)
    for match in re.finditer(r"[.!?]\s", segment[:-1]):
        cut = max(cut, match.end() - 1)
    return cut + 1


def sentence_before(text: str, token_start: int) -> str:
    """The sentence a citation token is attached to: the text between the
    previous sentence boundary / citation token and this token."""
    return strip_citations(text[_sentence_start(text, token_start) : token_start].rstrip()).strip()


def remove_cited_sentences(text: str, indices: set[int]) -> tuple[str, set[int]]:
    """Drops the sentence each citation in `indices` (its index in token order)
    is attached to, together with every token attached to that sentence -- a
    token right after another token belongs to the same sentence. Returns the
    text and the indices of all the tokens removed."""
    groups: list[tuple[int, int, list[int]]] = []  # (sentence start, last token end, token indices)
    for index, citation in enumerate(parse_citations(text)):
        start = _sentence_start(text, citation.start)
        if groups and not strip_citations(text[start : citation.start]).strip():
            first, _, members = groups[-1]
            groups[-1] = (first, citation.end, [*members, index])
        else:
            groups.append((start, citation.end, [index]))
    removed: set[int] = set()
    pieces, cursor = [], 0
    for start, end, members in groups:
        if indices & set(members):
            pieces.append(text[cursor:start])
            cursor = end
            removed.update(members)
    pieces.append(text[cursor:])
    cleaned = re.sub(r"[ \t]+\n", "\n", re.sub(r"[ \t]{2,}", " ", "".join(pieces)))
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip(), removed


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
