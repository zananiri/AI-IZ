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
