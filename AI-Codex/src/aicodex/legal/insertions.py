"""Provisions that an amending law inserts into another law (for chunking.py).

A Spanish amending provision such as 'Se añade un nuevo artículo 14 bis a la Ley 29/1994, ..., con
la siguiente redacción: «Artículo 14 bis. ...»' or 'Se modifica el artículo 9, que queda redactado
como sigue: «Artículo 9. ...»' carries whole new provisions of the amended law. Chunked as one
unit, a long insertion (several articles quoted together) sprawls across parts that every question
about any of them drags in together, and it can only be cited by the amending law's own number
("disposición final tercera") -- never by the number a lawyer asks about ("artículo 14 bis").
`find_insertion` picks those provisions out so each gets its own chunk, labelled with both numbers.

The BOE's consolidated texts already merge these changes into the amended law; this matters for
amending laws ingested as PDFs (the BOE's own daily publication, autonomous bulletins).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SUFFIXES = "bis|ter|quater|quinquies|sexies|septies|octies|nonies|novies|decies"
_INSTRUCTION_RE = re.compile(
    r"(?i)(?:redactad[oa]s?\s+(?:como\s+sigue|del\s+siguiente\s+modo|en\s+los\s+siguientes\s+t[ée]rminos)"
    r"|(?:la\s+)?siguiente\s+redacci[óo]n|en\s+los\s+siguientes\s+t[ée]rminos|con\s+el\s+siguiente\s+(?:contenido|tenor))"
    r"\s*[:.]?\s*$"
)
_HEAD_RE = re.compile(
    rf"^[«“\"]?\s*Art[íi]culo\s+(?P<num>\d{{1,4}}(?:\s+(?:{_SUFFIXES}))?|[úu]nico)\s*\.\s*(?P<rest>.*)$"
)
_APARTADO_HEAD_RE = re.compile(r"^[«“\"]\s*(?P<num>\d{1,3})\.\s+(?P<rest>.*)$")
_CHAPTER_RE = re.compile(r"^[«“\"]?\s*(?:CAP[ÍI]TULO|Cap[íi]tulo|Secci[óo]n|SECCI[ÓO]N|T[ÍI]TULO)\s")
_CLOSE_RE = re.compile(r"[»”]\s*[.;,]?\s*$")
_SPACED_SUFFIX_RE = re.compile(rf"(?<=\d)(?=(?:{_SUFFIXES})\b)")
_REPLACED_APARTADO_RE = re.compile(
    rf"(?i)(?:se\s+modifica|se\s+añade|se\s+introduce)\s+(?:el|un\s+nuevo)\s+apartado\s+(?P<sub>\d{{1,3}})\s+"
    rf"(?:al|del|en\s+el)\s+art[íi]culo\s+(?P<num>\d{{1,4}}(?:\s+(?:{_SUFFIXES}))?)"
)


def join_spaced_section_numbers(text: str) -> str:
    """'14bis' -> '14 bis' (the spelling the index uses); everything else unchanged."""
    return _SPACED_SUFFIX_RE.sub(" ", text)


@dataclass
class InsertedProvision:
    number: str  # "14 bis"; "9.2" for a replaced apartado; "" if it couldn't be recovered
    title: str | None
    lines: list[str]
    inferred_number: bool = False


@dataclass
class Insertion:
    context: list[str]  # the amending instruction, plus an inserted chapter heading
    provisions: list[InsertedProvision] = field(default_factory=list)
    trailing: list[str] = field(default_factory=list)  # further instructions after the quoted block


def _strip_quotes(line: str) -> str:
    return line.strip().lstrip("«“\"").rstrip("»”\"").strip()


def _replaced_apartado(context: list[str], body: list[str]) -> Insertion | None:
    """'Se modifica el apartado 2 del artículo 9, que queda redactado como sigue: «2. ...»': the new
    apartado is the inserted provision 9.2."""
    match = _REPLACED_APARTADO_RE.search(" ".join(context))
    if not match or not body:
        return None
    head = _APARTADO_HEAD_RE.match(body[0].strip())
    if not head or head.group("num") != match.group("sub"):
        return None
    close = next((i for i in range(len(body)) if _CLOSE_RE.search(body[i])), len(body) - 1)
    lines = [_strip_quotes(line) for line in body[: close + 1]]
    article = re.sub(r"\s+", " ", match.group("num")).lower()
    number = f"{article}.{match.group('sub')}"
    provision = InsertedProvision(number, None, [line for line in lines if line])
    return Insertion(context=context, provisions=[provision], trailing=body[close + 1 :])


def find_insertion(text: str) -> Insertion | None:
    """The provisions `text` (one amending article or apartado) inserts, or None when it inserts
    none -- replacing a few words ('se sustituye la expresión «X» por «Y»') isn't inserting a
    provision."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for start, line in enumerate(lines):
        if not _INSTRUCTION_RE.search(line):
            continue
        body = lines[start + 1 :]
        heads = [i for i, candidate in enumerate(body) if _HEAD_RE.match(candidate)]
        if not heads:
            replaced = _replaced_apartado(lines[: start + 1], body)
            if replaced:
                return replaced
            continue
        close = next((i for i in range(heads[-1], len(body)) if _CLOSE_RE.search(body[i])), len(body) - 1)
        insertion = Insertion(context=lines[: start + 1], trailing=body[close + 1 :])
        body = body[: close + 1]
        i = 0
        while i < heads[0] and _CHAPTER_RE.match(body[i]):
            insertion.context.append(_strip_quotes(body[i]))
            i += 1
        for k, h in enumerate(heads):
            match = _HEAD_RE.match(body[h])
            end = heads[k + 1] if k + 1 < len(heads) else len(body)
            rest = _strip_quotes(match.group("rest"))
            title = None
            title_match = re.match(r"^(?P<title>[^.]{2,120}?)\.\s*(?P<body>.*)$", rest)
            if title_match and len(title_match.group("title").split()) <= 14:
                title, rest = title_match.group("title").strip(), title_match.group("body").strip()
            provision_lines = ([rest] if rest else []) + [_strip_quotes(l) for l in body[h + 1 : end]]
            number = re.sub(r"\s+", " ", match.group("num")).lower()
            insertion.provisions.append(InsertedProvision(number, title, [l for l in provision_lines if l]))
        return insertion
    return None
