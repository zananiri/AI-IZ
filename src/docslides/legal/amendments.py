"""Amendment links between laws in the Legal index.

Israeli law changes mostly through separate amending laws ("in section 62(ג)
of the Elections Law, replace 'ה־40' with 'ה־43'"), not by republishing the
principal law. An index holding both a principal law and its amending laws
therefore holds two texts that disagree, and nothing says which applies when.
This module links them -- with dates, never a bare "outdated" flag, because a
lawyer usually needs the version in force when the facts happened:

  * At chunking, every section of an amending law is tagged with what it amends
    (`amends`): the target law (full name, resolved through the law's own list
    of indirect amendments, "תיקונים עקיפים"), the amendment number, the target
    sections it touches, and whether it's a temporary provision (הוראת שעה).
  * Every chunk carries a normalized `law_key`, so "חוק הבחירות לכנסת [נוסח
    משולב], התשכ"ט-1969" matches its amendments whatever the spelling around it.
  * At query time, `notes_for()` finds, for each retrieved provision, indexed
    amendments to its law that took effect after that version started. They
    reach the model as dated `amended_by` notes, the footnotes and the audit
    log; a citation of a section the amendment itself changed escalates.

Nothing rewrites a principal law's text by applying an amendment: a wrong
automatic merge is worse than a flagged, dated conflict.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

_TOC_LINE_RE = re.compile(
    r"^(?P<name>(?:חוק|פקודת|פקודה|תקנות)\s.+?)\s*-\s*(?:(?P<temp>הוראת שעה)\s*-\s*)?מס['׳]\s*(?P<num>\d+)\s*$"
)
_TITLE_RE = re.compile(
    r"^תיקון\s+(?P<name>.+?)\s*-\s*(?:(?P<temp>הוראת שעה)\s*-\s*)?מס['׳]\s*(?P<num>\d+)"
)
_OPENING_LAW_RE = re.compile(r"^\s*ב(?P<name>(?:חוק|פקודת|תקנות)\s[^,]+?,\s*ה?תש[^,\s]*-\s*\d{4})")
_SECTION_REF_RE = re.compile(
    r"(?:^|[\s(])(?:ו|ב|ל|מ|ש|אחרי |לפני |במקום )?סעי(?:פים|פי|ף)\s+"
    r"(?P<list>\d{1,4}[א-ת]{0,3}\d{0,3}(?:\([^)]{1,4}\))*(?:\s*(?:,|ו־?|עד)\s*\d{1,4}[א-ת]{0,3}\d{0,3}(?:\([^)]{1,4}\))*)*)"
)
_INSERTED_SECTION_RE = re.compile(r"(?m)^(\d{1,4}[א-ת]{1,3}\d{0,3})\.\s")
_OTHER_LAW_AFTER_RE = re.compile(r"^\s*(?:ל|ב)?(?:חוק|פקודת|תקנות)\s")


def law_key(name: str) -> str:
    """Law identity without the parts that vary between citations of it: the
    "[נוסח משולב]" tag, the Hebrew/Gregorian year, punctuation, spacing."""
    key = re.sub(r"\[[^\]]*\]", " ", name or "")
    key = re.split(r",\s*ה?תש", key)[0]
    key = re.sub(r"[-–]\s*(?:19|20)\d\d.*$", "", key)
    key = re.sub(r"[\"״׳'(),.]", " ", key)
    return re.sub(r"\s+", " ", key).strip()


@dataclass
class AmendmentRef:
    target: str  # full name of the amended law
    target_key: str
    number: str  # amendment number ("79"), "" if not stated
    sections: list[str]  # target-law sections touched ("62", "24א")
    temporary: bool

    def describe(self) -> str:
        parts = [self.target]
        if self.number:
            parts.append(f"מס' {self.number}")
        if self.temporary:
            parts.append("הוראת שעה")
        if self.sections:
            parts.append("סעיפים " + ", ".join(self.sections))
        return " — ".join(parts)


def parse_toc(preamble_text: str) -> list[tuple[str, str, bool]]:
    """(law name, amendment number, temporary) from a gazette's list of
    indirect amendments, e.g. 'חוק הבחירות לכנסת [נוסח משולב], התשכ"ט-1969 - מס' 79'."""
    entries = []
    for line in preamble_text.splitlines():
        match = _TOC_LINE_RE.match(line.strip())
        if match:
            entries.append((match.group("name").strip(), match.group("num"), bool(match.group("temp"))))
    return entries


def _resolve(short_name: str, number: str, toc: list[tuple[str, str, bool]]) -> str:
    """Full name of the law a title refers to by its short name ("חוק
    התעמולה"), using the amendment number to pick the TOC entry."""
    candidates = [name for name, num, _ in toc if num == number]
    if len(candidates) == 1:
        return candidates[0]
    words = set(law_key(short_name).split())
    scored = sorted(
        ((len(words & set(law_key(name).split())), name) for name in candidates or [n for n, _, _ in toc]),
        reverse=True,
    )
    return scored[0][1] if scored and scored[0][0] >= 2 else short_name


def _target_sections(text: str) -> list[str]:
    sections: list[str] = []
    for match in _SECTION_REF_RE.finditer(text):
        if _OTHER_LAW_AFTER_RE.match(text[match.end() : match.end() + 25]):
            continue  # "סעיף 5 לחוק X" -- a third law, not the one being amended
        listing = re.sub(r"\([^)]*\)", "", match.group("list"))
        for number in re.findall(r"\d{1,4}[א-ת]{0,3}\d{0,3}", listing):
            if number not in sections:
                sections.append(number)
    for number in _INSERTED_SECTION_RE.findall(text):
        if number not in sections:
            sections.append(number)
    return sections


def extract_amendment(title: str | None, text: str, toc: list[tuple[str, str, bool]]) -> AmendmentRef | None:
    """What an amending section (or one chunk of it) amends, from its margin
    title ("תיקון חוק הבחירות לכנסת - מס' 79") or, without a title, its
    opening words ("בחוק המפלגות, התשנ"ב-1992, בסעיף ..."). None for a
    section that amends nothing."""
    body = text.split("\n\n", 1)[-1]  # never the breadcrumb: its "סעיף 7(4)" is this law's own number
    match = _TITLE_RE.match(title or "")
    if match:
        number, temporary = match.group("num"), bool(match.group("temp"))
        target = _resolve(match.group("name").strip(), number, toc)
    else:
        opening = _OPENING_LAW_RE.match(body)
        if not opening:
            return None
        target, number, temporary = opening.group("name").strip(), "", False
        for name, num, temp in toc:
            if law_key(name) == law_key(target):
                number, temporary = num, temp
                break
    return AmendmentRef(target, law_key(target), number, _target_sections(body), temporary)


def encode(refs: list[AmendmentRef]) -> list[dict]:
    return [asdict(r) for r in refs]


def decode(data) -> list[AmendmentRef]:
    items = json.loads(data) if isinstance(data, str) else (data or [])
    return [AmendmentRef(**item) for item in items]


# --- query time ---------------------------------------------------------------


@dataclass
class AmendmentNote:
    """A later indexed amendment to a retrieved provision's law."""

    amending_law: str
    gazette: str | None
    effective: str  # the amending law's effective date
    ref: AmendmentRef
    touches_section: bool  # the amendment names this provision's section

    def describe(self) -> str:
        where = f" ({self.gazette})" if self.gazette else ""
        what = f"מס' {self.ref.number}" if self.ref.number else "amendment"
        scope = "this section" if self.touches_section else (
            "sections " + ", ".join(self.ref.sections) if self.ref.sections else "this law"
        )
        kind = ", temporary provision" if self.ref.temporary else ""
        return f"{what}{kind} by {self.amending_law}{where}, effective {self.effective}, amended {scope}"


def _base_section(number: str) -> str:
    """"62(ג)" -> "62", "116יז10(ד)" -> "116יז10"."""
    match = re.match(r"\d{1,4}[א-ת]{0,3}\d{0,3}", number or "")
    return match.group(0) if match else (number or "")


def build_index(metadatas: list[dict]) -> dict[str, list[tuple[AmendmentRef, dict]]]:
    """target law key -> [(amendment, amending chunk's metadata)] over the index."""
    index: dict[str, list[tuple[AmendmentRef, dict]]] = {}
    for meta in metadatas:
        for ref in decode(meta.get("amends") or "[]"):
            index.setdefault(ref.target_key, []).append((ref, meta))
    return index


def notes_for(meta, index: dict[str, list[tuple[AmendmentRef, dict]]]) -> list[AmendmentNote]:
    """Amendments that took effect after `meta`'s version started (so that
    version's text may have changed since), one note per amending law +
    number, section-specific first."""
    key = getattr(meta, "law_key", "") or law_key(getattr(meta, "law_name", ""))
    section = _base_section(getattr(meta, "section_number", ""))
    notes: dict[tuple[str, str], AmendmentNote] = {}
    for ref, amending in index.get(key, []):
        if amending.get("law_id") == getattr(meta, "law_id", None):
            continue
        if amending.get("effective_date_start", "") <= getattr(meta, "effective_date_start", ""):
            continue  # this version already post-dates (so includes) the amendment
        touches = section in {_base_section(s) for s in ref.sections}
        note_key = (amending.get("law_name", ""), ref.number)
        existing = notes.get(note_key)
        if existing is None:
            own_ref = AmendmentRef(**{**asdict(ref), "sections": list(ref.sections)})  # never mutate the index
            notes[note_key] = AmendmentNote(
                amending.get("law_name", ""), amending.get("gazette") or None,
                amending.get("effective_date_start", ""), own_ref, touches,
            )
        else:
            existing.touches_section = existing.touches_section or touches
            existing.ref.sections += [s for s in ref.sections if s not in existing.ref.sections]
    return sorted(notes.values(), key=lambda n: (not n.touches_section, n.effective))
