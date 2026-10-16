"""Amendment links between laws in the Legal index.

Spanish law changes through separate amending provisions -- often hidden in the final dispositions
of an unrelated law ("Disposición final tercera. Modificación de la Ley 29/1994, de 24 de noviembre,
de Arrendamientos Urbanos. Se modifica el artículo 9, que queda redactado como sigue: «...»") --
which the BOE then merges into its consolidated text. An index holding both a principal law (as a
PDF of its original publication, say) and an amending law therefore holds two texts that disagree,
and nothing says which applies when. This module links them -- with dates, never a bare
"outdated" flag, because a lawyer usually needs the version in force when the facts happened:

  * At chunking, every amending article or disposition is tagged with what it amends (`amends`):
    the target law (its rank and number, "Ley 29/1994", or its name), the articles it touches, and
    whether the change is temporary ("con carácter temporal", "con vigencia hasta").
  * Every chunk carries a normalized `law_key` ("ley 29/1994"), so the target matches whatever the
    spelling around its number.
  * At query time, `notes_for()` finds, for each retrieved provision, indexed amendments to its law
    that took effect after that version started. They reach the model as dated `amended_by` notes,
    the footnotes and the audit log; a citation of an article the amendment itself changed escalates.

Nothing rewrites a principal law's text by applying an amendment: a wrong automatic merge is worse
than a flagged, dated conflict. (BOE consolidated texts, from fetch_laws.py, are already current.)
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from aicodex.legal_data.law_names import name_key

_SUFFIXES = "bis|ter|quater|quinquies|sexies|septies|octies|nonies|novies|decies"
_MONTHS = "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre"
_RANKS = (r"Ley\s+Org[áa]nica|Ley\s+Foral|Ley|Real\s+Decreto\s+Legislativo|Real\s+Decreto-ley|Real\s+Decreto"
          r"|Decreto\s+Legislativo|Decreto-ley|Decreto|Orden|Reglamento\s+\((?:UE|CE)\)|Directiva\s+\(?(?:UE|CE)\)?")
# "la Ley 29/1994, de 24 de noviembre, de Arrendamientos Urbanos" / "el Código Civil" / "texto refundido de ..."
_LAW_NAME_RE = re.compile(
    rf"(?P<name>(?:{_RANKS})\s+(?:n\.?º\s*)?\d{{1,4}}/\d{{2,4}}(?:,\s*de\s+\d{{1,2}}\s+de\s+(?:{_MONTHS}))?"
    rf"(?:,\s*(?:de|del|sobre|por\s+la\s+que|por\s+el\s+que|reguladora)\s+[^,:;«(]{{2,140}})?"
    rf"|C[óo]digo\s+(?:Civil|Penal|de\s+Comercio)|Ley\s+de\s+Enjuiciamiento\s+(?:Civil|Criminal)|Constituci[óo]n\s+Espa[ñn]ola"
    rf"|Estatuto\s+de\s+los\s+Trabajadores)"
)
_TITLE_RE = re.compile(r"(?i)^\s*Modificaci[óo]n\s+(?:parcial\s+)?(?:de\s+)?(?:la|el|los|las)?\s*(?P<rest>.+)$")
_OPENING_RE = re.compile(
    r"(?i)^\s*(?:(?:Uno|Dos|Tres|\d+)\.\s+)?(?:Se\s+(?:modifica|modifican|añade|añaden|introduce|introducen|suprime|suprimen|"
    r"deroga|derogan|da\s+nueva\s+redacci[óo]n)|(?:La|El)\s+.{3,160}?\s+queda(?:n)?\s+modificad[oa]s?)"
)
_SECTION_REF_RE = re.compile(
    rf"(?i)\bart[íi]culos?\s+(?P<list>\d{{1,4}}(?:\s+(?:{_SUFFIXES}))?(?:\.\d{{1,3}})?"
    rf"(?:\s*(?:,|y|e)\s*\d{{1,4}}(?:\s+(?:{_SUFFIXES}))?(?:\.\d{{1,3}})?)*)"
)
_OTHER_LAW_AFTER_RE = re.compile(rf"^\s*(?:,\s*)?(?:de|del)\s+(?:la\s+|el\s+)?(?:{_RANKS}|C[óo]digo|Constituci)")
_INSERTED_RE = re.compile(rf"(?m)^[«“]\s*Art[íi]culo\s+(\d{{1,4}}(?:\s+(?:{_SUFFIXES}))?)\.")
_TEMPORARY_RE = re.compile(r"(?i)con\s+car[áa]cter\s+(?:temporal|transitorio)|con\s+vigencia\s+hasta|durante\s+el\s+a[ñn]o\s+\d{4}")


def law_key(name: str) -> str:
    """Law identity without the parts that vary between citations of it: "ley 1/2000" (rank and
    number), else the accent-folded name without its date clause."""
    return name_key(name or "")


@dataclass
class AmendmentRef:
    target: str  # the amended law as named
    target_key: str
    number: str  # kept for the index format; Spanish amendments carry no number ("")
    sections: list[str]  # target-law articles touched ("9", "14 bis")
    temporary: bool
    # "amends": changes the other law's text. "reads_as": applies the other law with modifications
    # within this law's own scope -- the other law itself is unchanged.
    relation: str = "amends"

    def describe(self) -> str:
        if self.relation == "reads_as":
            scope = f" -- artículos {', '.join(self.sections)}" if self.sections else ""
            return f"{self.target} -- aplicada con modificaciones solo en el ámbito de esta norma, no modificada{scope}"
        parts = [self.target]
        if self.temporary:
            parts.append("con carácter temporal")
        if self.sections:
            parts.append(("artículo " if len(self.sections) == 1 else "artículos ") + ", ".join(self.sections))
        return " — ".join(parts)


def parse_toc(preamble_text: str) -> list[tuple[str, str, bool]]:
    """Israeli gazettes list the laws an act amends in its header; Spanish acts don't. Kept for the
    chunker's interface: always empty."""
    return []


def _target_sections(text: str) -> list[str]:
    sections: list[str] = []
    for match in _SECTION_REF_RE.finditer(text):
        if _OTHER_LAW_AFTER_RE.match(text[match.end() : match.end() + 40]) and not _LAW_NAME_RE.search(
                text[: match.start()]):
            continue
        for number in re.findall(rf"\d{{1,4}}(?:\s+(?:{_SUFFIXES}))?", match.group("list")):
            number = re.sub(r"\s+", " ", number)
            if number not in sections:
                sections.append(number)
    for number in _INSERTED_RE.findall(text):
        number = re.sub(r"\s+", " ", number)
        if number not in sections:
            sections.append(number)
    return sections


def extract_amendment(title: str | None, text: str, toc: list[tuple[str, str, bool]] | None = None) -> AmendmentRef | None:
    """What an amending article or disposition (or one chunk of it) amends, from its title
    ("Modificación de la Ley 29/1994, de 24 de noviembre, de Arrendamientos Urbanos") or its opening
    words ("Se modifica el artículo 9 de la Ley 29/1994 ..."). None for a provision that amends nothing."""
    body = text.split("\n\n", 1)[-1]  # never the breadcrumb: its "Artículo 7" is this law's own number
    target = None
    match = _TITLE_RE.match(title or "")
    if match:
        named = _LAW_NAME_RE.search(match.group("rest"))
        target = named.group("name") if named else match.group("rest").strip().rstrip(".")
    elif _OPENING_RE.match(body):
        named = _LAW_NAME_RE.search(body[:600].split("«", 1)[0])
        target = named.group("name") if named else None
    if not target:
        return None
    target = re.sub(r"\s+", " ", target).strip().rstrip(",.")
    return AmendmentRef(target, law_key(target), "", _target_sections(body.split("«", 1)[0] + "\n" + body),
                        bool(_TEMPORARY_RE.search(body)))


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
        what = "amendment"
        scope = "this article" if self.touches_section else (
            "articles " + ", ".join(self.ref.sections) if self.ref.sections else "this law"
        )
        kind = ", temporary provision" if self.ref.temporary else ""
        return f"{what}{kind} by {self.amending_law}{where}, effective {self.effective}, amended {scope}"


def _base_section(number: str) -> str:
    """"62(2)" -> "62", "14 bis.3" -> "14 bis", "Disposición adicional primera" unchanged."""
    match = re.match(rf"\d{{1,4}}(?:\s+(?:{_SUFFIXES}))?", number or "")
    return match.group(0) if match else (number or "")


def build_index(metadatas: list[dict]) -> dict[str, list[tuple[AmendmentRef, dict]]]:
    """target law key -> [(amendment, amending chunk's metadata)] over the index."""
    index: dict[str, list[tuple[AmendmentRef, dict]]] = {}
    for meta in metadatas:
        for ref in decode(meta.get("amends") or "[]"):
            if ref.relation == "amends":  # applying a law with modifications doesn't change it
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
