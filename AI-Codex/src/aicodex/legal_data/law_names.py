"""Spanish law names: normalization, the official number ("Ley 1/2000"), and the short names and
abbreviations lawyers use ("LEC", "Código Penal", "ET").

The same law is written 'Ley 1/2000, de 7 de enero, de Enjuiciamiento Civil.' in the BOE,
'Ley 1/2000 de Enjuiciamiento Civil' in a brief and 'LEC' in a question. name_key() erases case,
accents, punctuation and the date clause; law_number_key() keeps only rank + number, which is the
law's identity whenever it has one."""

from __future__ import annotations

import re
import unicodedata

from aicodex.legal_data.spanish import fold_accents

_MONTHS = "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre"
_DATE_CLAUSE_RE = re.compile(rf",?\s*de\s+\d{{1,2}}\s+de\s+(?:{_MONTHS})(?:\s+de\s+\d{{4}})?,?", re.IGNORECASE)
_RANKS = (
    "Ley Orgánica", "Ley Foral", "Ley", "Real Decreto Legislativo", "Real Decreto-ley", "Real Decreto",
    "Decreto Legislativo", "Decreto-ley Foral", "Decreto Foral Legislativo", "Decreto-ley", "Decreto Foral",
    "Decreto", "Orden", "Resolución", "Circular", "Instrucción", "Reglamento", "Directiva", "Decisión",
)
_RANK_RE = "|".join(re.escape(r) for r in sorted(_RANKS, key=len, reverse=True))
_NUMBER_RE = re.compile(rf"(?i)\b(?P<rank>{_RANK_RE})\s+(?:\((?:UE|CE)\)\s+)?(?:n\.?º\s*)?(?P<num>\d{{1,4}}/\d{{2,4}})")

# Short names and abbreviations -> the canonical BOE identifier of the consolidated text.
# Used to resolve a question's "art. 1902 CC" to the right law; extend freely.
ABBREVIATIONS: dict[str, str] = {
    "CE": "BOE-A-1978-31229",          # Constitución Española
    "CONSTITUCION": "BOE-A-1978-31229",
    "CC": "BOE-A-1889-4763",           # Código Civil
    "CODIGO CIVIL": "BOE-A-1889-4763",
    "CP": "BOE-A-1995-25444",          # Código Penal (LO 10/1995)
    "CODIGO PENAL": "BOE-A-1995-25444",
    "CCOM": "BOE-A-1885-6627",         # Código de Comercio
    "CODIGO DE COMERCIO": "BOE-A-1885-6627",
    "LEC": "BOE-A-2000-323",           # Ley 1/2000, de Enjuiciamiento Civil
    "LECRIM": "BOE-A-1882-6036",       # Ley de Enjuiciamiento Criminal
    "LOPJ": "BOE-A-1985-12666",        # LO 6/1985, del Poder Judicial
    "LJCA": "BOE-A-1998-16718",        # Ley 29/1998, Jurisdicción Contencioso-administrativa
    "LRJS": "BOE-A-2011-15936",        # Ley 36/2011, Jurisdicción Social
    "ET": "BOE-A-2015-11430",          # RDLeg 2/2015, Estatuto de los Trabajadores
    "LGSS": "BOE-A-2015-11724",        # RDLeg 8/2015, Ley General de la Seguridad Social
    "LPAC": "BOE-A-2015-10565",        # Ley 39/2015, Procedimiento Administrativo Común
    "LRJSP": "BOE-A-2015-10566",       # Ley 40/2015, Régimen Jurídico del Sector Público
    "LAU": "BOE-A-1994-26003",         # Ley 29/1994, Arrendamientos Urbanos
    "LPH": "BOE-A-1960-10906",         # Ley 49/1960, Propiedad Horizontal
    "LSC": "BOE-A-2010-10544",         # RDLeg 1/2010, Sociedades de Capital
    "TRLC": "BOE-A-2020-4859",         # RDLeg 1/2020, Ley Concursal
    "LC": "BOE-A-2020-4859",
    "LGT": "BOE-A-2003-23186",         # Ley 58/2003, General Tributaria
    "LIRPF": "BOE-A-2006-20764",       # Ley 35/2006, IRPF
    "LIVA": "BOE-A-1992-28740",        # Ley 37/1992, IVA
    "LIS": "BOE-A-2014-12328",         # Ley 27/2014, Impuesto sobre Sociedades
    "TRLGDCU": "BOE-A-2007-20555",     # RDLeg 1/2007, Defensa de Consumidores y Usuarios
    "LOPDGDD": "BOE-A-2018-16673",     # LO 3/2018, Protección de Datos
    "LCSP": "BOE-A-2017-12902",        # Ley 9/2017, Contratos del Sector Público
    "LOTC": "BOE-A-1979-23709",        # LO 2/1979, del Tribunal Constitucional
    "LOREG": "BOE-A-1985-11672",       # LO 5/1985, Régimen Electoral General
    "LH": "BOE-A-1946-2453",           # Ley Hipotecaria
    "LCS": "BOE-A-1980-22501",         # Ley 50/1980, Contrato de Seguro
    "LCCI": "BOE-A-2019-3814",         # Ley 5/2019, Contratos de Crédito Inmobiliario
    "LEX": "BOE-A-1954-15431",         # Ley de Expropiación Forzosa
    "LOEX": "BOE-A-2000-544",          # LO 4/2000, Extranjería
    "LOVG": "BOE-A-2004-21760",        # LO 1/2004, Violencia de Género
    "LRC": "BOE-A-2011-12628",         # Ley 20/2011, Registro Civil
}


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFC", name or "").strip().rstrip(".")
    text = re.sub(r"\s*,\s*", ", ", text)
    return re.sub(r"\s+", " ", text)


def law_number_key(name: str) -> str | None:
    """"ley 1/2000", "real decreto 35/2023" -- the rank and official number, or None."""
    match = _NUMBER_RE.search(name or "")
    if not match:
        return None
    return f"{fold_accents(match.group('rank')).lower()} {match.group('num')}"


# Codes known by name: the key their full title or official number gives, so "Código Penal" and
# "Ley Orgánica 10/1995, de 23 de noviembre, del Código Penal" match.
_NAMED_CODES = {
    "codigo penal": "ley organica 10/1995",
    "codigo civil": "codigo civil",
    "codigo de comercio": "codigo de comercio",
    "ley de enjuiciamiento criminal": "ley de enjuiciamiento criminal",
    "ley de enjuiciamiento civil": "ley 1/2000",
    "constitucion española": "constitucion española",
    "estatuto de los trabajadores": "real decreto legislativo 2/2015",
    "ley hipotecaria": "ley hipotecaria",
}


def name_key(name: str) -> str:
    """The law's identity for matching: its rank + number when it has one, else its folded name
    without the date clause (a known code by its canonical key)."""
    number = law_number_key(name)
    if number:
        return number
    text = _DATE_CLAUSE_RE.sub(" ", normalize_name(name))
    text = re.sub(r"[^\w/]+", " ", fold_accents(text).lower()).strip()
    for code, key in _NAMED_CODES.items():
        if code in text:
            return key
    return text


def has_prefix(title: str, prefix: str) -> bool:
    return fold_accents(normalize_name(title)).lower().startswith(fold_accents(normalize_name(prefix)).lower())


def resolve_abbreviation(token: str) -> str | None:
    """'LEC' / 'Código Civil' -> its BOE identifier, else None."""
    key = re.sub(r"[^\w ]", "", fold_accents(token or "")).upper().strip()
    return ABBREVIATIONS.get(key)


class NameIndex:
    def __init__(self) -> None:
        self._ids: dict[str, set[str]] = {}

    def add(self, name: str | None, registry_id: str) -> None:
        if name:
            self._ids.setdefault(name_key(name), set()).add(registry_id)

    def lookup(self, name: str) -> str | None:
        ids = self._ids.get(name_key(name))
        return next(iter(ids)) if ids and len(ids) == 1 else None

    def __len__(self) -> int:
        return len(self._ids)
