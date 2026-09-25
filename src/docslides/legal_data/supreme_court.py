"""Supreme Court rulings from the Hugging Face dataset LevMuchnik/SupremeCourtOfIsrael (a 2022
snapshot, one 1.5 GB parquet file, license OpenRAIL).

Kept: full judgments (Type in legal_data.judgment_types, "פסק-דין") and decisions marked
non-technical (Technical == false). Everything else is counted, not written.

Privacy (strict, legal_data.privacy in config): a kept document is excluded when
  1. publication_restriction -- its text mentions a publication restriction anywhere
     (אסור בפרסום, צו איסור פרסום, חסוי, בדלתיים סגורות, הותר לפרסום ...);
  2. family_case_type -- its case number or case type starts with a family prefix (בע"מ ...),
     or its department is a configured family value;
  3. anonymized_parties -- the case name or a party is פלוני/אלמוני (the court already hid them);
  4. topic_keywords -- adoption, minors, sexual offenses or family topics in the case name, case
     type or the first `topic_scan_chars` characters.
Rules are checked in that order; the report counts both the first rule that removed a document
and every rule that would have.

Anonymizer (optional): natural-person parties from meta_side_nm become [צד 1], [צד 2] ... in the
text and case name; the State, public bodies and organizations (legal_data.privacy.
public_body_patterns) are kept. It is a heuristic over the dataset's party list, not a guarantee.
Party and lawyer names are never copied into record fields.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import quote

from docslides.config import LegalDataPrivacyConfig
from docslides.legal_data.hebrew import repair_text, strip_points
from docslides.legal_data.http import PoliteClient
from docslides.legal_data.records import CorpusRecord, Quality, now_iso

LICENSE = "OpenRAIL (per the dataset card: huggingface.co/datasets/LevMuchnik/SupremeCourtOfIsrael)"
COURT = "בית המשפט העליון"
COLUMNS = [
    "case_id", "CaseId", "CaseNum", "meta_case_nbr", "CaseName", "Type", "TypeCode", "Technical", "meta_is_technical",
    "meta_court_nm", "meta_judge", "VerdictDt", "VerdictsDt", "meta_verdict_dt", "Year", "Pages", "meta_verdict_pages",
    "meta_inyan_nm", "meta_mador_nm", "meta_side_nm", "meta_side_ty", "document_hash", "DocName", "text",
]
RULES = ("publication_restriction", "family_case_type", "anonymized_parties", "topic_keywords")
_QUOTES = str.maketrans({"״": '"', "׳": "'", "“": '"', "”": '"', "‘": "'", "’": "'"})


def fold(text) -> str:
    """Comparable Hebrew: no points, ASCII quote marks, one hyphen kind, single spaces."""
    text = strip_points(str(text or "")).translate(_QUOTES)
    return re.sub(r"[ \t]+", " ", re.sub(r"[‐‑‒–—―־]", "-", text))


def _normalized_type(value) -> str:
    return re.sub(r"\s+", " ", fold(value).replace("-", " ")).strip()


def _as_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower() if value is not None else ""
    if text in ("true", "1", "כן", "yes", "y"):
        return True
    if text in ("false", "0", "לא", "no", "n"):
        return False
    return None


def classify(row: dict, judgment_types: list[str]) -> tuple[str | None, str]:
    """(authority_level or None if not kept, bucket for the report)."""
    if _normalized_type(row.get("Type")) in {_normalized_type(t) for t in judgment_types}:
        return "judgment", "judgment"
    technical = _as_bool(row.get("Technical"))
    if technical is None:
        technical = _as_bool(row.get("meta_is_technical"))
    if technical is False:
        return "decision", "decision_non_technical"
    return None, "decision_technical" if technical else "technical_flag_missing"


def _names(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value if v]


class PrivacyFilter:
    def __init__(self, cfg: LegalDataPrivacyConfig) -> None:
        if not cfg.publication_restriction_patterns or not cfg.topic_keywords:
            raise ValueError("legal_data.privacy lists are empty -- refusing to process court documents without them")
        self.publication = [re.compile(p) for p in cfg.publication_restriction_patterns]
        self.family_prefixes = [fold(p) for p in cfg.family_case_prefixes]
        self.family_subjects = {fold(v).strip() for v in cfg.family_subject_values}
        self.anonymized = [re.compile(p) for p in cfg.anonymized_party_patterns]
        keywords = "|".join(f"(?:{k})" for k in cfg.topic_keywords)
        self.topics = re.compile(rf"(?<![א-ת])[ובהלמשכ]{{0,3}}(?:{keywords})(?![א-ת])")
        self.scan_chars = cfg.topic_scan_chars

    def hits(self, row: dict) -> list[str]:
        text = fold(row.get("text"))
        case_name, case_number, case_type = fold(row.get("CaseName")), fold(row.get("meta_case_nbr")), fold(row.get("meta_inyan_nm"))
        found = []
        if any(p.search(text) for p in self.publication):
            found.append("publication_restriction")
        if (any(v.strip().startswith(prefix) for v in (case_number, case_type) for prefix in self.family_prefixes)
                or fold(row.get("meta_mador_nm")).strip() in self.family_subjects):
            found.append("family_case_type")
        if any(p.search(name) for name in [case_name, *map(fold, _names(row.get("meta_side_nm")))] for p in self.anonymized):
            found.append("anonymized_parties")
        if self.topics.search(" ".join([case_name, case_type, text[: self.scan_chars]])):
            found.append("topic_keywords")
        return found


class Anonymizer:
    def __init__(self, public_body_patterns: list[str]) -> None:
        self.public = re.compile("|".join(f"(?:{p})" for p in public_body_patterns)) if public_body_patterns else None

    def private_parties(self, parties: list[str]) -> list[str]:
        private: list[str] = []
        for name in parties:
            name = re.sub(r"\s+", " ", name).strip()
            if len(name) < 3 or (self.public and self.public.search(fold(name))):
                continue
            if name not in private:
                private.append(name)
        return private

    def apply(self, text: str, case_name: str, parties: list[str]) -> tuple[str, str, int, int]:
        """(text, case name, replacements made, private parties found)."""
        private = self.private_parties(parties)
        variants: list[tuple[str, str]] = []
        for i, name in enumerate(private, 1):
            placeholder = f"[צד {i}]"
            tokens = name.split()
            forms = {name, " ".join(reversed(tokens))} if len(tokens) == 2 else {name}
            variants += [(form, placeholder) for form in forms]
        replaced = 0
        for form, placeholder in sorted(variants, key=lambda v: -len(v[0])):
            pattern = re.compile(rf"(?<![א-ת]){re.escape(form)}(?![א-ת])")
            text, n = pattern.subn(placeholder, text)
            case_name, m = pattern.subn(placeholder, case_name)
            replaced += n + m
        return text, case_name, replaced, len(private)


def _iso_date(value) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (datetime, date)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    text = str(value)
    if text.isdigit() and len(text) >= 12:  # epoch milliseconds (datasets-server JSON)
        return datetime.fromtimestamp(int(text) / 1000, tz=timezone.utc).date().isoformat()
    return text[:10] if re.match(r"\d{4}-\d{2}-\d{2}", text) else None


def _int(value) -> int | None:
    try:
        return int(float(value)) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def to_record(row: dict, level: str, repo: str, snapshot: str, anonymizer: Anonymizer | None) -> tuple[CorpusRecord, int]:
    """(record, anonymization replacements)."""
    text, encoding, notes = repair_text(str(row.get("text") or ""))
    case_name = str(row.get("CaseName") or "").strip()
    replaced = 0
    if anonymizer is not None:
        text, case_name, replaced, private = anonymizer.apply(text, case_name, _names(row.get("meta_side_nm")))
        notes.append(f"anonymized: {private} private part(y/ies), {replaced} replacement(s)")
    case_number = str(row.get("meta_case_nbr") or row.get("CaseNum") or "").strip()
    document_hash = row.get("document_hash") or hashlib.sha256(text.encode("utf-8")).hexdigest()
    record = CorpusRecord(
        id=f"supreme_court:{document_hash}",
        source=f"Hugging Face {repo} ({snapshot})",
        source_url=f"https://huggingface.co/datasets/{repo}",
        license=LICENSE,
        attribution=f"{repo} (Lev Muchnik et al.), Hugging Face",
        category="supreme_court",
        title=" ".join(p for p in (case_number, case_name) if p) or document_hash[:16],
        authority_level=level,
        status="unknown",
        effective_date=_iso_date(row.get("VerdictDt") or row.get("VerdictsDt") or row.get("meta_verdict_dt")),
        retrieved_at=now_iso(),
        text=text,
        quality=Quality(encoding=encoding, extraction="n/a", notes=notes),
        case_number=case_number or None,
        court=str(row.get("meta_court_nm") or COURT),
        judges=_names(row.get("meta_judge")),
        decision_date=_iso_date(row.get("VerdictDt") or row.get("VerdictsDt") or row.get("meta_verdict_dt")),
        doc_type=str(row.get("Type") or "") or None,
        technical=_as_bool(row.get("Technical")),
        case_name=case_name or None,
        year=_int(row.get("Year")),
        pages=_int(row.get("Pages")) or _int(row.get("meta_verdict_pages")),
        source_case_id=str(row.get("case_id") or row.get("CaseId") or row.get("DocName") or "") or None,
        anonymized=anonymizer is not None,
    )
    return record.finalize(), replaced


# --- reading the dataset ---------------------------------------------------------------------------


def iter_parquet(path: Path, batch_size: int = 256) -> Iterator[dict]:
    import pyarrow.parquet as pq

    file = pq.ParquetFile(path)
    columns = [c for c in COLUMNS if c in set(file.schema_arrow.names)]
    for batch in file.iter_batches(batch_size=batch_size, columns=columns):
        yield from batch.to_pylist()


def hub_file_info(http: PoliteClient, repo: str, revision: str, filename: str) -> dict:
    """{size, sha256, commit} of a dataset file, from the Hub API."""
    data = http.get_json(f"https://huggingface.co/api/datasets/{repo}/revision/{quote(revision, safe='')}?blobs=true")
    sibling = next((s for s in data.get("siblings", []) if s.get("rfilename") == filename), None)
    if sibling is None:
        raise FileNotFoundError(f"{filename} is not in {repo}@{revision}")
    lfs = sibling.get("lfs") or {}
    return {"size": lfs.get("size") or sibling.get("size"), "sha256": lfs.get("sha256"), "commit": data.get("sha"),
            "last_modified": data.get("lastModified")}


def resolve_url(repo: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{quote(revision, safe='')}/{quote(filename)}"


def sample_rows(http: PoliteClient, repo: str, n: int, spread: int = 5) -> list[dict]:
    """n rows from `spread` places across the dataset, through the Hub's datasets-server API --
    a sample without the 1.5 GB download."""
    base = "https://datasets-server.huggingface.co/rows"
    per = max(1, min(100, math.ceil(n / spread)))

    def page(offset: int) -> dict:
        return http.get_json(f"{base}?dataset={quote(repo, safe='')}&config=default&split=train"
                             f"&offset={offset}&length={per}")

    first = page(0)
    total = int(first.get("num_rows_total") or 0)
    rows = [r["row"] for r in first.get("rows", [])]
    for k in range(1, spread):
        if len(rows) >= n or not total:
            break
        rows += [r["row"] for r in page(k * (total // spread)).get("rows", [])]
    return rows[:n]
