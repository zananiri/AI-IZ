"""Running per-document glossary of key terms and their translations.

Extracted incrementally per chunk during translation and injected into the
prompt for every subsequent chunk of the same document, so terminology stays
consistent across the whole translation rather than being decided fresh (and
potentially inconsistently) chunk by chunk.
"""

from __future__ import annotations

import json
from pathlib import Path

from docslides.config import get_config
from docslides.llm.schemas import GlossaryTerm
from docslides.logging_setup import get_logger

logger = get_logger(__name__)


class Glossary:
    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        self._terms: dict[str, GlossaryTerm] = {}
        self._max_terms = get_config().translation.glossary_max_terms

    @property
    def terms(self) -> list[GlossaryTerm]:
        return list(self._terms.values())

    def add_terms(self, new_terms: list[GlossaryTerm]) -> None:
        for term in new_terms:
            key = term.source_term.strip().lower()
            if not key:
                continue
            if key not in self._terms and len(self._terms) >= self._max_terms:
                logger.debug("glossary_capacity_reached_dropping_term", term=term.source_term)
                continue
            self._terms[key] = term

    def _path(self) -> Path:
        return Path(get_config().paths.glossary_dir) / f"{self.document_id}.json"

    def save(self) -> None:
        path = self._path()
        path.write_text(
            json.dumps([t.model_dump() for t in self.terms], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, document_id: str) -> "Glossary":
        glossary = cls(document_id)
        path = glossary._path()
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            glossary.add_terms([GlossaryTerm.model_validate(t) for t in raw])
        return glossary
