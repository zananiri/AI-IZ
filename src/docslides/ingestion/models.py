"""Shared data structures produced by ingestion, consumed by OCR/cleaning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PageKind(Enum):
    NATIVE_TEXT = "native_text"
    SCANNED = "scanned"
    MIXED = "mixed"  # some native text blocks, some image blocks needing OCR


@dataclass
class PageImage:
    """A rendered page image, only populated for pages that need OCR."""

    png_bytes: bytes
    dpi: int


@dataclass
class Page:
    index: int
    kind: PageKind
    native_text: str = ""          # text extracted directly from the text layer, if any
    image: PageImage | None = None  # rendered bitmap, populated only if kind != NATIVE_TEXT
    language: str | None = None     # filled in by language_detect
    source_engine: str = "unknown"  # which parser produced this page (mineru | pymupdf)


@dataclass
class ParsedDocument:
    source_path: str
    pages: list[Page] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)
