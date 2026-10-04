"""OCR router: a crashing engine falls through to the next one in the language's chain."""

from __future__ import annotations

import asyncio

import pytest

from docslides.ocr import router
from docslides.ocr.engines import OCRResult


class _Broken:
    name = "surya"
    gpu_resident = False

    def recognize(self, png_bytes: bytes, lang: str) -> OCRResult:
        raise ImportError("cannot import name 'segformer' from 'surya.model.detection'")


class _Ok:
    name = "tesseract"
    gpu_resident = False

    def recognize(self, png_bytes: bytes, lang: str) -> OCRResult:
        return OCRResult(text="שלום", mean_confidence=0.9, engine=self.name)


def test_broken_primary_falls_back_to_cpu_engine(monkeypatch):
    monkeypatch.setitem(router.ENGINE_REGISTRY, "surya", _Broken())
    monkeypatch.setitem(router.ENGINE_REGISTRY, "tesseract", _Ok())
    result = asyncio.run(router.recognize_page(b"", "he", page_index=0))
    assert result.engine == "tesseract"
    assert result.text == "שלום"


def test_all_engines_broken_raises_with_every_reason(monkeypatch):
    for name in ("surya", "tesseract", "paddleocr_vl"):
        monkeypatch.setitem(router.ENGINE_REGISTRY, name, _Broken())
    with pytest.raises(RuntimeError, match="all OCR engines failed"):
        asyncio.run(router.recognize_page(b"", "he", page_index=0))
