"""OCR engine wrappers behind one common interface.

Latin-script languages (en/fr/es/it/de): PaddleOCR PP-OCRv6 primary,
Tesseract fallback, both CPU.

Arabic: PaddleOCR-VL (falls back to Surya) -- handles contextual letter
shaping and RTL reading order natively; GPU-resident, loaded opportunistically.

Hebrew: Surya primary, Tesseract (heb.traineddata) fallback. EasyOCR/classic
PaddleOCR are deliberately not used for Hebrew -- neither ships a Hebrew model.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Protocol

from docslides.config import get_config
from docslides.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class OCRResult:
    text: str
    mean_confidence: float
    engine: str


class OCREngine(Protocol):
    name: str
    gpu_resident: bool

    def recognize(self, png_bytes: bytes, lang: str) -> OCRResult: ...


class PaddleOCREngine:
    """CPU PP-OCRv6 for Latin-script languages."""

    name = "paddleocr"
    gpu_resident = False
    _instances: dict[str, object] = {}

    def _get_instance(self, lang: str):
        # PaddleOCR keeps separate model instances per language.
        paddle_lang = {"en": "en", "fr": "fr", "es": "es", "it": "it", "de": "german"}.get(lang, "en")
        if paddle_lang not in self._instances:
            from paddleocr import PaddleOCR  # deferred import: heavy, optional dep

            # paddleocr 3.x removed use_angle_cls/use_gpu/show_log (now
            # use_textline_orientation/device, or just rely on defaults) --
            # unrecognized kwargs raise ValueError rather than being ignored.
            #
            # enable_mkldnn=False works around a paddlepaddle>=3.3.0 regression
            # in its oneDNN PIR (Program IR) executor on CPU: real inference
            # crashes with "NotImplementedError: (Unimplemented)
            # ConvertPirAttribute2RuntimeAttribute not support
            # [pir::ArrayAttribute<pir::DoubleAttribute>]" the moment a model
            # with double-array attributes (e.g. detection box thresholds)
            # actually runs -- construction succeeds either way, so this only
            # surfaces once you OCR something. Confirmed fixed upstream by
            # pinning paddlepaddle==3.2.0, but disabling mkldnn avoids needing
            # a downgrade. See https://github.com/PaddlePaddle/Paddle/issues/77340
            self._instances[paddle_lang] = PaddleOCR(lang=paddle_lang, enable_mkldnn=False)
        return self._instances[paddle_lang]

    def recognize(self, png_bytes: bytes, lang: str) -> OCRResult:
        import numpy as np
        from PIL import Image

        engine = self._get_instance(lang)
        image = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
        # paddleocr 3.x: .ocr(cls=...) is a deprecated .predict() alias that
        # no longer accepts `cls`; .predict() also returns a differently
        # shaped result (per-image dict-like objects with rec_texts/
        # rec_scores lists) instead of the old [[box, (text, conf)], ...]
        # nested-tuple format.
        results = engine.predict(image)

        lines: list[str] = []
        confidences: list[float] = []
        for page_result in results or []:
            texts = page_result.get("rec_texts", []) if hasattr(page_result, "get") else []
            scores = page_result.get("rec_scores", []) if hasattr(page_result, "get") else []
            lines.extend(texts)
            confidences.extend(scores)

        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return OCRResult(text="\n".join(lines), mean_confidence=mean_conf, engine=self.name)


class TesseractEngine:
    """Lightweight CPU fallback for any configured language."""

    name = "tesseract"
    gpu_resident = False

    def recognize(self, png_bytes: bytes, lang: str) -> OCRResult:
        import pytesseract
        from PIL import Image

        lang_codes = get_config().ocr.tesseract.lang_codes
        tess_lang = lang_codes.get(lang, "eng")
        image = Image.open(io.BytesIO(png_bytes))

        data = pytesseract.image_to_data(
            image, lang=tess_lang, output_type=pytesseract.Output.DICT
        )
        words = [w for w in data["text"] if w.strip()]
        confs = [float(c) for w, c in zip(data["text"], data["conf"]) if w.strip() and c != "-1"]

        mean_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
        return OCRResult(text=" ".join(words), mean_confidence=mean_conf, engine=self.name)


class PaddleOCRVLEngine:
    """GPU-resident VLM-OCR: primary for Arabic (contextual shaping + RTL
    reading order), and the GPU fallback escalation path for Latin scripts."""

    name = "paddleocr_vl"
    gpu_resident = True
    _instance = None

    def _get_instance(self):
        if self._instance is None:
            from paddleocr import PaddleOCRVL  # deferred import: heavy, optional dep, GPU-resident

            # enable_mkldnn=False: same paddlepaddle>=3.3.0 CPU oneDNN
            # regression as PaddleOCREngine above -- applies here too since
            # this falls back to CPU on any machine without a GPU.
            self._instance = PaddleOCRVL(enable_mkldnn=False)
        return self._instance

    def recognize(self, png_bytes: bytes, lang: str) -> OCRResult:
        from PIL import Image

        engine = self._get_instance()
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        result = engine.predict(image)

        lines = [r["text"] for r in result.get("blocks", [])]
        confidences = [r.get("confidence", 0.0) for r in result.get("blocks", [])]
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return OCRResult(text="\n".join(lines), mean_confidence=mean_conf, engine=self.name)


class SuryaEngine:
    """GPU-resident: primary for Hebrew, fallback escalation for Arabic."""

    name = "surya"
    gpu_resident = True
    _det_model = None
    _rec_model = None

    def _get_models(self):
        if self._rec_model is None:
            from surya.model.detection import segformer  # deferred import
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor

            self._det_model = segformer.load_model()
            self._rec_model = (load_rec_model(), load_rec_processor())
        return self._det_model, self._rec_model

    def recognize(self, png_bytes: bytes, lang: str) -> OCRResult:
        from PIL import Image
        from surya.detection import batch_text_detection
        from surya.recognition import batch_recognition

        det_model, (rec_model, rec_processor) = self._get_models()
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")

        det_result = batch_text_detection([image], det_model)[0]
        surya_lang = {"he": "he", "ar": "ar"}.get(lang, lang)
        rec_result = batch_recognition(
            [image], [[surya_lang]] * 1, rec_model, rec_processor, [det_result.bboxes]
        )[0]

        lines = [line.text for line in rec_result.text_lines]
        confidences = [line.confidence for line in rec_result.text_lines]
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return OCRResult(text="\n".join(lines), mean_confidence=mean_conf, engine=self.name)


ENGINE_REGISTRY: dict[str, OCREngine] = {
    "paddleocr": PaddleOCREngine(),
    "tesseract": TesseractEngine(),
    "paddleocr_vl": PaddleOCRVLEngine(),
    "surya": SuryaEngine(),
}
