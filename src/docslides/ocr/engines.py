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
        import numpy as np
        from PIL import Image

        engine = self._get_instance()
        image = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
        # paddleocr 3.x: .predict() returns a list of per-page result objects;
        # recognised text lives in parsing_res_list (PaddleOCRVLBlock objects
        # with .label/.content, in reading order). The VLM emits no per-text
        # confidence, so the layout detector's per-block scores stand in for it.
        results = engine.predict(image)

        lines: list[str] = []
        confidences: list[float] = []
        for page_result in results or []:
            for block in page_result.get("parsing_res_list", []) or []:
                content = (getattr(block, "content", "") or "").strip()
                if content:
                    lines.append(content)
            layout = page_result.get("layout_det_res") or {}
            confidences.extend(float(b["score"]) for b in layout.get("boxes", []) if "score" in b)

        mean_conf = sum(confidences) / len(confidences) if confidences and lines else 0.0
        return OCRResult(text="\n".join(lines), mean_confidence=mean_conf, engine=self.name)


class SuryaEngine:
    """GPU-resident: primary for Hebrew, fallback escalation for Arabic."""

    name = "surya"
    gpu_resident = True
    _det_predictor = None
    _rec_predictor = None

    def _get_predictors(self):
        # surya-ocr >= 0.14 API (pyproject pins 0.16.7-0.19): one shared FoundationPredictor
        # behind RecognitionPredictor, DetectionPredictor passed per call. The old
        # surya.model.detection.segformer / batch_recognition functions are long gone.
        if self._rec_predictor is None:
            from surya.detection import DetectionPredictor  # deferred import: heavy, optional dep
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor

            SuryaEngine._det_predictor = DetectionPredictor()
            SuryaEngine._rec_predictor = RecognitionPredictor(FoundationPredictor())
        return self._det_predictor, self._rec_predictor

    def recognize(self, png_bytes: bytes, lang: str) -> OCRResult:
        from PIL import Image

        det_predictor, rec_predictor = self._get_predictors()
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")

        # The recognition model is multilingual: it no longer takes a language list, so `lang`
        # only picks this engine upstream (router), it isn't passed to Surya.
        rec_result = rec_predictor([image], det_predictor=det_predictor, sort_lines=True)[0]

        lines = [line.text for line in rec_result.text_lines]
        confidences = [line.confidence or 0.0 for line in rec_result.text_lines]
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return OCRResult(text="\n".join(lines), mean_confidence=mean_conf, engine=self.name)


ENGINE_REGISTRY: dict[str, OCREngine] = {
    "paddleocr": PaddleOCREngine(),
    "tesseract": TesseractEngine(),
    "paddleocr_vl": PaddleOCRVLEngine(),
    "surya": SuryaEngine(),
}
