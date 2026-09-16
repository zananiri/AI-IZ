#!/usr/bin/env python3
"""Idempotent post-install patches for magic-pdf (MinerU) on the current
machine. Both are real upstream bugs discovered while getting MinerU
working here, not something wrong with this project's own code -- re-run
safely any time after (re)installing the `ingestion-mineru` extra.

1. magic-pdf 1.3.12's bundled OCR language config
   (paddleocr2pytorch/pytorchocr/utils/resources/models_config.yml) still
   references PP-OCRv3 detection weights (e.g. ch_PP-OCRv3_det_infer.pth)
   for ch_lite/ch_lite_v4/ch_server/ch_server_v4/ch, but the
   opendatalab/PDF-Extract-Kit-1.0 repo on Hugging Face has moved on to
   PP-OCRv5 weights and no longer hosts those v3 files -- so
   CustomPEKModel's OCR sub-model (constructed unconditionally, even when
   ocr=False/apply_ocr=False) fails to load with a bare FileNotFoundError.
   Fix: repoint those five entries' "det" field at the v5 detection weight
   that IS present (their "rec"/"dict" pairs already matched v5/v4, only
   "det" was stale). Deliberately NOT touched: latin/arabic/korean/etc --
   their v5 rec weights need v5-vocabulary dict files this magic-pdf
   release doesn't bundle, so remapping them risks silently wrong OCR text
   instead of a clean failure. This project's own OCR routing
   (src/docslides/ocr/router.py) handles those languages instead; MinerU is
   only ever used here for layout-aware native-text extraction (ocr=False),
   never to actually OCR non-Chinese text.

2. fasttext-wheel 0.9.2 (pulled in by magic-pdf's fast-langdetect
   dependency, used internally for per-block language detection during
   paragraph splitting) calls `np.array(x, copy=False)` in a few places.
   NumPy 2.0 made `copy=False` strict (raises instead of silently copying
   when a copy is unavoidable), and no newer fasttext-wheel release exists
   on PyPI to fix it. Fix: those call sites to `np.asarray(x)`, which is
   what the NumPy 2.0 migration guide itself recommends as the equivalent.

3. magic-pdf's vendored UnimerNet formula-recognition model
   (mfr/unimernet/unimernet_hf/unimer_mbart/modeling_unimer_mbart.py,
   `UnimerMBartForCausalLM.forward`) predates transformers' `cache_position`
   generation kwarg. Recent transformers (confirmed on 4.57.x) always injects
   `cache_position` into the kwargs passed to the model during `.generate()`,
   but this fixed-signature `forward()` has no `cache_position` (or `**kwargs`)
   param to absorb it, so every formula-recognition call dies with
   `TypeError: UnimerMBartForCausalLM.forward() got an unexpected keyword
   argument 'cache_position'` and MinerU silently falls back to plain
   pymupdf text extraction (no OCR, no formula recognition, worse quality)
   for every document. This class doesn't use HF's `Cache`/`cache_position`
   machinery at all (it manages `past_key_values` itself as a plain tuple),
   so `cache_position` is safe to accept and ignore. Fix: add
   `cache_position: Optional[torch.LongTensor] = None,` to the `forward()`
   signature.

4. Same vendored UnimerNet model, one layer up
   (mfr/unimernet/unimernet_hf/modeling_unimernet.py, `UnimernetModel`, a
   `VisionEncoderDecoderModel` subclass): once fix #3 lets `.generate()`
   actually reach the decoder, recent transformers' `generate()` also
   pre-instantiates an empty `EncoderDecoderCache`/`DynamicCache` object (not
   `None`) as `past_key_values` before the first decode step, wherever
   `_supports_default_dynamic_cache()` returns `True` (the base-class
   default -- this model never overrides it). The vendored decoder
   (`unimer_mbart/modeling_unimer_mbart.py`) predates that Cache machinery
   and indexes `past_key_values` as a legacy tuple-of-tuples directly (e.g.
   `past_key_values[0][0].shape[2]`); an empty Cache's per-layer slots are
   `None` rather than absent, so that first step dies with `AttributeError:
   'NoneType' object has no attribute 'shape'`, and MinerU again falls back
   to pymupdf. Fix: override `UnimernetModel._supports_default_dynamic_cache`
   to return `False`, which makes `generate()` skip pre-instantiating a
   Cache and leaves `past_key_values` unset/`None` until this model's own
   forward populates it as a real tuple on the first step -- restoring the
   pre-Cache-era behavior this vendored code was written against.

Usage:
    python scripts/patch_mineru.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CH_DET_ENTRIES = ["ch_lite", "ch_lite_v4", "ch_server", "ch_server_v4", "ch"]
STALE_DET_FILE = "ch_PP-OCRv3_det_infer.pth"
CURRENT_DET_FILE = "ch_PP-OCRv5_det_infer.pth"


def patch_models_config() -> str:
    try:
        import magic_pdf.model.sub_modules.ocr.paddleocr2pytorch.pytorch_paddle as pp
    except ImportError:
        return "[skip] magic_pdf not installed -- nothing to patch"

    config_path = Path(pp.__file__).parent / "pytorchocr" / "utils" / "resources" / "models_config.yml"
    if not config_path.exists():
        return f"[skip] models_config.yml not found at {config_path}"

    text = config_path.read_text(encoding="utf-8")
    if STALE_DET_FILE not in text:
        return f"[ok] {config_path.name} already patched (or upstream fixed it)"

    lines = text.splitlines(keepends=True)
    current_entry = None
    changed = 0
    for i, line in enumerate(lines):
        m = re.match(r"^  (\S+):\s*$", line)
        if m:
            current_entry = m.group(1)
            continue
        if current_entry in CH_DET_ENTRIES and STALE_DET_FILE in line:
            lines[i] = line.replace(STALE_DET_FILE, CURRENT_DET_FILE)
            changed += 1

    if changed == 0:
        return f"[skip] no matching det entries found under {CH_DET_ENTRIES} in {config_path.name}"

    config_path.write_text("".join(lines), encoding="utf-8")
    return f"[fixed] {config_path}: repointed {changed} det entr{'y' if changed == 1 else 'ies'} to {CURRENT_DET_FILE}"


def patch_fasttext() -> str:
    try:
        import fasttext
    except ImportError:
        return "[skip] fasttext(-wheel) not installed -- nothing to patch"

    ft_path = Path(fasttext.__file__).parent / "FastText.py"
    if not ft_path.exists():
        return f"[skip] FastText.py not found at {ft_path}"

    text = ft_path.read_text(encoding="utf-8")
    pattern = re.compile(r"np\.array\((\w+),\s*copy=False\)")
    if not pattern.search(text):
        return f"[ok] {ft_path.name} already patched (or upstream fixed it)"

    new_text, count = pattern.subn(r"np.asarray(\1)", text)
    ft_path.write_text(new_text, encoding="utf-8")
    return f"[fixed] {ft_path}: replaced {count} numpy-2.x-incompatible copy=False call(s)"


UNIMER_MBART_ANCHOR = "        count_gt: Optional[torch.LongTensor] = None,\n"


def patch_unimer_mbart_cache_position() -> str:
    try:
        import magic_pdf.model.sub_modules.mfr.unimernet.unimernet_hf.unimer_mbart.modeling_unimer_mbart as um
    except ImportError:
        return "[skip] magic_pdf (unimer_mbart) not installed -- nothing to patch"

    model_path = Path(um.__file__)
    text = model_path.read_text(encoding="utf-8")

    if "cache_position" in text:
        return f"[ok] {model_path.name} already patched (or upstream fixed it)"

    if text.count(UNIMER_MBART_ANCHOR) != 1:
        return f"[skip] UnimerMBartForCausalLM.forward signature not found (as expected) in {model_path.name}"

    new_text = text.replace(
        UNIMER_MBART_ANCHOR,
        UNIMER_MBART_ANCHOR + "        cache_position: Optional[torch.LongTensor] = None,\n",
    )
    model_path.write_text(new_text, encoding="utf-8")
    return f"[fixed] {model_path}: added cache_position param to UnimerMBartForCausalLM.forward"


UNIMERNET_MODEL_ANCHOR = "        assert self.config.pad_token_id == tokenizer.pad_token_id\n"

UNIMERNET_MODEL_DYNAMIC_CACHE_OVERRIDE = '''
    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        # The vendored UnimerMBartForCausalLM decoder indexes past_key_values
        # as a legacy tuple-of-tuples (e.g. past_key_values[0][0].shape[2])
        # and predates transformers' Cache classes. Without this override,
        # generate() pre-instantiates an empty EncoderDecoderCache as
        # past_key_values before the first decode step; that cache's
        # per-layer slots are None (not absent), so the legacy indexing
        # crashes with "'NoneType' object has no attribute 'shape'".
        # Returning False here makes generate() leave past_key_values
        # unset/None until this model's own forward populates it as a tuple.
        return False
'''


def patch_unimernet_dynamic_cache() -> str:
    try:
        import magic_pdf.model.sub_modules.mfr.unimernet.unimernet_hf.modeling_unimernet as un
    except ImportError:
        return "[skip] magic_pdf (modeling_unimernet) not installed -- nothing to patch"

    model_path = Path(un.__file__)
    text = model_path.read_text(encoding="utf-8")

    if "_supports_default_dynamic_cache" in text:
        return f"[ok] {model_path.name} already patched (or upstream fixed it)"

    if text.count(UNIMERNET_MODEL_ANCHOR) != 1:
        return f"[skip] UnimernetModel._post_check anchor not found (as expected) in {model_path.name}"

    new_text = text.replace(
        UNIMERNET_MODEL_ANCHOR,
        UNIMERNET_MODEL_ANCHOR + UNIMERNET_MODEL_DYNAMIC_CACHE_OVERRIDE,
    )
    model_path.write_text(new_text, encoding="utf-8")
    return f"[fixed] {model_path}: disabled default dynamic cache on UnimernetModel"


def main() -> None:
    results = [
        patch_models_config(),
        patch_fasttext(),
        patch_unimer_mbart_cache_position(),
        patch_unimernet_dynamic_cache(),
    ]
    for r in results:
        print(r)
    if any(r.startswith("[fixed]") for r in results):
        print("\nDone -- re-run this script any time after reinstalling ingestion-mineru.")
    sys.exit(0)


if __name__ == "__main__":
    main()
