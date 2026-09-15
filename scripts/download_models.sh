#!/usr/bin/env bash
# One-time, ONLINE model download step. Run this once with internet access
# (e.g. on the build host or inside the app container before disconnecting
# it from the network) -- nothing in the runtime path calls out afterwards.
#
# Usage: ./scripts/download_models.sh [models_dir]

set -euo pipefail

MODELS_DIR="${1:-./models}"
mkdir -p "$MODELS_DIR"

echo "== Qwen3-32B (quantized) weights =="
echo "Verify the exact repo name against the current Qwen3-32B AWQ/GPTQ"
echo "checkpoint listing (see scripts/verify_vllm_launch.py), then:"
echo "  huggingface-cli download <QWEN_MODEL_REPO> --local-dir ./hf_cache/qwen3-32b"
echo

echo "== fastText language-id model (lid.176) =="
if [ ! -f "$MODELS_DIR/lid.176.bin" ]; then
  curl -L -o "$MODELS_DIR/lid.176.bin" \
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
else
  echo "already present: $MODELS_DIR/lid.176.bin"
fi
echo

echo "== spaCy small pipelines (en, fr, es, it, de) =="
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
python -m spacy download es_core_news_sm
python -m spacy download it_core_news_sm
python -m spacy download de_core_news_sm
echo

echo "== Stanza pipelines (ar, he) =="
python -c "import stanza; stanza.download('ar'); stanza.download('he')"
echo

echo "== PaddleOCR PP-OCRv6 / PaddleOCR-VL model weights =="
echo "PaddleOCR downloads its detection/recognition weights to ~/.paddleocr on"
echo "first use per language. Trigger that download once, online, per language:"
cat <<'PY'
python - <<'EOF'
from paddleocr import PaddleOCR, PaddleOCRVL
for lang in ["en", "fr", "es", "it", "de", "german"]:
    PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=False, show_log=False)
PaddleOCRVL()
EOF
PY
echo

echo "== Surya OCR (Hebrew primary, Arabic GPU fallback) =="
echo "Surya downloads its detection/recognition weights from Hugging Face on"
echo "first use. Trigger once, online:"
cat <<'PY'
python - <<'EOF'
from surya.model.detection import segformer
from surya.model.recognition.model import load_model
from surya.model.recognition.processor import load_processor
segformer.load_model()
load_model()
load_processor()
EOF
PY
echo

echo "== Tesseract language packs =="
echo "Installed via apt in docker/Dockerfile.app (tesseract-ocr-{eng,fra,spa,ita,deu,ara,heb})."
echo

echo "== MinerU (magic-pdf) layout/OCR-classification models =="
echo "Follow MinerU's own model-download instructions for the installed"
echo "magic-pdf version (its download script/CLI varies by release):"
echo "  https://github.com/opendatalab/MinerU"
echo

echo "Done. All of the above must complete online, ahead of time; nothing in"
echo "the runtime path (src/docslides/**) makes network calls at inference time."
