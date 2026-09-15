#!/usr/bin/env bash
# One-time, ONLINE model download step. Run this once with internet access
# (e.g. on the build host or inside the app container before disconnecting
# it from the network) -- nothing in the runtime path calls out afterwards.
#
# IMPORTANT: this script downloads model *weights* for packages that must
# already be installed. It does not install the Python packages themselves.
# Run this first, in the same environment you'll run the app in:
#   pip install -e ".[ocr,lang,dev]"
#
# Usage: ./scripts/download_models.sh [models_dir]
# (On native Windows, run this via Git Bash: `bash scripts/download_models.sh`
#  -- a bare PowerShell/cmd session won't execute a .sh file directly.)
#
# If you're deploying with docker-compose.yml, run this INSIDE the app
# container instead of on the host, so the Hugging Face / PaddleOCR / Stanza
# caches land in the same paths (/root/.cache/huggingface, /root/.paddleocr,
# /root/stanza_resources) that docker-compose.yml bind-mounts to
# ./hf_cache, ./paddleocr_cache, ./stanza_cache on the host:
#   docker compose run --rm --no-deps app bash scripts/download_models.sh ./models
# Running it directly on the host (no Docker) is fine too -- it just uses
# your normal user cache directories, which is what a non-Docker run of the
# app expects.

set -uo pipefail  # no -e: one missing optional package must not silently kill every later step

MODELS_DIR="${1:-./models}"
mkdir -p "$MODELS_DIR"

SKIPPED=()

require_module() {
  # Usage: require_module <import-name>
  python -c "import $1" >/dev/null 2>&1
}

QWEN_MODEL_REPO="${QWEN_MODEL_REPO:-Qwen/Qwen3-32B-AWQ}"

echo "== Qwen3-32B (quantized) weights: $QWEN_MODEL_REPO (~20GB) =="
echo "Verify this is still the repo you want (see scripts/verify_vllm_launch.py"
echo "to check it against the current Hugging Face listing) before this runs."
# `huggingface-cli` was renamed to `hf` in newer huggingface_hub releases;
# prefer `hf` and fall back to the old name for older installs.
if command -v hf >/dev/null 2>&1; then
  HF_DOWNLOAD_CMD=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_DOWNLOAD_CMD=(huggingface-cli download)
else
  HF_DOWNLOAD_CMD=()
fi

if [ "${#HF_DOWNLOAD_CMD[@]}" -gt 0 ]; then
  # Deliberately NOT --local-dir: this downloads into the standard HF hub
  # cache layout (respecting $HF_HOME if set), which is what lets vLLM
  # resolve the model BY REPO ID while offline (HF_HUB_OFFLINE=1) -- both
  # locally and via the ./hf_cache bind mount in docker-compose.yml.
  "${HF_DOWNLOAD_CMD[@]}" "$QWEN_MODEL_REPO" || SKIPPED+=("Qwen weights: $QWEN_MODEL_REPO")
else
  echo "[skip] Neither 'hf' nor 'huggingface-cli' found. Install with:"
  echo "       pip install -U \"huggingface_hub[cli]\""
  echo "       then re-run, or run directly:"
  echo "       hf download $QWEN_MODEL_REPO"
  SKIPPED+=("Qwen weights: $QWEN_MODEL_REPO")
fi
echo

echo "== fastText language-id model (lid.176, ~125MB) =="
if [ -f "$MODELS_DIR/lid.176.bin" ]; then
  echo "already present: $MODELS_DIR/lid.176.bin"
else
  echo "Downloading -- this is ~125MB and may take a few minutes depending on"
  echo "your connection. curl's progress meter updates in place (carriage"
  echo "returns), so it can look frozen in some terminals/log viewers even"
  echo "while it's actively transferring; check the file size growing in"
  echo "another terminal if you're unsure (ls -la \"$MODELS_DIR\")."
  if curl -fL -o "$MODELS_DIR/lid.176.bin.part" \
      "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"; then
    mv "$MODELS_DIR/lid.176.bin.part" "$MODELS_DIR/lid.176.bin"
    echo "done: $MODELS_DIR/lid.176.bin"
  else
    echo "[skip] fastText download failed -- check connectivity and retry."
    rm -f "$MODELS_DIR/lid.176.bin.part"
    SKIPPED+=("fastText lid.176")
  fi
fi
echo

echo "== spaCy small pipelines (en, fr, es, it, de) =="
if require_module spacy; then
  for model in en_core_web_sm fr_core_news_sm es_core_news_sm it_core_news_sm de_core_news_sm; do
    python -m spacy download "$model" || SKIPPED+=("spacy:$model")
  done
else
  echo "[skip] spacy is not installed in this Python environment."
  echo "       Run: pip install -e \".[lang]\"  (then re-run this script)"
  SKIPPED+=("spacy (all models)")
fi
echo

echo "== Stanza pipelines (ar, he) =="
if require_module stanza; then
  python -c "import stanza; stanza.download('ar'); stanza.download('he')" || SKIPPED+=("stanza:ar/he")
else
  echo "[skip] stanza is not installed in this Python environment."
  echo "       Run: pip install -e \".[lang]\"  (then re-run this script)"
  SKIPPED+=("stanza (ar/he)")
fi
echo

echo "== PaddleOCR PP-OCRv6 / PaddleOCR-VL model weights =="
if require_module paddleocr; then
  echo "Triggering first-use download for each language (writes to ~/.paddleocr)..."
  python - <<'EOF' || true
from paddleocr import PaddleOCR, PaddleOCRVL
for lang in ["en", "fr", "es", "it", "de", "german"]:
    PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=False, show_log=False)
PaddleOCRVL()
EOF
else
  echo "[skip] paddleocr is not installed in this Python environment."
  echo "       Run: pip install -e \".[ocr]\"  (then re-run this script)"
  SKIPPED+=("paddleocr/paddleocr-vl weights")
fi
echo

echo "== Surya OCR (Hebrew primary, Arabic GPU fallback) =="
if require_module surya; then
  echo "Triggering first-use download from Hugging Face..."
  python - <<'EOF' || true
from surya.model.detection import segformer
from surya.model.recognition.model import load_model
from surya.model.recognition.processor import load_processor
segformer.load_model()
load_model()
load_processor()
EOF
else
  echo "[skip] surya (surya-ocr) is not installed in this Python environment."
  echo "       Run: pip install -e \".[ocr]\"  (then re-run this script)"
  SKIPPED+=("surya weights")
fi
echo

echo "== Tesseract language packs =="
echo "Installed via apt in docker/Dockerfile.app (tesseract-ocr-{eng,fra,spa,ita,deu,ara,heb})."
echo "Outside Docker, install tesseract-ocr plus those language packs via your"
echo "OS package manager (or the UB-Mannheim installer on Windows, selecting"
echo "the same language packs during setup)."
echo

echo "== MinerU (magic-pdf) layout/OCR-classification models =="
echo "Follow MinerU's own model-download instructions for the installed"
echo "magic-pdf version (its download script/CLI varies by release):"
echo "  https://github.com/opendatalab/MinerU"
echo

if [ "${#SKIPPED[@]}" -eq 0 ]; then
  echo "Done. Everything downloaded successfully."
else
  echo "Finished with ${#SKIPPED[@]} step(s) skipped:"
  for item in "${SKIPPED[@]}"; do
    echo "  - $item"
  done
  echo "Install the missing extras (see messages above) and re-run this script"
  echo "-- it's safe to re-run; already-downloaded files are left in place."
fi
echo "Nothing in the runtime path (src/docslides/**) makes network calls at inference time."
