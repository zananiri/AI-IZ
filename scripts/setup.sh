#!/usr/bin/env bash
# One-shot setup for macOS/Linux: creates the venv, installs every Python
# dependency (core + OCR + language + MinerU ingestion extras), sets up the
# LLM backend appropriate for THIS machine's hardware (vLLM on an NVIDIA GPU,
# Ollama everywhere else -- CPU, AMD, Apple Silicon), and downloads every
# model weight the app needs (the chat model, fastText, spaCy, Stanza,
# PaddleOCR, Surya, MinerU).
#
# This is the ONLY online step. Nothing under src/docslides/** calls out to
# the internet at request time -- see README.md.
#
# Usage:
#   ./scripts/setup.sh [models_dir]
#
# Env overrides:
#   QWEN_MODEL_REPO      vLLM path model repo. default: Qwen/Qwen3-32B-AWQ
#   OLLAMA_MODEL         Ollama path model tag (general + Legal orchestrator).
#                        default: qwen3:32b on a >=48GB-RAM host, auto-
#                        downgraded to qwen3:8b below that (set this env var
#                        to override either way). qwen3:32b's ~20GB GGUF plus
#                        llama.cpp's CPU "repack" buffer (another ~14-20GB,
#                        briefly resident while loading) reliably hits
#                        std::bad_alloc under ~48GB RAM, which looks like the
#                        app "not responding" rather than a load failure.
#   FORCE_BACKEND        "vllm" or "ollama" -- skip GPU auto-detection
#   SKIP_MINERU=1     skip the magic-pdf (MinerU) extra -- it has a known
#                     dependency conflict with gradio's huggingface-hub pin
#                     (see the warning this script prints). PDF ingestion
#                     still works without it via the PyMuPDF fallback;
#                     DOCX/PPTX/XLSX/image ingestion will not.
#   SKIP_HEAVY_OCR=1  skip paddleocr/paddlepaddle/surya-ocr (large, slow to
#                     build on some platforms)

set -uo pipefail  # no -e: one missing optional package must not kill every later step

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$(pwd)"
MODELS_DIR="${1:-./models}"
mkdir -p "$MODELS_DIR"

SKIPPED=()
OS_NAME="$(uname -s)"

echo "== [1/7] Locating Python (3.10-3.12) =="
PYTHON_BIN=""
for cand in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then
    ver="$("$cand" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null)"
    case "$ver" in
      3.10|3.11|3.12) PYTHON_BIN="$cand"; break ;;
    esac
  fi
done
if [ -z "$PYTHON_BIN" ]; then
  echo "[fatal] No Python 3.10-3.12 found on PATH. Install one (e.g. 'brew install python@3.11' on macOS) and re-run."
  exit 1
fi
echo "using $("$PYTHON_BIN" --version) at $(command -v "$PYTHON_BIN")"
echo

echo "== [2/7] Creating virtual environment (.venv) =="
if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
python -m ensurepip --upgrade >/dev/null 2>&1 || true
echo "venv ready: $(python --version)"
echo

echo "== [3/7] Installing Python dependencies =="
echo "-- core + ocr + lang + dev extras --"
pip install -e ".[ocr,lang,dev]" || { echo "[fatal] core dependency install failed"; exit 1; }
pip install -U "huggingface_hub[cli]" -q || true

if [ "${SKIP_MINERU:-0}" != "1" ]; then
  echo "-- ingestion-mineru extra (magic-pdf) --"
  echo "NOTE: magic-pdf's own deps pin huggingface-hub<1.0, which conflicts with"
  echo "      gradio's huggingface-hub>=1.16 requirement. pip will install both"
  echo "      packages but print a dependency-conflict warning -- this is"
  echo "      expected (see src/docslides/ingestion/parser.py adapter notes)."
  echo "      Set SKIP_MINERU=1 to skip this extra entirely (PDF still works"
  echo "      via the PyMuPDF fallback; DOCX/PPTX/XLSX/image ingestion needs it)."
  pip install -e ".[ingestion-mineru]" || SKIPPED+=("magic-pdf (MinerU) package")
else
  echo "-- skipping ingestion-mineru extra (SKIP_MINERU=1) --"
  SKIPPED+=("magic-pdf (MinerU) package [skipped by request]")
fi
echo

echo "== [4/7] Detecting LLM backend for this machine =="
BACKEND="${FORCE_BACKEND:-}"
if [ -z "$BACKEND" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    BACKEND="vllm"
  else
    BACKEND="ollama"
  fi
fi
echo "selected backend: $BACKEND"
if [ "$BACKEND" = "ollama" ] && [ "$OS_NAME" = "Darwin" ]; then
  echo "[note] Apple Silicon/Intel Macs have no NVIDIA GPU -- using Ollama, which"
  echo "       runs natively on the host and uses Metal acceleration automatically"
  echo "       (Docker on macOS cannot pass Metal through to a container)."
fi
echo

if [ "$BACKEND" = "vllm" ]; then
  echo "== [5/7] vLLM path: pulling serving image + downloading the model =="
  if command -v docker >/dev/null 2>&1; then
    docker pull vllm/vllm-openai:latest || SKIPPED+=("docker pull vllm/vllm-openai:latest")
  else
    echo "[skip] docker not found on PATH -- install Docker Desktop, then run:"
    echo "       docker pull vllm/vllm-openai:latest"
    SKIPPED+=("docker pull vllm/vllm-openai:latest")
  fi
  python scripts/verify_vllm_launch.py || true
  # config/config.yaml already defaults to backend: vllm -- no override file needed.
  rm -f .env.local
else
  echo "== [5/7] Ollama path: installing Ollama + pulling the model =="
  if [ -z "${OLLAMA_MODEL:-}" ]; then
    TOTAL_RAM_GB=0
    if [ "$OS_NAME" = "Darwin" ]; then
      TOTAL_RAM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
    elif command -v free >/dev/null 2>&1; then
      TOTAL_RAM_GB=$(( $(free -b | awk '/^Mem:/{print $2}') / 1073741824 ))
    fi
    if [ "$TOTAL_RAM_GB" -gt 0 ] && [ "$TOTAL_RAM_GB" -lt 48 ]; then
      OLLAMA_MODEL="qwen3:8b"
      echo "[note] ${TOTAL_RAM_GB}GB RAM detected -- defaulting Ollama chat model to qwen3:8b"
      echo "       instead of qwen3:32b (which needs ~48GB+ RAM to load reliably under"
      echo "       Ollama's CPU backend). Set OLLAMA_MODEL=qwen3:32b to override."
    else
      OLLAMA_MODEL="qwen3:32b"
    fi
  fi
  if ! command -v ollama >/dev/null 2>&1; then
    if [ "$OS_NAME" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
      brew install ollama || SKIPPED+=("ollama (brew)")
    elif [ "$OS_NAME" = "Linux" ]; then
      echo "Installing Ollama via its official install script..."
      curl -fsSL https://ollama.com/install.sh | sh || SKIPPED+=("ollama (install.sh)")
    else
      echo "[skip] Could not auto-install Ollama. Get it from https://ollama.com/download"
      SKIPPED+=("ollama (manual install)")
    fi
  else
    echo "ollama already installed: $(ollama --version 2>&1 | head -1)"
  fi

  if command -v ollama >/dev/null 2>&1; then
    # The app can call more than one Ollama model back to back (the general
    # model, the Legal orchestrator if it's set to a different tag, and the
    # judge model in evaluations). Left at Ollama's default of "as many as
    # fit", two can end up loaded at once and the second load fails with the
    # same allocation error as an oversized single model on a modest host.
    # Pin it to one resident model at a time so the second call evicts the
    # first instead of fighting it for RAM.
    export OLLAMA_MAX_LOADED_MODELS=1
    for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
      if [ -f "$rc" ] && ! grep -q "^export OLLAMA_MAX_LOADED_MODELS=" "$rc" 2>/dev/null; then
        echo 'export OLLAMA_MAX_LOADED_MODELS=1' >> "$rc"
      fi
    done
    if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
      echo "Starting 'ollama serve' in the background..."
      nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
      for _ in $(seq 1 15); do
        curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && break
        sleep 1
      done
    else
      echo "[note] ollama is already running -- if it wasn't just started by this"
      echo "       script, restart it (e.g. 'brew services restart ollama', or quit"
      echo "       and reopen the app) so OLLAMA_MAX_LOADED_MODELS=1 takes effect."
    fi
    echo "Pulling $OLLAMA_MODEL (this is a large download, comparable to the vLLM weights)..."
    ollama pull "$OLLAMA_MODEL" || {
      echo "[warn] 'ollama pull $OLLAMA_MODEL' failed. Check the exact tag at"
      echo "       https://ollama.com/library/qwen3 and retry: ollama pull <tag>"
      SKIPPED+=("ollama pull $OLLAMA_MODEL")
    }
  fi

  # Read by the GUI launcher / any local (non-Docker) run of the app so
  # config/config.yaml's vLLM defaults (general llm: + Legal orchestrator)
  # are overridden without editing it. docker-compose.
  # portable.yml sets the same vars itself, so it does not read this file.
  cat > .env.local <<EOF
DOCSLIDES_LLM_BACKEND=ollama
DOCSLIDES_LLM_BASE_URL=http://localhost:11434
DOCSLIDES_LLM_MODEL=$OLLAMA_MODEL
DOCSLIDES_LEGAL_ORCHESTRATOR_BACKEND=ollama
DOCSLIDES_LEGAL_ORCHESTRATOR_BASE_URL=http://localhost:11434
DOCSLIDES_LEGAL_ORCHESTRATOR_MODEL=$OLLAMA_MODEL
EOF
  echo "wrote $REPO_ROOT/.env.local (backend=ollama, model=$OLLAMA_MODEL)"
fi
echo

echo "== [6/7] Downloading language/OCR model weights =="
if [ "$BACKEND" = "ollama" ]; then
  SKIP_QWEN=1 bash scripts/download_models.sh "$MODELS_DIR"
else
  bash scripts/download_models.sh "$MODELS_DIR"
fi
echo

echo "== [7/7] Tesseract OCR language packs (OS-level, not pip) =="
if command -v tesseract >/dev/null 2>&1; then
  echo "tesseract already installed: $(tesseract --version 2>&1 | head -1)"
else
  if [ "$OS_NAME" = "Darwin" ]; then
    if command -v brew >/dev/null 2>&1; then
      brew install tesseract tesseract-lang || SKIPPED+=("tesseract (brew)")
    else
      echo "[skip] Homebrew not found. Install from https://brew.sh, then:"
      echo "       brew install tesseract tesseract-lang"
      SKIPPED+=("tesseract (brew)")
    fi
  else
    echo "[skip] Install via your distro's package manager, e.g.:"
    echo "       sudo apt-get install tesseract-ocr tesseract-ocr-{eng,fra,spa,ita,deu,ara,heb}"
    SKIPPED+=("tesseract (apt)")
  fi
fi
echo

echo "== LibreOffice (needed by MinerU for DOCX/PPTX/XLSX -> PDF conversion) =="
if command -v soffice >/dev/null 2>&1; then
  echo "soffice already installed: $(soffice --version 2>&1 | head -1)"
elif [ "${SKIP_MINERU:-0}" = "1" ]; then
  echo "[skip] MinerU extra was skipped (SKIP_MINERU=1) -- not needed."
else
  if [ "$OS_NAME" = "Darwin" ]; then
    if command -v brew >/dev/null 2>&1; then
      brew install --cask libreoffice || SKIPPED+=("libreoffice (brew)")
    else
      echo "[skip] Homebrew not found. Install from https://brew.sh, then:"
      echo "       brew install --cask libreoffice"
      SKIPPED+=("libreoffice (brew)")
    fi
  else
    echo "[skip] Install via your distro's package manager, e.g.:"
    echo "       sudo apt-get install libreoffice"
    SKIPPED+=("libreoffice (apt)")
  fi
fi
echo

if [ "${#SKIPPED[@]}" -eq 0 ]; then
  echo "Done. Everything installed and downloaded successfully."
else
  echo "Finished with ${#SKIPPED[@]} step(s) skipped/failed:"
  for item in "${SKIPPED[@]}"; do
    echo "  - $item"
  done
  echo "Re-run this script after addressing them -- it's safe to re-run."
fi
echo
echo "Next steps ($BACKEND backend selected):"
if [ "$BACKEND" = "vllm" ]; then
  echo "  1. GPU host: docker compose up -d          # starts vllm + app"
  echo "  2. Or dev-run the app only:  uvicorn docslides.api.main:run --factory"
else
  echo "  1. Docker:      docker compose -f docker-compose.portable.yml up -d"
  echo "  2. Or locally:  set -a; source .env.local; set +a; .venv/bin/python -m uvicorn docslides.api.main:app --host 0.0.0.0 --port 8456"
  echo "  3. Or just use gui/DocSlides.command -- it reads .env.local automatically."
fi
