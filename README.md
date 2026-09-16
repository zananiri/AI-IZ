# docslides

An offline, multilingual document-to-PowerPoint generation system with a
chat interface, powered by a locally-hosted Qwen3. **Nothing in the runtime
path calls out to the internet** -- model/package downloads are the only
online step, done once ahead of time.

The chat model is served by one of two interchangeable backends
(`config.llm.backend`, see [Hardware requirements](#hardware-requirements)):
**vLLM** (NVIDIA GPU only, fastest) or **Ollama** (CPU, or whatever
acceleration the host has -- CUDA/ROCm/Metal), so the app runs on any
computer, not just NVIDIA-GPU machines. `scripts/setup.sh`/`setup.ps1`
auto-detect which one to use.

Supported languages: English, French, Spanish, Italian, German, Arabic,
Hebrew (Arabic/Hebrew get explicit RTL handling throughout).

## Architecture

```
docs/images/PDFs  --> ingestion (MinerU / PyMuPDF fallback, per-page scan detection)
                  --> language detection (fastText lid.176 / py3langid, per page)
                  --> OCR routing (ocr/router.py): CPU primary + CPU fallback,
                      escalating to a GPU VLM-OCR fallback only on low confidence,
                      arbitrated against vLLM (ocr/gpu_arbiter.py)
                  --> cleaning (unicode/control-char normalization, header/footer
                      stripping, de-hyphenation, sentence segmentation, masking
                      of non-translatable spans)
                  --> chunking (paragraph/sentence-safe, token-budgeted)
                  --> translation (Qwen3-32B itself, per-chunk, with a running
                      glossary injected into every subsequent chunk's prompt)
                  --> Stage 1: outline generation (Qwen3-32B, thinking ON,
                      guided JSON)
                  --> Stage 2: per-slide fill (Qwen3-32B, thinking OFF,
                      guided JSON)
                  --> PPTX assembly (python-pptx + direct OOXML RTL handling)
```

All of the above is driven by `config/config.yaml` -- no language, engine,
or threshold is hardcoded in application code.

Module layout (`src/docslides/`):

| Module | Responsibility |
|---|---|
| `ingestion/` | Document parsing (MinerU primary, PyMuPDF fallback), per-page scan detection, language detection |
| `ocr/` | OCR engine wrappers, language/script routing, confidence-based GPU escalation, GPU arbitration vs. vLLM |
| `cleaning/` | Unicode/control-char cleanup, header/footer stripping, sentence segmentation, masking, chunking, token counting |
| `translation/` | Per-chunk translation via Qwen3-32B, running glossary |
| `llm/` | Dual-backend chat client (vLLM OpenAI-compatible API or Ollama's native API), guided-JSON schemas, prompts |
| `slides/` | Two-stage outline/fill generation, PPTX building, explicit RTL OOXML handling |
| `tone/` | Reusable tone-control component (professionalism + creativity sliders) for any rewrite request |
| `pipeline/` | End-to-end orchestration with SSE status events |
| `api/` | FastAPI backend: upload, generation job kickoff, SSE streaming, chat, tone-rewrite |
| `ui/` | Gradio chat interface (mounted at `/ui` on the FastAPI app) |

## Hardware requirements

Two interchangeable serving backends, selected by `config.llm.backend`
(`scripts/setup.sh`/`setup.ps1` auto-detect the right one for the machine
they run on; override with `FORCE_BACKEND`/`-ForceBackend`):

| | **vLLM** | **Ollama** |
|---|---|---|
| Hardware | NVIDIA GPU only (24GB-class, e.g. RTX 4090/5090) | Any: CPU, NVIDIA, AMD (ROCm), Apple Silicon (Metal) |
| Install | Docker image (`vllm/vllm-openai`) | Native host install (not Docker -- see `docker-compose.portable.yml`'s header comment for why, esp. on Mac) |
| Model | Qwen3-32B AWQ/GPTQ 4-bit (~20GB) via Hugging Face | `qwen3:32b` (or smaller, e.g. `qwen3:8b`) via `ollama pull` |
| Speed | Fastest -- purpose-built for concurrent GPU serving | Slower, especially CPU-only; scales with whatever acceleration the host has |
| Structured JSON / thinking toggle | `guided_json` extra_body / `chat_template_kwargs` | top-level `format` JSON Schema / `think` field |

Both are driven through the same `src/docslides/llm/client.py` interface --
nothing above the LLM client needs to know which backend is active.

Qwen3-32B is heavy even quantized (~20GB); on a CPU-only or modest machine,
override `llm.model` to a smaller Ollama tag (e.g. `qwen3:8b`) for usable
latency -- set `OLLAMA_MODEL`/`-OllamaModel` when running the setup script.

OCR runs on CPU by default (PaddleOCR PP-OCRv6, Tesseract). Only the
low-confidence VLM-OCR fallback (PaddleOCR-VL / Surya) touches the GPU, and
only opportunistically between vLLM generations -- see
`src/docslides/ocr/gpu_arbiter.py`. That arbitration is vLLM-specific (it
polls vLLM's Prometheus metrics) and is a no-op under the Ollama backend.

## Setup

### One-shot setup (recommended)

`scripts/setup.sh` (macOS/Linux) and `scripts/setup.ps1` (Windows) each do
everything below in one pass: create the venv, install all extras, **detect
whether this machine has an NVIDIA GPU and pick vLLM or Ollama accordingly**,
pull/install the right serving engine, and download every model weight
(chat model, fastText, spaCy, Stanza, PaddleOCR, Surya, MinerU).
Re-running either script is safe -- already-downloaded files are left in place.

```bash
# macOS/Linux
./scripts/setup.sh [models_dir]          # SKIP_MINERU=1 / SKIP_HEAVY_OCR=1 / FORCE_BACKEND=ollama / OLLAMA_MODEL=qwen3:8b
```

```powershell
# Windows
.\scripts\setup.ps1                      # -SkipMineru / -SkipHeavyOcr / -ForceBackend ollama / -OllamaModel qwen3:8b
```

When the Ollama backend is selected, the script writes `.env.local` with the
three env vars (`DOCSLIDES_LLM_BACKEND`/`_BASE_URL`/`_MODEL`) that override
`config/config.yaml`'s vLLM defaults -- see [Hardware requirements](#hardware-requirements).

After setup, use **`gui/DocSlides.bat`** (Windows) or **`gui/DocSlides.command`**
(macOS) to start/stop everything and watch live status (backend, app, Docker)
from a desktop window -- see [gui/launcher.py](gui/launcher.py). It reads
`.env.local` automatically.

### 1. Install (manual)

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[ocr,lang,dev]"
```

The `ocr` and `lang` extras pull in PaddleOCR, Tesseract bindings, Surya,
fastText, spaCy, and Stanza -- all heavy, all optional at the Python-import
level so core modules stay testable without them (see each extra in
`pyproject.toml`).

MinerU (`magic-pdf`) is not pinned as a hard dependency because its API has
moved across releases; install the version matching your setup and see the
adapter notes in `src/docslides/ingestion/parser.py`. Without it, PDF still
works via the PyMuPDF fallback; DOCX/PPTX/XLSX/image inputs require MinerU.
It also needs **LibreOffice** (`soffice`) on PATH, for its DOCX/PPTX/XLSX
-> PDF conversion step -- `scripts/setup.*` install it automatically; see
`docker-compose.yml`'s app image if you're building your own container.

The `ingestion-mineru` extra installs `magic-pdf[full]` (the bare install is
missing the packages the layout/table/formula models actually need at
runtime) and pins `stringzilla`/`pycocotools` to versions with prebuilt
Windows wheels (see the comments in `pyproject.toml` for why). After
installing it, run `python scripts/patch_mineru.py` once (also wired into
`scripts/setup.*` automatically) -- it fixes two real upstream bugs found
getting this working: magic-pdf 1.3.12's bundled OCR config still points at
PP-OCRv3 weight filenames the current `PDF-Extract-Kit-1.0` Hugging Face
repo no longer hosts, and `fasttext-wheel` 0.9.2 (a transitive dependency)
uses a NumPy API that NumPy 2.x made stricter. Both are documented in detail
in that script's docstring.

**Known conflict:** `magic-pdf` (any recent release) pins
`huggingface-hub<1.0`, while `gradio>=6` requires `huggingface-hub>=1.16`.
Installing `.[ingestion-mineru]` into the same venv as the rest of the app
will print a pip dependency-conflict warning -- both packages still import,
but this is a real, unresolved upstream clash, not a false positive. If it
causes problems in practice, run MinerU ingestion as a separate process/venv
instead of sharing one with the FastAPI/Gradio app.

### 2. Download models (ONE-TIME, ONLINE step)

First, verify the configured Qwen3-32B repo still matches the current
Hugging Face listing (repo names/quantizations do change) and see the
resulting `vllm serve` command:

```bash
pip install -U "huggingface_hub[cli]"
python scripts/verify_vllm_launch.py
```

Then download everything -- the Qwen3-32B checkpoint (~20GB, via `hf
download` -- the `huggingface_hub` CLI was renamed from `huggingface-cli` to
`hf` in newer releases; `download_models.sh` uses whichever is installed --
into the standard HF hub cache so vLLM can resolve it by repo id while
offline), the fastText `lid.176` language-ID
model, spaCy's small pipelines for en/fr/es/it/de, Stanza's pipelines for
ar/he, and PaddleOCR/PaddleOCR-VL/Surya's first-run weights:

```bash
./scripts/download_models.sh ./models
```

The script skips (with a clear message) anything whose Python package isn't
installed yet -- e.g. `pip install -e ".[lang]"` before it can fetch spaCy/
Stanza models, `".[ocr]"` before PaddleOCR/Surya. It's safe to re-run after
installing the missing extras; already-downloaded files are left in place.
Tesseract's language packs are installed via apt in `docker/Dockerfile.app`
(or your OS package manager outside Docker).

**If you're deploying with `docker compose`**, run the script *inside* the
app container instead of on the host, so the caches land in the same paths
docker-compose.yml bind-mounts to `./hf_cache`, `./paddleocr_cache`, and
`./stanza_cache` on disk:

```bash
docker compose build app
docker compose run --rm --no-deps app bash scripts/download_models.sh ./models
```

This also downloads the Qwen weights into `./hf_cache`, which the `vllm`
service mounts at the same path -- so one download step covers both
services.

**No NVIDIA GPU?** Use Ollama instead of the two steps above: install it
(https://ollama.com/download or your package manager), then
`ollama pull qwen3:32b` (or a smaller tag, e.g. `qwen3:8b`). No Hugging
Face download needed for the chat model.

### 3. Configure

Edit `config/config.yaml`:
- `llm.backend` -- `"vllm"` or `"ollama"` (see [Hardware requirements](#hardware-requirements))
- `llm.model` / `llm.base_url` -- the model repo id/tag and endpoint for
  whichever backend is selected
- `paths.template_potx` -- your organization's slide template (see
  `assets/templates/README.md`)
- `ocr.confidence_threshold` -- when to escalate to the GPU OCR fallback
- Any per-language model paths under `languages.*`

Or leave `config.yaml` as-is and override just the LLM section via env vars
(`DOCSLIDES_LLM_BACKEND`/`_BASE_URL`/`_MODEL`) -- this is what `.env.local`
(written by the setup scripts) and `docker-compose.portable.yml` do.

### 4. Run

**NVIDIA GPU (vLLM):**

```bash
docker compose up --build
```

This starts:
- `vllm` -- serves Qwen3-32B on port 8000 (OpenAI-compatible API)
- `app` -- FastAPI backend on port 8456, with the Gradio chat UI mounted at
  `http://localhost:8456/ui`

**Any other machine (Ollama):** start Ollama first (`ollama serve`, usually
already running as a background service after install), then:

```bash
docker compose -f docker-compose.portable.yml up --build
```

This starts only `app`, pointed at the host's Ollama via
`host.docker.internal:11434` -- see that file's header comment for why
Ollama runs on the host rather than in its own container (Docker on macOS
can't pass Metal through to a container). Or skip Docker entirely and run
the app directly -- see `gui/DocSlides.bat`/`.command` or
`scripts/setup.*`'s printed "Next steps".

Reference `vllm serve` command (also printed by
`scripts/verify_vllm_launch.py`):

```bash
vllm serve <Qwen3-32B-AWQ-or-GPTQ-repo> \
  --quantization awq \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --guided-decoding-backend xgrammar \
  --port 8000
```

### Running without Docker

**vLLM (NVIDIA GPU):**

```bash
# terminal 1
vllm serve <repo> --quantization awq --max-model-len 8192 --gpu-memory-utilization 0.90 --port 8000

# terminal 2
docslides-api   # FastAPI + Gradio UI on :8456 (UI at /ui)
```

**Ollama (any machine):**

```bash
# terminal 1 (often already running as a background service after install)
ollama serve

# terminal 2
set -a; source .env.local; set +a   # written by scripts/setup.sh; PowerShell: see scripts/setup.ps1's "Next steps"
docslides-api
```

Or just use `gui/DocSlides.bat`/`.command`, which does both of the above
from one window and shows live status.

## Tests

```bash
pytest
```

Unit tests cover the parts that don't require GPU/model downloads: masking
round-trips, text cleaning, tone-control sampling-parameter resolution, RTL
OOXML manipulation, and PDF scan-detection. Fixtures live in
`tests/fixtures/` (regenerate with `python tests/fixtures/generate_fixtures.py`):

- `scanned_arabic.pdf` -- image-only PDF (no text layer) to validate
  scan-detection and Arabic OCR routing
- `native_hebrew.docx` -- native-text Hebrew document to validate Hebrew
  language detection and RTL handling without OCR
- `mixed_language.docx` -- English/French/Arabic paragraphs in one document,
  to validate per-segment (not per-document) language detection

End-to-end OCR/translation/generation behavior requires the models from
step 2 and a running vLLM instance, and is exercised manually via the
Gradio UI rather than in the unit test suite.

## Non-goals

- No cloud OCR/translation/LLM API calls anywhere in the runtime path.
- No hardcoded Latin-script/English assumptions -- every OCR, cleaning, and
  chunking step is explicitly language-aware via `config/config.yaml`.
