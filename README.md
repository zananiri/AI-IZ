# docslides

An offline, multilingual document-to-PowerPoint generation system with a
chat interface, powered by a locally-hosted Qwen3-32B served through vLLM.
**Nothing in the runtime path calls out to the internet** -- model/package
downloads are the only online step, done once ahead of time.

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
| `llm/` | vLLM OpenAI-compatible client, guided-JSON schemas, prompts |
| `slides/` | Two-stage outline/fill generation, PPTX building, explicit RTL OOXML handling |
| `tone/` | Reusable tone-control component (professionalism + creativity sliders) for any rewrite request |
| `pipeline/` | End-to-end orchestration with SSE status events |
| `api/` | FastAPI backend: upload, generation job kickoff, SSE streaming, chat, tone-rewrite |
| `ui/` | Gradio chat interface (mounted at `/ui` on the FastAPI app) |

## Hardware requirements

- One 24GB-class consumer GPU (RTX 4090/5090) for the target deployment.
  Qwen3-32B AWQ/GPTQ 4-bit (~20GB) + `--max-model-len 8192` +
  `--gpu-memory-utilization 0.90` fits this comfortably.
- OCR runs on CPU by default (PaddleOCR PP-OCRv6, Tesseract). Only the
  low-confidence VLM-OCR fallback (PaddleOCR-VL / Surya) touches the GPU,
  and only opportunistically between vLLM generations -- see
  `src/docslides/ocr/gpu_arbiter.py`.
- Quantization is configurable (`bf16` / `awq` / `gptq` -- see
  `vllm_launch.quantization` in `config/config.yaml`) so the same codebase
  scales up to a multi-GPU server (e.g. bf16 across more VRAM) without a
  rewrite -- only the launch flags change.

## Setup

### 1. Install

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

### 2. Download models (ONE-TIME, ONLINE step)

```bash
./scripts/download_models.sh ./models
```

This downloads: the fastText `lid.176` language-ID model, spaCy's small
pipelines for en/fr/es/it/de, Stanza's pipelines for ar/he, and triggers
PaddleOCR/PaddleOCR-VL/Surya's first-run weight downloads. Tesseract's
language packs are installed via apt in `docker/Dockerfile.app`.

Separately, download the Qwen3-32B quantized checkpoint:

```bash
huggingface-cli download <QWEN_MODEL_REPO> --local-dir ./hf_cache/qwen3-32b
```

Verify the exact repo name and the resulting launch command against the
current model listing before deploying:

```bash
python scripts/verify_vllm_launch.py
```

### 3. Configure

Edit `config/config.yaml`:
- `llm.model` / `llm.base_url` -- the model repo id and vLLM endpoint
- `paths.template_potx` -- your organization's slide template (see
  `assets/templates/README.md`)
- `ocr.confidence_threshold` -- when to escalate to the GPU OCR fallback
- Any per-language model paths under `languages.*`

### 4. Run

```bash
docker compose up --build
```

This starts:
- `vllm` -- serves Qwen3-32B on port 8000 (OpenAI-compatible API)
- `app` -- FastAPI backend on port 8080, with the Gradio chat UI mounted at
  `http://localhost:8080/ui`

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

```bash
# terminal 1
vllm serve <repo> --quantization awq --max-model-len 8192 --gpu-memory-utilization 0.90 --port 8000

# terminal 2
docslides-api   # FastAPI + Gradio UI on :8080 (UI at /ui)
```

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
