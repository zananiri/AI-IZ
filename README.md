# docslides

An offline, multilingual document-to-PowerPoint generation system with a
chat interface, powered by locally-hosted Gemma 4 and TranslateGemma. **Nothing in the runtime
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
                  --> translation (TranslateGemma per chunk; Gemma 4 builds a
                      glossary first and fixes any glossary term a chunk's
                      translation missed)
                  --> Stage 1: outline generation (Gemma 4, guided JSON)
                  --> Stage 2: per-slide fill (Gemma 4, guided JSON)
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
| `translation/` | Per-chunk translation via TranslateGemma, with a Gemma 4-built glossary |
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
| Hardware | NVIDIA GPU only, above 40GB (48GB+ workstation/data-center cards) | Any: CPU, NVIDIA, AMD (ROCm), Apple Silicon (Metal) |
| Install | Docker image (`vllm/vllm-openai`) | Native host install (not Docker -- see `docker-compose.portable.yml`'s header comment for why, esp. on Mac) |
| Model | Gemma 4 31B, `google/gemma-4-31B-it` (gated; FP8-quantized on load, ~33GB) via Hugging Face, for the general chat and the Legal tab; Gemma 4 also translates | General chat and Legal tab: the same Gemma tag (`gemma4:31b`, `gemma4:12b` or `gemma3:4b-it-qat` by host memory). Translation: TranslateGemma of the matching size (`translategemma:27b` / `:12b` / `:4b`). All via `ollama pull` |
| Speed | Fastest -- purpose-built for concurrent GPU serving | Slower, especially CPU-only; scales with whatever acceleration the host has |
| Structured JSON / thinking toggle | `guided_json` extra_body / `chat_template_kwargs` | top-level `format` JSON Schema / `think` field |

Both are driven through the same `src/docslides/llm/client.py` interface --
nothing above the LLM client needs to know which backend is active.

The general chat (chat, rewrite, slides) runs on the same Gemma as the Legal tab, picked by
the same rule. Under Ollama that is Gemma 4 31B dense (`gemma4:31b`, 4-bit; ~25GB resident
with the 16k context); `scripts/setup.*` fall back to Gemma 4 12B (`gemma4:12b`) under 32GB of
RAM or without an NVIDIA/AMD GPU, and to `gemma3:4b-it-qat` under 12GB. `OLLAMA_MODEL` /
`-OllamaModel` picks another tag for both. Both get a 16384-token context
(`DOCSLIDES_LLM_MAX_MODEL_LEN` / `DOCSLIDES_LEGAL_ORCHESTRATOR_MAX_MODEL_LEN`, Ollama's
`num_ctx`). Gemma runs with thinking off (Gemma 3 has no thinking mode, and Gemma 4's stays
off), so the setup scripts write `DOCSLIDES_LLM_SUPPORTS_THINKING=false` and
`DOCSLIDES_LEGAL_ORCHESTRATOR_SUPPORTS_THINKING=false` to `.env.local` (every call then goes
out with an explicit `"think": false`: Gemma 4 thinks by default when the field is missing; a
server that rejects the field, as some do for Gemma 3, is asked again without it). Under vLLM
both run on `google/gemma-4-31B-it`.

**Translation** pairs two models (`src/docslides/translation/translator.py`): Gemma 4 reads
the document once and builds a glossary of terms to render consistently, TranslateGemma
translates each chunk with its own prompt, and any glossary term a chunk's translation missed
goes back to Gemma 4 to fix. Every chat request that wants text in another language goes through
TranslateGemma: translating an attached document or pasted text, and any other reply asked for in
another language (a summary, an answer), which Gemma 4 writes in the material's own language and
TranslateGemma then translates (`src/docslides/api/routes_chat.py`). Under Ollama the setup scripts pull the TranslateGemma size that
matches the Gemma above -- `translategemma:27b` with `gemma4:31b`, `:12b` with `gemma4:12b`,
`:4b` with `gemma3:4b-it-qat` (`OLLAMA_TRANSLATE_MODEL` / `-OllamaTranslateModel` override) --
and write it as `DOCSLIDES_TRANSLATOR_*` (config `translation.translator`). Under vLLM it is
unset and Gemma 4 translates on its own. The glossary is saved per document and target
language under `paths.glossary_dir`.

The setup scripts also set
`OLLAMA_MAX_LOADED_MODELS=1` so that when two Ollama models are called back
to back (e.g. the Legal orchestrator and an evaluation's judge model) the
second evicts the first instead of both trying to stay resident at once, and
`OLLAMA_FLASH_ATTENTION=1` + `OLLAMA_KV_CACHE_TYPE=q8_0`, which halve the
context's memory.

**Context window.** The Legal tab runs with a 16k-token window
(`legal.orchestrator.max_model_len: 16384`): up to 5k tokens of evidence
(`legal.retrieval.max_evidence_tokens`) plus the thinking analysis pass and
the memo/draft outputs. vLLM serves 16k (`VLLM_MAX_MODEL_LEN` in `docker-compose.yml`; set
`VLLM_KV_CACHE_DTYPE=fp8` if it doesn't fit next to the ~33GB FP8 weights); under Ollama it is the
`num_ctx` sent with every request. `DOCSLIDES_LEGAL_ORCHESTRATOR_MAX_MODEL_LEN`
/ `DOCSLIDES_LLM_MAX_MODEL_LEN` override it per host (`LEGAL_CONTEXT_LENGTH` /
`-LegalContextLength` in the setup scripts).
The Legal tab uses one model, the Gemma 4 orchestrator, for every stage and
answers in the question's language, Hebrew included.

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
./scripts/setup.sh [models_dir]          # SKIP_MINERU=1 / SKIP_HEAVY_OCR=1 / FORCE_BACKEND=ollama / OLLAMA_MODEL=gemma4:31b / OLLAMA_TRANSLATE_MODEL=translategemma:27b / LEGAL_CONTEXT_LENGTH=16384
```

```powershell
# Windows
.\scripts\setup.ps1                      # -SkipMineru / -SkipHeavyOcr / -ForceBackend ollama / -OllamaModel gemma4:31b / -OllamaTranslateModel translategemma:27b / -LegalContextLength 16384
```

When the Ollama backend is selected, the script writes `.env.local` with the
`DOCSLIDES_LLM_BACKEND`/`_BASE_URL`/`_MODEL` triple (general chat model) plus
a matching `DOCSLIDES_LEGAL_ORCHESTRATOR_*` set for the Legal tab's
independent orchestrator deployment (backend, URL, model and
`_MAX_MODEL_LEN`, its context window) and a `DOCSLIDES_TRANSLATOR_*` set for
TranslateGemma -- all overriding
`config/config.yaml`'s vLLM defaults. Both scripts also install the
`legal` and `legal-data` extras the Legal tab needs. See
[Hardware requirements](#hardware-requirements).

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

First, verify the configured Gemma repo still matches the current
Hugging Face listing (repo names/quantizations do change) and see the
resulting `vllm serve` command:

```bash
pip install -U "huggingface_hub[cli]"
python scripts/verify_vllm_launch.py
```

Then download everything -- the Gemma 4 31B checkpoint (`google/gemma-4-31B-it`, ~62GB, gated:
accept its license on huggingface.co and run `hf auth login` first; via `hf
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

This also downloads the Gemma weights into `./hf_cache`, which the `vllm`
service mounts at the same path -- so one download step covers both
services.

**No NVIDIA GPU?** Use Ollama instead of the two steps above: install it
(https://ollama.com/download or your package manager), then
`ollama pull gemma4:31b` (general chat and Legal tab) and `ollama pull translategemma:27b`
(translation) -- or the smaller pair for a smaller host (see above). No Hugging Face download
needed for either.

### 3. Configure

Edit `config/config.yaml`:
- `llm.backend` -- `"vllm"` or `"ollama"` (see [Hardware requirements](#hardware-requirements))
- `llm.model` / `llm.base_url` -- the model repo id/tag and endpoint for
  whichever backend is selected
- `paths.template_potx` -- your organization's slide template (see
  `assets/templates/README.md`)
- `ocr.confidence_threshold` -- when to escalate to the GPU OCR fallback
- Any per-language model paths under `languages.*`

Or leave `config.yaml` as-is and override the LLM sections via env vars --
`DOCSLIDES_LLM_BACKEND`/`_BASE_URL`/`_MODEL` for the general model, and
`DOCSLIDES_LEGAL_ORCHESTRATOR_*` for the Legal tab's deployment, `DOCSLIDES_TRANSLATOR_*` for
TranslateGemma --
this is what
`.env.local` (written by the setup scripts) and `docker-compose.portable.yml`
do.

### Legal tab: where it searches

By default (`legal.retrieval.source: "corpus"` in `config/config.yaml`) the
Legal tab runs its full pipeline over the bulk legal corpus: the laws and
procedural regulations built on Kaggle and installed with
`python scripts/legal_data/install_corpus.py legal_corpus_vectordb.zip` into
`data/legal_corpus_vectordb`. Each question is first planned (issues, governing
laws and sections), then searched and reranked the same way as the bulk eval
(`scripts/legal_data/eval_run.py`). The law PDFs in `legal_txt/` are not used.
Set `source: "signed_index"` to answer from the signed index below instead.

### Legal tab: Supreme Court case law

Alongside whichever source answers the question (bulk corpus or signed index), the Legal tab can
also draw on Israeli Supreme Court judgments decided before 2022 (`caselaw/`, see
`caselaw/README.md`), built on Kaggle or Colab from the `LevMuchnik/SupremeCourtOfIsrael` dataset
and searched by `src/docslides/legal/caselaw.py`. It's off until you point the app at a built
index:

```bash
pip install -e ".[legal]"   # pulls in lancedb and bm25s, needed to read the index
```

Unzip the ingest's output (the Output of `notebooks/kaggle_caselaw_ingest.ipynb`, or
`israeli_caselaw_ingest all --root ./out --embed` run locally -- see `caselaw/README.md`)
somewhere on this machine, e.g. `data/legal_caselaw/`. It must contain `lancedb/` and
`state/embed.done.json`; `bm25/` is optional (dense search alone still works without it). Then in
`config/config.yaml`:

```yaml
legal:
  corpus:
    caselaw_dir: "./data/legal_caselaw"
```

and restart the app. The embedding model named in `state/embed.done.json` must match
`legal.retrieval.embedding_model` (`BAAI/bge-m3` by default) -- a mismatch is logged
(`caselaw_model_mismatch`) and case law is skipped for that run, statute answers are unaffected.
When it's on, each question's best-matching judgment excerpts (reranked, one per judgment, capped
by `caselaw_top_k` / `caselaw_max_tokens`) go into the prompt as their own `<case_law>` block,
labelled as judgments the model may cite but not treat as the statute text itself.

### Legal tab: building the signed Israeli-law index

With `legal.retrieval.source: "signed_index"`, the Legal tab answers only from
sources that were staged, reviewed and signed into its local index, so it has
nothing to cite until you add some.
It never fetches from the web: get official texts yourself, within each
site's terms of use.

```bash
pip install -e ".[legal]"
export DOCSLIDES_LEGAL_BUNDLE_KEY=<long random secret>   # same value for the API process
# Official texts: data/legal/sources/<file>.txt|.docx|.pdf + <file>.meta.json
python scripts/ingest_legal.py stage data/legal/sources/<file>.txt --dry-run   # check the parsed structure
python scripts/ingest_legal.py stage-sources
# Memos / firm / client documents: drop them in uploads/
python scripts/ingest_legal.py stage-uploads
python scripts/ingest_legal.py list --status pending
python scripts/ingest_legal.py approve <batch_id> --reviewer "Adv. Name"      # uploads require --reviewer
python scripts/ingest_legal.py verify
```

**Quick path for law PDFs:** drop them into `legal_txt/` at the project root
and run `python scripts/ingest_legal_txt.py` (add `--watch` to keep polling,
`--dry-run` to preview, `--prune` to remove laws whose PDF you deleted). Each
new or changed PDF is indexed straight away as an official source. A PDF
without a `.meta.json` gets one derived from its title (law name, 1 January of
the title's year, statute/Knesset or regulation/Reshumot). Check it, fix
`effective_date_start` in particular, and re-run to re-index. Scanned PDFs, and
PDFs whose Hebrew text layer is stored in reversed (visual) order, are refused.

**Superseding is automatic:** approving a newer version of a law (same law,
later `effective_date_start`) sets the older version's end date to the day
before the new one takes effect and marks it `amended`. Both stay searchable
for questions about past dates. `ingest_legal.py retract <law_id> <date>`
removes a version and re-opens the one before it.

**Amending laws are linked to the laws they amend.** Each section of an
amending law is tagged with the law, amendment number and sections it changes
(read from its title and the gazette's "תיקונים עקיפים" list). When an answer
draws on a provision that a later indexed law amended, the model is told the
amendment and its effective date -- the old text isn't discarded, because a
lawyer usually needs the version in force when the facts happened -- and an
answer citing a section the amendment itself changed escalates. Run
`python scripts/ingest_legal.py retag` once for laws indexed before this existed.

**Only verified statements reach the answer.** After the redraft, a sentence
whose cited source doesn't state it (or isn't a source the research memorandum
established) is removed. If nothing cited is left, no answer is given and the
question escalates. A partly supported citation stays, marked unverified.
Words the model wrote in another script ('מ報導', 'октяבר') are replaced word
by word; the rest of the text is left as it is. When provisions of several
indexed laws match a question that names none, the memorandum and the draft
are sent back to cover each law, and an answer that still leaves one out says
so first. A question naming a section whose text isn't in the index (an
amending law that only refers to it) is told so up front.

Every answer is appended to `data/legal/audit/<date>.jsonl` (question,
model, retrieved chunks, memorandum/draft attempts,
verification results). Those files hold users' questions verbatim.

### 4. Run

**NVIDIA GPU (vLLM):**

```bash
docker compose up --build
```

This starts:
- `vllm` -- serves Gemma 4 31B on port 8000 (OpenAI-compatible API)
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
vllm serve google/gemma-4-31B-it \
  --quantization fp8 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --guided-decoding-backend xgrammar \
  --port 8000
```

### Running without Docker

**vLLM (NVIDIA GPU):**

```bash
# terminal 1
vllm serve google/gemma-4-31B-it --quantization fp8 --max-model-len 16384 --gpu-memory-utilization 0.90 --port 8000

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

### Legal evals (100 questions + paralegal cases)

The Israeli legal eval set (`legal_txt/Evals/israeli_legal_eval.zip`) runs against the bulk corpus:

| Notebook | Where | What |
|---|---|---|
| `notebooks/colab_legal_eval_qwen_gemma.ipynb` | Google Colab, free T4 | qwen3:14b and gemma4:12b answer the same 100 questions + 5 cases. **No judging**: answers, full reasoning traces (`llm_trace/`, `reasoning_report*.md`), judge-free scores and a summary, written to Google Drive as it goes (resumable) |
| `notebooks/kaggle_legal_eval_gemma27b.ipynb` | Kaggle, 2× T4 | gemma4:31b on the same 100 questions + 5 cases, with retrieval on the GPU and BM25; **no judging**. Prints a judge-free comparison with the 29 Sept gemma3:12b run (multiple choice, yes/no labels, governing section retrieved, out-of-scope sources, timing) |
| `notebooks/kaggle_legal_eval_gemma4_caselaw.ipynb` | Kaggle, 2× T4 | gemma4:31b on the same 100 questions + 5 cases as the 29 Sept Gemma 3 27B run, in two arms: with the Supreme Court case-law index (`legal.corpus.caselaw_dir`, `src/docslides/legal/caselaw.py`) and without; **no judging**. Needs the Output of `notebooks/kaggle_caselaw_ingest.ipynb` as Input |
| `notebooks/kaggle_legal_eval_bulk500.ipynb` | Kaggle, 2× T4 | answer + judge + score; qwen3:14b by default (qwen3:32b for milestone runs) |
| `notebooks/kaggle_legal_eval_dictalm.ipynb` | Kaggle, 2× T4 | the same test with DictaLM 3.0 answering |

The judge is `legal.judge_model` (default `gpt-oss:20b`, overridden by `DOCSLIDES_LEGAL_JUDGE_MODEL`): a
family other than the models under test, so no model grades its own answers. Saved answers
can be re-judged without re-answering: `score.py prepare`, then `scripts/legal_data/eval_run.py judge`,
then `score.py report` (see `eval_run.py`'s docstring). `scripts/legal_data/check_corpus_coverage.py`
checks every law and section the gold answers cite against the installed corpus.

## Non-goals

- No cloud OCR/translation/LLM API calls anywhere in the runtime path.
- No hardcoded Latin-script/English assumptions -- every OCR, cleaning, and
  chunking step is explicitly language-aware via `config/config.yaml`.
