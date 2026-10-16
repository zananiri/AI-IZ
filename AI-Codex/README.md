# AI-Codex

**AI-Codex is the Spanish-market edition of the AI-IZ legal assistant**: the same offline system --
grounded legal research and drafting (Legal GPT), case-file chat, and a general chat that writes
Excel, PowerPoint and PDF files -- retargeted from Israeli to **Spanish law**. The Legal tab answers
only from Spanish and EU law and Tribunal Constitucional case law fetched from the official
open-data sources (BOE, EUR-Lex/CELLAR, the TC's own search system), in the question's language,
with every claim cited and verified against its source. See
[Spanish legal corpus](#spanish-legal-corpus) and [Spanish pipeline choices](#spanish-pipeline-choices).

It is an offline, multilingual document-to-PowerPoint generation system with a
chat interface, powered by locally-hosted Gemma 4 and TranslateGemma. **Nothing in the runtime
path calls out to the internet** -- model/package downloads are the only
online step, done once ahead of time.

The chat model is served by one of two interchangeable backends
(`config.llm.backend`, see [Hardware requirements](#hardware-requirements)):
**vLLM** (NVIDIA GPU only, fastest) or **Ollama** (CPU, or whatever
acceleration the host has -- CUDA/ROCm/Metal), so the app runs on any
computer, not just NVIDIA-GPU machines. `scripts/setup.sh`/`setup.ps1`
auto-detect which one to use.

Supported languages: Spanish first, Catalan, Galician, Basque, English, French, Portuguese,
Italian, German, Arabic and Hebrew (Arabic/Hebrew keep their explicit RTL handling).

## Spanish legal corpus

```bash
pip install -e ".[legal,legal-data]"
export AICODEX_LEGAL_DATA_CONTACT_EMAIL=you@example.com    # goes in the fetchers' User-Agent
python scripts/legal_data/fetch_all.py --dry-run            # what each source would fetch, and how long
python scripts/legal_data/fetch_laws.py                     # BOE: laws, regulations, regional law (~3.5 h first run)
python scripts/legal_data/fetch_eu_law.py                   # EU law in Spanish
python scripts/legal_data/fetch_constitutional_court.py     # TC sentencias (~9 h first run, resumable)
python scripts/legal_data/check_corpus_coverage.py          # CE, CC, CP, LEC, ET, LAU ... present and parsed
python scripts/legal_data/vectorize.py --export-lexical     # embeddings + BM25 copy (GPU, or Kaggle)
```

| Category | Source | What |
|---|---|---|
| `laws` | BOE open-data API (`legislacion-consolidada`) | Constitución, leyes orgánicas, leyes, RDLeg, RDL, the codes (CC, CCom, LECrim ...), treaties -- consolidated, current wording, article by article |
| `regulations` | same | reales decretos, órdenes, resoluciones, circulares |
| `regional_laws` | same | the autonomous communities' laws, decree-laws and regulations (derecho foral included) |
| `eu_law` | CELLAR + SPARQL (Publications Office) | RGPD, AI Act, DSA/DMA, Bruselas I bis, Roma I/II, consumer directives, the Charter and Treaties ... in Spanish, newest consolidated version |
| `constitutional_court` | hj.tribunalconstitucional.es | sentencias (autos optional) with ECLI, chamber, judges; private parties anonymized |

All fetchers honour robots.txt (with wildcards: the BOE withdraws documents that way), stop on any
block, run at one request per second, and are incremental: re-runs fetch only what changed. CENDOJ
(Tribunal Supremo) is deliberately not crawled -- its terms forbid bulk downloads without CGPJ
authorization. Details: [`scripts/legal_data/README.md`](scripts/legal_data/README.md).
`notebooks/kaggle_spanish_corpus_build.ipynb` builds the whole store on a Kaggle GPU.

## Spanish pipeline choices

Each component was checked against the published Spanish evidence, and where it mattered,
measured on the real corpus: `scripts/legal_data/benchmark_embeddings.py` over the 4,816 articles
of the CE, CC, CP, LEC, ET and LAU as the BOE publishes them today, with the 30 questions of
`legal_txt/Evals/es_retrieval_seed.jsonl` (each gold article checked against the BOE text).

| | top-1 | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|---|
| BM25 alone (accent-folded, stemmed) | 60.0% | 80.0% | 83.3% | 0.697 |
| BM25 + reranker | 80.0% | 83.3% | 83.3% | 0.811 |
| **bge-m3** dense | **86.7%** | 93.3% | **100%** | **0.901** |
| Qwen3-Embedding-0.6B dense | 73.3% | 93.3% | 96.7% | 0.814 |
| **bge-m3 + BM25 + bge-reranker-v2-m3** (the Legal tab's path) | **86.7%** | **100%** | **100%** | **0.919** |
| Qwen3-Embedding-0.6B + BM25 + bge-reranker-v2-m3 | 86.7% | 96.7% | 96.7% | 0.903 |

(Documents truncated to 384 tokens, reranker over the fused top 20, CPU run on 16 Oct 2026. 30
questions is a small set: treat differences of one question, 3.3 points, as noise.)

- **Embeddings: bge-m3 stays.** Third-party Spanish results (SINAI's legal-embedding evaluation)
  put Qwen3-Embedding-0.6B ahead of bge-m3 on most Spanish sets, so it was the candidate to switch
  to -- but on Spanish statute text it ranked the governing article first 4 times fewer in 30, and
  was slower to encode on a CPU (64 min for the corpus, against under 50 min for bge-m3 including its reranking). The embedding layer now supports instruction-tuned
  and prompt-based models (`src/aicodex/rag/embedding.py`: Qwen3's task instruction, e5's
  "query:/passage:", a model's own "query" prompt), so a model can be swapped by config and
  re-measured with the benchmark; `legal.retrieval.query_instruction` overrides the instruction.
  A Spanish legal-domain model (SINAI/ALIA-MrBERT-es-legal-administrative-embeddings, Apache 2.0)
  wins on in-domain pairs in its own card but loses on QA; it doesn't load with the
  transformers < 5 pin this project needs for MinerU, so it is left for a later test.
- **Reranker: bge-reranker-v2-m3 stays.** An independent Spanish evaluation (LAMAR, 2026) scores it
  86.8 MRR@10 on Spanish against 60.2 for Qwen3-Reranker-0.6B; here it lifts BM25's top-1 from 60%
  to 80% and brings the full path to 100% recall@5.
- **Keyword search: rebuilt for Spanish** (`legal/keyword.py`): lowercased and accent-folded
  ("artículo" = "articulo"), a light plural/gender stemmer (light stemming matches or beats
  aggressive Snowball stemming for Spanish retrieval), law numbers ("29/1994") and "14 bis" kept as
  single tokens, Spanish stopwords. It is the safety net for exact identifiers; the reranker, not
  the fusion, decides the order (equal-weight fusion alone lowers MRR; the fusion weight is being measured).
- **LLM: Gemma 4 stays.** The Spanish public models (BSC's ALIA-40b, Apache 2.0, trained with
  Spanish and co-official languages oversampled; the Salamandra family) are worth watching, but
  ALIA-40b has no instruction-tuned release yet and publishes no legal or Gemma comparison. Any
  Ollama tag can be tried through `model_sizes` in `config/config.yaml`.
- **Anonymization:** the TC already pseudonymizes minors and victims; the remaining "don/doña"
  party names are replaced by a deterministic rule that keeps the magistrates. Open Spanish judicial
  NER models exist (AymurAI's Flair models) but their licences and evaluations aren't published, so
  they weren't adopted.

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

Module layout (`src/aicodex/`):

| Module | Responsibility |
|---|---|
| `ingestion/` | Document parsing (MinerU primary, PyMuPDF fallback), per-page scan detection, language detection |
| `ocr/` | OCR engine wrappers, language/script routing, confidence-based GPU escalation, GPU arbitration vs. vLLM |
| `cleaning/` | Unicode/control-char cleanup, header/footer stripping, sentence segmentation, masking, chunking, token counting |
| `translation/` | Per-chunk translation via TranslateGemma, with a Gemma 4-built glossary |
| `llm/` | Dual-backend chat client (vLLM OpenAI-compatible API or Ollama's native API), guided-JSON schemas, prompts |
| `slides/` | Two-stage outline/fill generation, PPTX building, explicit RTL OOXML handling |
| `documents/` | Excel / PowerPoint / PDF generation from a chat request: Gemma 4 writes a guided-JSON spec, openpyxl / python-pptx / PyMuPDF render it |
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

Both are driven through the same `src/aicodex/llm/client.py` interface --
nothing above the LLM client needs to know which backend is active.

The general chat (chat, rewrite, slides) runs on the same Gemma as the Legal tab, picked by
the same rule. Under Ollama that is Gemma 4 31B dense (`gemma4:31b`, 4-bit; ~25GB resident
with the 16k context); `scripts/setup.*` fall back to Gemma 4 12B (`gemma4:12b`) under 32GB of
RAM or without an NVIDIA/AMD GPU, and to `gemma3:4b-it-qat` under 12GB. `OLLAMA_MODEL` /
`-OllamaModel` picks another tag for both. Both get a 16384-token context
(`AICODEX_LLM_MAX_MODEL_LEN` / `AICODEX_LEGAL_ORCHESTRATOR_MAX_MODEL_LEN`, Ollama's
`num_ctx`). Gemma runs with thinking off (Gemma 3 has no thinking mode, and Gemma 4's stays
off), so the setup scripts write `AICODEX_LLM_SUPPORTS_THINKING=false` and
`AICODEX_LEGAL_ORCHESTRATOR_SUPPORTS_THINKING=false` to `.env.local` (every call then goes
out with an explicit `"think": false`: Gemma 4 thinks by default when the field is missing; a
server that rejects the field, as some do for Gemma 3, is asked again without it). Under vLLM
both run on `google/gemma-4-31B-it`.

**Translation** pairs two models (`src/aicodex/translation/translator.py`): Gemma 4 reads
the document once and builds a glossary of terms to render consistently, TranslateGemma
translates each chunk with its own prompt, and any glossary term a chunk's translation missed
goes back to Gemma 4 to fix. Every chat request that wants text in another language goes through
TranslateGemma: translating an attached document or pasted text, and any other reply asked for in
another language (a summary, an answer), which Gemma 4 writes in the material's own language and
TranslateGemma then translates (`src/aicodex/api/routes_chat.py`). Under Ollama the setup scripts pull the TranslateGemma size that
matches the Gemma above -- `translategemma:27b` with `gemma4:31b`, `:12b` with `gemma4:12b`,
`:4b` with `gemma3:4b-it-qat` (`OLLAMA_TRANSLATE_MODEL` / `-OllamaTranslateModel` override) --
and write it as `AICODEX_TRANSLATOR_*` (config `translation.translator`). Under vLLM it is
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
`num_ctx` sent with every request. `AICODEX_LEGAL_ORCHESTRATOR_MAX_MODEL_LEN`
/ `AICODEX_LLM_MAX_MODEL_LEN` override it per host (`LEGAL_CONTEXT_LENGTH` /
`-LegalContextLength` in the setup scripts).
The Legal tab uses one model, the Gemma 4 orchestrator, for every stage and
answers in the question's language (Spanish by default; Catalan, Galician, Basque and English too).

OCR runs on CPU by default (PaddleOCR PP-OCRv6, Tesseract). Only the
low-confidence VLM-OCR fallback (PaddleOCR-VL / Surya) touches the GPU, and
only opportunistically between vLLM generations -- see
`src/aicodex/ocr/gpu_arbiter.py`. That arbitration is vLLM-specific (it
polls vLLM's Prometheus metrics) and is a no-op under the Ollama backend.
If an OCR engine fails to load or crashes (e.g. surya-ocr missing or an incompatible
version), the router logs `ocr_engine_failed` and moves on to the next engine for that
language (Spanish: PaddleOCR -> Tesseract `spa` -> PaddleOCR-VL); a page only fails when every
engine in its chain does.

## Generating Excel, PowerPoint and PDF files

Ask the General tab's chat (Clara) for a file and it comes back as a download link:
"create an Excel budget for a 3-day trip to Rome", "make a presentation about solar energy",
"write a PDF report summarizing this" (with a 📎 attachment, the file is built from the
document). The chat's intent classifier (`ChatIntent.document_format`) decides which file was
asked for; `src/aicodex/documents/generator.py` does the rest:

1. **Gemma 4 writes the content as structured JSON** -- `SpreadsheetSpec` (sheets, header row,
   numeric cells, `=SUM(...)`-style formulas), `PresentationSpec` (titled slides with bullets,
   speaker notes and layouts) or `PdfSpec` (sections with paragraphs, bullets and tables) in
   `src/aicodex/llm/schemas.py`. The schema is enforced by guided decoding -- xgrammar under
   vLLM serving the Hugging Face checkpoint `google/gemma-4-31B-it`, a JSON Schema `format`
   under Ollama -- with the client's validate-and-retry on top.
2. **A deterministic builder renders it**: openpyxl (`documents/xlsx_builder.py`: bold frozen
   header, auto-filter, numbers stored as numbers, column widths, right-to-left sheets for
   Hebrew/Arabic), python-pptx through the slide pipeline's own `slides/pptx_builder.py` (so the
   configured `.potx` template and RTL handling apply, plus a title slide), and PyMuPDF's HTML
   Story engine (`documents/pdf_builder.py`: multi-page A4 layout with tables, shaping and
   right-to-left text from MuPDF's bundled Noto fonts).

The model never emits file bytes or code, so every file opens cleanly and nothing it generates is
executed -- and all of it runs offline on libraries already in the core dependencies
(`openpyxl`, `python-pptx`, `PyMuPDF`); there is nothing extra to install. Hugging Face's
`transformers.pipeline` has no task that writes Office or PDF files: the model's job is the
content, served from the same Hugging Face Gemma 4 checkpoint the rest of the app uses. Files go
to `paths.output_dir` and download from `/api/download/{job_id}`.

A deck from an *attached* document still goes through the full slide pipeline (translate ->
outline -> fill). Generated files are written directly in the requested language, since each
spec is one structured reply rather than prose TranslateGemma can translate chunk by chunk.

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
`AICODEX_LLM_BACKEND`/`_BASE_URL`/`_MODEL` triple (general chat model) plus
a matching `AICODEX_LEGAL_ORCHESTRATOR_*` set for the Legal tab's
independent orchestrator deployment (backend, URL, model and
`_MAX_MODEL_LEN`, its context window) and a `AICODEX_TRANSLATOR_*` set for
TranslateGemma -- all overriding
`config/config.yaml`'s vLLM defaults. Both scripts also install the
`legal` and `legal-data` extras the Legal tab needs. See
[Hardware requirements](#hardware-requirements).

After setup, use **`gui/AICodex.bat`** (Windows) or **`gui/AICodex.command`**
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
adapter notes in `src/aicodex/ingestion/parser.py`. Without it, PDF still
works via the PyMuPDF fallback; DOCX/PPTX/XLSX/image inputs require MinerU.
It also needs **LibreOffice** (`soffice`) on PATH, for its DOCX/PPTX/XLSX
-> PDF conversion step -- `scripts/setup.*` install it automatically; see
`docker-compose.yml`'s app image if you're building your own container.

The `ingestion-mineru` extra installs `magic-pdf[full]` (the bare install is
missing the packages the layout/table/formula models actually need at
runtime) and pins `stringzilla`/`pycocotools` to versions with prebuilt
Windows wheels (see the comments in `pyproject.toml` for why). After
installing it, run `python scripts/patch_mineru.py` once (also wired into
`scripts/setup.*` automatically) -- it fixes real upstream bugs found
getting this working: magic-pdf 1.3.12's bundled OCR config still points at
PP-OCRv3 weight filenames the current `PDF-Extract-Kit-1.0` Hugging Face
repo no longer hosts; `fasttext-wheel` 0.9.2 (a transitive dependency)
uses a NumPy API that NumPy 2.x made stricter; and magic-pdf's vendored
UnimerNet formula model uses `transformers` APIs newer releases removed
(`cache_position`, `find_pruneable_heads_and_indices`, `get_head_mask`).
Each is documented in detail in that script's docstring. The patches edit
files inside the installed packages, so **re-run it after any reinstall or
upgrade** of magic-pdf or `transformers`. If the log shows
`mineru_parse_failed_falling_back` with an `ImportError`/`TypeError` from
`transformers`, the patch hasn't been applied to that venv.

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
`AICODEX_LLM_BACKEND`/`_BASE_URL`/`_MODEL` for the general model, and
`AICODEX_LEGAL_ORCHESTRATOR_*` for the Legal tab's deployment, `AICODEX_TRANSLATOR_*` for
TranslateGemma --
this is what
`.env.local` (written by the setup scripts) and `docker-compose.portable.yml`
do.

### Legal tab: where it searches

By default (`legal.retrieval.source: "corpus"` in `config/config.yaml`) the
Legal tab runs its full pipeline over the bulk Spanish legal corpus: the BOE laws and
regulations and EU law (`legal.corpus.categories`; add `regional_laws` for autonomous-community
law), built on Kaggle or locally and installed with
`python scripts/legal_data/install_corpus.py legal_corpus_vectordb.zip` into
`data/legal_corpus_vectordb`. Each question is first planned (issues, governing
laws -- by rank and number or abbreviation, "LAU", "art. 1902 CC" -- and articles), then searched
(dense + BM25), reranked and answered. The law PDFs in `legal_txt/` are not used.
Set `source: "signed_index"` to answer from the signed index below instead.

### Legal tab: analyzing a case folder

Pick **Case analysis** as the Mode and a **📂 Browse case folder** button appears next to it.
Choose the folder that holds the case's files (subfolders included) and the analysis starts right
away; anything typed in the message box goes along as the case description. Every PDF, DOCX,
PPTX, XLSX, image and .txt file in the folder is uploaded (other files are listed as not
analyzed; at most 25 documents, by name). Each one is then read on the server: parsed, OCR'd and
cleaned. If they all fit the case file's budget (`legal.pipeline.case_material_max_tokens`) they
go to the model whole. If not, the model first writes a factual digest of each document (parties,
key facts, dates, deadlines, amounts) and the work file is written from the digests, so no
document is dropped (`src/aicodex/legal/pipeline.py`, `_case_documents_material`). A file that
can't be read is named in the work file's notes, and the audit log lists every document and
digest.

### Choosing the model size (12B / 31B)

A **Model size** selector at the top of the page, shared by General GPT and Legal GPT, picks the
model every request from either tab runs on: **LLM 12 Billion** (the default) or **LLM 31 Billion**.
The page shows only those labels, never model names. Each size is an entry under `model_sizes` in
`config/config.yaml` (its key, `12B` / `31B`, is what requests carry):

| Size | Shown as | Chat and Legal model | Translation model | Knowledge cutoff |
|---|---|---|---|---|
| 12B | LLM 12 Billion | `gemma4:12b` | `translategemma:12b` | January 2025 |
| 31B | LLM 31 Billion | `gemma4:31b` | `translategemma:27b` | January 2025 |

The choice travels with each request (`model_size`) and is applied per job by
`src/aicodex/llm/client.py` (`model_size()`), so two users can run different sizes at the same
time. It needs the Ollama backend: vLLM serves a single model, so under vLLM the selector is shown
but disabled. `scripts/setup.*` pull the 12B tags when the host has 12GB+ of RAM and the 31B tags
with 32GB+. A size whose model isn't pulled answers with Ollama's "model not found" error;
`ollama pull <tag>` fixes it.

### Refreshing the page stops its running requests

Refreshing or closing the page cancels everything that page still had running on the server:
chat turns, rewrites, slide and document generation, Legal questions and case analyses. The
reloaded page then starts fresh. Each request carries the page's session id (the
`X-Client-Session` header). Gradio's unload event then calls `POST /api/sessions/{session}/cancel`
(`src/aicodex/api/routes_session.py`), which cancels those jobs and with them their in-flight
calls to the model server.

### Chatting with the lawyer's case files (`legal_data/`)

The lawyer's own files -- demandas, contestaciones, sentencias, contracts, actas de la vista,
e-mails, letters, exhibits -- go in `legal_data/` at the project root, one subfolder per matter.
They are chunked and indexed into their own store (`data/legal_case_files_vectordb`), separate from
the law corpus, and then the user can chat with them:

```bash
pip install -e ".[legal]"
python scripts/ingest_case_files.py                    # index new/changed files (re-run after adding files)
python scripts/ingest_case_files.py index --dry-run    # classify + chunk only, write nothing
python scripts/ingest_case_files.py show "legal_data/Cohen v. Levi/claim.pdf"   # inspect one file's chunks
python scripts/ingest_case_files.py chat --matter "Cohen v. Levi"
python scripts/ingest_case_files.py ask "When was the termination notice sent?"
```

**In the app:** Legal GPT tab → Mode **Case files chat**. Pick a **Matter** (or All matters) and ask;
follow-up questions keep the conversation, and the Citations panel shows each cited file, page,
paragraph and the passage itself. **🔄 Index legal_data** indexes files added since the last run
(the same as running the script). Files indexed by the script while the app is running are picked up
on the next question, no restart needed.

Answers come from the Legal tab's model (`legal.orchestrator`), only from the retrieved excerpts,
with every statement cited to file, page and paragraph; parties' allegations are attributed to the
party, not stated as fact. Settings: `legal.case_files` in `config/config.yaml`. Everything in
`legal_data/` except its README is git-ignored -- case files are privileged and must not be pushed.

**Chunking strategy** (`src/aicodex/legal/case_files.py`):

| Principle | What it does |
|---|---|
| Structure before size | The unit is the numbered paragraph / clause / speaker turn / e-mail -- what a lawyer cites ("para. 14 of the claim", "clause 7.2"). Units are never split unless one alone exceeds the budget (then by sentence); chunks are packed from whole units up to ~450 tokens |
| Per document type | Each file is classified (pleading, judgment/decision, affidavit, contract, transcript, e-mail, letter, exhibit, memo, other) from its name and opening, Spanish and English. Pleadings/judgments/affidavits split on numbered paragraphs ("1.", "PRIMERO.-", "a)") and headings (a heading starts a new chunk); contracts on clauses (cláusulas, estipulaciones) and sub-clauses; transcripts on speaker turns with each question kept with its answer (P:/R:, Pregunta/Respuesta, Q/A); e-mail threads one message at a time, each chunk repeating its sender, recipients, date and subject; everything else by paragraph |
| Context header on every chunk | Matter, file, document type, document date, page range, section heading and paragraph numbers, embedded with the text -- the claim and the defence describe the same events in similar words, and the header is what tells them apart and what the answer cites |
| Pages kept | PDF pages (OCRed scans too) per chunk, for "p. 4" citations. Word's auto-numbered paragraphs get their numbers back |
| Small overlap | ~60 tokens of whole units between consecutive chunks of the same document (never across e-mails or sections) |
| Hybrid retrieval, small-to-big | bge-m3 dense search + BM25 (names, case numbers, amounts, dates), fused and reranked (bge-reranker-v2-m3); then the chunks right before and after the best hits are added, within a 5k-token evidence budget |

Re-runs skip unchanged files (content hash); changing the chunk settings re-chunks everything;
`index --prune` removes files deleted from the folder.

### Legal tab: Tribunal Constitucional case law

Alongside whichever source answers the question, the Legal tab draws on the Tribunal
Constitucional's sentencias: the `constitutional_court` category of the same corpus
(`fetch_constitutional_court.py`, then `vectorize.py --categories constitutional_court
--export-lexical`), searched by `src/aicodex/legal/caselaw.py` (dense + BM25, reranked, one excerpt
per judgment, capped by `caselaw_top_k` / `caselaw_max_tokens`). The excerpts go into the prompt
as their own `<case_law>` block, labelled as judgments the model may cite (STC number and ECLI)
but not treat as statute text. `legal.corpus.caselaw_category: null` turns it off.

### Legal tab: building the signed Spanish-law index

With `legal.retrieval.source: "signed_index"`, the Legal tab answers only from
sources that were staged, reviewed and signed into its local index, so it has
nothing to cite until you add some.
It never fetches from the web: get official texts yourself, within each
site's terms of use.

```bash
pip install -e ".[legal]"
export AICODEX_LEGAL_BUNDLE_KEY=<long random secret>   # same value for the API process
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
without a `.meta.json` gets one derived from its text (law name and number, the bulletin's
date -- "BOE núm. 7, de 8 de enero de 2000" -- or the title's date, statute or regulation,
BOE or regional bulletin). Check it, fix `effective_date_start` in particular, and re-run to
re-index. Scanned PDFs are refused (OCR them first).

**Superseding is automatic:** approving a newer version of a law (same law,
later `effective_date_start`) sets the older version's end date to the day
before the new one takes effect and marks it `amended`. Both stay searchable
for questions about past dates. `ingest_legal.py retract <law_id> <date>`
removes a version and re-opens the one before it.

**Amending laws are linked to the laws they amend.** Each article or final
disposition of an amending law ("Modificación de la Ley 29/1994 ...", "Se modifica
el artículo 9 ..., que queda redactado como sigue: «...»") is tagged with the law
(by rank and number) and the articles it changes, and each quoted article it
inserts gets its own chunk under both numbers. When an answer
draws on a provision that a later indexed law amended, the model is told the
amendment and its effective date -- the old text isn't discarded, because a
lawyer usually needs the version in force when the facts happened -- and an
answer citing a section the amendment itself changed escalates. Run
`python scripts/ingest_legal.py retag` once for laws indexed before this existed.

**Only verified statements reach the answer.** After the redraft, a sentence
whose cited source doesn't state it (or isn't a source the research memorandum
established) is removed. If nothing cited is left, no answer is given and the
question escalates. A partly supported citation stays, marked unverified.
Words the model wrote in another script ('報導', a Cyrillic 'о' in 'oктubre') are replaced word
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
the app directly -- see `gui/AICodex.bat`/`.command` or
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
aicodex-api   # FastAPI + Gradio UI on :8456 (UI at /ui)
```

**Ollama (any machine):**

```bash
# terminal 1 (often already running as a background service after install)
ollama serve

# terminal 2
set -a; source .env.local; set +a   # written by scripts/setup.sh; PowerShell: see scripts/setup.ps1's "Next steps"
aicodex-api
```

Or just use `gui/AICodex.bat`/`.command`, which does both of the above
from one window and shows live status.

### Hosting on Colab with a public link

`notebooks/colab_host_app.ipynb` runs the whole app (all tabs) on a free Colab T4 under Ollama and
puts it online through an ngrok tunnel (`NGROK_AUTHTOKEN` secret). Put `legal_corpus_vectordb.zip`
(from `notebooks/kaggle_spanish_corpus_build.ipynb`) on Google Drive and set `CORPUS_ZIP` to use the
Legal tab. The link has no login, so share it with care. Setup steps are at the top of the notebook.

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
- `native_hebrew.docx` -- native-text Hebrew document to validate RTL
  language detection and handling without OCR (kept: the general tabs still support RTL)
- `mixed_language.docx` -- English/French/Arabic paragraphs in one document,
  to validate per-segment (not per-document) language detection

End-to-end OCR/translation/generation behavior requires the models from
step 2 and a running vLLM instance, and is exercised manually via the
Gradio UI rather than in the unit test suite.

### Legal evals (Spanish)

- `evals/legal/lau_2026_es.json` -- 16 questions on the Ley de Arrendamientos Urbanos (exact periods and
  amounts, exceptions, two unanswerable), gold answers taken from the current consolidated BOE text, with
  key-fact, evidence and trap patterns. `scripts/eval_retrieval.py` (retrieval only, seconds) and
  `scripts/eval_legal.py` (before/after RAG, judged) use it by default.
- `legal_txt/Evals/es_retrieval_seed.jsonl` -- 30 questions across CE, CC, CP, LEC, ET and LAU with their
  governing article, checked against the BOE text; `scripts/legal_data/benchmark_embeddings.py` scores
  embedding models (alone and with BM25) on it.

The judge is `legal.judge_model` (default `gpt-oss:20b`, overridden by `AICODEX_LEGAL_JUDGE_MODEL`): a
family other than the models under test, so no model grades its own answers. Have a Spanish lawyer review
and extend these sets before relying on scores from them.

## Non-goals

- No cloud OCR/translation/LLM API calls anywhere in the runtime path.
- No hardcoded Latin-script/English assumptions -- every OCR, cleaning, and
  chunking step is explicitly language-aware via `config/config.yaml`.
