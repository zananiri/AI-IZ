# Spanish legal corpus: acquisition and vectorization

Builds an offline corpus of Spanish legislation (state and autonomous communities), EU law in Spanish and
Tribunal Constitucional case law under `./legal_txt/`, normalized to JSONL, and vectorizes it into a Chroma
store **separate from the Legal tab's signed index** (`data/legal_corpus_vectordb`, one collection per
category). Code: `src/aicodex/legal_data/`.

## Sources (the only hosts contacted)

| Source | What | Method | License / attribution |
|---|---|---|---|
| **BOE** datos abiertos — `www.boe.es/datosabiertos/api/legislacion-consolidada` | Consolidated text of every state law and regulation, and of the autonomous communities' law the BOE consolidates (~12,400 acts), in force and repealed | The BOE's official JSON/XML API: the catalogue (paged, 10,000 per request) gives every act's rank, scope, department, dates, in-force flag and **last-update stamp**; `/id/{id}/texto` gives the text as `<bloque>`s (one per article, heading, disposition or annex) with every historical `<version>` — the last one is the wording in force | Reuse authorised with attribution (Ley 37/2007; BOE *aviso legal*). Consolidated texts are informative; only the BOE publication is official — stated in every record |
| **Tribunal Constitucional** — `hj.tribunalconstitucional.es` (Sistema HJ) | Sentencias (default), autos and declaraciones since 1980, with ECLI, chamber, judges, BOE publication, procedure and the court's síntesis | One public page per resolution id (`/es/Resolucion/Show/{n}`); the newest id is found by galloping + bisection | Reuse with attribution (Ley 37/2007). Private parties anonymized by default (see below) |
| **EU Publications Office** — CELLAR `publications.europa.eu/resource/celex/{CELEX}` + SPARQL `publications.europa.eu/webapi/rdf/sparql` | The EU acts a Spanish practice needs (RGPD, AI Act, DSA, Bruselas I bis, Roma I/II, consumer directives, the Charter and the Treaties …), in Spanish, newest consolidated version | SPARQL finds the latest consolidated CELEX (`0…-YYYYMMDD`); content negotiation (`Accept: application/xhtml+xml`, `Accept-Language: spa`) returns the Spanish XHTML. EUR-Lex's own HTML pages sit behind a bot challenge and are not used | © European Union, reuse authorised (Decision 2011/833/EU); only the Official Journal text is authentic |

**Not crawled, on purpose:** CENDOJ (`poderjudicial.es`, Tribunal Supremo and the other courts). Its terms
forbid automated bulk downloading without CGPJ authorization. Judgments you obtain from it (individually or
under a CGPJ reuse licence) can go in `legal_data/` (case files) or in `legal_txt/` as PDFs for the signed
index. The autonomous-community bulletins (DOGC, BOJA, BOCM …) are covered through the BOE's consolidation
of regional law; a bulletin PDF can likewise be dropped into `legal_txt/`.

## Setup

```bash
pip install -e ".[legal,legal-data]"      # or: pip install -r requirements-legal-data.txt
```

Set your contact address in `config/config.yaml` (`legal_data.contact_email`) or in the
`AICODEX_LEGAL_DATA_CONTACT_EMAIL` environment variable. It goes in the User-Agent, and the fetchers refuse
to send requests without it.

## Run order

```bash
python scripts/legal_data/fetch_all.py --dry-run                 # counts, requests and hours per source
python scripts/legal_data/fetch_laws.py                          # 1. BOE: laws, regulations, regional_laws (~3.5 h first run)
python scripts/legal_data/fetch_eu_law.py                        # 2. EU law (~1 min)
python scripts/legal_data/fetch_constitutional_court.py          # 3. TC sentencias (~9 h first run, resumable)
python scripts/legal_data/check_corpus_coverage.py               # the core codes are there, in force, with articles
python scripts/legal_data/vectorize.py --dry-run                 # chunk counts, index size, GPU hours
python scripts/legal_data/vectorize.py --export-lexical          # embed + the BM25 copy (GPU or Kaggle)
```

`fetch_all.py` runs the three fetchers in order. Useful variants:

```bash
python scripts/legal_data/fetch_laws.py --categories laws                       # only rank-of-law norms (~1,650 acts)
python scripts/legal_data/fetch_laws.py --ids BOE-A-1889-4763 BOE-A-2000-323    # specific acts (Código Civil, LEC)
python scripts/legal_data/fetch_laws.py --since 20260101                        # acts updated since a date
python scripts/legal_data/fetch_laws.py --exclude-repealed --with-subjects      # in force only; add materias
python scripts/legal_data/fetch_constitutional_court.py --doc-types SENTENCIA AUTO --from-id 25000
python scripts/legal_data/fetch_eu_law.py --celex 32016R0679 32024R1689
```

Every fetcher takes `--dry-run` (metadata only; reports what it would fetch and how long it would take),
`--sample N` (a small real run under `legal_txt/_sample/`) and `--output DIR`.

**Incremental by design.** BOE acts are cached as normalized records under `legal_txt/boe/cache/`, keyed
by the catalogue's `fecha_actualizacion`: a re-run fetches only acts whose consolidated text changed (a few
dozen a week). TC ids are cached once seen (including missing ids and skipped types), so later runs fetch
only new resolutions. Schedule `fetch_all.py` weekly to keep the corpus current; `vectorize.py` then
re-embeds only records whose content hash changed.

## Crawler behaviour

- robots.txt is read for every host before its first request and honoured, **with wildcards** (`*`, `$`,
  longest match wins — RFC 9309). The standard library's parser ignores wildcards, and the BOE lists the
  documents it has withdrawn (right-to-be-forgotten requests) exactly that way (`Disallow: /*BOE-A-2026-4920$`);
  those ids are collected once per run and never fetched.
- At most one request per second per host (`legal_data.min_interval_s`), longer if robots.txt sets a
  Crawl-delay. 429/5xx back off exponentially, honouring Retry-After.
- **A 401/403, an access-denied page, or a host that keeps resetting connections stops the run** (exit
  code 2) with the reason in the manifest. Nothing retries or works around a block.
- XML is parsed with entity resolution and network access off. UTF-8 everywhere; mojibake (`artÃ­culo`) is
  repaired and noted.

## Output

```
legal_txt/
  laws/                 laws-00001.jsonl ...                  Constitución, leyes orgánicas, leyes, RDLeg, RDL, codes, treaties
  regulations/          regulations-00001.jsonl ...           reales decretos, órdenes, resoluciones, circulares
  regional_laws/        regional_laws-00001.jsonl ...         the autonomous communities' laws, decree-laws and regulations
  eu_law/               eu_law.jsonl
  constitutional_court/ constitutional_court-00001.jsonl ...  + cache/
  boe/cache/            one normalized record per act (.json.gz), keyed by its update stamp
  _manifests/           <run>_<script>.json + .log (sources, licenses, robots decisions, counts, warnings)
  _sample/              the same layout, for --sample runs
  Evals/                es_retrieval_seed.jsonl -- 30 questions with gold BOE articles (benchmark_embeddings.py)
```

These folders are gitignored, and `scripts/ingest_legal_txt.py` skips them.

### Record fields (`src/aicodex/legal_data/records.py`)

All records: `id, source, source_url, license, attribution, category, title, authority_level, status
(in_force | repealed | unknown), status_source, effective_date, retrieved_at, language, text, text_sha256,
quality{encoding, extraction, notes}`.

`authority_level` follows the Spanish hierarchy of norms: `constitution, organic_law, law,
legislative_decree, decree_law, international_treaty, regulation, order, resolution, circular,
regional_law, regional_decree_law, regional_regulation, eu_treaty, eu_regulation, eu_directive,
eu_decision, judgment, decision`.

Legislation adds `law_id` (BOE identifier or CELEX), `rank, official_number, department, scope (estatal |
autonomico | ue), region, eli_url, journal, journal_number, disposition_date, publication_date,
latest_amendment_date, source_updated_at, consolidated, subjects`, and `sections[{kind (section |
disposition | schedule | preamble), number ("14", "14 bis", "Disposición adicional primera"), title,
division (Libro > Título), chapter (Capítulo), subchapter (Sección), note (the BOE's footnote on the current
wording: "Se modifica por …"), intro, subsections[{label, text}] (apartados), text, version_date}]`.

TC judgments add `ecli, case_number, court, chamber, judges, decision_date, doc_type, resolution_number,
procedure_type, boe_publication, summary, year, source_case_id, anonymized`.

### Privacy (Tribunal Constitucional)

The TC already anonymizes minors and victims. With `legal_data.constitutional_court.anonymize_parties`
(default **on**), every other natural person introduced as "don/doña …" becomes `[persona N]`
consistently within a resolution; the magistrates (from the ficha técnica, and any "magistrado don …") are
kept. A research corpus needs the doctrine, not the parties (RGPD art. 5.1.c). It is a heuristic, not a
guarantee: review before publishing anything built from it.

## Vectorization

`vectorize.py` reuses the repo's RAG stack:
- **Embeddings:** `legal.retrieval.embedding_model` (see *Embedding model* below). Instruction-tuned models
  get their query prompt automatically (`src/aicodex/rag/embedding.py`).
- **Store:** Chroma with cosine distance, one store per category: `data/legal_corpus_vectordb/<category>/`,
  collection `spanish_law_corpus_<category>`.
- **Legislation chunking:** `legal/chunking.chunk_sections` — one chunk per article (split at its apartados
  when long), each starting with its breadcrumb `Ley … > TÍTULO I > CAPÍTULO II > Artículo 14.2 — Título`,
  with `legal.corpus.chunk_overlap_tokens` of overlap between the parts of a long article. Dispositions and
  annexes are chunked the same way.
- **Judgments:** paragraph groups with overlap, each prefixed `Tribunal Constitucional | SENTENCIA 84/2025 |
  Recurso de amparo 6194-2020 | date | ECLI`.
- The displayed text keeps the original; the embedded copy unifies quotes and spaces; the lexical copy
  (`--export-lexical`, for the BM25 index) is also lowercased and accent-folded.

Metadata for filtering: `category, authority_level, status, effective_ymd, decision_ymd` (integers
YYYYMMDD), plus `law_id, rank, scope, region, ecli, chamber, resolution_number, procedure_type, case_number,
doc_type, record_id, section_number, source_url`. For example, only state law in force:
`--where '{"$and": [{"status": "in_force"}, {"scope": "estatal"}]}'`.

Indexing is incremental by record hash and state commits after every batch, so an interrupted run resumes.
On Kaggle: `package_for_kaggle.py` zips the JSONL; `--shard i/n` splits a big category across sessions and
`--merge-from` combines them without re-embedding. `install_corpus.py` installs a built store locally.

## Embedding model: benchmark before you change it

`scripts/legal_data/benchmark_embeddings.py` compares embedding models (alone and fused with BM25, as the
Legal tab searches) on the articles of the fetched corpus, against `legal_txt/Evals/es_retrieval_seed.jsonl`:
30 Spanish questions whose governing article (CE, CC, CP, LEC, ET, LAU) was checked against the BOE text.

```bash
python scripts/legal_data/benchmark_embeddings.py --models BAAI/bge-m3 Qwen/Qwen3-Embedding-0.6B
```

The results of the run made while building AI-Codex, and the model chosen from them, are in the main
README (*Spanish pipeline choices*). Changing `legal.retrieval.embedding_model` requires re-running
`vectorize.py --rebuild`: vectors from different models can't share a store.
