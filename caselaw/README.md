# israeli_caselaw_ingest

Turns Israeli Supreme Court judgments decided **before 1 January 2022** into cleaned documents,
structure-aware retrieval chunks, embeddings, and offline search indexes (LanceDB or FAISS, plus
BM25) for an offline RAG legal assistant.

- Colab notebook: [`../notebooks/colab_ingest.ipynb`](../notebooks/colab_ingest.ipynb)
- CLI: `python -m israeli_caselaw_ingest download|filter|clean|chunk|bm25|embed|validate|peek|all`
- Settings: [`config.yaml`](config.yaml) (every threshold, model and path; `sample_n` for dry runs)

## Data provenance

The one and only source is the Hugging Face dataset
[`LevMuchnik/SupremeCourtOfIsrael`](https://huggingface.co/datasets/LevMuchnik/SupremeCourtOfIsrael):
a 2022 crawl of the official Supreme Court website (supreme.court.gov.il), about 751K documents in
one file, `cases_all.parquet` (~1.5 GB), mostly Hebrew. The pipeline downloads that file and model
weights and makes no other network call. It does no scraping, uses no commercial databases (Nevo,
Takdin), and draws on no third-party republishers.

Every chunk keeps `source_dataset` and a `source_url`. The URL is the official download link
rebuilt from the dataset's `Path` and `FileName`, in the form
`https://supremedecisions.court.gov.il/Home/Download?path=HebrewVerdicts\…&fileName=…&type=4`.
It is there **for citation only**; the pipeline never fetches it. The rebuild is best-effort, so
check one or two links against the site before relying on them.

**Keep source attribution** in anything built on these outputs, and cite the dataset's curators:

> Muchnik, L., Yahav, I., Nevo, A., Chriqui, E., & Shektov, S. (2023). *Supreme Court of Israel
> judgments dataset* (LevMuchnik/SupremeCourtOfIsrael). Hugging Face.

### Known gaps

- **Supreme Court only.** No district, magistrate, labour or other courts.
- **Nothing from 2022 onward.** The filter keeps decisions strictly before `max_decision_date`
  (2022-01-01), and the crawl ends in 2022 anyway.
- **Older coverage is thin.** There are few documents from before 2000, and fewer the further back you go.
- **A frozen 2022 snapshot.** Later redactions, corrections, anonymisations or publication bans
  are not reflected. Check a judgment on the official site before quoting it.
- **Decision dates are resolved heuristically.** Some rows carry bogus dates, such as 1920 on a
  2003 case. Rows whose date came from a fallback field are counted in the filter report, and
  examples are written to `reports/date_fixes.jsonl`.
- **Header parsing is best-effort.** When it fails, the dataset metadata is used instead. The
  success rate is in the validation report.

## What each stage does

| stage | output | notes |
|---|---|---|
| `download` | `<root>/cache/cases_all.parquet` | `hf_hub_download`; skipped when present with the right size |
| `filter` | `filtered/part-*.parquet`, `reports/filter_report.md`, `reports/schema.md` | streamed with `iter_batches` (5,000 rows); columns checked against the file |
| `clean` | `documents.parquet` | mojibake repair, NFC, bidi marks removed, spaces collapsed, boilerplate removed, header parsed |
| `chunk` | `chunks/year=YYYY/part-*.parquet` | structure-aware; 400–600 tokens, 80-token overlap, bge-m3 tokenizer |
| `bm25` | `bm25/shard-*/` | bm25s shards (50k chunks each) over prefix + text |
| `embed` | `embeddings/shard-*.npy`, `lancedb/` or `faiss/`, `<root>/models/` | float16 shards act as checkpoints; estimate first; `--yes` needed above 3 h |
| `validate` | `reports/validation_report.md` | counts, histograms, top citations, date checks, sanity queries |

### Filter rules

These are the defaults; all of them are in `config.yaml`.

- **Decision date before 2022-01-01.** The date is taken from `meta_verdict_dt`, then
  `VerdictsDt`, then `VerdictDt`, then `Year`.
  - `VerdictDt` holds a local midnight stored with a UTC offset (`2003-07-08T19:00` means 9 July),
    so it is rounded to the nearest midnight.
  - A date before 1948, or before `meta_case_dt`, is treated as bogus and the next field is tried.
  - Rows whose date can't be resolved at all are dropped and counted.
- **Type** is one of `פסק-דין`, `פד"י`, `פסקי דין באנגלית`, `תקצירים`.
- **`החלטה`** is kept only when it is non-technical (`Technical == False`) and at least 1,500 characters long.
- **No empty text and no exact duplicates.** Duplicates are found by a hash of the normalised text.

### Cleaning

- **Mojibake:** a line is repaired when its letters are mostly Latin-1 or Greek look-alikes of
  cp1255 Hebrew, for example `á áéú äîùôè` or `α αιϊ δξωτθ`. It is re-encoded as latin-1, cp1252
  or cp1253, then decoded as cp1255, and the version with the most Hebrew letters is kept. ftfy
  is the fallback. The method used is recorded in `encoding_repair`.
- **Header:** the parser finds the court line, the panel (`בפני:`), the parties on each side of
  `נגד`, and where the body starts (after `פסק-דין` / `החלטה`, or at paragraph 1). The parsed
  judges and parties are used when parsing succeeds; otherwise the dataset metadata is used.

### Chunking

1. The body is split into segments at numbered paragraphs (`12.`), Hebrew-letter items (`ב.`),
   section headings (`רקע`, `דיון והכרעה`, `סוף דבר`, …) and blank lines.
2. Segments are split into sentences, which are packed into chunks.
3. A chunk closes at a paragraph boundary once it holds 400 tokens, or before the sentence that
   would push it past 600.
4. The next chunk repeats up to 80 tokens of whole sentences from the end of the previous one.
5. A sentence is only ever split in the middle when that single sentence is longer than 600
   tokens; it is then cut at word boundaries.
6. The operative section (`סוף דבר` / `אשר על כן` / `התוצאה היא`) always starts a new chunk,
   flagged `is_holding` with `section = "holding"`.

Chunk 0 of every document is its `header` (court, citation, panel, parties). Every chunk also
has a `context_prefix` such as `[בג"ץ 5856/03 | פסק-דין | 2003-11-30 | דורנר, נאור, חיות]`,
stored separately, which is embedded together with the text. Citations are extracted into
`citations`: case numbers like `ע"א 1234/99`, `בג"ץ 5856/03`, `ת"א (ת"א) 1234/98` and
`12345-06-15`, and report references like `פ"ד מט(3) 355`.

## Output schema

`chunks/year=YYYY/part-*.parquet` is zstd-compressed, with one file per document batch per year:

| column | type | |
|---|---|---|
| `chunk_id` | string | `{document_hash}:{chunk_index}`: stable across runs |
| `doc_id` | string | dataset `Id` |
| `case_citation`, `case_name` | string | e.g. `בג"ץ 5856/03` |
| `doc_type` | string | `Type` |
| `proceeding_type` | string | `meta_inyan_nm` (ע"א, ע"פ, רע"א, בג"ץ …) |
| `division` | string | `meta_mador_nm` |
| `decision_date` | date | resolved as above |
| `judges`, `parties` | list<string> | |
| `lang` | string | `he` / `en` |
| `section` | string | `header` / `body` / `holding` |
| `is_holding` | bool | |
| `chunk_index`, `n_chunks` | int | |
| `char_start`, `char_end` | int | offsets into the document's cleaned `text` (0/0 for a header built from metadata) |
| `token_count` | int | in the embedding model's tokens |
| `context_prefix` | string | |
| `text` | string | |
| `citations` | list<string> | |
| `source_url`, `source_dataset` | string | |
| `ingested_at` | timestamp | |

`documents.parquet` has one row per kept document. It holds the full cleaned `text` plus the
metadata above, `header_text`, `header_parsed`, `body_start` and `encoding_repair`, so chunks can
be regenerated with `chunk --force` without cleaning again.

`lancedb/` holds the chunk columns plus a float16 `vector` column (bge-m3: 1024 dimensions,
L2-normalised; search it with cosine). `embeddings/` holds the same vectors as `.npy` shards,
with `id_map.parquet` giving each vector's chunk id.

## Running it

In Colab, open `notebooks/colab_ingest.ipynb`. It mounts Drive, installs the dependencies, runs a
2,000-document dry run, then the full run and validation, and zips the artifacts. Every cell
resumes where it stopped after a disconnect.

Locally:

```bash
cd caselaw
pip install -r requirements.txt && pip install -e .
python -m pytest -q
python -m israeli_caselaw_ingest all --root ./out --sample-n 2000 --embed --yes   # dry run
python -m israeli_caselaw_ingest peek --root ./out --sample-n 2000 --n 5
python -m israeli_caselaw_ingest all --root ./out --embed                          # full run
```

Each stage writes `state/<stage>.done.json` and skips work already done. `--force` redoes a
stage; changing a setting it depends on also redoes it. A sample run writes to
`<root>/sample_<n>/`, apart from the full run.

### Sizing a free Colab session

These are estimates to check against the dry run.

- **Chunk count:** the filtered corpus is a few hundred thousand documents, which gives a few
  million chunks.
- **Embedding time:** bge-m3 on a T4 in fp16 manages a few hundred chunks per second, so the
  full corpus takes several sessions. `embed` prints its own estimate, needs `--yes` above
  3 hours, and resumes shard by shard.
- **Vector store:** use LanceDB, which is the default. FAISS `IndexHNSWFlat` keeps every vector
  in RAM as float32, which is more than a free session's 12 GB for the full corpus.

## Using the outputs offline

```python
from israeli_caselaw_ingest.config import load_config
from israeli_caselaw_ingest.search import Searcher

s = Searcher(load_config(root="/path/to/israeli_caselaw"))
s.hybrid('עקרון המידתיות', k=5)        # [(chunk_id, score)], BM25 + dense via reciprocal-rank fusion
s.bm25('ע"א 6821/93', k=5)
s.rows([cid for cid, _ in s.dense("חובת ההנמקה", 5)])
```

Dense search loads the model saved under `<root>/models/`, so copy that folder along with the
indexes. Nothing is fetched at query time.
