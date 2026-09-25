# Israeli legal corpus: acquisition and vectorization

Builds an offline corpus of Israeli legislation and Supreme Court rulings under `./legal_txt/`,
normalized to JSONL, and vectorizes it into a Chroma store **separate from the Legal tab's signed
index** (`data/legal_corpus_vectordb`, one collection per category). Code:
`src/docslides/legal_data/`.

## Sources (the only hosts contacted)

| Source | What | License / attribution |
|---|---|---|
| Knesset OData V4 `knesset.gov.il/OdataV4/ParliamentInfo` (+ PDFs on `fs.knesset.gov.il`) | law and regulation registry | Knesset open data (see knesset.gov.il) |
| Hebrew Wikisource Open Law Book, via the **official Wikimedia dump** (`dumps.wikimedia.org/hewikisource`) | consolidated text of laws and regulations | CC BY-SA 4.0 — attribution in every record |
| Hugging Face `LevMuchnik/SupremeCourtOfIsrael` (2022 snapshot) | Supreme Court rulings | OpenRAIL (use restrictions apply; read the dataset card) |
| ISCD `iscd.huji.ac.il/data` | case metadata files, as-is | per the ISCD site |

The dump was chosen over the `gkour/israeli_law` Hugging Face dataset: that dataset was last
updated on 7 Jan 2026, while the dump is refreshed monthly (20260901 at the time of writing).

## Setup

```bash
pip install -e ".[legal,legal-data]"      # or: pip install -r requirements-legal-data.txt
```

Set your contact address in `config/config.yaml` (`legal_data.contact_email`) or in the
`DOCSLIDES_LEGAL_DATA_CONTACT_EMAIL` environment variable. It goes in the User-Agent, and the
fetchers refuse to send requests without it.

## Run order

```bash
python scripts/legal_data/fetch_metadata.py           # 1. Knesset registry tables (+ ISCD files)
python scripts/legal_data/fetch_laws.py               # 2. laws (downloads the Wikisource dump)
python scripts/legal_data/fetch_procedural_rules.py   # 3. regulations (reuses the dump) + registry PDFs
python scripts/legal_data/fetch_supreme_court.py      # 4. rulings (1.5 GB parquet)
python scripts/legal_data/vectorize.py --dry-run      # 5. chunk counts and size; then embed (Kaggle)
```

Every fetcher takes:
- `--dry-run`: only metadata requests (robots.txt, sizes, counts). Reports what it would download
  and the estimated size and time; writes nothing but a manifest.
- `--sample N`: a small real run, written under `legal_txt/_sample/`. The laws and regulations
  samples stream the start of the dump instead of downloading it; the Supreme Court sample reads
  rows through the Hub's rows API.
- `--output DIR`: another output root.

Re-running is incremental: files whose hash matches are skipped without a request, the dump and
parquet are re-downloaded only when their published hash changes, and OData tables are fetched
from their last `LastUpdatedDate` watermark (`fetch_metadata.py --full` fetches everything again,
which is also the only way to see rows deleted at the source).

## Crawler behaviour

- robots.txt is read for every host before its first request, and honoured. A missing robots.txt
  allows everything; an unreachable one, or one that answers 401/403, blocks the host.
- At most one request per second per host (`legal_data.min_interval_s`), longer if robots.txt sets
  a Crawl-delay. 429/5xx back off exponentially, honouring Retry-After.
- **A 401/403, an access-denied page, or a host that keeps resetting connections stops the run**
  (exit code 2) with the reason in the manifest. Nothing retries or works around a block.
- OData queries never contain `;` or `aggregate(` (the Knesset firewall rejects them).
- UTF-8 everywhere. Files are named by numeric ids, never by Hebrew titles.

## Output

```
legal_txt/
  laws/              laws.jsonl, unmatched_registry_laws.json, raw/wikisource/<dump>.xml.bz2, raw/pdf/
  procedural_rules/  procedural_rules.jsonl, coverage_report.json, raw/pdf/<SecondaryLawId>/<DocumentId>.pdf
  supreme_court/     supreme_court-00001.jsonl ... (50,000 per shard), filter_report.json, raw/cases_all.parquet
  metadata/          knesset/<Table>.jsonl + _state.json, iscd/
  _manifests/        <run>_<script>.json + .log (sources, licenses, robots decisions, files + hashes, counts),
                     _downloads.json (download ledger)
  _sample/           the same layout, for --sample runs
```

These folders are gitignored, and `scripts/ingest_legal_txt.py` skips them: the curated PDFs
directly in `legal_txt/` stay the Legal tab's own index.

### Record fields (`src/docslides/legal_data/records.py`)

All records: `id, source, source_url, license, attribution, category, title, authority_level
(basic_law | law | ordinance | regulation | judgment | decision), status (in_force | repealed |
unknown), status_source, effective_date, retrieved_at, language, text, text_sha256,
quality{encoding: ok|repaired|flagged, extraction: ok|low|n/a, notes}`.

Laws and regulations add `law_id` (KNS_IsraelLaw.Id, or KNS_SecondaryLaw.Id for regulations),
`law_id_source, knesset_name, is_basic_law, authorizing_law_ids, sections[{kind, number, title,
division, chapter, subchapter, note, intro, subsections[{label, text}], text}], section_numbers,
publication_date, latest_amendment_date, gazette_citations[{ref, name, url}], wikisource_*`, and
`pdf_files`.

Rulings add `case_number, court, judges, decision_date, doc_type, technical, case_name, year,
pages, source_case_id, anonymized`.

## Category rules

**laws**: Open Law Book pages titled חוק, חוק-יסוד (or flagged IsBasicLaw) and פקודה/פקודת
(Mandate-era ordinances included). Pages join the registry by their `ח:מאגר` id, or failing that
by normalized name. Status comes from `LawValidityDesc`; without it, a repeal marker on the page;
otherwise `unknown`. `effective_date` is `ValidityStartDate`, else `PublicationDate`, else the
title's year (noted in `quality.notes`).

**procedural_rules**: every other Open Law Book page (תקנות, צו, כללים, הודעה ...), joined to
`KNS_SecondaryLaw` by name, plus the `KNS_DocumentSecondaryLaw` PDFs. `coverage_report.json` checks
the Civil Procedure Regulations 2018, the criminal procedure regulations and the court-fee
regulations (`legal_data.required_regulations`), and the manifest warns about any that are missing.
A regulation Wikisource lacks gets a record from its registry PDFs only for the group types in
`legal_data.secondary_text_group_types`. That list is empty by default, because the registry mostly
holds committee background material. Pick the right types from a run's manifest (`pdf_group_types`).

**supreme_court**: keeps `Type = פסק-דין` and decisions with `Technical = false`. Strict privacy
exclusions (`legal_data.privacy`): publication restrictions anywhere in the text, family case types,
anonymized parties (פלוני/אלמוני), and adoption, minors, sexual-offense or family keywords in the
case name, case type and first 5,000 characters. `filter_report.json` counts what each rule removed
(first rule and every rule), and lists the case types and departments seen, so the family mapping
can be checked. `--anonymize` replaces natural-person parties with `[צד N]` and keeps the State,
public bodies and organizations. It is a heuristic, not a guarantee.

## Vectorization

`vectorize.py` reuses the repo's RAG stack:
- **Embeddings:** bge-m3 (`legal.retrieval.embedding_model`).
- **Store:** Chroma with cosine distance.
- **Laws and regulations chunking:** `legal/chunking.chunk_sections`. Every chunk carries the law
  name, headings and section number, with `legal.corpus.chunk_overlap_tokens` of overlap between
  the parts of a long section.
- **Rulings chunking:** paragraph groups with overlap, each prefixed with
  `court | case number | case name | date`.

The displayed text keeps the original. Embeddings use the niqqud-stripped, quote-unified copy.
Final letters are folded only in the lexical copy (`--export-lexical`), because folded spellings
are unseen by bge-m3. `legal.corpus.fold_final_letters_for_embedding` changes that.

Metadata for filtering: `category, authority_level, status, effective_ymd, decision_ymd`
(integers YYYYMMDD, since Chroma range filters need numbers), plus `law_id, court, case_number,
doc_type, record_id, section_number`.

Each category gets its own store: `data/legal_corpus_vectordb/<category>/`, holding collection
`israeli_law_corpus_<category>` and its state database. Indexing is incremental by record hash,
and state commits after every batch, so an interrupted run resumes.

On Kaggle (`notebooks/kaggle_legal_corpus_vectorize.ipynb`):
1. `python scripts/legal_data/package_for_kaggle.py` zips only the JSONL.
2. Upload the zip as a private Kaggle Dataset and run the notebook, one category group per session
   (Kaggle keeps 20 GB of output and runs 12 h per session).
3. A category too big for one session: split it with `--shard 0/2`, `1/2` ..., then combine the
   downloaded stores with `vectorize.py --merge-from <shard dir> --categories supreme_court`. The
   merge copies the vectors; nothing is re-embedded.
4. `--dry-run` reports chunk counts, index size and GPU hours before you commit to a full run.
   If the rulings turn out too big, `--chunk-tokens 1000` halves the chunk count.

## Known gaps (Sept 2026)

- `KNS_DocumentIsraelLaw` is empty in OData V4, so there are no registry PDFs for laws.
  `fetch_laws.py --wikisource-pdfs` downloads the fs.knesset.gov.il PDFs the pages cite. Links to
  other sites (e.g. olaw.org.il) are recorded, never fetched.
- `KNS_Law` doesn't exist in OData V4 and is reported as unavailable.
- `iscd.huji.ac.il` reset connections during planning. If it keeps doing so, download the metadata
  files by hand and run `fetch_metadata.py --only iscd --import-dir <folder>`.
- A regulation's status is usually unknown: OData doesn't record its validity.
