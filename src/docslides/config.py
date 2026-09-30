"""Central configuration loader.

Everything model-path-, threshold-, and language-related lives in
`config/config.yaml`. Nothing downstream should hardcode a language list,
engine name, or sampling default -- it must come through here so the system
stays language- and environment-agnostic.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path(os.environ.get("DOCSLIDES_CONFIG", "config/config.yaml"))


class ThinkingDefaults(BaseModel):
    outline_generation: bool = True
    slide_fill: bool = False
    translation: bool = False
    chat_general: bool = True
    tone_rewrite: bool = True
    chunk_summary: bool = False
    legal_language_id: bool = False
    # The one Legal call that thinks: free-text notes, so no JSON grammar competes with the
    # reasoning, and the reasoning is logged (llm/trace.py). See legal/pipeline.analyze_question.
    legal_analysis: bool = True
    # Case mode's single call: the work file is free text too, so it can think the same way.
    legal_case_analysis: bool = True
    legal_research_memo: bool = False
    legal_draft: bool = False
    legal_citation_verification: bool = False
    legal_script_repair: bool = False
    legal_eval_baseline: bool = False
    legal_eval_judge: bool = False
    legal_eval_plan: bool = False
    legal_retrieval_plan: bool = False
    legal_eval_rewrite: bool = False
    legal_eval_label_check: bool = False
    legal_eval_scope: bool = False
    legal_eval_toc: bool = False
    legal_eval_extract: bool = False
    legal_eval_completeness: bool = False
    legal_eval_repair: bool = False


class SamplingDefaults(BaseModel):
    temperature: float = 0.6
    top_p: float = 0.92
    max_tokens: int = 2048


class LLMConfig(BaseModel):
    # "vllm" talks to vLLM's OpenAI-compatible /v1/chat/completions with its
    # guided_json/chat_template_kwargs extensions (NVIDIA GPU only). "ollama"
    # talks to Ollama's native /api/chat, which runs on CPU or whatever
    # acceleration the host exposes (CUDA/ROCm/Metal) -- see
    # src/docslides/llm/client.py for the protocol differences.
    backend: Literal["vllm", "ollama"] = "vllm"
    base_url: str
    model: str
    api_key: str = "not-needed"
    request_timeout_s: int = 600
    max_model_len: int = 8192
    guided_decoding_backend: str = "xgrammar"
    # False for a model with no thinking mode (Gemma 3): every call then goes out with thinking
    # off, instead of asking for it and relying on the server's 400 + a retry (llm/client.py).
    supports_thinking: bool = True
    thinking_defaults: ThinkingDefaults = Field(default_factory=ThinkingDefaults)
    default_sampling: SamplingDefaults = Field(default_factory=SamplingDefaults)


class LegalRetrievalConfig(BaseModel):
    # What the Legal tab searches: "corpus" = the bulk corpus (legal.corpus, legal/corpus_retrieval.py),
    # "signed_index" = the signed index of drop-in law PDFs and reviewed sources (vectordb_dir below).
    source: Literal["corpus", "signed_index"] = "corpus"
    vectordb_dir: str = "./data/legal_vectordb"
    embedding_model: str = "BAAI/bge-m3"
    # Where the bulk-corpus path runs the embedder and reranker: None = the library default (a GPU
    # when one is visible), "cpu", or "cuda" (loaded in fp16). legal/corpus_retrieval.warm_up_retrieval
    # falls back to the CPU when the GPU hasn't room beside the LLM.
    device: str | None = None
    # BM25 over the corpus's lexical copy (lexical_<category>.jsonl), fused with the dense search on
    # the corpus path: exact terms of art ("עושק", "פקודת הנזיקין") that embeddings rank loosely.
    corpus_lexical: bool = True
    top_k: int = 6
    fetch_k: int = 24  # candidates considered before dedupe + MMR cut them to top_k
    mmr_lambda: float = 0.7  # 1.0 = pure relevance; lower = more diversity
    # Cosine distance above which a hit counts as low-relevance; fewer than
    # `min_relevant_chunks` hits under it triggers the thin-coverage escalation.
    low_relevance_distance: float = 0.55
    min_relevant_chunks: int = 2
    max_cross_refs: int = 4
    # Precision -- what actually reaches the model. A small model handed a full
    # top_k of loosely related provisions drops or garbles the one that answers,
    # and every extra token costs CPU time. None disables each limit.
    #   relevance_margin: search hits farther than this (cosine distance) from
    #     the best hit are cut; the best hit always stays.
    #   sibling_margin: another part of a split provision is added only if it is
    #     itself within this distance of the best hit.
    #   max_evidence_tokens: budget for all evidence (hits, sibling parts,
    #     cross-references), filled best-first; counted like chunk budgets.
    relevance_margin: float | None = 0.08
    sibling_margin: float | None = 0.12
    max_evidence_tokens: int | None = 5000
    # Hebrew-aware keyword search fused with the embedding ranking (legal/keyword.py).
    keyword_search: bool = True
    # Chunks fetched for each section number the question names ("סעיף 132א"); 0 disables.
    section_lookup_max: int = 2
    # Cross-encoder that reorders candidates and scores how directly each answers the
    # question (0-1). None, or a model that can't load, falls back to embedding distance.
    reranker_model: str | None = "BAAI/bge-reranker-v2-m3"
    rerank_candidates: int = 16
    rerank_max_length: int = 1024
    # Calibrated on the 16-question elections eval with scripts/eval_retrieval.py: every
    # answerable question's best score was >= 0.48, the unanswerable C2's 0.23.
    rerank_margin: float | None = 0.6  # keep hits scoring within this of the best one...
    rerank_floor: float = 0.1  # ...and at least this (the best hit always stays)
    min_rerank_score: float = 0.35  # best score under this = thin coverage (flag + prompt note)
    # Several laws "in play" (the model is told to answer per law or flag the ambiguity) when a
    # question names no law and hits from 2+ laws score at least this, within this of the best.
    ambiguity_min_score: float = 0.5
    ambiguity_margin: float = 0.3
    # A section the question names ('סעיף 25') found in another law counts at this lower
    # reranker score: the named number itself is the ambiguity.
    ambiguity_lookup_floor: float = 0.2


class LegalIngestionConfig(BaseModel):
    sources_dir: str = "./data/legal/sources"
    legal_txt_dir: str = "./legal_txt"
    uploads_dir: str = "./uploads"
    staging_dir: str = "./data/legal/staging"
    bundle_manifest: str = "./data/legal/signed_bundle.json"
    chunk_max_tokens: int = 500


class LegalPipelineConfig(BaseModel):
    max_memo_revisions: int = 2
    max_draft_revisions: int = 1
    entailment_concurrency: int = 4
    # Pass 0: the model reads the evidence against the question with thinking on and writes
    # notes the memorandum and the draft both start from. Its budget covers thinking + notes,
    # and is capped further to fit the orchestrator's max_model_len.
    analysis_pass: bool = True
    analysis_max_tokens: int = 4096
    # Case mode (legal/pipeline.run_case_turn): the work file's budget covers thinking and all eight
    # sections; the case material (typed text + attached documents) is capped so that it, the
    # evidence and that budget fit the orchestrator's max_model_len together.
    case_max_tokens: int = 6144
    case_material_max_tokens: int = 3000


class LegalCorpusConfig(BaseModel):
    """The bulk corpus (legal_txt/ JSONL, vectorized by scripts/legal_data/vectorize.py): its own
    Chroma store, one collection per category (<collection_prefix>_<category>), separate from the
    signed, reviewed index the Legal tab answers from. Same embedding model as retrieval."""

    vectordb_dir: str = "./data/legal_corpus_vectordb"
    collection_prefix: str = "israeli_law_corpus"
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 64
    embed_batch_size: int = 64
    # Folding ך ם ן ף ץ into their regular forms makes spellings bge-m3 never saw: off for the
    # embedded text by default. The lexical copy (for a keyword index) is always folded.
    fold_final_letters_for_embedding: bool = False
    # What the Legal tab searches when legal.retrieval.source is "corpus".
    categories: list[str] = Field(default_factory=lambda: ["laws", "procedural_rules"])
    top_k: int = 12  # chunks kept after reranking, before the max_evidence_tokens budget
    # Records that aren't Israeli law in force inside Israel, matched against the title and never
    # retrieved: military commanders' orders for Judea & Samaria, the Jordanian criminal law, and
    # drafts/proposals. (Knesset laws *about* those areas -- חוק להסדרת ההתיישבות ביהודה והשומרון,
    # the Jordan peace-treaty law -- don't match.) The 29 Sept eval review traced wrong answers to
    # each of these.
    exclude_title_patterns: list[str] = Field(default_factory=lambda: [
        r"^(?:צו|תקנות) בדבר .*יהודה וה?שומרון",
        r"\(יהודה וה?שומרון\)\s*\(מס",
        r"החוק הפלילי הירדני",
        r"^הצעת ",
    ])
    # Repealed records (status "repealed": the 1984 Civil Procedure Regulations, 66 laws) are left
    # out of free search, and come back only when the retrieval plan names that very law -- a
    # question about what applied *before* can still reach them.
    repealed_only_when_named: bool = True
    # Retrieval and answering variants from the 30 Sept root-cause review, each off by default so a
    # dev-split run can measure it alone (scripts/legal_data/eval_run.py answer --variant ...).
    # whole_sections: a retrieved chunk of a split section brings the rest of that section, merged
    # in order (the per-section cap cut exceptions and provisos off 49 half-credit answers).
    whole_sections: bool = False
    whole_section_max_tokens: int = 1500  # a longer section keeps only its retrieved parts
    # toc_navigation: the model reads the table of contents of the top laws retrieved and names the
    # sections that govern; those are added whole (29.5 points were lost on the right law, wrong section).
    toc_navigation: bool = False
    toc_laws: int = 2
    toc_max_sections: int = 4
    # cross_references: "סעיף 5", "בכפוף לסעיף 12" in a retrieved section fetch those sections of the same law.
    cross_references: bool = False
    cross_reference_max: int = 3
    # regulation_cap: at most this many regulation (procedural_rules) hits unless the plan names a
    # regulation; None = no cap. They took 24% of the context slots in the 29 Sept 27B run.
    regulation_cap: int | None = None
    # law_grouped_context: the context lists each law's excerpts together, in section order.
    law_grouped_context: bool = False
    # extract_then_answer: a first call quotes the governing provisions and lists every element,
    # condition and exception; the answer call must cover them. completeness_check: a last call
    # compares the answer with that list and restores anything left out.
    extract_then_answer: bool = False
    completeness_check: bool = False
    # Doctrine cards (legal_txt/doctrine_cards.jsonl): short, labelled notes on case-law doctrines and
    # amendment timelines the statute text doesn't state, matched by keyword. Drafts until a lawyer
    # has reviewed them; None = off.
    doctrine_cards_path: str | None = None
    doctrine_cards_max: int = 2


class LegalConfig(BaseModel):
    """Legal tab: grounded RAG over Israeli law (see src/docslides/legal/).
    `orchestrator` (Qwen) does research, drafting and every verification
    step. It is a full `LLMConfig` -- an independent deployment from the
    general `llm:` section."""

    orchestrator: LLMConfig
    retrieval: LegalRetrievalConfig = Field(default_factory=LegalRetrievalConfig)
    ingestion: LegalIngestionConfig = Field(default_factory=LegalIngestionConfig)
    pipeline: LegalPipelineConfig = Field(default_factory=LegalPipelineConfig)
    corpus: LegalCorpusConfig = Field(default_factory=LegalCorpusConfig)
    audit_dir: str = "./data/legal/audit"
    # Evals only (legal/evaluation.get_judge_client): the grading model, served by the same Ollama
    # as the orchestrator. A family other than the models under test (qwen3, gemma3), so neither
    # is graded by itself. None = the orchestrator grades its own answers.
    # DOCSLIDES_LEGAL_JUDGE_MODEL overrides it.
    judge_model: str | None = "gpt-oss:20b"


class LegalDataSourcesConfig(BaseModel):
    """The only hosts the corpus fetchers (scripts/legal_data/) may contact."""

    knesset_odata: str = "https://knesset.gov.il/OdataV4/ParliamentInfo"
    knesset_tables: list[str] = Field(default_factory=list)
    # Requested name -> the table that holds it in OData V4 (KNS_DocumentLaw is KNS_DocumentIsraelLaw).
    knesset_table_aliases: dict[str, str] = Field(default_factory=dict)
    knesset_pdf_host: str = "fs.knesset.gov.il"
    wikisource_dumps: str = "https://dumps.wikimedia.org/hewikisource"
    wikisource_dump_date: str = "latest"  # "latest" = newest completed dump, or e.g. "20260901"
    wikisource_wiki: str = "https://he.wikisource.org/wiki/"
    supreme_court_repo: str = "LevMuchnik/SupremeCourtOfIsrael"
    supreme_court_revision: str = "main"
    supreme_court_file: str = "cases_all.parquet"


class LegalDataPrivacyConfig(BaseModel):
    """Supreme Court exclusions (legal_data/supreme_court.py). Regexes over Hebrew text with
    ״/׳ already folded to ASCII quotes. Empty lists make fetch_supreme_court.py refuse to run."""

    publication_restriction_patterns: list[str] = Field(default_factory=list)
    family_case_prefixes: list[str] = Field(default_factory=list)
    family_subject_values: list[str] = Field(default_factory=list)
    anonymized_party_patterns: list[str] = Field(default_factory=list)
    topic_keywords: list[str] = Field(default_factory=list)
    topic_scan_chars: int = 5000
    public_body_patterns: list[str] = Field(default_factory=list)


class LegalDataConfig(BaseModel):
    """Corpus acquisition (scripts/legal_data/fetch_*.py) -- see scripts/legal_data/README.md."""

    contact_email: str = ""  # goes in the User-Agent; the fetchers refuse to run without it
    output_dir: str = "./legal_txt"
    min_interval_s: float = 1.0  # per host
    max_retries: int = 6
    backoff_base_s: float = 2.0
    backoff_max_s: float = 300.0
    timeout_s: float = 120.0
    sources: LegalDataSourcesConfig = Field(default_factory=LegalDataSourcesConfig)
    # label -> title prefixes (compared after law_names.normalize_name) that must be in procedural_rules.
    required_regulations: dict[str, list[str]] = Field(default_factory=dict)
    # KNS_DocumentSecondaryLaw.GroupTypeDesc values whose PDFs hold a regulation's own text; only
    # those are text-extracted for regulations Wikisource lacks. Empty = none (confirm from a sample).
    secondary_text_group_types: list[str] = Field(default_factory=list)
    judgment_types: list[str] = Field(default_factory=lambda: ["פסק-דין"])
    privacy: LegalDataPrivacyConfig = Field(default_factory=LegalDataPrivacyConfig)


class PathsConfig(BaseModel):
    data_dir: str
    upload_dir: str
    output_dir: str
    cache_dir: str
    template_potx: str
    glossary_dir: str

    def ensure_exist(self) -> None:
        for p in [self.data_dir, self.upload_dir, self.output_dir, self.cache_dir, self.glossary_dir]:
            Path(p).mkdir(parents=True, exist_ok=True)


class LanguagesConfig(BaseModel):
    supported: list[str]
    rtl: list[str]
    spacy_models: dict[str, str] = Field(default_factory=dict)
    stanza_models: dict[str, str] = Field(default_factory=dict)
    fasttext_model_path: str = ""

    def is_rtl(self, lang: str) -> bool:
        return lang in self.rtl


class ScriptEngineConfig(BaseModel):
    languages: list[str]
    primary: str
    fallback_cpu: str
    fallback_gpu: str


class GPUArbiterConfig(BaseModel):
    poll_interval_s: float = 1.0
    max_wait_s: float = 30.0


class TesseractConfig(BaseModel):
    lang_codes: dict[str, str]


class OCRConfig(BaseModel):
    confidence_threshold: float = 0.80
    engines_by_script: dict[str, ScriptEngineConfig]
    tesseract: TesseractConfig
    gpu_arbiter: GPUArbiterConfig = Field(default_factory=GPUArbiterConfig)

    def script_for_language(self, lang: str) -> ScriptEngineConfig:
        for script_cfg in self.engines_by_script.values():
            if lang in script_cfg.languages:
                return script_cfg
        raise KeyError(f"No OCR engine routing configured for language '{lang}'")


class ScanDetectionConfig(BaseModel):
    min_chars_per_page_area: float = 0.0008


class CleaningConfig(BaseModel):
    header_footer_repetition_threshold: float = 0.6
    max_chunk_tokens: int = 1500
    mask_categories: list[str] = Field(default_factory=list)


class TranslationConfig(BaseModel):
    glossary_max_terms: int = 200


class SlidesConfig(BaseModel):
    default_bullet_count: int = 4
    max_bullets_per_slide: int = 6
    layouts: list[str]
    schema_retry_attempts: int = 2


class ProfessionalismLevel(BaseModel):
    label: str
    temp_cap: float


class CreativityLevel(BaseModel):
    label: str
    temperature: float
    top_p: float


class ToneControlConfig(BaseModel):
    professionalism_levels: dict[int, ProfessionalismLevel]
    creativity_levels: dict[int, CreativityLevel]


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_output: bool = Field(default=True, alias="json")
    # One JSON line per LLM call (llm/trace.py): call site, the model's reasoning, its output,
    # why it stopped, token counts -- and the full prompt when llm_trace_prompts is on.
    # None disables the file; the Legal audit log still records each turn's calls.
    llm_trace_dir: str | None = "./data/llm_trace"
    llm_trace_prompts: bool = True

    model_config = {"populate_by_name": True}


class VLLMLaunchConfig(BaseModel):
    quantization: str = "awq"
    gpu_memory_utilization: float = 0.90
    port: int = 8000


class AppConfig(BaseModel):
    llm: LLMConfig
    legal: LegalConfig
    legal_data: LegalDataConfig = Field(default_factory=LegalDataConfig)
    vllm_launch: VLLMLaunchConfig = Field(default_factory=VLLMLaunchConfig)
    paths: PathsConfig
    languages: LanguagesConfig
    ocr: OCRConfig
    scan_detection: ScanDetectionConfig = Field(default_factory=ScanDetectionConfig)
    cleaning: CleaningConfig
    translation: TranslationConfig
    slides: SlidesConfig
    tone_control: ToneControlConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# Lets the setup scripts / GUI launcher / docker-compose.portable.yml switch
# between the vLLM and Ollama backends without maintaining a second full
# config.yaml -- config/config.yaml stays the single source of truth for
# everything else (languages, OCR routing, sampling defaults, ...). The Legal
# tab's orchestrator is an independent deployment (see LegalConfig) with its
# own override set, so it can be pointed at Ollama separately from -- or
# together with -- the general `llm:` section. *_REQUEST_TIMEOUT_S sets how long
# one model call may take (a CPU-only host needs far more than config.yaml's
# GPU-sized limits). *_MAX_MODEL_LEN sets the context
# window (Ollama's num_ctx; under vLLM it must not exceed the served
# --max-model-len), so a host can match it to its memory.
_LLM_ENV_OVERRIDES = {
    "DOCSLIDES_LLM_BACKEND": "backend",
    "DOCSLIDES_LLM_BASE_URL": "base_url",
    "DOCSLIDES_LLM_MODEL": "model",
    "DOCSLIDES_LLM_MAX_MODEL_LEN": "max_model_len",
    "DOCSLIDES_LLM_SUPPORTS_THINKING": "supports_thinking",
    "DOCSLIDES_LLM_REQUEST_TIMEOUT_S": "request_timeout_s",
}
_LEGAL_ORCHESTRATOR_ENV_OVERRIDES = {
    "DOCSLIDES_LEGAL_ORCHESTRATOR_BACKEND": "backend",
    "DOCSLIDES_LEGAL_ORCHESTRATOR_BASE_URL": "base_url",
    "DOCSLIDES_LEGAL_ORCHESTRATOR_MODEL": "model",
    "DOCSLIDES_LEGAL_ORCHESTRATOR_MAX_MODEL_LEN": "max_model_len",
    "DOCSLIDES_LEGAL_ORCHESTRATOR_SUPPORTS_THINKING": "supports_thinking",
    "DOCSLIDES_LEGAL_ORCHESTRATOR_REQUEST_TIMEOUT_S": "request_timeout_s",
}


def _env_overrides(env_map: dict[str, str]) -> dict[str, str]:
    return {key: os.environ[env] for env, key in env_map.items() if env in os.environ}


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    llm_overrides = _env_overrides(_LLM_ENV_OVERRIDES)
    if llm_overrides:
        raw = {**raw, "llm": {**raw.get("llm", {}), **llm_overrides}}

    orchestrator_overrides = _env_overrides(_LEGAL_ORCHESTRATOR_ENV_OVERRIDES)
    if orchestrator_overrides:
        legal = raw.get("legal", {})
        legal = {**legal, "orchestrator": {**legal.get("orchestrator", {}), **orchestrator_overrides}}
        raw = {**raw, "legal": legal}

    if os.environ.get("DOCSLIDES_LEGAL_DATA_CONTACT_EMAIL"):
        legal_data = {**(raw.get("legal_data") or {}), "contact_email": os.environ["DOCSLIDES_LEGAL_DATA_CONTACT_EMAIL"]}
        raw = {**raw, "legal_data": legal_data}

    if "DOCSLIDES_LLM_TRACE_DIR" in os.environ:  # "" disables the trace file
        trace_dir = os.environ["DOCSLIDES_LLM_TRACE_DIR"] or None
        raw = {**raw, "logging": {**(raw.get("logging") or {}), "llm_trace_dir": trace_dir}}

    return raw


@lru_cache(maxsize=1)
def get_config(path: str | Path | None = None) -> AppConfig:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = _apply_env_overrides(_load_yaml(cfg_path))
    cfg = AppConfig.model_validate(raw)
    cfg.paths.ensure_exist()
    for legal_dir in (
        cfg.legal.retrieval.vectordb_dir,
        cfg.legal.ingestion.sources_dir,
        cfg.legal.ingestion.legal_txt_dir,
        cfg.legal.ingestion.uploads_dir,
        cfg.legal.ingestion.staging_dir,
        cfg.legal.audit_dir,
    ):
        Path(legal_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def reload_config(path: str | Path | None = None) -> AppConfig:
    get_config.cache_clear()
    return get_config(path)
