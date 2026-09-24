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
    legal_query_normalization: bool = False
    legal_research_memo: bool = False
    legal_draft: bool = False
    legal_citation_verification: bool = False
    legal_hebrew_polish: bool = False
    legal_equivalence_check: bool = False
    legal_eval_baseline: bool = False
    legal_eval_judge: bool = False
    canon_orchestration: bool = True
    canon_answer: bool = False


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
    thinking_defaults: ThinkingDefaults = Field(default_factory=ThinkingDefaults)
    default_sampling: SamplingDefaults = Field(default_factory=SamplingDefaults)


class DictaTierConfig(BaseModel):
    """One user-selectable DictaLM size for the Legal tab's Hebrew stages
    (query normalization + final polish). `min_memory_gb` drives the tab's
    "insufficient RAM" suggestion banner -- see legal/resources.py."""

    label: str
    min_memory_gb: float
    llm: LLMConfig


class LegalRetrievalConfig(BaseModel):
    vectordb_dir: str = "./data/legal_vectordb"
    embedding_model: str = "BAAI/bge-m3"
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
    max_evidence_tokens: int | None = 2500
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
    max_polish_attempts: int = 2
    entailment_concurrency: int = 4


class LegalConfig(BaseModel):
    """Legal tab: grounded RAG over Israeli law (see src/docslides/legal/).
    `orchestrator` (Qwen) does research, drafting and every verification
    step; `dicta_tiers` are the DictaLM sizes the user can pick between for
    the Hebrew-only normalization/polish stages. Each model is a full
    `LLMConfig` -- independent deployments from the general `llm:` section."""

    orchestrator: LLMConfig
    dicta_tiers: dict[str, DictaTierConfig]
    default_dicta_tier: str = "heavy"
    retrieval: LegalRetrievalConfig = Field(default_factory=LegalRetrievalConfig)
    ingestion: LegalIngestionConfig = Field(default_factory=LegalIngestionConfig)
    pipeline: LegalPipelineConfig = Field(default_factory=LegalPipelineConfig)
    audit_dir: str = "./data/legal/audit"

    def dicta_tier(self, key: str | None) -> tuple[str, DictaTierConfig]:
        key = key or self.default_dicta_tier
        if key not in self.dicta_tiers:
            raise KeyError(f"Unknown DictaLM tier '{key}' (configured: {sorted(self.dicta_tiers)})")
        return key, self.dicta_tiers[key]


class CanonConfig(BaseModel):
    """Canon GPT tab: RAG over the CIC and CCEO canon-law codes -- see
    src/docslides/canon/ and scripts/ingest_canon_law.py. `generation` is a
    full `LLMConfig`, independent from the general `llm:` section like
    `LegalConfig`'s models are, but defaults to pointing at the same
    deployment since no dedicated fine-tuned model is needed here."""

    vectordb_dir: str = "./data/canon_vectordb"
    embedding_model: str = "BAAI/bge-m3"
    top_k: int = 8
    generation: LLMConfig


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

    model_config = {"populate_by_name": True}


class VLLMLaunchConfig(BaseModel):
    quantization: str = "awq"
    gpu_memory_utilization: float = 0.90
    port: int = 8000


class AppConfig(BaseModel):
    llm: LLMConfig
    legal: LegalConfig
    canon: CanonConfig
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
# tab's orchestrator and DictaLM tiers are independent deployments (see
# LegalConfig) and get their own override triples so they can be pointed at
# Ollama separately from -- or together with -- the general `llm:` section.
# DOCSLIDES_LEGAL_HEBREW_* keeps its old name (setup scripts write it) and
# now targets the Heavy tier; DOCSLIDES_LEGAL_DICTA_LIGHT_* targets Light.
_LLM_ENV_OVERRIDES = {
    "DOCSLIDES_LLM_BACKEND": "backend",
    "DOCSLIDES_LLM_BASE_URL": "base_url",
    "DOCSLIDES_LLM_MODEL": "model",
}
_LEGAL_ORCHESTRATOR_ENV_OVERRIDES = {
    "DOCSLIDES_LEGAL_ORCHESTRATOR_BACKEND": "backend",
    "DOCSLIDES_LEGAL_ORCHESTRATOR_BASE_URL": "base_url",
    "DOCSLIDES_LEGAL_ORCHESTRATOR_MODEL": "model",
}
_LEGAL_HEBREW_ENV_OVERRIDES = {
    "DOCSLIDES_LEGAL_HEBREW_BACKEND": "backend",
    "DOCSLIDES_LEGAL_HEBREW_BASE_URL": "base_url",
    "DOCSLIDES_LEGAL_HEBREW_MODEL": "model",
}
_LEGAL_DICTA_LIGHT_ENV_OVERRIDES = {
    "DOCSLIDES_LEGAL_DICTA_LIGHT_BACKEND": "backend",
    "DOCSLIDES_LEGAL_DICTA_LIGHT_BASE_URL": "base_url",
    "DOCSLIDES_LEGAL_DICTA_LIGHT_MODEL": "model",
}
_CANON_GENERATION_ENV_OVERRIDES = {
    "DOCSLIDES_CANON_BACKEND": "backend",
    "DOCSLIDES_CANON_BASE_URL": "base_url",
    "DOCSLIDES_CANON_MODEL": "model",
}


def _env_overrides(env_map: dict[str, str]) -> dict[str, str]:
    return {key: os.environ[env] for env, key in env_map.items() if env in os.environ}


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    llm_overrides = _env_overrides(_LLM_ENV_OVERRIDES)
    if llm_overrides:
        raw = {**raw, "llm": {**raw.get("llm", {}), **llm_overrides}}

    orchestrator_overrides = _env_overrides(_LEGAL_ORCHESTRATOR_ENV_OVERRIDES)
    tier_overrides = {
        "heavy": _env_overrides(_LEGAL_HEBREW_ENV_OVERRIDES),
        "light": _env_overrides(_LEGAL_DICTA_LIGHT_ENV_OVERRIDES),
    }
    if orchestrator_overrides or any(tier_overrides.values()):
        legal = raw.get("legal", {})
        if orchestrator_overrides:
            legal = {**legal, "orchestrator": {**legal.get("orchestrator", {}), **orchestrator_overrides}}
        tiers = dict(legal.get("dicta_tiers", {}))
        for tier_key, overrides in tier_overrides.items():
            if overrides and tier_key in tiers:
                tier = tiers[tier_key]
                tiers[tier_key] = {**tier, "llm": {**tier.get("llm", {}), **overrides}}
        raw = {**raw, "legal": {**legal, "dicta_tiers": tiers}}

    canon_overrides = _env_overrides(_CANON_GENERATION_ENV_OVERRIDES)
    if canon_overrides:
        canon = raw.get("canon", {})
        canon = {**canon, "generation": {**canon.get("generation", {}), **canon_overrides}}
        raw = {**raw, "canon": canon}

    return raw


@lru_cache(maxsize=1)
def get_config(path: str | Path | None = None) -> AppConfig:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = _apply_env_overrides(_load_yaml(cfg_path))
    cfg = AppConfig.model_validate(raw)
    cfg.paths.ensure_exist()
    Path(cfg.canon.vectordb_dir).mkdir(parents=True, exist_ok=True)
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
