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
    legal_orchestration: bool = True
    legal_hebrew_analysis: bool = True
    legal_verification: bool = False


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


class LegalConfig(BaseModel):
    """Model configs for the Legal tab's 3-step pipeline (see
    src/docslides/legal/pipeline.py): `orchestrator` plans the research and
    reformulates the question in Hebrew and later verifies/translates the
    final answer; `hebrew_analyst` (DictaLM) does the actual legal analysis,
    in Hebrew. Each is a full `LLMConfig` -- they're independent deployments
    (different model, possibly different host/backend) from the general
    `llm:` section above, not a variant of it."""

    orchestrator: LLMConfig
    hebrew_analyst: LLMConfig


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
# tab's orchestrator/hebrew_analyst are independent deployments (see
# LegalConfig) and get their own override triples so they can be pointed at
# Ollama separately from -- or together with -- the general `llm:` section.
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


def _env_overrides(env_map: dict[str, str]) -> dict[str, str]:
    return {key: os.environ[env] for env, key in env_map.items() if env in os.environ}


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    llm_overrides = _env_overrides(_LLM_ENV_OVERRIDES)
    if llm_overrides:
        raw = {**raw, "llm": {**raw.get("llm", {}), **llm_overrides}}

    orchestrator_overrides = _env_overrides(_LEGAL_ORCHESTRATOR_ENV_OVERRIDES)
    hebrew_overrides = _env_overrides(_LEGAL_HEBREW_ENV_OVERRIDES)
    if orchestrator_overrides or hebrew_overrides:
        legal = raw.get("legal", {})
        if orchestrator_overrides:
            legal = {**legal, "orchestrator": {**legal.get("orchestrator", {}), **orchestrator_overrides}}
        if hebrew_overrides:
            legal = {**legal, "hebrew_analyst": {**legal.get("hebrew_analyst", {}), **hebrew_overrides}}
        raw = {**raw, "legal": legal}

    return raw


@lru_cache(maxsize=1)
def get_config(path: str | Path | None = None) -> AppConfig:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = _apply_env_overrides(_load_yaml(cfg_path))
    cfg = AppConfig.model_validate(raw)
    cfg.paths.ensure_exist()
    return cfg


def reload_config(path: str | Path | None = None) -> AppConfig:
    get_config.cache_clear()
    return get_config(path)
