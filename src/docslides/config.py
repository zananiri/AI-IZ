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
from typing import Any

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


class SamplingDefaults(BaseModel):
    temperature: float = 0.6
    top_p: float = 0.92
    max_tokens: int = 2048


class LLMConfig(BaseModel):
    base_url: str
    model: str
    api_key: str = "not-needed"
    request_timeout_s: int = 600
    max_model_len: int = 8192
    guided_decoding_backend: str = "xgrammar"
    thinking_defaults: ThinkingDefaults = Field(default_factory=ThinkingDefaults)
    default_sampling: SamplingDefaults = Field(default_factory=SamplingDefaults)


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


@lru_cache(maxsize=1)
def get_config(path: str | Path | None = None) -> AppConfig:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = _load_yaml(cfg_path)
    cfg = AppConfig.model_validate(raw)
    cfg.paths.ensure_exist()
    return cfg


def reload_config(path: str | Path | None = None) -> AppConfig:
    get_config.cache_clear()
    return get_config(path)
