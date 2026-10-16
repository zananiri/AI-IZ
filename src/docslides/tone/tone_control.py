"""Shared tone-control component for any rewrite request (emails, messages,
slide text, etc.) -- not hardcoded to a single use case.

Two independent sliders:
  * professionalism (1-5): shapes the SYSTEM PROMPT (salutation/sign-off
    style, contraction use, sentence complexity, directness vs hedging) and
    caps the usable sampling temperature.
  * creativity (1-5): shapes SAMPLING PARAMETERS only.

final_temperature = min(creativity_temperature, professionalism_temp_cap)
"""

from __future__ import annotations

from dataclasses import dataclass

from docslides.config import get_config
from docslides.llm.client import SamplingParams

PROFESSIONALISM_GUIDANCE: dict[int, str] = {
    1: (
        "Write in a casual tone: contractions are welcome, short punchy sentences, "
        "friendly and informal. Skip formal salutations/sign-offs -- a simple greeting "
        "and sign-off (or none) is fine."
    ),
    2: (
        "Write in a conversational tone: natural contractions, approachable phrasing, "
        "light and friendly but still coherent. A casual greeting and sign-off is fine."
    ),
    3: (
        "Write in a standard business tone: clear, polite, moderately formal. Minimal "
        "contractions. Use a conventional greeting and sign-off appropriate for a "
        "workplace audience."
    ),
    4: (
        "Write in a formal tone: no contractions, precise and structured sentences, "
        "measured and respectful phrasing, hedge claims appropriately, and use a formal "
        "salutation and sign-off."
    ),
    5: (
        "Write in an executive/legal register: no contractions, highly precise and "
        "unambiguous language, complex but well-structured sentences, maximal formality "
        "and directness where appropriate, formal salutation and sign-off, and careful "
        "hedging of any non-guaranteed claims."
    ),
}


@dataclass(frozen=True)
class ToneSettings:
    professionalism: int
    creativity: int

    def __post_init__(self) -> None:
        if not 1 <= self.professionalism <= 5:
            raise ValueError("professionalism must be between 1 and 5")
        if not 1 <= self.creativity <= 5:
            raise ValueError("creativity must be between 1 and 5")


def build_tone_system_prompt_fragment(tone: ToneSettings) -> str:
    return PROFESSIONALISM_GUIDANCE[tone.professionalism]


def resolve_sampling_params(tone: ToneSettings) -> SamplingParams:
    cfg = get_config().tone_control
    creativity_cfg = cfg.creativity_levels[tone.creativity]
    professionalism_cfg = cfg.professionalism_levels[tone.professionalism]

    final_temperature = min(creativity_cfg.temperature, professionalism_cfg.temp_cap)
    return SamplingParams(temperature=final_temperature, top_p=creativity_cfg.top_p)


def compose_rewrite_system_prompt(tone: ToneSettings, task_description: str) -> str:
    return (
        f"{task_description}\n\n"
        f"Tone guidance: {build_tone_system_prompt_fragment(tone)}\n"
        "Preserve the original meaning and all factual content; only adjust tone, register, "
        "and phrasing."
    )
