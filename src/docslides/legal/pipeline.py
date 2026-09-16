"""Three-step Legal tab pipeline: an orchestrator model (Qwen) plans the
research and reformulates the question in Hebrew, a Hebrew legal-domain
model (DictaLM, an Israeli-law specialist) analyzes it in Hebrew, and the
orchestrator verifies the result and translates the final answer back into
whichever language the user asked in. Every prompt sent to the Hebrew
analyst is in Hebrew -- it's only ever asked a Hebrew question and only
ever answers one; citations and relevant-law names are left in Hebrew (their
natural language) all the way through, since they're shown in their own
panel rather than translated prose.
"""

from __future__ import annotations

from docslides.llm.client import ChatMessage, LLMCallSite, QwenClient, SamplingParams
from docslides.llm.schemas import HebrewLegalFindings, LegalFinalAnswer, LegalQueryPlan

_ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are the orchestrator for a legal-research assistant specializing in Israeli law. "
    "Given the conversation so far, reformulate the user's latest question into a precise, "
    "self-contained legal-research query written in Hebrew -- this will be handed verbatim to "
    "a Hebrew-language Israeli-law analysis model with no other context, so it must include "
    "anything from earlier turns the model would need, and be phrased the way a real Israeli "
    "legal research query would be. Do not answer the question yourself."
)

_HEBREW_ANALYST_SYSTEM_PROMPT = (
    "אתה עוזר משפטי מומחה בדין הישראלי. ענה על השאלה המשפטית הבאה בעברית בלבד, "
    "באופן מדויק, מבוסס ומנומק. ציין בנפרד את כל האסמכתאות (פסקי דין ומקורות משפטיים) "
    "ואת כל החוקים/הסעיפים הרלוונטיים עליהם התבססת."
)


async def analyze_legal_query(client: QwenClient, messages: list[ChatMessage]) -> LegalQueryPlan:
    system = ChatMessage(role="system", content=_ORCHESTRATOR_SYSTEM_PROMPT)
    return await client.complete_json(
        [system, *messages],
        LLMCallSite("legal_orchestration"),
        schema=LegalQueryPlan,
        sampling=SamplingParams(temperature=0.2, max_tokens=512),
    )


async def run_hebrew_legal_analysis(client: QwenClient, hebrew_query: str) -> HebrewLegalFindings:
    system = ChatMessage(role="system", content=_HEBREW_ANALYST_SYSTEM_PROMPT)
    return await client.complete_json(
        [system, ChatMessage(role="user", content=hebrew_query)],
        LLMCallSite("legal_hebrew_analysis"),
        schema=HebrewLegalFindings,
        sampling=SamplingParams(temperature=0.2, max_tokens=4096),
    )


async def verify_and_finalize(
    client: QwenClient,
    original_message: str,
    user_lang: str,
    hebrew_query: str,
    findings: HebrewLegalFindings,
) -> LegalFinalAnswer:
    system = ChatMessage(
        role="system",
        content=(
            "You are the verification stage of a legal-research assistant. You are given the user's "
            "original question, the Hebrew legal query it was translated into, and a Hebrew-language "
            "legal analysis (with citations and relevant laws) produced by an Israeli-law specialist "
            "model. Verify the analysis actually answers the original question, then write the final "
            f"answer in this language (ISO 639-1 code): '{user_lang}'. Keep citations and relevant law "
            "names exactly as given, in Hebrew -- translate only the prose answer."
        ),
    )
    user = ChatMessage(
        role="user",
        content=(
            f"Original question:\n{original_message}\n\n"
            f"Hebrew query sent to the legal model:\n{hebrew_query}\n\n"
            f"Hebrew legal analysis:\n{findings.analysis_hebrew}\n\n"
            f"Citations:\n{findings.citations}\n\n"
            f"Relevant laws:\n{findings.relevant_laws}"
        ),
    )
    return await client.complete_json(
        [system, user],
        LLMCallSite("legal_verification"),
        schema=LegalFinalAnswer,
        sampling=SamplingParams(temperature=0.2, max_tokens=2048),
    )
