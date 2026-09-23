"""Canon GPT's 3-step RAG pipeline: an orchestrator model reformulates the
user's question into a search query and guesses which code(s) it's about,
plain-Python retrieval fetches the actual source text from the local vector
store (canon/retrieval.py, populated offline by scripts/ingest_canon_law.py),
and the orchestrator answers grounded strictly in what was retrieved.

Unlike legal/pipeline.py's 3-step shape, there is only one model in the
loop here (no separate domain-specialist model) and citations are never
model-invented -- api/routes_canon.py builds the citations panel straight
from each retrieved chunk's own metadata (real vatican.va source URLs).
"""

from __future__ import annotations

from docslides.canon.retrieval import RetrievedChunk, retrieve
from docslides.llm.client import ChatMessage, LLMCallSite, QwenClient, SamplingParams
from docslides.llm.schemas import CanonFinalAnswer, CanonQueryPlan

_ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are the orchestrator for a canon-law research assistant covering the Code of Canon Law (CIC "
    "1983, Latin Church) and the Code of Canons of the Eastern Churches (CCEO 1990). Given the "
    "conversation so far, reformulate the user's "
    "latest question into a precise, self-contained search query suitable for embedding-based retrieval "
    "over that text, and name which code(s) it is most likely about -- leave it empty if genuinely "
    "unclear. Do not answer the question yourself."
)

_ANSWER_SYSTEM_PROMPT_TEMPLATE = (
    "You are the answering stage of a canon-law research assistant. You are given the user's original "
    "question and a set of retrieved canon-law provisions, each already labeled with its code "
    "and canon number. Answer using ONLY the provided provisions -- if they don't actually cover "
    "the question, say so plainly rather than relying on outside knowledge. Write the answer in this "
    "language (ISO 639-1 code): '{user_lang}'. CCEO provisions are in Latin -- translate their "
    "substance into the answer language, but keep canon numbers exactly as given so they match the "
    "citations panel."
)


async def plan_canon_query(client: QwenClient, messages: list[ChatMessage]) -> CanonQueryPlan:
    system = ChatMessage(role="system", content=_ORCHESTRATOR_SYSTEM_PROMPT)
    return await client.complete_json(
        [system, *messages],
        LLMCallSite("canon_orchestration"),
        schema=CanonQueryPlan,
        sampling=SamplingParams(temperature=0.2, max_tokens=512),
    )


def retrieve_context(plan: CanonQueryPlan) -> list[RetrievedChunk]:
    return retrieve(plan.search_query, codes=plan.likely_codes or None)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no matching provisions were found in the local canon-law index)"
    return "\n\n---\n\n".join(f"{c.breadcrumb or c.code}\n{c.text}" for c in chunks)


async def answer_from_context(
    client: QwenClient, original_message: str, user_lang: str, chunks: list[RetrievedChunk]
) -> CanonFinalAnswer:
    system = ChatMessage(
        role="system", content=_ANSWER_SYSTEM_PROMPT_TEMPLATE.format(user_lang=user_lang)
    )
    user = ChatMessage(
        role="user",
        content=f"Question:\n{original_message}\n\nRetrieved provisions:\n\n{_format_context(chunks)}",
    )
    return await client.complete_json(
        [system, user],
        LLMCallSite("canon_answer"),
        schema=CanonFinalAnswer,
        sampling=SamplingParams(temperature=0.2, max_tokens=2048),
    )
