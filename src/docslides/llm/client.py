"""Thin client around the configured LLM backend: vLLM's OpenAI-compatible
/v1/chat/completions (NVIDIA GPU only) or Ollama's native /api/chat (CPU/
CUDA/ROCm/Metal -- the portable path). See `config.llm.backend`.

Responsibilities:
  * Toggle the model's "thinking" mode per call site, however the active backend
    exposes that knob (vLLM: chat_template_kwargs; Ollama: `think`).
  * Drive structured JSON decoding for every content-generation call --
    callers never regex-parse free text (vLLM: guided_json extra_body;
    Ollama: top-level `format` JSON Schema).
  * Validate JSON responses against a pydantic schema and retry with an
    error-correction prompt on failure. An output cut off at max_tokens --
    at temperature 0, almost always a loop repeating one sentence -- is
    retried fresh with non-greedy sampling instead.
  * Keep every request inside the context window: max_tokens is capped so
    prompt + output fit max_model_len (Ollama otherwise shifts the context
    mid-answer, silently dropping the system prompt and evidence).
  * Record every call -- reasoning, output, stop reason, tokens -- in the
    trace (llm/trace.py).
  * Stream chat turns and split reasoning from content for the UI, using
    each backend's own wire format (SSE + <think> tags for vLLM; newline-
    delimited JSON with a separate `thinking` field for Ollama).
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from docslides.config import LLMConfig, get_config
from docslides.llm import trace
from docslides.logging_setup import get_logger

logger = get_logger(__name__)

# Output-budget arithmetic (see LLMClient._fit_context). Deliberately pessimistic
# about Hebrew/Arabic, which subword tokenizers split finely. Fitted on the 26 Sept bulk500
# trace (931 calls): 2.0/3.5 undercounted mixed Hebrew/English prompts by up
# to 39%, 1.5/3.0 by at most 12% -- the margin covers the rest on a 4k-token prompt.
_CHARS_PER_TOKEN_NON_LATIN = 1.5
_CHARS_PER_TOKEN_LATIN = 3.0
_TOKENS_PER_MESSAGE = 8
_CONTEXT_MARGIN = 512
_MIN_OUTPUT_TOKENS = 256

# How much of a malformed reply goes back to the model with the correction request.
_MAX_ECHOED_CHARS = 8000

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def _parse_json_reply(content: str):
    """json.loads, tolerating what models that don't honor the grammar exactly (gpt-oss, some
    Ollama builds) wrap around the object: a ```json fence, or a sentence before or after it."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        fenced = _FENCED_JSON_RE.search(content)
        if fenced:
            return json.loads(fenced.group(1))
        start, end = content.find("{"), content.rfind("}")
        if 0 <= start < end:
            return json.loads(content[start:end + 1])
        raise

_RUNAWAY_NOTE = (
    "(Your previous reply to this was cut off at the length limit because it kept repeating itself. "
    "Reply again, concisely: state each point once, never repeat a sentence, and finish the JSON.)"
)


class LLMTimeoutError(RuntimeError):
    """A model call ran past request_timeout_s. The message names the model, the step
    and the setting to change: httpx's own ReadTimeout has an empty message, which is
    all the chat and Legal tabs used to show."""


class SchemaValidationFailed(Exception):
    def __init__(self, raw: str, errors: str):
        super().__init__(f"LLM JSON output failed schema validation: {errors}")
        self.raw = raw
        self.errors = errors


@dataclass
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class StreamDelta:
    kind: Literal["reasoning", "content"]
    text: str


@dataclass
class SamplingParams:
    temperature: float = 0.6
    top_p: float = 0.92
    max_tokens: int = 2048  # thinking tokens count against it too, on both backends
    top_k: int | None = None
    seed: int | None = None


@dataclass
class Completion:
    content: str
    reasoning: str = ""
    done_reason: str | None = None  # "stop", or "length" when max_tokens cut it off
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def truncated(self) -> bool:
        return self.done_reason == "length"


@dataclass
class LLMCallSite:
    """Identifies which config-driven `enable_thinking` default applies."""

    name: Literal[
        "outline_generation",
        "slide_fill",
        "translation",
        "chat_general",
        "document_generation",
        "tone_rewrite",
        "chunk_summary",
        "legal_language_id",
        "legal_analysis",
        "legal_case_analysis",
        "legal_case_document",
        "legal_research_memo",
        "legal_draft",
        "legal_citation_verification",
        "legal_script_repair",
        "legal_eval_baseline",
        "legal_eval_judge",
        "legal_eval_plan",
        "legal_retrieval_plan",
        "legal_eval_rewrite",
        "legal_eval_label_check",
        "legal_eval_scope",
        "legal_eval_toc",
        "legal_eval_extract",
        "legal_eval_completeness",
        "legal_eval_repair",
    ]


def _with_note(messages: list[ChatMessage], note: str) -> list[ChatMessage]:
    """`messages` with `note` appended to the last user turn."""
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].role == "user":
            out[i] = ChatMessage("user", f"{out[i].content}\n\n{note}")
            return out
    return [*out, ChatMessage("user", note)]


def _loop_breaking(sampling: SamplingParams, attempt: int) -> SamplingParams:
    """Non-greedy loop-breaking sampling (temperature 0.7, top_p 0.8,
    top_k 20), with a new seed each retry so the next one isn't a replay."""
    return dataclasses.replace(
        sampling, temperature=max(sampling.temperature, 0.7), top_p=0.8, top_k=20,
        seed=(sampling.seed or 0) + attempt + 1,
    )


class LLMClient:
    """Talks to either vLLM's OpenAI-compatible API or Ollama's native API,
    selected by `config.llm.backend`. vLLM is NVIDIA-GPU-only but fastest;
    Ollama runs on CPU or whatever acceleration the host exposes (CUDA/ROCm/
    Metal), which is what makes the app portable to non-NVIDIA machines. The
    two APIs differ enough (endpoint path, request shape, structured-JSON
    mechanism, streaming wire format, and how "thinking" tokens are exposed)
    that this class branches on `self._backend` rather than pretending
    they're the same protocol.
    """

    def __init__(self, llm_cfg: LLMConfig | None = None) -> None:
        """`llm_cfg` picks which model this client talks to -- defaults to
        the general-purpose `config.llm` section, but callers needing a
        different deployment (e.g. the Legal tab's orchestrator/Hebrew-
        analyst models, see llm/client.py's `get_legal_*_client`) pass their
        own `LLMConfig` instead."""
        cfg = get_config()
        self._cfg = cfg
        self._llm_cfg = llm_cfg or cfg.llm
        self._backend = self._llm_cfg.backend
        self._chat_path = "/chat/completions" if self._backend == "vllm" else "/api/chat"
        self._client = httpx.AsyncClient(
            base_url=self._llm_cfg.base_url,
            timeout=self._llm_cfg.request_timeout_s,
            headers={"Authorization": f"Bearer {self._llm_cfg.api_key}"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _enable_thinking(self, call_site: LLMCallSite, override: bool | None) -> bool:
        if not self._llm_cfg.supports_thinking:
            return False
        if override is not None:
            return override
        return getattr(self._llm_cfg.thinking_defaults, call_site.name)

    def _build_payload(
        self,
        messages: list[ChatMessage],
        call_site: LLMCallSite,
        sampling: SamplingParams | None,
        enable_thinking: bool | None,
        guided_json_schema: dict[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        sampling = sampling or SamplingParams(**self._llm_cfg.default_sampling.model_dump())
        thinking = self._enable_thinking(call_site, enable_thinking)
        common: dict[str, Any] = {
            "model": self._llm_cfg.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
        }

        if self._backend == "vllm":
            payload: dict[str, Any] = {
                **common,
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "max_tokens": sampling.max_tokens,
                "chat_template_kwargs": {"enable_thinking": thinking},
            }
            payload.update({k: v for k, v in (("top_k", sampling.top_k), ("seed", sampling.seed)) if v is not None})
            if guided_json_schema is not None:
                payload["extra_body"] = {
                    "guided_json": guided_json_schema,
                    "guided_decoding_backend": self._llm_cfg.guided_decoding_backend,
                }
            return payload

        # Ollama's native /api/chat: sampling params live under "options"
        # (Modelfile PARAMETER names), "think" toggles reasoning directly
        # (no chat-template-string juggling), and structured output is a
        # top-level "format" field holding the raw JSON Schema dict.
        # "think" is sent even as false for supports_thinking: false: a model that thinks by default
        # (Gemma 4) otherwise spends max_tokens reasoning and returns empty content. A server that
        # rejects the field (Gemma 3 on some Ollama versions, HTTP 400) is asked again without it.
        payload = {
            **common,
            "think": thinking,
            "options": {
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "num_predict": sampling.max_tokens,
                "num_ctx": self._llm_cfg.max_model_len,
            },
        }
        extra = (("top_k", sampling.top_k), ("seed", sampling.seed))
        payload["options"].update({k: v for k, v in extra if v is not None})
        if guided_json_schema is not None:
            payload["format"] = guided_json_schema
        return payload

    @staticmethod
    def _split_thinking(text: str) -> tuple[str, str]:
        """Split a full (non-streamed) completion into (reasoning, content). Handles <think>...</think>
        (Gemma 4 and other <think>-tag models) and [THINK]...[/THINK] (Mistral-based reasoning models such as DictaLM 3.0 Thinking),
        and a reply that has only the closing tag because the chat template already opened the block."""
        for open_tag, close_tag in (("<think>", "</think>"), ("[THINK]", "[/THINK]")):
            if close_tag in text:
                end = text.index(close_tag)
                start = text.index(open_tag) + len(open_tag) if open_tag in text[:end] else 0
                return text[start:end].strip(), text[end + len(close_tag):].strip()
            if text.lstrip().startswith(open_tag):  # cut off while still thinking: there is no answer
                return text.lstrip()[len(open_tag):].strip(), ""
        return "", text.strip()

    @property
    def model(self) -> str:
        return self._llm_cfg.model

    @staticmethod
    def _estimate_tokens(messages: list[ChatMessage]) -> int:
        chars = "".join(m.content for m in messages)
        non_latin = sum(1 for ch in chars if ord(ch) > 0x024F)
        return (
            int(non_latin / _CHARS_PER_TOKEN_NON_LATIN + (len(chars) - non_latin) / _CHARS_PER_TOKEN_LATIN)
            + _TOKENS_PER_MESSAGE * len(messages)
        )

    def output_room(self, messages: list[ChatMessage]) -> int:
        """The output tokens left after `messages` in max_model_len (can be negative): what
        _fit_context caps max_tokens to. Callers that build long prompts trim their evidence by it."""
        return self._llm_cfg.max_model_len - self._estimate_tokens(messages) - _CONTEXT_MARGIN

    def _fit_context(self, messages: list[ChatMessage], sampling: SamplingParams) -> SamplingParams:
        """`sampling` with max_tokens lowered, if need be, so prompt + output fit
        max_model_len. Ollama runs with --context-shift: a generation that
        reaches num_ctx discards the oldest tokens -- the system prompt and the
        evidence -- and carries on without them. vLLM rejects the request."""
        budget = self.output_room(messages)
        if sampling.max_tokens <= budget:
            return sampling
        capped = max(budget, _MIN_OUTPUT_TOKENS)
        # Under half the budget asked for, the answer is likely cut off (case_03 on 1 Oct: 5,120 -> 256).
        log = logger.warning if capped < sampling.max_tokens // 2 else logger.info
        log("llm_max_tokens_capped", requested=sampling.max_tokens, capped=capped,
            max_model_len=self._llm_cfg.max_model_len)
        return dataclasses.replace(sampling, max_tokens=capped)

    async def _post_completion(self, payload: dict[str, Any]) -> Completion:
        """Non-streamed completion, with the reasoning split from the content."""
        resp = await self._client.post(self._chat_path, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if self._backend == "vllm":
            choice = data["choices"][0]
            message = choice["message"]
            # A server started with --reasoning-parser returns the reasoning separately;
            # otherwise it arrives inline as <think> tags.
            inline_reasoning, content = self._split_thinking(message.get("content") or "")
            usage = data.get("usage") or {}
            return Completion(
                content=content,
                reasoning=message.get("reasoning_content") or message.get("reasoning") or inline_reasoning,
                done_reason=choice.get("finish_reason"),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )
        message = data["message"]
        content = message.get("content") or ""
        reasoning = message.get("thinking") or ""
        if not reasoning:
            # Older Ollama builds / models that ignore "think" still
            # emit reasoning inline as <think> tags in content.
            reasoning, content = self._split_thinking(content)
        return Completion(
            content=content,
            reasoning=reasoning,
            done_reason=data.get("done_reason"),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )

    async def _complete(
        self,
        messages: list[ChatMessage],
        call_site: LLMCallSite,
        sampling: SamplingParams | None,
        enable_thinking: bool | None,
        guided_json_schema: dict[str, Any] | None,
        attempt: int = 1,
    ) -> Completion:
        """One non-streamed call, recorded in the trace whatever its outcome."""
        sampling = self._fit_context(
            messages, sampling or SamplingParams(**self._llm_cfg.default_sampling.model_dump())
        )
        thinking = self._enable_thinking(call_site, enable_thinking)
        started = time.monotonic()
        completion: Completion | None = None
        error: str | None = None
        try:
            payload = self._build_payload(messages, call_site, sampling, thinking, guided_json_schema, stream=False)
            try:
                completion = await self._post_completion(payload)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 400:
                    raise
                if thinking:
                    # A model or server that can't think ("does not support thinking"): answer without it.
                    logger.warning("llm_thinking_unsupported", call_site=call_site.name, error=exc.response.text[:300])
                    thinking = False
                    payload = self._build_payload(messages, call_site, sampling, False, guided_json_schema, stream=False)
                elif "think" in payload and "think" in exc.response.text.lower():
                    # A model with no thinking mode that rejects even "think": false (e.g. Gemma 3 on some
                    # Ollama versions): leave the field out, which such a model can only read as off.
                    logger.warning("llm_think_field_rejected", call_site=call_site.name, error=exc.response.text[:300])
                    payload = {k: v for k, v in payload.items() if k != "think"}
                else:
                    raise
                completion = await self._post_completion(payload)
            return completion
        except httpx.TimeoutException as exc:
            timeout = self._timeout_error(call_site)
            error = f"{type(exc).__name__}: {timeout}"
            raise timeout from exc
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            trace.record(
                {
                    "call_site": call_site.name,
                    "model": self._llm_cfg.model,
                    "backend": self._backend,
                    "thinking": thinking,
                    "json": guided_json_schema is not None,
                    "attempt": attempt,
                    "sampling": dataclasses.asdict(sampling),
                    "seconds": round(time.monotonic() - started, 1),
                    "done_reason": completion.done_reason if completion else None,
                    "prompt_tokens": completion.prompt_tokens if completion else None,
                    "completion_tokens": completion.completion_tokens if completion else None,
                    "reasoning": completion.reasoning if completion else "",
                    "content": completion.content if completion else "",
                    "error": error,
                },
                messages=[{"role": m.role, "content": m.content} for m in messages],
            )

    async def complete_text(
        self,
        messages: list[ChatMessage],
        call_site: LLMCallSite,
        sampling: SamplingParams | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        """Free-text generation, for call sites whose output is prose that
        must not be forced through a JSON schema. With thinking on, the
        reasoning goes to the trace and only the answer is returned."""
        completion = await self._complete(messages, call_site, sampling, enable_thinking, guided_json_schema=None)
        return completion.content.strip()

    async def complete_json(
        self,
        messages: list[ChatMessage],
        call_site: LLMCallSite,
        schema: type[BaseModel],
        sampling: SamplingParams | None = None,
        enable_thinking: bool | None = None,
        max_retries: int | None = None,
        salvage: Callable[[str], BaseModel | None] | None = None,
    ) -> BaseModel:
        """Structured-JSON generation with schema validation + error-correction retry.

        `salvage` gets each failed reply, in order, once the retries are spent;
        the first model it recovers is returned instead of raising."""
        cfg = self._cfg
        attempts = 1 + (max_retries if max_retries is not None else cfg.slides.schema_retry_attempts)
        last_error: Exception | None = None
        failed_replies: list[str] = []
        working_messages = list(messages)
        working_sampling = sampling
        guided = schema.model_json_schema()

        for attempt in range(attempts):
            completion = await self._complete(
                working_messages, call_site, working_sampling, enable_thinking, guided, attempt=attempt + 1
            )
            content = completion.content

            try:
                parsed = _parse_json_reply(content)
                return schema.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = SchemaValidationFailed(raw=content, errors=str(exc))
                failed_replies.append(content)
                logger.warning(
                    "llm_schema_validation_failed",
                    call_site=call_site.name,
                    attempt=attempt + 1,
                    truncated=completion.truncated,
                    error=str(exc),
                )
                if completion.truncated:
                    # Cut off at max_tokens. Greedy decoding (temperature 0) loops -- the model vendors' own
                    # guidance warns of it -- and feeding the loop back would only prime it and eat
                    # the context: start over, sampling as recommended for non-thinking mode.
                    working_messages = _with_note(messages, _RUNAWAY_NOTE)
                    working_sampling = _loop_breaking(sampling or SamplingParams(), attempt)
                    continue
                working_messages = [
                    *messages,
                    ChatMessage(role="assistant", content=content[:_MAX_ECHOED_CHARS]),
                    ChatMessage(
                        role="user",
                        content=(
                            "Your previous response did not match the required JSON schema. "
                            f"Validation error: {exc}\n"
                            f"Required schema: {guided}\n"
                            "Return ONLY corrected JSON matching the schema, no commentary."
                        ),
                    ),
                ]

        if salvage is not None:
            for reply in failed_replies:
                rescued = salvage(reply)
                if rescued is not None:
                    logger.warning("llm_output_salvaged", call_site=call_site.name)
                    return rescued
        assert last_error is not None
        raise last_error

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        call_site: LLMCallSite,
        sampling: SamplingParams | None = None,
        enable_thinking: bool | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """Stream a chat turn, yielding separate reasoning/content deltas. Like the
        non-streamed calls: the output budget is fitted to the context window, a
        server that rejects thinking is asked again without it, and a timeout
        says which model and step were too slow."""
        sampling = self._fit_context(
            messages, sampling or SamplingParams(**self._llm_cfg.default_sampling.model_dump())
        )
        thinking = self._enable_thinking(call_site, enable_thinking)
        payload = self._build_payload(messages, call_site, sampling, thinking, guided_json_schema=None, stream=True)
        try:
            for attempt in (1, 2):
                async with self._client.stream("POST", self._chat_path, json=payload) as resp:
                    if resp.status_code == 400 and attempt == 1 and "think" in payload:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        if thinking:
                            logger.warning("llm_thinking_unsupported", call_site=call_site.name, error=body[:300])
                            payload = self._build_payload(messages, call_site, sampling, False,
                                                          guided_json_schema=None, stream=True)
                            continue
                        if "think" in body.lower():
                            logger.warning("llm_think_field_rejected", call_site=call_site.name, error=body[:300])
                            payload = {k: v for k, v in payload.items() if k != "think"}
                            continue
                    if resp.is_error:
                        await resp.aread()  # so the error carries the server's message
                    resp.raise_for_status()
                    if self._backend == "vllm":
                        async for delta in self._stream_vllm(resp):
                            yield delta
                    else:
                        async for delta in self._stream_ollama(resp):
                            yield delta
                    return
        except httpx.TimeoutException as exc:
            raise self._timeout_error(call_site) from exc

    def _timeout_error(self, call_site: LLMCallSite) -> LLMTimeoutError:
        return LLMTimeoutError(
            f"{self._llm_cfg.model} ({self._backend}) did not finish the '{call_site.name}' step within "
            f"{self._llm_cfg.request_timeout_s:g} s. The model is too slow for this machine: on a CPU without a "
            "supported GPU use a smaller model (e.g. gemma4:12b), or raise the limit with "
            "DOCSLIDES_LLM_REQUEST_TIMEOUT_S / DOCSLIDES_LEGAL_ORCHESTRATOR_REQUEST_TIMEOUT_S."
        )

    @staticmethod
    async def _stream_vllm(resp: httpx.Response) -> AsyncIterator[StreamDelta]:
        """vLLM's OpenAI-compatible SSE stream embeds reasoning inline as
        <think>...</think> tags in the content deltas -- split on them as
        they arrive, buffering only across a tag boundary."""
        in_thinking = False
        thinking_closed = False
        buffer = ""

        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                break
            chunk = json.loads(data_str)
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content")
            if not token:
                continue
            buffer += token

            while buffer:
                if not in_thinking and not thinking_closed and buffer.lstrip().startswith("<think>"):
                    idx = buffer.index("<think>")
                    buffer = buffer[idx + len("<think>") :]
                    in_thinking = True
                    continue
                if in_thinking and "</think>" in buffer:
                    idx = buffer.index("</think>")
                    if idx > 0:
                        yield StreamDelta(kind="reasoning", text=buffer[:idx])
                    buffer = buffer[idx + len("</think>") :]
                    in_thinking = False
                    thinking_closed = True
                    continue
                kind = "reasoning" if in_thinking else "content"
                yield StreamDelta(kind=kind, text=buffer)
                buffer = ""

        if buffer:
            kind = "reasoning" if in_thinking else "content"
            yield StreamDelta(kind=kind, text=buffer)

    @staticmethod
    async def _stream_ollama(resp: httpx.Response) -> AsyncIterator[StreamDelta]:
        """Ollama's /api/chat stream is newline-delimited JSON (not SSE); each
        line already separates `message.thinking` from `message.content`, so
        no tag-splitting is needed here -- only models that actually honor
        the "think" request field populate `thinking` distinctly."""
        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            message = chunk.get("message") or {}
            thinking = message.get("thinking")
            if thinking:
                yield StreamDelta(kind="reasoning", text=thinking)
            content = message.get("content")
            if content:
                yield StreamDelta(kind="content", text=content)
            if chunk.get("done"):
                break


# The model size the current request asked for (the UI's 12B/27B selector, config.model_sizes), or
# None for the configured models. Set by the API routes around the job they start: asyncio tasks
# copy the context they're created in, so everything the job does sees it.
_model_size: ContextVar[str | None] = ContextVar("model_size", default=None)
# One client per (deployment, model tag): each size gets its own, created on first use.
_clients: dict[tuple[str, str], LLMClient] = {}


@contextmanager
def model_size(size: str | None) -> Iterator[None]:
    """Within this block (and in tasks started inside it) get_client, get_legal_orchestrator_client
    and get_translator_client talk to `size`'s models. An unknown size or None keeps the
    configured models."""
    token = _model_size.set(size)
    try:
        yield
    finally:
        _model_size.reset(token)


def _sized(llm_cfg: LLMConfig, translator: bool = False) -> LLMConfig:
    """`llm_cfg` with the model of the size the current request picked. Ollama only: a vLLM
    server serves one model, whatever the request names."""
    size = get_config().model_sizes.get(_model_size.get() or "")
    if size is None or llm_cfg.backend != "ollama":
        return llm_cfg
    tag = size.translator_model if translator else size.model
    return llm_cfg.model_copy(update={"model": tag}) if tag else llm_cfg


def _client_for(deployment: str, llm_cfg: LLMConfig) -> LLMClient:
    key = (deployment, llm_cfg.model)
    if key not in _clients:
        _clients[key] = LLMClient(llm_cfg)
    return _clients[key]


def get_client() -> LLMClient:
    return _client_for("general", _sized(get_config().llm))


def get_legal_orchestrator_client() -> LLMClient:
    """The Legal tab's research, drafting and verification model (Gemma). See
    legal/pipeline.py."""
    return _client_for("legal_orchestrator", _sized(get_config().legal.orchestrator))


def get_translator_client() -> LLMClient | None:
    """TranslateGemma, the dedicated translation model (config translation.translator), or None
    when none is configured -- then the general model translates on its own. See
    translation/translator.py."""
    translator_cfg = get_config().translation.translator
    if translator_cfg is None:
        return None
    return _client_for("translator", _sized(translator_cfg, translator=True))


async def aclose_all_clients() -> None:
    """Closes whichever of the above clients were actually instantiated,
    without creating new ones just to close them -- called once at app
    shutdown (see api/main.py's lifespan)."""
    for client in list(_clients.values()):
        await client.aclose()
    _clients.clear()
