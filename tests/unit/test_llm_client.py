"""The LLM client without a model server: request payloads, the context budget, a
runaway reply retried fresh, salvage, and the trace of every call's reasoning."""

import asyncio

import pytest

from docslides.config import get_config
from docslides.llm import trace
from docslides.llm.client import (
    ChatMessage,
    Completion,
    LLMCallSite,
    QwenClient,
    SamplingParams,
    SchemaValidationFailed,
)
from docslides.llm.schemas import EntailmentVerdict, EvalJudgement, LegalDraft


@pytest.fixture
def ollama(monkeypatch, tmp_path):
    monkeypatch.setattr(get_config().logging, "llm_trace_dir", str(tmp_path))
    cfg = get_config().legal.orchestrator.model_copy(
        update={"backend": "ollama", "base_url": "http://localhost:1", "model": "qwen3:8b", "max_model_len": 8192}
    )
    return QwenClient(cfg)


def _scripted(client, monkeypatch, replies):
    payloads = []

    async def post(payload):
        payloads.append(payload)
        return replies.pop(0)

    monkeypatch.setattr(client, "_post_completion", post)
    return payloads


def test_payload_carries_top_k_seed_thinking_and_context(ollama):
    hi = [ChatMessage("user", "hi")]
    sampling = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=100, seed=0)
    payload = ollama._build_payload(hi, LLMCallSite("legal_analysis"), sampling, None, None, stream=False)
    assert payload["think"] is True  # the analysis pass thinks by default
    assert (payload["options"]["top_k"], payload["options"]["seed"], payload["options"]["num_ctx"]) == (20, 0, 8192)
    plain = ollama._build_payload(hi, LLMCallSite("legal_draft"), SamplingParams(), None, None, stream=False)
    assert plain["think"] is False and "top_k" not in plain["options"]


def test_max_tokens_is_capped_to_fit_the_context(ollama):
    long_prompt = [ChatMessage("user", "א" * 12000)]  # ~6000 tokens by the pessimistic estimate
    assert 256 <= ollama._fit_context(long_prompt, SamplingParams(max_tokens=4096)).max_tokens < 4096
    assert ollama._fit_context([ChatMessage("user", "hi")], SamplingParams(max_tokens=4096)).max_tokens == 4096


def test_context_window_can_be_set_per_host_from_the_environment(monkeypatch):
    from docslides.config import _apply_env_overrides

    monkeypatch.setenv("DOCSLIDES_LEGAL_ORCHESTRATOR_MAX_MODEL_LEN", "20480")
    monkeypatch.setenv("DOCSLIDES_LLM_MAX_MODEL_LEN", "12288")
    raw = _apply_env_overrides({"llm": {"max_model_len": 8192}, "legal": {"orchestrator": {"max_model_len": 16384}}})
    assert raw["llm"]["max_model_len"] == "12288"
    assert raw["legal"]["orchestrator"]["max_model_len"] == "20480"
    orchestrator = type(get_config().legal.orchestrator).model_validate(
        {**get_config().legal.orchestrator.model_dump(), **raw["legal"]["orchestrator"]})
    assert orchestrator.max_model_len == 20480


def test_a_runaway_reply_is_retried_fresh_with_non_greedy_sampling(ollama, monkeypatch):
    payloads = _scripted(ollama, monkeypatch, [
        Completion('{"answer_draft": "loop loop loop', done_reason="length"),
        Completion('{"answer_draft": "ok", "escalation_flag": false}', done_reason="stop"),
    ])
    messages = [ChatMessage("system", "s"), ChatMessage("user", "q")]
    draft = asyncio.run(ollama.complete_json(messages, LLMCallSite("legal_draft"), LegalDraft,
                                             SamplingParams(temperature=0.0, max_tokens=2048)))
    assert draft.answer_draft == "ok"
    retry = payloads[1]
    assert (retry["options"]["temperature"], retry["options"]["top_k"]) == (0.7, 20)
    assert [m["role"] for m in retry["messages"]] == ["system", "user"]  # the loop isn't fed back
    assert "kept repeating itself" in retry["messages"][-1]["content"]


def test_salvage_rescues_a_reply_every_retry_left_malformed(ollama, monkeypatch):
    rescued = LegalDraft(answer_draft="kept", escalation_flag=True)
    _scripted(ollama, monkeypatch, [Completion("{bad", done_reason="stop") for _ in range(3)])
    result = asyncio.run(ollama.complete_json([ChatMessage("user", "q")], LLMCallSite("legal_draft"), LegalDraft,
                                              salvage=lambda raw: rescued))
    assert result is rescued
    _scripted(ollama, monkeypatch, [Completion("{bad", done_reason="stop") for _ in range(3)])
    with pytest.raises(SchemaValidationFailed):
        asyncio.run(ollama.complete_json([ChatMessage("user", "q")], LLMCallSite("legal_draft"), LegalDraft))


def test_every_call_is_traced_with_its_reasoning(ollama, monkeypatch, tmp_path):
    _scripted(ollama, monkeypatch, [Completion("notes", reasoning="thinking it through", done_reason="stop",
                                               prompt_tokens=10, completion_tokens=5)])
    with trace.collect("job-1") as calls:
        text = asyncio.run(ollama.complete_text([ChatMessage("user", "q")], LLMCallSite("legal_analysis")))
    assert text == "notes"
    [call] = calls
    assert (call["reasoning"], call["thinking"], call["job_id"], call["done_reason"]) == (
        "thinking it through", True, "job-1", "stop")
    [trace_file] = list(tmp_path.iterdir())
    [line] = trace_file.read_text(encoding="utf-8").splitlines()
    assert '"messages"' in line and "thinking it through" in line


def test_reasoning_inline_in_think_tags_is_split_from_the_answer():
    assert QwenClient._split_thinking("<think>why</think>answer") == ("why", "answer")
    assert QwenClient._split_thinking("<think>cut off mid-thought") == ("cut off mid-thought", "")
    assert QwenClient._split_thinking("just an answer") == ("", "just an answer")


def test_the_judge_explains_before_its_verdict_and_the_verifier_after():
    judge = list(EvalJudgement.model_json_schema()["properties"])
    assert judge.index("explanation") < judge.index("verdict")
    verifier = list(EntailmentVerdict.model_json_schema()["properties"])  # verdict first: see its docstring
    assert verifier.index("verdict") < verifier.index("explanation")


def test_a_json_reply_wrapped_in_a_fence_or_prose_still_parses():
    from docslides.llm.client import _parse_json_reply

    assert _parse_json_reply('{"a": 1}') == {"a": 1}
    assert _parse_json_reply('Here it is:\n```json\n{"a": 2}\n```') == {"a": 2}
    assert _parse_json_reply('Verdict follows. {"a": 3} Done.') == {"a": 3}
    with pytest.raises(ValueError):
        _parse_json_reply("no object here")


def test_the_judge_defaults_to_a_family_not_under_test(monkeypatch):
    from docslides.legal.evaluation import get_judge_client

    monkeypatch.delenv("DOCSLIDES_LEGAL_JUDGE_MODEL", raising=False)
    assert get_judge_client().model == get_config().legal.judge_model == "gpt-oss:20b"
    monkeypatch.setenv("DOCSLIDES_LEGAL_JUDGE_MODEL", "mistral-small3.2:24b")
    assert get_judge_client().model == "mistral-small3.2:24b"
