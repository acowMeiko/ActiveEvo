import asyncio
from types import SimpleNamespace

import modules.stage2.model_client as model_client_module
from modules.stage2.model_client import Stage2ModelClient


def _config() -> SimpleNamespace:
    return SimpleNamespace(stage2=SimpleNamespace(
        generator_model="generator",
        generator_url="https://example.test/v1/chat/completions",
        generator_key_env="TEST_GENERATOR_KEY",
        generation_max_tokens=3072,
        generation_analysis_max_tokens=8192,
        generator_timeout=600,
        generator_retries=2,
        generator_thinking_enabled=False,
        generator_synthesis_thinking_enabled=False,
        generator_seed=42,
        qwen_generation_analysis_max_tokens=6144,
        qwen_generation_max_tokens=4096,
        qwen_generation_retry_max_tokens=8192,
        qwen_generator_model="qwen",
        qwen_generator_urls=(
            "http://127.0.0.1:8200/v1/chat/completions",
            "http://127.0.0.1:8201/v1/chat/completions",
        ),
        qwen_generator_key_env="TEST_QWEN_KEY",
        qwen_generator_temperature=1.0,
        qwen_generator_top_p=0.95,
        qwen_generator_top_k=20,
        qwen_generator_min_p=0.0,
        qwen_generator_presence_penalty=0.0,
        qwen_generator_repetition_penalty=1.0,
        qwen_generator_reasoning_effort="medium",
        auditor_model="qwen",
        auditor_url="http://127.0.0.1:8200/v1/chat/completions",
        auditor_urls=(
            "http://127.0.0.1:8200/v1/chat/completions",
            "http://127.0.0.1:8201/v1/chat/completions",
        ),
        auditor_key_env="TEST_AUDITOR_KEY",
        audit_max_tokens=3072,
        audit_retry_max_tokens=4096,
        auditor_timeout=300,
        auditor_retries_per_endpoint=1,
        auditor_thinking_enabled=True,
        auditor_reasoning_effort="medium",
        audit_formatter_max_tokens=3072,
        kimi_thinking_temperature=1.0,
        kimi_thinking_top_p=0.95,
        kimi_nonthinking_temperature=0.6,
        kimi_nonthinking_top_p=0.95,
    ))


def test_auditor_round_robins_between_two_endpoints(monkeypatch) -> None:
    calls = []

    async def fake_call(**kwargs):
        calls.append(kwargs)
        return {"raw": {"choices": [{"finish_reason": "stop", "message": {"content": '{"decision":"pass"}'}}]}}

    monkeypatch.setattr(model_client_module, "async_llm_call_with_metadata", fake_call)
    client = Stage2ModelClient(_config())
    first = asyncio.run(client.auditor("first", auditor_id="qwen"))
    second = asyncio.run(client.auditor("second", auditor_id="qwen"))

    assert first["endpoint"].startswith("http://127.0.0.1:8200")
    assert second["endpoint"].startswith("http://127.0.0.1:8201")
    assert [item["retries"] for item in calls] == [1, 1]


def test_qwen_auditor_expands_on_same_endpoint_for_missing_final_content(monkeypatch) -> None:
    calls = []

    async def fake_call(**kwargs):
        calls.append(kwargs)
        content = "" if len(calls) == 1 else '{"decision":"pass"}'
        return {"raw": {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}}

    monkeypatch.setattr(model_client_module, "async_llm_call_with_metadata", fake_call)
    client = Stage2ModelClient(_config())
    result = asyncio.run(client.auditor("audit", auditor_id="qwen"))

    assert result["endpoint"].startswith("http://127.0.0.1:8200")
    assert len(result["attempted_endpoints"]) == 1
    assert calls[0]["url"] == calls[1]["url"]
    assert [item["max_tokens"] for item in calls] == [3072, 4096]
    assert all(item["retries"] == 1 for item in calls)


def test_qwen_auditor_fails_over_only_without_provider_response(monkeypatch) -> None:
    calls = []

    async def fake_call(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {"raw": {}}
        return {"raw": {"choices": [{
            "finish_reason": "stop", "message": {"content": '{"decision":"pass"}'},
        }]}}

    monkeypatch.setattr(model_client_module, "async_llm_call_with_metadata", fake_call)
    result = asyncio.run(Stage2ModelClient(_config()).auditor("audit", auditor_id="qwen"))

    assert result["endpoint"].startswith("http://127.0.0.1:8201")
    assert len(calls) == 2
    assert calls[0]["url"] != calls[1]["url"]


def test_qwen_generator_keeps_thinking_and_parses_only_final_content(monkeypatch) -> None:
    calls = []

    async def fake_call(**kwargs):
        calls.append(kwargs)
        return {"raw": {"choices": [{
            "finish_reason": "stop",
            "message": {
                "reasoning_content": "private diagnosis that is not JSON",
                "content": '{"protocol_ok":true,"note":"complete"}',
            },
        }]}}

    monkeypatch.setattr(model_client_module, "async_llm_call_with_metadata", fake_call)
    result = asyncio.run(Stage2ModelClient(_config(), generator_id="qwen").generator("probe"))

    assert result["parsed"]["protocol_ok"] is True
    assert result["reasoning"] == "private diagnosis that is not JSON"
    assert result["parse_source"] == "message.content"
    assert result["thinking_enabled"] is True
    assert calls[0]["chat_template_kwargs"] == {
        "enable_thinking": True, "preserve_thinking": False,
    }
    assert calls[0]["reasoning_effort"] == "medium"
    assert calls[0]["temperature"] == 1.0


def test_qwen_length_retry_never_disables_thinking(monkeypatch) -> None:
    calls = []

    async def fake_call(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {"raw": {"choices": [{
                "finish_reason": "length",
                "message": {"reasoning_content": "unfinished", "content": ""},
            }]}}
        return {"raw": {"choices": [{
            "finish_reason": "stop",
            "message": {"reasoning_content": "complete reasoning", "content": '{"ok":true}'},
        }]}}

    monkeypatch.setattr(model_client_module, "async_llm_call_with_metadata", fake_call)
    result = asyncio.run(Stage2ModelClient(_config(), generator_id="qwen").generator("probe"))

    assert result["parsed"] == {"ok": True}
    assert [item["max_tokens"] for item in calls] == [4095, 8191]
    assert all(item["chat_template_kwargs"]["enable_thinking"] is True for item in calls)


def test_qwen_diagnosis_accepts_first_nonempty_reasoning_trace(monkeypatch) -> None:
    calls = []

    async def fake_call(**kwargs):
        calls.append(kwargs)
        return {"raw": {"choices": [{
            "finish_reason": "length",
            "message": {"reasoning_content": "usable diagnosis", "content": ""},
        }]}}

    monkeypatch.setattr(model_client_module, "async_llm_call_with_metadata", fake_call)
    result = asyncio.run(
        Stage2ModelClient(_config(), generator_id="qwen").generator(
            "diagnose", max_tokens=6144, accept_reasoning_only=True,
        )
    )

    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 6143
    assert result["reasoning"] == "usable diagnosis"
    assert result["reasoning_only_accepted"] is True


def test_kimi_critic_and_formatter_use_separate_fixed_protocols(monkeypatch) -> None:
    calls = []

    async def fake_call(**kwargs):
        calls.append(kwargs)
        return {"raw": {"choices": [{
            "finish_reason": "stop",
            "message": {"reasoning_content": "audit", "content": '{"ok":true}'},
        }]}}

    monkeypatch.setattr(model_client_module, "async_llm_call_with_metadata", fake_call)
    client = Stage2ModelClient(_config())
    critic = asyncio.run(client.auditor("critic", auditor_id="kimi", thinking_enabled=True))
    formatter = asyncio.run(client.format_audit("formatter"))

    assert critic["parsed"] == {"ok": True}
    assert formatter["parsed"] == {"ok": True}
    assert calls[0]["thinking"] == {"type": "enabled"}
    assert calls[0]["temperature"] == 1.0
    assert calls[1]["thinking"] == {"type": "disabled"}
    assert calls[1]["temperature"] == 0.6
    assert calls[1]["max_tokens"] == 3072
