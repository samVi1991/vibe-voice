import json

import pytest

from vibe_voice.errors import LlmError
from vibe_voice.llm import (
    ChatMessage,
    OpenAICompatibleClient,
    build_chat_request,
    build_messages,
    chat_completions_url,
    parse_chat_response,
)


def test_chat_url_joins_openai_ollama_and_vllm_bases():
    assert chat_completions_url("https://api.openai.com/v1") == "https://api.openai.com/v1/chat/completions"
    assert chat_completions_url("http://127.0.0.1:11434/v1/") == "http://127.0.0.1:11434/v1/chat/completions"
    assert chat_completions_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1/chat/completions"
    assert (
        chat_completions_url("http://127.0.0.1:8000/v1/chat/completions")
        == "http://127.0.0.1:8000/v1/chat/completions"
    )


def test_request_omits_authorization_when_key_is_empty():
    messages = build_messages("你是小助", [ChatMessage("user", "上一句"), ChatMessage("assistant", "上一答")], "现在几点")
    url, headers, body = build_chat_request(
        base_url="http://127.0.0.1:11434/v1",
        api_key="",
        model="qwen2.5:7b",
        messages=messages,
        temperature=0.2,
    )
    assert url.endswith("/chat/completions")
    assert "Authorization" not in headers
    assert body["model"] == "qwen2.5:7b"
    assert body["messages"][0] == {"role": "system", "content": "你是小助"}
    assert body["messages"][-1] == {"role": "user", "content": "现在几点"}
    assert "max_tokens" not in body


def test_request_sends_bearer_token_and_max_tokens():
    _, headers, body = build_chat_request(
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "你好"}],
        temperature=0.7,
        max_tokens=128,
    )
    assert headers["Authorization"] == "Bearer sk-test"
    assert body["max_tokens"] == 128


def test_client_posts_built_request_and_reads_text(monkeypatch):
    captured = {}

    def transport(url, headers, body, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = body
        captured["timeout"] = timeout
        return {"choices": [{"message": {"content": "  今天晴天。  "}}]}

    client = OpenAICompatibleClient(
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
        model="qwen2.5:7b",
        temperature=0.4,
        system_prompt="简短回答",
        timeout_s=12,
        transport=transport,
    )
    assert client.complete("天气呢", [ChatMessage("user", "早")]) == "今天晴天。"
    assert captured["url"] == "http://127.0.0.1:11434/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer ollama"
    assert captured["body"]["messages"][0]["role"] == "system"
    assert captured["body"]["messages"][1]["content"] == "早"
    assert captured["body"]["messages"][2]["content"] == "天气呢"
    assert captured["timeout"] == 12


def test_parse_list_content_and_reject_empty_payload():
    payload = {"choices": [{"message": {"content": [{"type": "text", "text": "你好"}]}}]}
    assert parse_chat_response(payload) == "你好"
    with pytest.raises(LlmError):
        parse_chat_response({"choices": []})
    with pytest.raises(LlmError):
        parse_chat_response({"choices": [{"message": {"content": "  "}}]})


def test_urllib_transport_is_not_called_when_transport_is_injected():
    def transport(url, headers, body, timeout):
        raise AssertionError("should not use the network")

    client = OpenAICompatibleClient(
        base_url="https://api.openai.com/v1",
        api_key="",
        model="gpt-4o-mini",
        transport=transport,
    )
    with pytest.raises(AssertionError):
        client.complete("hi")


def test_request_body_is_json_serializable():
    _, _, body = build_chat_request(
        base_url="https://api.openai.com/v1",
        api_key="",
        model="m",
        messages=[{"role": "user", "content": "你好小助"}],
        temperature=0.7,
    )
    assert json.loads(json.dumps(body))["messages"][0]["content"] == "你好小助"
