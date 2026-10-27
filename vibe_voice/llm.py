"""OpenAI-compatible Chat Completions client. No vendor SDK."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from vibe_voice.errors import LlmError

Transport = Callable[[str, dict[str, str], dict, float], dict]


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


def chat_completions_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        raise LlmError("OPENAI_BASE_URL 为空")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def build_messages(
    system_prompt: str,
    history: Sequence[ChatMessage],
    user_text: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for item in history:
        messages.append({"role": item.role, "content": item.content})
    messages.append({"role": "user", "content": user_text})
    return messages


def build_chat_request(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: Sequence[dict[str, str]],
    temperature: float,
    max_tokens: int | None = None,
) -> tuple[str, dict[str, str], dict]:
    if not model:
        raise LlmError("OPENAI_MODEL 为空")
    url = chat_completions_url(base_url)
    body: dict = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return url, headers, body


def parse_chat_response(payload: dict) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmError("大模型响应里没有 choices[0].message.content") from exc
    text = _content_text(content)
    if not text:
        raise LlmError("大模型返回了空内容")
    return text


def urllib_transport(url: str, headers: dict[str, str], body: dict, timeout: float) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise LlmError(f"大模型接口返回 HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LlmError(f"无法连接大模型接口 {url}: {exc.reason}") from exc
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise LlmError("大模型接口返回的不是 JSON") from exc
    if not isinstance(parsed, dict):
        raise LlmError("大模型接口返回的 JSON 不是对象")
    return parsed


class OpenAICompatibleClient:
    """POST ``{base}/chat/completions`` and return the assistant text."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        system_prompt: str = "",
        timeout_s: float = 60.0,
        max_tokens: int | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self._transport = transport or urllib_transport

    def complete(self, user_text: str, history: Sequence[ChatMessage] | None = None) -> str:
        messages = build_messages(self.system_prompt, history or [], user_text)
        url, headers, body = build_chat_request(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        payload = self._transport(url, headers, body, self.timeout_s)
        return parse_chat_response(payload)


def append_history(
    history: list[ChatMessage],
    user_text: str,
    answer: str,
    limit: int,
) -> None:
    history.append(ChatMessage("user", user_text))
    history.append(ChatMessage("assistant", answer))
    if limit > 0 and len(history) > limit:
        del history[: len(history) - limit]


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in (None, "text"):
                    parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    raise LlmError("无法读取大模型返回的文本")
