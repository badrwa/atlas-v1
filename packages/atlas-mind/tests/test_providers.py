"""Provider wire-protocol tests — no network, no keys.

`httpx.MockTransport` gives us a real HTTP layer with a canned body, so SSE
parsing, error mapping and tool-call reassembly are all tested for real.
"""

from __future__ import annotations

import json

import httpx
import pytest

from atlas_core.config import ProviderConfig
from atlas_core.contracts import (
    LlmRequest,
    Message,
    Role,
    StreamEnd,
    TextDelta,
    ToolCallRequest,
    ToolSpec,
)
from atlas_core.errors import AuthFailed, ProviderUnavailable, RateLimited
from atlas_mind.providers.gemini import GeminiProvider
from atlas_mind.providers.openai_compatible import OpenAiCompatibleProvider

OPENAI_SSE = "\n".join(
    [
        'data: {"choices":[{"delta":{"content":"salam "}}]}',
        'data: {"choices":[{"delta":{"content":"sahbi"}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":4}}',
        "data: [DONE]",
    ]
)

GEMINI_SSE = "\n".join(
    [
        'data: {"candidates":[{"content":{"parts":[{"text":"salam "}]}}]}',
        'data: {"candidates":[{"content":{"parts":[{"text":"sahbi"}]}}]}',
        'data: {"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":3}}',
    ]
)


def _client(body: str, status: int = 200) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body, headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _request() -> LlmRequest:
    return LlmRequest(
        messages=[Message(role=Role.USER, content="salam")],
        system="you are Atlas",
        max_output_tokens=64,
    )


# ── OpenAI-compatible (Groq / OpenRouter / Together / HF / Ollama / llama.cpp) ──
async def test_openai_compatible_streams_text_and_usage() -> None:
    provider = OpenAiCompatibleProvider(
        ProviderConfig(name="groq", kind="openai_compatible", model="m", base_url="https://x/v1"),
        api_key="k",
        client=_client(OPENAI_SSE),
    )
    events = [event async for event in provider.stream(_request())]
    kinds = [event.kind for event in events]
    assert kinds == ["text", "text", "usage", "end"]
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "salam sahbi"
    last = events[-1]
    assert isinstance(last, StreamEnd) and last.reason == "stop"


async def test_openai_compatible_requires_authorization_header() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(request.headers)
        return httpx.Response(200, text=OPENAI_SSE)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAiCompatibleProvider(
            ProviderConfig(name="groq", model="m", base_url="https://x/v1"),
            api_key="secret",
            client=client,
        )
        [event async for event in provider.stream(_request())]
    assert captured["authorization"] == "Bearer secret"


async def test_openai_compatible_maps_http_errors() -> None:
    for status, expected in ((401, AuthFailed), (429, RateLimited), (503, ProviderUnavailable)):
        provider = OpenAiCompatibleProvider(
            ProviderConfig(name="groq", model="m", base_url="https://x/v1"),
            api_key="k",
            client=_client('{"error":"nope"}', status=status),
        )
        with pytest.raises(expected):
            [event async for event in provider.stream(_request())]


async def test_openai_compatible_reassembles_tool_calls() -> None:
    body = "\n".join(
        [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"get_weather","arguments":"{\\"city\\":"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"Casa\\"}"}}]}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]
    )
    provider = OpenAiCompatibleProvider(
        ProviderConfig(name="groq", model="m", base_url="https://x/v1", supports_tools=True),
        api_key="k",
        client=_client(body),
    )
    events = [event async for event in provider.stream(_request())]
    calls = [event for event in events if isinstance(event, ToolCallRequest)]
    assert calls and calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "Casa"}


async def test_openai_compatible_sends_tools_and_schema() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, text=OPENAI_SSE)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAiCompatibleProvider(
            ProviderConfig(name="groq", model="m", base_url="https://x/v1", supports_tools=True),
            api_key="k",
            client=client,
        )
        request = _request()
        request.tools = [ToolSpec(name="weather", description="weather", parameters={"type": "object"})]
        [event async for event in provider.stream(request)]

    assert captured["stream"] is True
    assert captured["model"] == "m"
    assert captured["tools"][0]["function"]["name"] == "weather"  # type: ignore[index]
    assert captured["messages"][0]["role"] == "system"  # type: ignore[index]


# ── Gemini ───────────────────────────────────────────────────────────
async def test_gemini_streams_text_and_maps_system_instruction() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["key"] = request.headers.get("x-goog-api-key")
        captured.update(json.loads(request.content))
        return httpx.Response(200, text=GEMINI_SSE)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = GeminiProvider(
            ProviderConfig(
                name="gemini",
                kind="gemini",
                model="gemini-2.5-flash-lite",
                base_url="https://gen.example/v1beta",
            ),
            api_key="gk",
            client=client,
        )
        events = [event async for event in provider.stream(_request())]

    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "salam sahbi"
    assert ":streamGenerateContent" in str(captured["url"])
    assert captured["key"] == "gk"
    assert captured["systemInstruction"] == {"parts": [{"text": "you are Atlas"}]}  # type: ignore[index]
    assert events[-1].kind == "end"


async def test_gemini_parses_function_call() -> None:
    body = (
        'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"weather","args":{"city":"Casa"}}}]}}]}'
    )
    provider = GeminiProvider(
        ProviderConfig(name="gemini", kind="gemini", model="m", base_url="https://g/v1beta"),
        api_key="k",
        client=_client(body),
    )
    events = [event async for event in provider.stream(_request())]
    call = next(event for event in events if isinstance(event, ToolCallRequest))
    assert call.name == "weather"
    assert call.arguments == {"city": "Casa"}


async def test_gemini_auth_error_is_not_retried() -> None:
    provider = GeminiProvider(
        ProviderConfig(name="gemini", kind="gemini", model="m", base_url="https://g/v1beta"),
        api_key="bad",
        client=_client("{}", status=403),
    )
    with pytest.raises(AuthFailed):
        [event async for event in provider.stream(_request())]
