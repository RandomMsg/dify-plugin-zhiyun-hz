"""Native Anthropic Messages API support for Zhiyun Hangzhou.

Selected per model via the `api_protocol` credential (openai | anthropic).

Protocol notes (Anthropic spec, which the vendor follows):
- Endpoint:  POST {base}/v1/messages   (SDK appends /v1/messages itself,
  so the credential base URL is the site ROOT, e.g.
  https://aistore.hztele.com.cn — no /v1)
- Auth:      x-api-key + anthropic-version headers (handled by the SDK)
- System prompts go into a top-level `system` string, not a message.
- Consecutive same-role messages must be merged (roles must alternate).
- The API is strict about unknown body fields, so only whitelisted model
  parameters are forwarded; thinking mode is expressed with the standard
  `thinking: {"type": "enabled", "budget_tokens": N}` object.
- Streaming uses the Anthropic SSE event protocol (message_start /
  content_block_start / content_block_delta / message_delta / message_stop);
  `thinking_delta` pieces are wrapped into  blocks so Dify renders
  them as a separate thinking section, matching the OpenAI-compat path.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any, Union

import ssl

import certifi
import httpx2
from anthropic import Anthropic
from dify_plugin.entities.model.llm import (
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageTool,
    SystemPromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)

DEFAULT_ANTHROPIC_BASE_URL = "https://aistore.hztele.com.cn"

#: model parameters forwarded verbatim to /v1/messages (Anthropic spec names)
PASSTHROUGH_PARAMS = ("temperature", "top_p", "top_k")

#: Anthropic stop_reason -> Dify finish_reason
_FINISH_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def build_client(credentials: dict) -> Anthropic:
    base_url = (
        credentials.get("endpoint_url") or DEFAULT_ANTHROPIC_BASE_URL
    ).strip().rstrip("/")
    # the SDK appends /v1/messages itself; tolerate a pasted .../v1
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    # Explicit certifi-backed context + trust_env=False:
    # httpcore2 defaults to truststore.SSLContext on macOS/Windows, which
    # triggers an infinite verify_mode recursion on conda-built Pythons, and
    # httpx2's system-proxy pickup otherwise yields opaque 502s. This keeps
    # TLS/proxy semantics identical to the requests-based OpenAI-compat path.
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    http_client = httpx2.Client(
        trust_env=False, timeout=120.0, verify=ssl_context
    )
    return Anthropic(
        api_key=credentials.get("api_key"), base_url=base_url, http_client=http_client
    )


def _text_of(prompt: PromptMessage) -> str:
    if isinstance(prompt.content, str):
        return prompt.content
    return "".join(
        item.data
        for item in (prompt.content or [])
        if isinstance(item, TextPromptMessageContent)
    )


def _parse_args(arguments: str) -> dict:
    try:
        return json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return {}


def _image_block(item: ImagePromptMessageContent) -> dict:
    """SDK 0.11 image content: url / base64_data are explicit fields,
    `data` is a computed property, mime_type is required."""
    url = getattr(item, "url", None)
    if url:
        return {"type": "image", "source": {"type": "url", "url": url}}
    b64 = getattr(item, "base64_data", None) or ""
    if not b64:
        # fall back to the synthesized data URI
        data = getattr(item, "data", "") or ""
        if data.startswith("data:"):
            header, _, payload = data.partition(",")
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": header[5:].split(";")[0] or "image/png",
                    "data": payload,
                },
            }
        raise ValueError("image content carries neither url nor base64 data")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": getattr(item, "mime_type", None) or "image/png",
            "data": b64,
        },
    }


def convert_messages(
    prompt_messages: list[PromptMessage],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Dify PromptMessages -> (system, Anthropic messages), merging
    consecutive same-role turns so roles alternate."""
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []

    def append(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})

    for m in prompt_messages:
        if isinstance(m, SystemPromptMessage):
            text = _text_of(m)
            if text:
                system_parts.append(text)
        elif isinstance(m, UserPromptMessage):
            if isinstance(m.content, str):
                blocks = [{"type": "text", "text": m.content}] if m.content else []
            else:
                blocks = []
                for item in m.content or []:
                    if isinstance(item, ImagePromptMessageContent):
                        blocks.append(_image_block(item))
                    elif isinstance(item, TextPromptMessageContent) and item.data:
                        blocks.append({"type": "text", "text": item.data})
            append("user", blocks)
        elif isinstance(m, AssistantPromptMessage):
            blocks = []
            text = m.content if isinstance(m.content, str) else _text_of(m)
            if text:
                blocks.append({"type": "text", "text": text})
            for call in m.tool_calls or []:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.function.name,
                        "input": _parse_args(call.function.arguments),
                    }
                )
            append("assistant", blocks)
        elif isinstance(m, ToolPromptMessage):
            append(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": _text_of(m),
                    }
                ],
            )
    return ("\n\n".join(system_parts) or None), messages


def build_request_args(
    model: str,
    credentials: dict,
    prompt_messages: list[PromptMessage],
    model_parameters: dict,
    tools: list[PromptMessageTool] | None,
    stop: list[str] | None,
) -> dict:
    system, messages = convert_messages(prompt_messages)
    params = dict(model_parameters or {})

    max_tokens = int(params.get("max_tokens") or 4096)
    args: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if system:
        args["system"] = system
    # anthropic SDK >= 2.x no longer types sampling params (temperature/top_p/
    # top_k are absent from messages.create); extra_body merges them verbatim
    # into the top level of the JSON body — the traditional Anthropic wire
    # format these gateways expect.
    sampling = {
        name: params[name]
        for name in PASSTHROUGH_PARAMS
        if params.get(name) is not None
    }
    if sampling:
        args["extra_body"] = sampling
    if stop:
        args["stop_sequences"] = stop

    if tools and credentials.get("function_calling_type", "tool_call") != "no_call":
        args["tools"] = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    if params.get("enable_thinking"):
        budget = int(params.get("thinking_budget") or 0)
        if budget < 1024:
            budget = min(8192, max_tokens - 1)
        if budget >= 1024 and max_tokens > budget:
            args["thinking"] = {"type": "enabled", "budget_tokens": budget}

    return args


def _usage(llm: Any, model: str, credentials: dict, inp: int, out: int):
    return llm._calc_response_usage(model, credentials, inp or 0, out or 0)


def to_llm_result(
    llm: Any,
    model: str,
    credentials: dict,
    prompt_messages: list[PromptMessage],
    response: Any,
) -> LLMResult:
    thinking: list[str] = []
    texts: list[str] = []
    calls: list[AssistantPromptMessage.ToolCall] = []
    for block in response.content or []:
        block_type = getattr(block, "type", "")
        if block_type == "thinking":
            thinking.append(getattr(block, "thinking", "") or "")
        elif block_type == "text":
            texts.append(getattr(block, "text", "") or "")
        elif block_type == "tool_use":
            calls.append(
                AssistantPromptMessage.ToolCall(
                    id=block.id,
                    type="function",
                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                        name=block.name,
                        arguments=json.dumps(
                            block.input or {}, ensure_ascii=False, default=str
                        ),
                    ),
                )
            )
    content = "".join(texts)
    if thinking:
        content = "<think>\n" + "\n".join(thinking) + "\n</think>" + content
    usage = _usage(
        llm,
        model,
        credentials,
        response.usage.input_tokens if response.usage else 0,
        response.usage.output_tokens if response.usage else 0,
    )
    return LLMResult(
        model=getattr(response, "model", None) or model,
        prompt_messages=list(prompt_messages),
        message=AssistantPromptMessage(content=content, tool_calls=calls),
        usage=usage,
    )


def stream_chunks(
    llm: Any,
    client: Anthropic,
    args: dict,
    model: str,
    credentials: dict,
    prompt_messages: list[PromptMessage],
) -> Generator[LLMResultChunk, None, None]:
    return _stream_inner(llm, client, args, model, credentials, prompt_messages)


def _stream_inner(
    llm: Any,
    client: Anthropic,
    args: dict,
    model: str,
    credentials: dict,
    prompt_messages: list[PromptMessage],
) -> Generator[LLMResultChunk, None, None]:
    del prompt_messages  # chunks carry deltas only; parity with OAICompat behaviour

    def chunk(text: str) -> LLMResultChunk:
        return LLMResultChunk(
            model=model,
            delta=LLMResultChunkDelta(
                index=0, message=AssistantPromptMessage(content=text)
            ),
        )

    in_thinking = False
    input_tokens = 0
    output_tokens = 0
    stop_reason = "stop"
    tool_fragments: dict[int, dict[str, str]] = {}

    events = client.messages.create(stream=True, **args)
    for event in events:
        event_type = getattr(event, "type", None)
        if event_type == "message_start":
            usage = getattr(event.message, "usage", None)
            input_tokens = getattr(usage, "input_tokens", 0) or 0
        elif event_type == "content_block_start":
            block = event.content_block
            if getattr(block, "type", "") == "tool_use":
                tool_fragments[event.index] = {
                    "id": getattr(block, "id", "") or "",
                    "name": getattr(block, "name", "") or "",
                    "arguments": "",
                }
        elif event_type == "content_block_delta":
            delta = event.delta
            delta_type = getattr(delta, "type", "")
            if delta_type == "thinking_delta":
                piece = getattr(delta, "thinking", "") or ""
                if piece:
                    if not in_thinking:
                        piece = "<think>\n" + piece
                        in_thinking = True
                    yield chunk(piece)
            elif delta_type == "text_delta":
                piece = getattr(delta, "text", "") or ""
                if piece:
                    if in_thinking:
                        piece = "\n</think>" + piece
                        in_thinking = False
                    yield chunk(piece)
            elif delta_type == "input_json_delta":
                fragment = tool_fragments.get(event.index)
                if fragment is not None:
                    fragment["arguments"] += getattr(delta, "partial_json", "") or ""
        elif event_type == "message_delta":
            stop_reason = (
                getattr(event.delta, "stop_reason", None) or stop_reason
            )
            usage = getattr(event, "usage", None)
            if usage is not None and getattr(usage, "output_tokens", None):
                output_tokens = usage.output_tokens

    if in_thinking:
        yield chunk("\n</think>")

    tool_calls = [
        AssistantPromptMessage.ToolCall(
            id=fragment["id"],
            type="function",
            function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                name=fragment["name"],
                arguments=fragment["arguments"] or "{}",
            ),
        )
        for _, fragment in sorted(tool_fragments.items())
    ]
    yield LLMResultChunk(
        model=model,
        delta=LLMResultChunkDelta(
            index=0,
            message=AssistantPromptMessage(content="", tool_calls=tool_calls),
            finish_reason=_FINISH_MAP.get(stop_reason, stop_reason or "stop"),
            usage=_usage(llm, model, credentials, input_tokens, output_tokens),
        ),
    )


def generate(
    llm: Any,
    model: str,
    credentials: dict,
    prompt_messages: list[PromptMessage],
    model_parameters: dict,
    tools: list[PromptMessageTool] | None = None,
    stop: list[str] | None = None,
    stream: bool = True,
) -> Union[LLMResult, Generator[LLMResultChunk, None, None]]:
    client = build_client(credentials)
    args = build_request_args(
        model, credentials, prompt_messages, model_parameters, tools, stop
    )
    if stream:
        return stream_chunks(llm, client, args, model, credentials, prompt_messages)
    response = client.messages.create(**args)
    return to_llm_result(llm, model, credentials, prompt_messages, response)
