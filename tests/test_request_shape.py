#!/usr/bin/env python3
"""Wire-format checks for the zhiyun-hz model plugin (run with project venv).

Proves the vendor's key requirement:
  1. `enable_thinking` is sent at the TOP LEVEL of the POSTed JSON body
  2. Authorization uses Bearer <AppKey>, path = <base>/chat/completions
  3. streaming reasoning deltas are surfaced (wrapped in <think> tags)
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dify_plugin.entities.model import ModelType  # noqa: E402  (just ctor arg)
from dify_plugin.entities.model.llm import LLMResult  # noqa: E402
from dify_plugin.entities.model.message import (  # noqa: E402
    SystemPromptMessage,
    UserPromptMessage,
)

from models.llm.llm import ZhiyunHZLargeLanguageModel  # noqa: E402

CAPTURED: list[dict] = []

SSE_EVENTS = [
    {
        "id": "1", "object": "chat.completion.chunk", "created": 1, "model": "qwen3-test",
        "choices": [{"index": 0, "delta": {"reasoning_content": "Let me think..."}}],
    },
    {
        "id": "2", "object": "chat.completion.chunk", "created": 1, "model": "qwen3-test",
        "choices": [{"index": 0, "delta": {"content": "The answer"}}],
    },
    {
        "id": "3", "object": "chat.completion.chunk", "created": 1, "model": "qwen3-test",
        "choices": [{"index": 0, "delta": {"content": " is 42."}, "finish_reason": "stop"}],
    },
    {
        "id": "4", "object": "chat.completion.chunk", "created": 1, "model": "qwen3-test",
        "choices": [],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
    },
]


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        CAPTURED.append(
            {"path": self.path, "auth": self.headers.get("Authorization"), "body": body}
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for ev in SSE_EVENTS:
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


class _JsonHandler(_Handler):
    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        CAPTURED.append(
            {"path": self.path, "auth": self.headers.get("Authorization"), "body": body}
        )
        resp = {
            "id": "x", "object": "chat.completion", "created": 1, "model": "qwen3-test",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "42", "reasoning_content": "hmm"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    llm = ZhiyunHZLargeLanguageModel([])
    credentials = {
        "api_key": "sk-appkey-test",
        "endpoint_url": f"http://127.0.0.1:{port}/v1",
        "context_size": "32768",
        "max_tokens": "16384",
        "function_calling_type": "tool_call",
    }
    messages = [
        SystemPromptMessage(content="You are helpful."),
        UserPromptMessage(content="What is 6*7?"),
    ]

    # ---- streaming ----
    chunks = list(
        llm._invoke(
            model="qwen3-test",
            credentials=dict(credentials),
            prompt_messages=messages,
            model_parameters={
                "enable_thinking": True, "temperature": 0.7,
                "max_tokens": 100, "thinking_budget": 2048,
            },
            stream=True,
        )
    )
    req = CAPTURED[-1]
    body = req["body"]
    print("POST", req["path"], "| Auth:", req["auth"])
    print("stream body:", json.dumps(body, ensure_ascii=False))
    assert req["path"] == "/v1/chat/completions", req["path"]
    assert req["auth"] == "Bearer sk-appkey-test", req["auth"]
    assert body.get("enable_thinking") is True, "enable_thinking must be TOP-LEVEL"
    assert body["model"] == "qwen3-test" and body["stream"] is True
    assert body["temperature"] == 0.7
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "thinking_budget" not in body, "Anthropic-only param must not leak to OpenAI body"

    text = "".join(
        c.delta.message.content
        for c in chunks
        if isinstance(c.delta.message.content, str)
    )
    usage = next((c.delta.usage for c in reversed(chunks) if c.delta.usage), None)
    print("stream text:", repr(text))
    assert "The answer is 42." in text, text
    assert "Let me think..." in text, "reasoning must be surfaced"
    assert usage is not None and usage.total_tokens == 19, usage
    server.shutdown()

    # ---- non-streaming (enable_thinking=False still top-level) ----
    server2 = HTTPServer(("127.0.0.1", 0), _JsonHandler)
    port2 = server2.server_address[1]
    threading.Thread(target=server2.serve_forever, daemon=True).start()
    credentials2 = {**credentials, "endpoint_url": f"http://127.0.0.1:{port2}/v1"}
    result = llm._invoke(
        model="qwen3-test",
        credentials=credentials2,
        prompt_messages=messages,
        model_parameters={"enable_thinking": False, "max_tokens": 100},
        stream=False,
    )
    body2 = CAPTURED[-1]["body"]
    print("blocking body:", json.dumps(body2, ensure_ascii=False))
    assert body2.get("enable_thinking") is False, "must stay top-level when False"
    assert isinstance(result.message.content, str) and "42" in result.message.content
    server2.shutdown()

    # ---- stream_mode = force_stream: app asked blocking, vendor streamed ----
    server3 = HTTPServer(("127.0.0.1", 0), _Handler)
    port3 = server3.server_address[1]
    threading.Thread(target=server3.serve_forever, daemon=True).start()
    credentials3 = {**credentials, "endpoint_url": f"http://127.0.0.1:{port3}/v1",
                    "stream_mode": "force_stream"}
    result3 = llm._invoke(
        model="qwen3-test",
        credentials=credentials3,
        prompt_messages=messages,
        model_parameters={"enable_thinking": True, "max_tokens": 100},
        stream=False,  # app wants blocking
    )
    body3 = CAPTURED[-1]["body"]
    assert body3["stream"] is True, "force_stream must call vendor streaming"
    assert isinstance(result3, LLMResult)
    print("force_stream aggregated:", repr(result3.message.content[:60]),
          "| usage:", result3.usage.completion_tokens)
    assert "The answer is 42." in result3.message.content
    assert result3.usage.completion_tokens > 0
    server3.shutdown()

    # ---- stream_mode = force_block: app asked streaming, vendor blocking ----
    server4 = HTTPServer(("127.0.0.1", 0), _JsonHandler)
    port4 = server4.server_address[1]
    threading.Thread(target=server4.serve_forever, daemon=True).start()
    credentials4 = {**credentials, "endpoint_url": f"http://127.0.0.1:{port4}/v1",
                    "stream_mode": "force_block"}
    out4 = list(llm._invoke(
        model="qwen3-test",
        credentials=credentials4,
        prompt_messages=messages,
        model_parameters={"max_tokens": 100},
        stream=True,  # app wants streaming
    ))
    body4 = CAPTURED[-1]["body"]
    assert body4["stream"] is False, "force_block must call vendor blocking"
    assert len(out4) == 1 and out4[0].delta.finish_reason == "stop"
    assert "42" in out4[0].delta.message.content
    print("force_block emulated chunk:", repr(out4[0].delta.message.content[:60]))
    server4.shutdown()

    # ---- customizable schema: openai-only rules, vision feature ----
    schema = llm.get_customizable_model_schema("qwen3-test", dict(credentials))
    assert schema is not None
    names = [r.name for r in schema.parameter_rules]
    assert "enable_thinking" in names
    assert any(f.value == "tool-call" for f in schema.features)
    assert "presence_penalty" in names and "frequency_penalty" in names
    assert "top_k" not in names, "top_k belongs to Anthropic protocol only"
    assert "thinking_budget" not in names, "budget belongs to Anthropic protocol only"
    assert not any(f.value == "vision" for f in schema.features)
    schema_vision = llm.get_customizable_model_schema(
        "qwen3-test", {**credentials, "use_vision": "true"}
    )
    assert any(f.value == "vision" for f in schema_vision.features)
    print("openai schema parameter rules:", names)

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
