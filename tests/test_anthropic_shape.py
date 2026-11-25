#!/usr/bin/env python3
"""Wire-format checks for the Anthropic-native protocol path of zhiyun-hz."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dify_plugin.entities.model.message import (  # noqa: E402
    AssistantPromptMessage,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
)

from models.llm.llm import ZhiyunHZLargeLanguageModel  # noqa: E402

CAPTURED: list[dict] = []

STREAM_EVENTS = [
    ("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": "claude-test", "content": [],
            "usage": {"input_tokens": 17, "output_tokens": 1},
        },
    }),
    ("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "thinking", "thinking": ""},
    }),
    ("content_block_delta", {
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "step1 "},
    }),
    ("content_block_delta", {
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "step2"},
    }),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("content_block_start", {
        "type": "content_block_start", "index": 1,
        "content_block": {"type": "text", "text": ""},
    }),
    ("content_block_delta", {
        "type": "content_block_delta", "index": 1,
        "delta": {"type": "text_delta", "text": "Hello "},
    }),
    ("content_block_delta", {
        "type": "content_block_delta", "index": 1,
        "delta": {"type": "text_delta", "text": "world"},
    }),
    ("content_block_stop", {"type": "content_block_stop", "index": 1}),
    ("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 9},
    }),
    ("message_stop", {"type": "message_stop"}),
]

TOOL_EVENTS = [
    ("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_2", "type": "message", "role": "assistant",
            "model": "claude-test", "content": [],
            "usage": {"input_tokens": 30, "output_tokens": 1},
        },
    }),
    ("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_time", "input": {}},
    }),
    ("content_block_delta", {
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": '{"zone":'},
    }),
    ("content_block_delta", {
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": '"UTC"}'},
    }),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"output_tokens": 20},
    }),
    ("message_stop", {"type": "message_stop"}),
]


def _sse(events) -> bytes:
    out = []
    for name, payload in events:
        out.append(f"event: {name}\ndata: {json.dumps(payload)}\n\n")
    out.append("\n")
    return "".join(out).encode()


class Handler(BaseHTTPRequestHandler):
    mode = "stream"  # or "tool" / "json"

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        CAPTURED.append({
            "path": self.path,
            "x_api_key": self.headers.get("x-api-key"),
            "anthropic_version": self.headers.get("anthropic-version"),
            "authorization": self.headers.get("Authorization"),
            "body": body,
        })
        if self.mode in ("stream", "tool"):
            events = STREAM_EVENTS if self.mode == "stream" else TOOL_EVENTS
            data = _sse(events)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
        else:
            resp = {
                "id": "msg_3", "type": "message", "role": "assistant",
                "model": "claude-test",
                "content": [
                    {"type": "thinking", "thinking": "because reasons", "signature": "sig"},
                    {"type": "text", "text": "final "},
                    {"type": "text", "text": "answer"},
                    {"type": "tool_use", "id": "toolu_9", "name": "lookup",
                     "input": {"q": "x"}},
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 11, "output_tokens": 4},
            }
            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def run_server(mode: str):
    handler = type("Bound", (Handler,), {"mode": mode})
    server = HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, port


def messages():
    return [
        SystemPromptMessage(content="Be terse."),
        UserPromptMessage(content="hi"),
        AssistantPromptMessage(
            content="ok",
            tool_calls=[AssistantPromptMessage.ToolCall(
                id="toolu_prev", type="function",
                function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                    name="get_time", arguments='{"zone":"UTC"}',
                ),
            )],
        ),
        ToolPromptMessage(content="12:00", tool_call_id="toolu_prev"),
        UserPromptMessage(content="and now?"),
    ]


PARAMS = {
    "enable_thinking": True,
    "thinking_budget": 2048,
    "max_tokens": 8192,
    "temperature": 0.5,
    "presence_penalty": 1.0,     # not in Anthropic spec -> must be dropped
    "frequency_penalty": 0.5,    # not in Anthropic spec -> must be dropped
}


def main() -> None:
    llm = ZhiyunHZLargeLanguageModel([])
    # deliberately entered WITH /v1 to prove protocol normalization
    credentials = {
        "api_key": "appkey-abc",
        "api_protocol": "anthropic",
        "endpoint_url": "https://placeholder/v1",  # replaced per server below
        "function_calling_type": "tool_call",
    }

    # ---------- streaming: thinking + text ----------
    server, port = run_server("stream")
    credentials["endpoint_url"] = f"http://127.0.0.1:{port}/v1"
    chunks = list(llm._invoke(
        model="claude-test", credentials=dict(credentials),
        prompt_messages=messages(), model_parameters=dict(PARAMS),
        stop=["END"], stream=True,
    ))
    req = CAPTURED[-1]
    body = req["body"]
    print("POST", req["path"])
    print("headers: x-api-key =", req["x_api_key"], "| anthropic-version =",
          req["anthropic_version"], "| Authorization =", req["authorization"])
    print("body:", json.dumps({k: v for k, v in body.items() if k != "messages"},
                              ensure_ascii=False))
    assert req["path"] == "/v1/messages", req["path"]  # /v1 stripped, SDK re-appends
    # anthropic SDK 2.14 authenticates with Authorization: Bearer + the current
    # anthropic-version header (the legacy x-api-key scheme is gone from the
    # spec the vendor claims compatibility with). If the live gateway rejects
    # this, add default_headers={"x-api-key": ...} in anthropic_impl.build_client.
    assert req["authorization"] == "Bearer appkey-abc", req["authorization"]
    assert req["anthropic_version"]

    assert body["system"] == "Be terse."
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert body["stop_sequences"] == ["END"]
    assert body["temperature"] == 0.5
    assert body["max_tokens"] == 8192
    for forbidden in ("enable_thinking", "thinking_budget",
                      "presence_penalty", "frequency_penalty"):
        assert forbidden not in body, f"{forbidden} leaked into Anthropic body"

    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "user"], roles  # tool result merged + alternating
    asst = body["messages"][1]["content"]
    assert {"type": "tool_use", "id": "toolu_prev", "name": "get_time",
            "input": {"zone": "UTC"}} in asst, asst
    assert body["messages"][2]["content"][0]["type"] == "tool_result"

    text = "".join(
        c.delta.message.content
        for c in chunks
        if isinstance(c.delta.message.content, str)
    )
    last = chunks[-1]
    print("stream text:", repr(text))
    print("finish:", last.delta.finish_reason,
          "| usage:", last.delta.usage.prompt_tokens, last.delta.usage.completion_tokens)
    assert text == "<think>\nstep1 step2\n</think>Hello world", text
    assert last.delta.finish_reason == "stop"
    assert last.delta.usage.prompt_tokens == 17 and last.delta.usage.completion_tokens == 9
    server.shutdown()

    # ---------- streaming: tool_use accumulation ----------
    server2, port2 = run_server("tool")
    credentials2 = {**credentials, "endpoint_url": f"http://127.0.0.1:{port2}/v1"}
    chunks2 = list(llm._invoke(
        model="claude-test", credentials=dict(credentials2),
        prompt_messages=[UserPromptMessage(content="what time?")],
        model_parameters={"max_tokens": 1024}, stream=True,
    ))
    final = chunks2[-1]
    calls = final.delta.message.tool_calls
    print("tool_calls:", [(c.id, c.function.name, c.function.arguments) for c in calls])
    assert final.delta.finish_reason == "tool_calls"
    assert calls and calls[0].id == "toolu_1" and calls[0].function.name == "get_time"
    assert json.loads(calls[0].function.arguments) == {"zone": "UTC"}
    # no system/tool params in this call
    body2 = CAPTURED[-1]["body"]
    assert "thinking" not in body2 and "system" not in body2
    server2.shutdown()

    # ---------- non-streaming ----------
    server3, port3 = run_server("json")
    credentials3 = {**credentials, "endpoint_url": f"http://127.0.0.1:{port3}"}  # root, no /v1
    result = llm._invoke(
        model="claude-test", credentials=dict(credentials3),
        prompt_messages=[UserPromptMessage(content="x")],
        model_parameters={"enable_thinking": False}, stream=False,
    )
    print("blocking content:", repr(result.message.content))
    print("blocking tool_calls:", [(c.function.name, c.function.arguments)
                                   for c in result.message.tool_calls])
    assert "because reasons" in result.message.content
    assert result.message.content.startswith("<think>")
    assert "final answer" in result.message.content
    assert result.message.tool_calls[0].function.name == "lookup"
    assert json.loads(result.message.tool_calls[0].function.arguments) == {"q": "x"}
    assert result.usage.prompt_tokens == 11
    assert "thinking" not in CAPTURED[-1]["body"]  # enable_thinking=False -> no thinking obj
    server3.shutdown()

    # ---------- validate_credentials hits /v1/messages ----------
    server4, port4 = run_server("json")
    credentials4 = {**credentials, "endpoint_url": f"http://127.0.0.1:{port4}/v1"}
    llm.validate_credentials("claude-test", dict(credentials4))
    vreq = CAPTURED[-1]
    print("validate ->", vreq["path"], "| small max_tokens:", vreq["body"]["max_tokens"])
    assert vreq["path"] == "/v1/messages"
    server4.shutdown()

    # ---- stream_mode force_block / force_stream (anthropic) ----
    server5, port5 = run_server("json")
    credentials5 = {**credentials, "endpoint_url": f"http://127.0.0.1:{port5}/v1",
                    "stream_mode": "force_block"}
    out5 = list(llm._invoke(
        model="claude-test", credentials=dict(credentials5),
        prompt_messages=[UserPromptMessage(content="x")],
        model_parameters={"max_tokens": 1024}, stream=True,
    ))
    assert CAPTURED[-1]["body"].get("stream") is not True, "force_block -> vendor blocking"
    assert len(out5) == 1 and "because reasons" in out5[0].delta.message.content
    server5.shutdown()

    server6, port6 = run_server("stream")
    credentials6 = {**credentials, "endpoint_url": f"http://127.0.0.1:{port6}/v1",
                    "stream_mode": "force_stream"}
    result6 = llm._invoke(
        model="claude-test", credentials=dict(credentials6),
        prompt_messages=[UserPromptMessage(content="x")],
        model_parameters={"max_tokens": 1024}, stream=False,
    )
    assert CAPTURED[-1]["body"]["stream"] is True, "force_stream -> vendor streaming"
    assert result6.message.content == "<think>\nstep1 step2\n</think>Hello world"
    print("anthropic stream_mode wrappers OK")
    server6.shutdown()

    # ---- protocol-specific parameter rules ----
    anth_names = [r.name for r in
                  llm.get_customizable_model_schema("m", dict(credentials)).parameter_rules]
    assert "top_k" in anth_names and "thinking_budget" in anth_names
    assert "presence_penalty" not in anth_names and "frequency_penalty" not in anth_names
    cred_openai = {**credentials, "api_protocol": "openai"}
    oai_names = [r.name for r in
                 llm.get_customizable_model_schema("m", cred_openai).parameter_rules]
    assert "presence_penalty" in oai_names and "top_k" not in oai_names
    assert "enable_thinking" in anth_names and "enable_thinking" in oai_names
    print("anthropic rules:", anth_names)
    print("openai rules:   ", oai_names)

    # ---- vision feature + image block conversion ----
    cred_vision = {**credentials, "use_vision": "true"}
    schema_v = llm.get_customizable_model_schema("m", cred_vision)
    assert any(f.value == "vision" for f in schema_v.features)
    server7, port7 = run_server("json")
    from dify_plugin.entities.model.message import (
        ImagePromptMessageContent,
        TextPromptMessageContent,
    )
    llm._invoke(
        model="claude-test",
        credentials={**cred_vision, "endpoint_url": f"http://127.0.0.1:{port7}/v1"},
        prompt_messages=[UserPromptMessage(content=[
            TextPromptMessageContent(data="look"),
            ImagePromptMessageContent(
                url="https://example.com/a.png", format="png", mime_type="image/png",
            ),
        ])],
        model_parameters={"max_tokens": 1024},
        stream=False,
    )
    blocks = CAPTURED[-1]["body"]["messages"][0]["content"]
    assert {"type": "text", "text": "look"} in blocks
    assert blocks[-1]["type"] == "image" and blocks[-1]["source"]["url"].endswith("a.png")
    print("vision image block:", json.dumps(blocks[-1]))
    llm._invoke(
        model="claude-test",
        credentials={**cred_vision, "endpoint_url": f"http://127.0.0.1:{port7}/v1"},
        prompt_messages=[UserPromptMessage(content=[
            ImagePromptMessageContent(
                base64_data="aGVsbG8=", format="jpeg", mime_type="image/jpeg",
            ),
        ])],
        model_parameters={"max_tokens": 1024},
        stream=False,
    )
    src = CAPTURED[-1]["body"]["messages"][0]["content"][-1]["source"]
    assert src == {"type": "base64", "media_type": "image/jpeg", "data": "aGVsbG8="}, src
    print("vision base64 block:", json.dumps(src))
    server7.shutdown()

    print("\nALL ANTHROPIC CHECKS PASSED")


if __name__ == "__main__":
    main()
