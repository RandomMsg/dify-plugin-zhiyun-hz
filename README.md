# Zhiyun Hangzhou

Model provider plugin for Dify, connecting to the **Zhiyun Hangzhou AI Store**
(`https://aistore.hztele.com.cn`) over **both** of its APIs: OpenAI-compatible and
Anthropic-native, selectable per model.

> **Community plugin** — independently developed; not affiliated with or endorsed by the Zhiyun
> Hangzhou platform. Requires your own AppKey from the vendor's console.

## Features

- **Dual protocol, per model** (credential field *API Protocol*):
  - **OpenAI-compatible** (default): `POST {base}/v1/chat/completions`, `Authorization: Bearer <AppKey>`.
    The thinking switch is sent as a **top-level `enable_thinking` field of the JSON request body**,
    exactly as this vendor requires (not nested); `delta.reasoning_content` is rendered by Dify as
    a separate thinking block.
  - **Anthropic-native**: `POST {base}/v1/messages` via the Anthropic SDK (auth headers exactly as
    the current SDK emits: `Authorization: Bearer <AppKey>` + `anthropic-version`).
    Enter the **site root** as the base URL (`https://aistore.hztele.com.cn`, no `/v1`; the plugin
    normalizes the suffix per protocol automatically). Thinking mode maps to the standard
    `thinking: {"type": "enabled", "budget_tokens": N}` object, with N = the *Thinking Budget* model
    parameter (which is never sent in OpenAI mode). Only Anthropic-spec fields are forwarded —
    non-spec ones like `presence_penalty` are dropped (strict API). `thinking` content blocks/deltas
    render as the same Dify thinking block, and `tool_use` blocks round-trip as tool calls.
- **Customizable-model mode**: add any model listed on the platform by its exact model name
  (e.g. `deepseek-r1`, `qwen3-32b`) — no need to wait for a plugin update.
- **Tool calling**: OpenAI-standard `tool_call` by default (also supports legacy `function_call`
  in OpenAI mode or disabled, per model).
- **Vision (image recognition)**: per-model *Vision Support* toggle adds the `vision` capability,
  so Dify apps can feed images to the model (URL and base64 inputs are both converted properly
  for each protocol).
- **API Streaming Mode**: per-model policy that pins the **plugin→vendor** call only (separate
  from the app's own *Streaming Response* toggle that controls what the end user sees). Note
  workflow/Chatflow LLM nodes have **no app-side streaming switch** — there the stream flag comes
  from the API's `response_mode`, so this setting is where you fix it. Choices:
  `Follow platform call` (default), `Always call vendor streaming` (aggregated when Dify asked
  for blocking; required for gateways where `enable_thinking` only works while streaming), or
  `Always call vendor blocking` (stream emulated; useful when the gateway's SSE is flaky).
  Works identically on both protocols.
- **Protocol-specific configuration is separated**: the parameter panel is generated per protocol
  — OpenAI mode exposes presence/frequency penalties and top-level `enable_thinking`; Anthropic
  mode exposes native `top_k`, `thinking_budget` (→ `thinking.budget_tokens`) and drops non-spec
  fields; the form itself hides the legacy `function_call` option outside OpenAI mode.

## Setup

1. Go to the Zhiyun Hangzhou console → **Key Management** page → copy your **AppKey**.
2. In Dify: **Plugins → Zhiyun Hangzhou → Model Settings → Add model**.
   - **API Protocol**: `OpenAI-compatible` or `Anthropic-native`, per model.
   - **Model Name**: exactly as listed on the platform.
   - **AppKey**: required. Sent as `Authorization: Bearer` on both protocols (Anthropic SDK 2.x
     current spec) plus the `anthropic-version` header in native mode.
   - **API Base URL**: defaults to `https://aistore.hztele.com.cn/v1`; suffix is auto-normalized
     per protocol (root for Anthropic, `/v1` for OpenAI).
   - **Context Size / Max Output Tokens**: match the values published for the model.
   - **Vision Support**: Yes for image-capable models (enables image inputs in apps).
   - **API Streaming Mode**: pin the vendor call to streaming/blocking — e.g. `enable_thinking`
     on gateways that only think while streaming, or workflows (no app-side streaming switch).
   - **Tool Calling Style**: keep `tool_call` unless the model lacks function calling.
3. Enable the model and select it in your apps.

## Thinking mode notes

- The toggle appears in each model's parameter panel (*Enable Thinking*).
- Request body shape:

  ```json
  { "model": "your-model", "stream": true, "enable_thinking": true, "temperature": 0.7, "messages": [...] }
  ```

- Many OpenAI-compatible gateways accept `enable_thinking=true` **only in streaming mode**;
  non-stream apps should leave it off.
- If a thinking-only model rejects `enable_thinking: false`, remove/disable the toggle usage for
  that model (or tell us and we can make the parameter optional per model).
- In **Anthropic-native** mode the same toggle produces the standard `thinking` object
  (`budget_tokens` taken from *Thinking Budget*, clamped below `max_tokens`); no `enable_thinking`
  field is ever sent on that path.

## Development

Requires Python 3.12 and the Dify Plugin CLI (`brew tap langgenius/dify && brew install dify`).

```bash
cp .env.example .env          # fill REMOTE_INSTALL_URL / REMOTE_INSTALL_KEY from Dify debug dialog
uv venv --python 3.12 && uv pip install -r requirements.txt
python -m main                # remote-debug the plugin against your workspace
dify plugin package ./zhiyun-hz
```

> The `author` field in `manifest.yaml` must be your GitHub handle, otherwise installing the
> package fails with `plugin_unique_identifier is not valid`.

## Privacy

See `PRIVACY.md`. Prompts are forwarded to the Zhiyun Hangzhou API only; the plugin stores nothing.

## Source

https://github.com/RandomMsg/dify-plugin-zhiyun-hz (issues and PRs welcome)
