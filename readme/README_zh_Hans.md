# 智云杭州（Zhiyun Hangzhou）

Dify 模型供应商插件，对接**智云杭州 AI 商店**（`https://aistore.hztele.com.cn`），**同一插件支持双协议，按模型选择**：OpenAI 兼容 与 Anthropic 原生。

> **社区插件声明**：本插件为第三方独立开发，与智云杭州平台无隶属或背书关系；需自行在厂商控制台获取 AppKey。

## 特性

- **双协议**（凭据表单「调用协议」字段）：
  - **OpenAI 兼容**（默认）：`POST {base}/v1/chat/completions`，`Authorization: Bearer <AppKey>`。
    思考开关按厂商要求以**请求体顶层 `enable_thinking` 字段**发送（非嵌套参数）；
    `delta.reasoning_content` 由 Dify 渲染为独立思考块。
  - **Anthropic 原生**：基于 Anthropic SDK 调 `POST {base}/v1/messages`（鉴权头以 SDK 现行行为为准：`Authorization: Bearer` + `anthropic-version`）。
    API 基础地址请填**站点根地址**（`https://aistore.hztele.com.cn`，不带 `/v1`，插件会按协议自动增删后缀）。
    思考模式映射为 Anthropic 标准 `thinking: {"type":"enabled","budget_tokens":N}` 对象
    （N 取模型参数「思考长度预算」，该参数在 OpenAI 模式下不会发送）。Anthropic API 严格校验字段：
    仅透传规范内参数（`presence_penalty` 等自动丢弃）；`thinking` 事件流与 Dify 思考块渲染对齐；
    `tool_use` 块与工具调用双向转换。
- **自定义模型模式**：按平台展示的模型名称逐个添加（如 `deepseek-r1`、`qwen3-32b`），无需等待插件更新模型清单。
- **工具调用**：默认标准 `tool_call`（OpenAI 模式下可切旧式 `function_call` 或关闭）。
- **图片识别（视觉输入）**：模型凭据里的「图片识别」开关为模型声明 vision 能力，应用即可向模型
  传入图片（URL 与 base64 两种来源均按各协议正确转换为图片块）。
- **接口流式策略**：只控制「插件→厂商接口」这一段的调用方式，与应用把回复逐字呈现给最终用户
  的「流式响应」开关是两层、互不影响。聊天/Agent 应用跟随其右侧面板的流式开关；
  **工作流/Chatflow 的 LLM 节点在应用端没有流式开关**（stream 值取决于 API 的 response_mode），
  要固定行为就在这里选：「跟随平台调用」（默认）/「始终流式调用厂商接口」（平台要非流式时插件
  聚合，适用于 `enable_thinking` 仅流式生效的模型）/「始终非流式调用厂商接口」（插件模拟流式，
  适用于网关 SSE 不稳定场景）。两种协议同样适用。
- **特有配置已按协议分离**：模型参数面板按协议动态生成——OpenAI 模式暴露 presence/frequency 惩罚
  与顶层 `enable_thinking`；Anthropic 模式暴露原生 `top_k`、`thinking_budget`（映射
  `thinking.budget_tokens`）且不出现非规范字段；凭据表单中旧式 `function_call` 选项仅在 OpenAI
  协议下可见。

## 配置步骤

1. 登录智云杭州控制台 → **密钥管理** 页，复制安全密钥（AppKey）。
2. Dify 中：**插件 → 智云杭州 → 模型设置 → 添加模型**：
   - **调用协议**：按模型选择「OpenAI 兼容」或「Anthropic 原生」；
   - **模型名称**：与平台展示完全一致；
   - **安全密钥（AppKey）**：必填，两种协议均作 `Authorization: Bearer` 发送（Anthropic SDK 2.x 现行规范）；
   - **API 基础地址**：默认 `https://aistore.hztele.com.cn/v1`（Anthropic 协议会自动去掉 `/v1` 后缀，填根地址或带 `/v1` 均可）；
   - **上下文长度 / 最大输出 tokens**：按平台公布的模型参数填写；
   - **图片识别**：模型支持图片输入时选「是」；
   - **接口流式策略**：需要固定插件→厂商的调用方式时选「始终流式 / 始终非流式」；
   - **工具调用协议**：一般保持 `tool_call`。
3. 启用模型后，即可在应用中选择使用。

## 思考模式说明

- 请求体形态：`{ "model": "...", "stream": true, "enable_thinking": true, ... }`（顶层字段）。
- 多数 OpenAI 兼容网关仅允许**流式模式**下开启思考；非流式应用请保持关闭。
- 若某思考专属模型不接受 `enable_thinking: false`，请勿开启该开关，或反馈我们调整参数定义。

## 开发与打包

需要 Python 3.12 与 Dify Plugin CLI（`brew tap langgenius/dify && brew install dify`）。

```bash
cp .env.example .env          # 从 Dify 调试弹窗填 REMOTE_INSTALL_URL / REMOTE_INSTALL_KEY
uv venv --python 3.12 && uv pip install -r requirements.txt
python -m main                # 远程调试
dify plugin package ./zhiyun-hz
```

> `manifest.yaml` 及各 YAML 的 `author` 必须与你的 GitHub 用户名一致，否则安装时报
> `plugin_unique_identifier is not valid`。

## 隐私

见 `PRIVACY.md`。仅将对话内容转发至智云杭州 API，插件本身不存储任何数据。

## 源码仓库

https://github.com/RandomMsg/dify-plugin-zhiyun-hz（欢迎提 Issue / PR）
