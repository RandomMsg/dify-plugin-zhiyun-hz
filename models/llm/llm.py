from collections.abc import Generator, Iterable
from typing import Any, Optional, Union

from dify_plugin import OAICompatLargeLanguageModel
from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    I18nObject,
    ModelFeature,
    ModelPropertyKey,
    ModelType,
    ParameterRule,
    ParameterType,
)
from dify_plugin.entities.model.llm import (
    LLMMode,
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
    LLMUsage,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageTool,
)
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)

from . import anthropic_impl

DEFAULT_ENDPOINT_URL = "https://aistore.hztele.com.cn/v1"


def _to_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


class ZhiyunHZLargeLanguageModel(OAICompatLargeLanguageModel):
    """LLM interface for Zhiyun Hangzhou (aistore.hztele.com.cn).

    The service is OpenAI-compatible. The SDK's OAICompatLargeLanguageModel
    builds the request body as:

        {"model": ..., "stream": ..., **model_parameters}

    so every parameter declared below in `parameter_rules` (including the
    thinking switch `enable_thinking`) is sent at the TOP LEVEL of the JSON
    request body, exactly as required by this vendor. Reasoning deltas
    (`delta.reasoning_content`) returned by thinking models are automatically
    wrapped into 思考标签 blocks by the base class.
    """

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        self._add_custom_parameters(credentials)
        protocol = credentials.get("api_protocol")
        stream_mode = credentials.get("stream_mode") or "auto"

        if protocol == "anthropic":
            # thinking params are translated into the standard Anthropic
            # `thinking` object inside anthropic_impl.
            def call(api_stream: bool):
                return anthropic_impl.generate(
                    self,
                    model,
                    credentials,
                    prompt_messages,
                    model_parameters,
                    tools,
                    stop,
                    api_stream,
                )
        else:
            def call(api_stream: bool):
                # thinking_budget only drives the Anthropic mapping, never
                # let it reach the /v1/chat/completions body.
                model_parameters.pop("thinking_budget", None)
                return OAICompatLargeLanguageModel._invoke(
                    self,
                    model,
                    credentials,
                    prompt_messages,
                    model_parameters,
                    tools,
                    stop,
                    api_stream,
                    user,
                )

        # Per-model streaming policy overrides what the app requested.
        if stream_mode == "force_stream" and not stream:
            return self._collapse_to_result(call(True), model, prompt_messages)
        if stream_mode == "force_block" and stream:
            return self._result_as_chunks(call(False))
        return call(stream)

    def _collapse_to_result(
        self,
        chunks: Iterable[LLMResultChunk],
        model: str,
        prompt_messages: list[PromptMessage],
    ) -> LLMResult:
        """Aggregate a streaming response into one LLMResult (force_stream mode)."""
        content = ""
        tool_calls: list[AssistantPromptMessage.ToolCall] = []
        usage: Optional[LLMUsage] = None
        for chunk in chunks:
            message = chunk.delta.message
            if isinstance(message.content, str):
                content += message.content
            if message.tool_calls:
                tool_calls = message.tool_calls
            if chunk.delta.usage:
                usage = chunk.delta.usage
        return LLMResult(
            model=model,
            prompt_messages=list(prompt_messages),
            message=AssistantPromptMessage(content=content, tool_calls=tool_calls),
            usage=usage or LLMUsage.empty_usage(),
        )

    def _result_as_chunks(self, result: LLMResult) -> Generator[LLMResultChunk, None, None]:
        """Emit a blocking response as a one-chunk generator (force_block mode)."""
        yield LLMResultChunk(
            model=result.model,
            system_fingerprint=result.system_fingerprint,
            delta=LLMResultChunkDelta(
                index=0,
                message=result.message,
                finish_reason="stop",
                usage=result.usage,
            ),
        )

    def validate_credentials(self, model: str, credentials: dict) -> None:
        self._add_custom_parameters(credentials)
        if credentials.get("api_protocol") == "anthropic":
            try:
                anthropic_impl.build_client(credentials).messages.create(
                    model=model,
                    max_tokens=5,
                    messages=[{"role": "user", "content": "ping"}],
                )
            except Exception as error:
                raise CredentialsValidateFailedError(
                    f"{type(error).__name__}: {error}"
                ) from error
            return
        super().validate_credentials(model, credentials)

    @classmethod
    def _add_custom_parameters(cls, credentials: dict) -> None:
        credentials["mode"] = LLMMode.CHAT.value
        protocol = credentials.get("api_protocol") or "openai"
        endpoint_url = (credentials.get("endpoint_url") or DEFAULT_ENDPOINT_URL).strip()
        endpoint_url = endpoint_url.rstrip("/")
        # OpenAI-compat: OAICompat posts to {endpoint_url}/chat/completions,
        # so the URL must carry the /v1 suffix.
        # Anthropic: the SDK appends /v1/messages itself, so it must NOT.
        if protocol == "anthropic":
            if endpoint_url.endswith("/v1"):
                endpoint_url = endpoint_url[:-3]
        elif not endpoint_url.endswith("/v1"):
            endpoint_url = f"{endpoint_url}/v1"
        credentials["endpoint_url"] = endpoint_url
        # `function_calling_type` comes straight from the model credential form;
        # both paths read the same key to decide tools handling.
        if not credentials.get("function_calling_type"):
            credentials["function_calling_type"] = "tool_call"

    @property
    def _invoke_error_mapping(self) -> dict:
        """Merge the base (requests/openai-compat) mapping with native
        anthropic SDK exceptions so the framework translates both paths."""
        import anthropic

        mapping = dict(super()._invoke_error_mapping)
        additions = {
            InvokeConnectionError: [
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
            ],
            InvokeServerUnavailableError: [anthropic.InternalServerError],
            InvokeRateLimitError: [anthropic.RateLimitError],
            InvokeAuthorizationError: [
                anthropic.AuthenticationError,
                anthropic.PermissionDeniedError,
            ],
            InvokeBadRequestError: [
                anthropic.BadRequestError,
                anthropic.NotFoundError,
                anthropic.UnprocessableEntityError,
                anthropic.APIStatusError,
                anthropic.APIError,
            ],
        }
        for error_type, sources in additions.items():
            merged = mapping.setdefault(error_type, [])
            for source in sources:
                if source not in merged:
                    merged.append(source)
        return mapping

    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> Optional[AIModelEntity]:
        """Declare the per-model metadata and parameter rules for models the
        user adds manually (customizable-model mode).

        The parameter panel is protocol-aware:
        - OpenAI-compat: `enable_thinking` is merged verbatim by OAICompat as a
          TOP-LEVEL request-body field (vendor requirement); presence/
          frequency penalties are exposed.
        - Anthropic-native: thinking mode becomes the standard `thinking`
          object sized by `thinking_budget`; `top_k` is native, while
          presence/frequency penalties are NOT part of the spec and stay hidden.
        """
        self._add_custom_parameters(credentials)
        protocol = credentials.get("api_protocol") or "openai"

        tool_calling = credentials.get("function_calling_type", "tool_call")
        features = (
            [
                ModelFeature.TOOL_CALL,
                ModelFeature.MULTI_TOOL_CALL,
                ModelFeature.STREAM_TOOL_CALL,
            ]
            if tool_calling == "tool_call"
            else []
        )
        if str(credentials.get("use_vision", "false")).lower() == "true":
            features.append(ModelFeature.VISION)

        max_tokens_upper = _to_int(credentials.get("max_tokens"), 16384)
        context_size = _to_int(credentials.get("context_size"), 32768)

        rules = [
            ParameterRule(
                name="temperature",
                use_template="temperature",
                label=I18nObject(en_US="Temperature", zh_Hans="温度"),
                type=ParameterType.FLOAT,
            ),
            ParameterRule(
                name="top_p",
                use_template="top_p",
                label=I18nObject(en_US="Top P", zh_Hans="核采样"),
                type=ParameterType.FLOAT,
            ),
            ParameterRule(
                name="max_tokens",
                use_template="max_tokens",
                default=4096,
                min=1,
                max=max_tokens_upper,
                label=I18nObject(en_US="Max Tokens", zh_Hans="最大标记"),
                type=ParameterType.INT,
                help=I18nObject(
                    en_US="Maximum number of tokens to generate, including thinking content for reasoning models.",
                    zh_Hans="生成的最大 token 数；思考类模型的思考过程同样计入。",
                ),
            ),
        ]

        if protocol == "anthropic":
            rules += [
                ParameterRule(
                    name="top_k",
                    type=ParameterType.INT,
                    default=None,
                    min=1,
                    required=False,
                    label=I18nObject(en_US="Top K", zh_Hans="Top K"),
                    help=I18nObject(
                        en_US="Anthropic-native sampling: only the top K tokens are considered.",
                        zh_Hans="Anthropic 原生采样参数：仅在前 K 个 token 中采样。",
                    ),
                ),
                ParameterRule(
                    name="thinking_budget",
                    type=ParameterType.INT,
                    default=8192,
                    min=1024,
                    max=max_tokens_upper,
                    required=False,
                    label=I18nObject(
                        en_US="Thinking Budget", zh_Hans="思考长度预算"
                    ),
                    help=I18nObject(
                        en_US=(
                            "Anthropic thinking.budget_tokens; only effective "
                            "when thinking mode is on and must stay below "
                            "max_tokens."
                        ),
                        zh_Hans=(
                            "映射为 Anthropic 标准 thinking.budget_tokens；"
                            "仅思考模式开启时生效，且必须小于 max_tokens。"
                        ),
                    ),
                ),
                ParameterRule(
                    name="enable_thinking",
                    type=ParameterType.BOOLEAN,
                    default=False,
                    required=False,
                    label=I18nObject(en_US="Enable Thinking", zh_Hans="思考模式"),
                    help=I18nObject(
                        en_US=(
                            "Anthropic-native mode: sends the standard "
                            "thinking={type:enabled, budget_tokens:N} object. "
                            "If the gateway only supports thinking while "
                            "streaming, set this model's API Streaming Mode to "
                            "always-streaming."
                        ),
                        zh_Hans=(
                            "Anthropic 原生协议下映射为标准 thinking 对象"
                            "（type=enabled，budget_tokens 取「思考长度预算」）。"
                            "若网关仅支持流式下开启思考，请将本模型的"
                            "「接口流式策略」设为始终流式。"
                        ),
                    ),
                ),
            ]
        else:
            rules += [
                ParameterRule(
                    name="presence_penalty",
                    use_template="presence_penalty",
                    label=I18nObject(en_US="Presence Penalty", zh_Hans="存在惩罚"),
                    type=ParameterType.FLOAT,
                ),
                ParameterRule(
                    name="frequency_penalty",
                    use_template="frequency_penalty",
                    label=I18nObject(en_US="Frequency Penalty", zh_Hans="频率惩罚"),
                    type=ParameterType.FLOAT,
                ),
                ParameterRule(
                    name="enable_thinking",
                    type=ParameterType.BOOLEAN,
                    default=False,
                    required=False,
                    label=I18nObject(en_US="Enable Thinking", zh_Hans="思考模式"),
                    help=I18nObject(
                        en_US=(
                            "OpenAI-compatible mode: sent as a top-level "
                            "`enable_thinking` field of the request body. Most "
                            "gateways accept enable_thinking=true only in "
                            "streaming mode — use API Streaming Mode "
                            "always-streaming if the app runs non-streaming."
                        ),
                        zh_Hans=(
                            "OpenAI 兼容协议下以请求体顶层 enable_thinking 字段传递。"
                            "多数网关仅支持流式模式下开启思考；应用非流式时请将"
                            "「接口流式策略」设为始终流式。"
                        ),
                    ),
                ),
            ]

        return AIModelEntity(
            model=model,
            label=I18nObject(en_US=model, zh_Hans=model),
            model_type=ModelType.LLM,
            features=features,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: context_size,
                ModelPropertyKey.MODE: LLMMode.CHAT.value,
            },
            parameter_rules=rules,
        )
