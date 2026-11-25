from dify_plugin import ModelProvider


class ZhiyunHZProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: dict) -> None:
        """
        This provider uses the `customizable-model` configuration method, so
        credentials (AppKey + model name) are attached to each model instance.
        Dify validates them by calling models/llm `validate_credentials`
        directly; there is nothing to check at provider level.
        """
        pass
