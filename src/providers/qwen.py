import os
from typing import Optional
from src.providers.openai_compatible import OpenAICompatibleProvider

class QwenProvider(OpenAICompatibleProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "qwen-turbo"):
        resolved_key = api_key or os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        super().__init__(
            api_key=resolved_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model=model,
            api_key_env_var="QWEN_API_KEY"
        )
