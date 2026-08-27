from typing import Optional
from src.providers.openai_compatible import OpenAICompatibleProvider

class DeepSeekProvider(OpenAICompatibleProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        super().__init__(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            model=model,
            api_key_env_var="DEEPSEEK_API_KEY"
        )
