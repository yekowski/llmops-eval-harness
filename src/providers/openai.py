from typing import Optional
from src.providers.openai_compatible import OpenAICompatibleProvider

class OpenAIProvider(OpenAICompatibleProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        super().__init__(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            model=model,
            api_key_env_var="OPENAI_API_KEY"
        )
