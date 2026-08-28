from typing import Optional
from src.providers.openai_compatible import OpenAICompatibleProvider

class GroqProvider(OpenAICompatibleProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "llama3-8b-8192", **kwargs):
        super().__init__(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            model=model,
            api_key_env_var="GROQ_API_KEY",
            **kwargs
        )
