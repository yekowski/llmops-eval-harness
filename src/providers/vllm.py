from typing import Optional
from src.providers.openai_compatible import OpenAICompatibleProvider

class VLLMProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8000/v1",
        model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        **kwargs
    ):
        super().__init__(
            api_key=api_key or "vllm",
            base_url=base_url,
            model=model,
            api_key_env_var="VLLM_API_KEY",
            **kwargs
        )
