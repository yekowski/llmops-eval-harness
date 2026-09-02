from typing import Optional
from src.providers.openai_compatible import OpenAICompatibleProvider

class OllamaProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3.2:3b",
        **kwargs
    ):
        super().__init__(
            api_key=api_key or "ollama",
            base_url=base_url,
            model=model,
            api_key_env_var="OLLAMA_API_KEY",
            **kwargs
        )

    @property
    def health_check_url(self) -> Optional[str]:
        return self.base_url.replace("/v1", "/api/tags")
