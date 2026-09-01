from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProviderResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    def __str__(self) -> str:
        return self.text

class ProviderRateLimitError(Exception):
    """Exception raised when a provider is rate limited (HTTP 429)."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code

class ProviderAPIError(Exception):
    """Exception raised for other API/HTTP errors from the provider."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code

class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ProviderResponse:
        """Asynchronously sends a generation request to the LLM model.

        Args:
            prompt: The formatted prompt to send to the LLM.
            **kwargs: Extra parameters to pass to the API.

        Returns:
            ProviderResponse containing text, token counts, and latency.
        """
        pass
