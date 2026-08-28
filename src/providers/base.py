import contextvars
from abc import ABC, abstractmethod

# Thread-safe & coroutine-safe context variable to store precise successful provider execution time (in seconds)
generation_latency = contextvars.ContextVar("generation_latency", default=0.0)

class ProviderRateLimitError(Exception):
    """Exception raised when a provider is rate limited (HTTP 429)."""
    pass

class ProviderAPIError(Exception):
    """Exception raised for other API/HTTP errors from the provider."""
    pass

class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Asynchronously sends a generation request to the LLM model.
        
        Args:
            prompt: The formatted prompt to send to the LLM.
            **kwargs: Extra parameters to pass to the API.

        Returns:
            The raw text response from the model.
        """
        pass
