from src.clients.base import SystemUnderTest
from src.schemas.models import SUTExecutionResult
from src.providers.base import LLMProvider, ProviderResponse

class LLMProviderSUT(SystemUnderTest):
    """Adapter that wraps an LLMProvider instance as a SystemUnderTest interface."""
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def execute(self, query: str) -> str:
        response = await self.provider.generate(query)
        if isinstance(response, ProviderResponse):
            return response.text
        return str(response)

    async def execute_detailed(self, query: str) -> SUTExecutionResult:
        response = await self.provider.generate(query)
        if isinstance(response, ProviderResponse):
            return SUTExecutionResult(
                text=response.text,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_ms=response.latency_ms
            )
        text = str(response)
        return SUTExecutionResult(
            text=text,
            prompt_tokens=max(1, len(query) // 4),
            completion_tokens=max(1, len(text) // 4),
            latency_ms=0.0
        )
