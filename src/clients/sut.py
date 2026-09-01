from src.clients.base import SystemUnderTest
from src.providers.base import LLMProvider

class LLMProviderSUT(SystemUnderTest):
    """Adapter that wraps an LLMProvider instance as a SystemUnderTest interface."""
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def execute(self, query: str) -> str:
        return await self.provider.generate(query)
