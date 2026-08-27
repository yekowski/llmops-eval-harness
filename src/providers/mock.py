from src.providers.base import LLMProvider

class MockProvider(LLMProvider):
    async def generate(self, prompt: str, **kwargs) -> str:
        """Instantly returns a mocked correct response string."""
        return '{"explanation": "Mocked test response", "passed": true}'
