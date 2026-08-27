from src.providers.base import LLMProvider

class MockProvider(LLMProvider):
    async def generate(self, prompt: str, **kwargs) -> str:
        """Instantly returns a mocked correct response string with scorecard metrics."""
        return (
            '{"faithfulness": 0.9, "faithfulness_reasoning": "Mocked test response", '
            '"answer_relevance": 0.9, "answer_relevance_reasoning": "Mocked test response", '
            '"correctness": 0.9, "correctness_reasoning": "Mocked test response"}'
        )
