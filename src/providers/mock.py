from src.providers.base import LLMProvider, ProviderResponse

class MockProvider(LLMProvider):
    async def generate(self, prompt: str, **kwargs) -> ProviderResponse:
        """Instantly returns a mocked correct response string with scorecard metrics."""
        if "context_precision" in prompt or "retrieved_contexts" in prompt:
            text = (
                '{"context_precision": 0.9, "context_precision_reasoning": "Mocked retrieval test response", '
                '"context_recall": 0.9, "context_recall_reasoning": "Mocked retrieval test response"}'
            )
        else:
            text = (
                '{"faithfulness": 0.9, "faithfulness_reasoning": "Mocked test response", '
                '"answer_relevance": 0.9, "answer_relevance_reasoning": "Mocked test response", '
                '"correctness": 0.9, "correctness_reasoning": "Mocked test response"}'
            )
        return ProviderResponse(text=text, prompt_tokens=len(prompt) // 4, completion_tokens=len(text) // 4, latency_ms=1.0)
