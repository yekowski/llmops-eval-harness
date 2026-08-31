from src.providers.base import LLMProvider, generation_latency

class MockProvider(LLMProvider):
    async def generate(self, prompt: str, **kwargs) -> str:
        """Instantly returns a mocked correct response string with scorecard metrics."""
        generation_latency.set(0.001)
        if "context_precision" in prompt or "retrieved_contexts" in prompt:
            return (
                '{"context_precision": 0.9, "context_precision_reasoning": "Mocked retrieval test response", '
                '"context_recall": 0.9, "context_recall_reasoning": "Mocked retrieval test response"}'
            )
        return (
            '{"faithfulness": 0.9, "faithfulness_reasoning": "Mocked test response", '
            '"answer_relevance": 0.9, "answer_relevance_reasoning": "Mocked test response", '
            '"correctness": 0.9, "correctness_reasoning": "Mocked test response"}'
        )
