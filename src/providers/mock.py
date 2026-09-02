from src.providers.base import LLMProvider, ProviderResponse

class MockProvider(LLMProvider):
    def __init__(self, model: str = "mock"):
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, prompt: str, **kwargs) -> ProviderResponse:
        """Instantly returns a mocked correct response string with scorecard metrics or echo for SUT."""
        if "context_precision" in prompt or "retrieved_contexts" in prompt:
            text = (
                '{"context_precision": 0.9, "context_precision_reasoning": "Mocked retrieval test response", '
                '"context_recall": 0.9, "context_recall_reasoning": "Mocked retrieval test response"}'
            )
        elif "LLM Judge" in prompt or "untrusted_rag_output" in prompt:
            text = (
                '{"faithfulness": 0.9, "faithfulness_reasoning": "Mocked test response", '
                '"answer_relevance": 0.9, "answer_relevance_reasoning": "Mocked test response", '
                '"correctness": 0.9, "correctness_reasoning": "Mocked test response"}'
            )
        else:
            text = prompt

        return ProviderResponse(
            text=text,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(text) // 4),
            latency_ms=1.0,
            provider_name="MockProvider",
            model_name=self.model,
            execution_mode="mock"
        )
