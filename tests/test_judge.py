import pytest
from unittest.mock import AsyncMock, MagicMock
from src.evaluation.judge import LLMJudge
from src.providers.base import LLMProvider, ProviderResponse, ProviderRateLimitError, ProviderAPIError
from src.providers.mock import MockProvider
from src.providers.router import ProviderRouter
from src.utils.cache import EvalCache

@pytest.mark.asyncio
async def test_judge_evaluate_markdown_json():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.model = "gemini-3.5-flash"
    mock_provider.api_key = "test-api-key"
    mock_provider.provider_name = "GeminiProvider"
    mock_provider.generate = AsyncMock(return_value=ProviderResponse(
        text='```json\n{"faithfulness": 0.95, "answer_relevance": 0.90, "correctness": 0.88, "faithfulness_reasoning": "Accurate", "answer_relevance_reasoning": "Relevant", "correctness_reasoning": "Correct"}\n```',
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=120.0,
        provider_name="GeminiProvider",
        model_name="gemini-3.5-flash",
        execution_mode="remote"
    ))

    judge = LLMJudge(provider=mock_provider)
    result = await judge.evaluate(
        context="France's capital is Paris.",
        expected_answer="Paris",
        generated_answer="The capital of France is Paris.",
        query="What is the capital of France?"
    )

    assert result["faithfulness"] == 0.95
    assert result["answer_relevance"] == 0.90
    assert result["correctness"] == 0.88
    assert result["passed"] is True
    assert result["judge_mode"] == "llm"
    assert judge.total_cost > 0.0

@pytest.mark.asyncio
async def test_judge_local_execution_mode_classified_as_fallback():
    """Verifies that local models (e.g. Ollama, vLLM) with execution_mode='local' are classified as judge_mode='fallback'."""
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.model = "llama3.2:3b"
    mock_provider.api_key = "local"
    mock_provider.provider_name = "OllamaProvider"
    mock_provider.generate = AsyncMock(return_value=ProviderResponse(
        text='{"faithfulness": 0.90, "answer_relevance": 0.90, "correctness": 0.90, "faithfulness_reasoning": "", "answer_relevance_reasoning": "", "correctness_reasoning": ""}',
        prompt_tokens=50,
        completion_tokens=25,
        provider_name="OllamaProvider",
        model_name="llama3.2:3b",
        execution_mode="local"
    ))

    judge = LLMJudge(provider=mock_provider)
    result = await judge.evaluate("ctx", "exp", "gen", query="query")
    assert result["judge_mode"] == "fallback"

@pytest.mark.asyncio
async def test_judge_mock_execution_mode_classified_as_fallback():
    """Verifies that mock providers with execution_mode='mock' are classified as judge_mode='fallback'."""
    mock_provider = MockProvider()
    judge = LLMJudge(provider=mock_provider)
    result = await judge.evaluate("ctx", "exp", "gen", query="query")
    assert result["judge_mode"] == "fallback"

@pytest.mark.asyncio
async def test_judge_secondary_remote_fallback_caching_and_identity(tmp_path):
    """Verifies that when primary fails and secondary remote succeeds:
    1. Response metadata and cost reflect the secondary remote provider.
    2. Cache is populated under the secondary remote provider/model.
    3. Repeated evaluation hits the cache without re-invoking the remote API.
    """
    class PrimaryFailingRemote(LLMProvider):
        def __init__(self):
            self.model = "gpt-4o"
            self.api_key = "test-remote-key-1"
            self.provider_name = "OpenAIProvider"
            self.call_count = 0
            self.execution_mode = "remote"
        async def generate(self, prompt, **kwargs):
            self.call_count += 1
            raise ProviderRateLimitError("Rate limit 429", status_code=429)

    class SecondaryWorkingRemote(LLMProvider):
        def __init__(self):
            self.model = "gemini-3.5-flash"
            self.api_key = "test-remote-key-2"
            self.provider_name = "GeminiProvider"
            self.call_count = 0
            self.execution_mode = "remote"
        async def generate(self, prompt, **kwargs):
            self.call_count += 1
            return ProviderResponse(
                text='{"faithfulness": 0.95, "answer_relevance": 0.95, "correctness": 0.95, "faithfulness_reasoning": "", "answer_relevance_reasoning": "", "correctness_reasoning": ""}',
                prompt_tokens=100,
                completion_tokens=50,
                provider_name="GeminiProvider",
                model_name="gemini-3.5-flash",
                execution_mode="remote"
            )

    p1 = PrimaryFailingRemote()
    p2 = SecondaryWorkingRemote()
    router = ProviderRouter([p1, p2])
    cache = EvalCache(cache_dir=str(tmp_path))

    judge = LLMJudge(provider=router, cache=cache)

    # First run: trips p1, succeeds on p2, populates cache
    res1 = await judge.evaluate("Context text", "Expected answer", "Generated answer", query="Query text")
    assert res1["judge_mode"] == "llm"
    assert res1["faithfulness"] == 0.95
    assert p1.call_count == 1
    assert p2.call_count == 1
    assert judge.total_cost > 0.0

    # Second run: must hit cache for the active judge routing and not repeat p2 call
    res2 = await judge.evaluate("Context text", "Expected answer", "Generated answer", query="Query text")
    assert res2["judge_mode"] == "cache"
    assert res2["faithfulness"] == 0.95
    assert p2.call_count == 1  # No repeated call!

@pytest.mark.asyncio
async def test_judge_cached_raw_scores_dynamic_sla_recomputation(tmp_path):
    """Verifies that cached evaluations dynamically recompute passed boolean against new active SLA thresholds."""
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.model = "gemini-3.5-flash"
    mock_provider.api_key = "test-api-key"
    mock_provider.provider_name = "GeminiProvider"
    mock_provider.generate = AsyncMock(return_value=ProviderResponse(
        text='{"faithfulness": 0.90, "answer_relevance": 0.90, "correctness": 0.90, "faithfulness_reasoning": "", "answer_relevance_reasoning": "", "correctness_reasoning": ""}',
        prompt_tokens=50,
        completion_tokens=25,
        latency_ms=100.0,
        provider_name="GeminiProvider",
        model_name="gemini-3.5-flash",
        execution_mode="remote"
    ))

    cache = EvalCache(cache_dir=str(tmp_path))

    # Run 1: with threshold 0.80 -> passes
    judge1 = LLMJudge(provider=mock_provider, cache=cache, sla_thresholds={"min_faithfulness": 0.80, "min_relevance": 0.80, "min_correctness": 0.80})
    res1 = await judge1.evaluate("ctx", "exp", "gen", query="query")
    assert res1["passed"] is True
    assert res1["judge_mode"] == "llm"

    # Run 2: with stricter threshold 0.95 -> uses cache, but passed is recomputed to False!
    judge2 = LLMJudge(provider=mock_provider, cache=cache, sla_thresholds={"min_faithfulness": 0.95, "min_relevance": 0.95, "min_correctness": 0.95})
    res2 = await judge2.evaluate("ctx", "exp", "gen", query="query")
    assert res2["judge_mode"] == "cache"
    assert res2["faithfulness"] == 0.90
    assert res2["passed"] is False  # Must fail dynamically against 0.95 threshold!

@pytest.mark.asyncio
async def test_judge_fallback_on_provider_error():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.model = "gemini-3.5-flash"
    mock_provider.api_key = "test-api-key"
    mock_provider.provider_name = "GeminiProvider"
    mock_provider.generate = AsyncMock(side_effect=ProviderRateLimitError("Rate limit 429", status_code=429))

    judge = LLMJudge(provider=mock_provider)
    result = await judge.evaluate(
        context="France's capital is Paris.",
        expected_answer="Paris",
        generated_answer="Paris is the capital of France.",
        query="What is the capital of France?"
    )

    assert result["judge_mode"] == "fallback"
    assert "faithfulness" in result
    assert "correctness" in result
    assert "answer_relevance" in result

@pytest.mark.asyncio
async def test_judge_malformed_json_fallback():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.model = "gemini-3.5-flash"
    mock_provider.api_key = "test-api-key"
    mock_provider.provider_name = "GeminiProvider"
    mock_provider.generate = AsyncMock(return_value=ProviderResponse(
        text='NOT JSON AT ALL',
        prompt_tokens=50,
        completion_tokens=25,
        provider_name="GeminiProvider",
        model_name="gemini-3.5-flash",
        execution_mode="remote"
    ))

    judge = LLMJudge(provider=mock_provider)
    result = await judge.evaluate("Context", "Paris", "Paris", "Query")
    assert result["judge_mode"] == "fallback"

@pytest.mark.asyncio
async def test_judge_out_of_range_score_validation():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.model = "gemini-3.5-flash"
    mock_provider.api_key = "test-api-key"
    mock_provider.provider_name = "GeminiProvider"
    mock_provider.generate = AsyncMock(return_value=ProviderResponse(
        text='{"faithfulness": 5.0, "answer_relevance": 0.8, "correctness": 0.8}',
        prompt_tokens=50,
        completion_tokens=25,
        provider_name="GeminiProvider",
        model_name="gemini-3.5-flash",
        execution_mode="remote"
    ))

    judge = LLMJudge(provider=mock_provider)
    result = await judge.evaluate("Context", "Paris", "Paris", "Query")
    assert result["judge_mode"] == "fallback"

@pytest.mark.asyncio
async def test_judge_evaluate_retrieval(tmp_path):
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.model = "gemini-3.5-flash"
    mock_provider.api_key = "test-api-key"
    mock_provider.provider_name = "GeminiProvider"
    mock_provider.generate = AsyncMock(return_value=ProviderResponse(
        text='{"context_precision": 0.92, "context_recall": 0.88, "context_precision_reasoning": "High precision", "context_recall_reasoning": "High recall"}',
        prompt_tokens=80,
        completion_tokens=40,
        latency_ms=150.0,
        provider_name="GeminiProvider",
        model_name="gemini-3.5-flash",
        execution_mode="remote"
    ))

    cache = EvalCache(cache_dir=str(tmp_path))
    judge = LLMJudge(provider=mock_provider, cache=cache)
    result = await judge.evaluate_retrieval(
        query="What is photosynthesis?",
        retrieved_contexts=["Photosynthesis converts light into energy.", "Plants produce glucose."],
        ground_truth="Photosynthesis is the biological process converting light to chemical energy."
    )

    assert result["context_precision"] == 0.92
    assert result["context_recall"] == 0.88
    assert result["judge_mode"] == "llm"
    assert result["judge_cost"] > 0.0
    assert judge.total_cost > 0.0
