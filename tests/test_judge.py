import pytest
from unittest.mock import AsyncMock, MagicMock
from src.evaluation.judge import LLMJudge
from src.providers.base import LLMProvider, ProviderResponse, ProviderRateLimitError, ProviderAPIError
from src.utils.cache import EvalCache

@pytest.mark.asyncio
async def test_judge_evaluate_markdown_json():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.model = "gemini-3.5-flash"
    mock_provider.api_key = "test-api-key"
    mock_provider.generate = AsyncMock(return_value=ProviderResponse(
        text='```json\n{"faithfulness": 0.95, "answer_relevance": 0.90, "correctness": 0.88, "faithfulness_reasoning": "Accurate"}\n```',
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=120.0
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
async def test_judge_dynamic_sla_thresholds():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.model = "gemini-3.5-flash"
    mock_provider.api_key = "test-api-key"
    mock_provider.generate = AsyncMock(return_value=ProviderResponse(
        text='{"faithfulness": 0.85, "answer_relevance": 0.85, "correctness": 0.85}',
        prompt_tokens=50,
        completion_tokens=25,
        latency_ms=100.0
    ))

    # Standard 0.80 thresholds pass with 0.85
    judge_pass = LLMJudge(provider=mock_provider, sla_thresholds={"min_faithfulness": 0.80, "min_relevance": 0.80, "min_correctness": 0.80})
    res_pass = await judge_pass.evaluate("ctx", "exp", "gen")
    assert res_pass["passed"] is True

    # Strict 0.90 threshold fails with 0.85
    judge_fail = LLMJudge(provider=mock_provider, sla_thresholds={"min_faithfulness": 0.90, "min_relevance": 0.80, "min_correctness": 0.80})
    res_fail = await judge_fail.evaluate("ctx", "exp", "gen")
    assert res_fail["passed"] is False

@pytest.mark.asyncio
async def test_judge_fallback_on_provider_error():
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.model = "gemini-3.5-flash"
    mock_provider.api_key = "test-api-key"
    mock_provider.generate = AsyncMock(side_effect=ProviderRateLimitError("Rate limit 429", status_code=429))

    judge = LLMJudge(provider=mock_provider)
    result = await judge.evaluate(
        context="France's capital is Paris.",
        expected_answer="Paris",
        generated_answer="Paris is the capital of France.",
        query="What is the capital of France?"
    )

    # Should gracefully degrade to local fallback
    assert result["judge_mode"] == "fallback"
    assert "faithfulness" in result
    assert "correctness" in result
    assert "answer_relevance" in result

@pytest.mark.asyncio
async def test_judge_evaluate_retrieval(tmp_path):
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.model = "gemini-3.5-flash"
    mock_provider.api_key = "test-api-key"
    mock_provider.generate = AsyncMock(return_value=ProviderResponse(
        text='{"context_precision": 0.92, "context_recall": 0.88, "context_precision_reasoning": "High precision", "context_recall_reasoning": "High recall"}',
        prompt_tokens=80,
        completion_tokens=40,
        latency_ms=150.0
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
