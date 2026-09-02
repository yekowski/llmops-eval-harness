import pytest
from unittest.mock import AsyncMock, MagicMock
from src.runners.async_runner import run_evaluation
from src.schemas.models import DatasetEntry, EvaluationResult, SUTExecutionResult
from src.clients.base import SystemUnderTest
from src.evaluation.judge import LLMJudge

@pytest.mark.asyncio
async def test_run_evaluation_with_judge():
    mock_sut = MagicMock(spec=SystemUnderTest)
    mock_sut.execute_detailed = AsyncMock(return_value=SUTExecutionResult(
        text="Answer from SUT",
        prompt_tokens=15,
        completion_tokens=20,
        latency_ms=100.0
    ))

    mock_judge = MagicMock(spec=LLMJudge)
    mock_judge.cache = None
    mock_judge.evaluate = AsyncMock(return_value={
        "passed": True,
        "faithfulness": 0.95,
        "answer_relevance": 0.90,
        "correctness": 0.92,
        "judge_mode": "llm",
        "judge_prompt_tokens": 100,
        "judge_completion_tokens": 50,
        "judge_cost": 0.0001
    })
    mock_judge.evaluate_retrieval = AsyncMock(return_value={
        "context_precision": 0.90,
        "context_recall": 0.85,
        "judge_mode": "llm",
        "judge_prompt_tokens": 80,
        "judge_completion_tokens": 40,
        "judge_cost": 0.00008
    })

    entries = [
        DatasetEntry(
            query=f"Query {i}",
            expected_context=f"Context {i}",
            expected_answer=f"Expected {i}",
            retrieved_contexts=[f"Chunk {i}"],
            ground_truth=f"Expected {i}"
        )
        for i in range(5)
    ]

    results = await run_evaluation(entries, mock_sut, mock_judge, concurrency_config={"max_workers": 2, "requests_per_second": 10.0})

    assert len(results) == 5
    for r in results:
        assert isinstance(r, EvaluationResult)
        assert r.passed is True
        assert r.faithfulness == 0.95
        assert r.answer_relevance == 0.90
        assert r.correctness == 0.92
        assert r.context_precision == 0.90
        assert r.context_recall == 0.85
        assert r.sut_prompt_tokens == 15
        assert r.sut_completion_tokens == 20
        assert r.judge_prompt_tokens == 180  # 100 + 80
        assert r.judge_completion_tokens == 90  # 50 + 40
        assert r.judge_mode == "llm"
        assert r.retrieval_judge_mode == "llm"

@pytest.mark.asyncio
async def test_run_evaluation_without_judge():
    mock_sut = MagicMock(spec=SystemUnderTest)
    mock_sut.execute_detailed = AsyncMock(return_value=SUTExecutionResult(
        text="Direct answer",
        prompt_tokens=5,
        completion_tokens=5,
        latency_ms=50.0
    ))

    entries = [
        DatasetEntry(query="Q1", expected_context="C1", expected_answer="A1"),
        DatasetEntry(query="Q2", expected_context="C2", expected_answer="A2")
    ]

    results = await run_evaluation(entries, mock_sut, judge=None)

    assert len(results) == 2
    assert results[0].passed is True
    assert results[0].faithfulness == 1.0
    assert results[1].passed is True

@pytest.mark.asyncio
async def test_run_evaluation_sut_error_handling():
    mock_sut = MagicMock(spec=SystemUnderTest)
    mock_sut.execute_detailed = AsyncMock(side_effect=RuntimeError("SUT connection failed"))

    entries = [
        DatasetEntry(query="Failed Query", expected_context="C", expected_answer="A")
    ]

    results = await run_evaluation(entries, mock_sut, judge=None)

    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].tokens == 0
