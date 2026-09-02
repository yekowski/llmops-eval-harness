import pytest
from unittest.mock import AsyncMock, MagicMock
from src.runners.async_runner import run_evaluation
from src.schemas.models import DatasetEntry, EvaluationResult
from src.clients.base import SystemUnderTest
from src.evaluation.judge import LLMJudge

@pytest.mark.asyncio
async def test_run_evaluation_with_judge():
    mock_sut = MagicMock(spec=SystemUnderTest)
    mock_sut.execute = AsyncMock(return_value="Answer from SUT")

    mock_judge = MagicMock(spec=LLMJudge)
    mock_judge.cache = None
    mock_judge.evaluate = AsyncMock(return_value={
        "passed": True,
        "faithfulness": 0.95,
        "answer_relevance": 0.90,
        "correctness": 0.92,
        "judge_mode": "llm"
    })
    mock_judge.evaluate_retrieval = AsyncMock(return_value={
        "context_precision": 0.90,
        "context_recall": 0.85,
        "judge_mode": "llm"
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

    results = await run_evaluation(entries, mock_sut, mock_judge)

    assert len(results) == 5
    for r in results:
        assert isinstance(r, EvaluationResult)
        assert r.passed is True
        assert r.faithfulness == 0.95
        assert r.answer_relevance == 0.90
        assert r.correctness == 0.92
        assert r.context_precision == 0.90
        assert r.context_recall == 0.85
        assert r.latency > 0.0

@pytest.mark.asyncio
async def test_run_evaluation_without_judge():
    mock_sut = MagicMock(spec=SystemUnderTest)
    mock_sut.execute = AsyncMock(return_value="Direct answer")

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
    mock_sut.execute = AsyncMock(side_effect=RuntimeError("SUT connection failed"))

    entries = [
        DatasetEntry(query="Failed Query", expected_context="C", expected_answer="A")
    ]

    results = await run_evaluation(entries, mock_sut, judge=None)

    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].tokens == 0
