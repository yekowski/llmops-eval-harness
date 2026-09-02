import pytest
from src.utils.cache import EvalCache
from src.evaluation.judge import LLMJudge
from src.evaluation.meta_eval import run_meta_evaluation, calculate_agreement_metrics
from src.providers.mock import MockProvider

def test_calculate_agreement_metrics():
    y_true = [True, True, False, False]
    y_pred = [True, True, False, True]
    acc, kappa = calculate_agreement_metrics(y_true, y_pred)
    assert acc == 0.75
    assert kappa > 0.0

@pytest.mark.asyncio
async def test_run_meta_evaluation_smoke(tmp_path):
    cache = EvalCache(cache_dir=str(tmp_path))
    judge = LLMJudge(provider=MockProvider(), cache=cache)

    dataset = [
        {
            "query": "What is Python?",
            "context": "Python is a programming language.",
            "expected_answer": "Python is a programming language.",
            "generated_answer": "Python is a programming language.",
            "expected_pass_boolean": True
        },
        {
            "query": "What is 2+2?",
            "context": "2+2 is 4.",
            "expected_answer": "4",
            "generated_answer": "5",
            "expected_pass_boolean": False
        }
    ]

    summary = await run_meta_evaluation(dataset, judge)
    assert "accuracy" in summary
    assert "cohens_kappa" in summary
    assert len(summary["details"]) == 2
    assert "raw_results" in summary
