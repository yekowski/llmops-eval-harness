import json
import pytest
from run_eval import enforce_slas_and_report, _compute_deltas, _evaluate_sla_gates
from src.evaluation.judge import LLMJudge
from src.providers.mock import MockProvider

@pytest.fixture
def sample_metrics():
    return {
        "pass_rate_pct": 100.0,
        "avg_latency_ms": 1500.0,
        "total_cost_usd": 0.001,
        "passed_count": 5,
        "total_count": 5,
        "avg_faithfulness": 0.95,
        "avg_relevance": 0.90,
        "avg_correctness": 0.92,
        "avg_context_precision": 0.88,
        "avg_context_recall": 0.85
    }

@pytest.fixture
def default_thresholds():
    return {
        "min_faithfulness": 0.85,
        "min_relevance": 0.80,
        "min_correctness": 0.85,
        "min_context_precision": 0.75,
        "min_context_recall": 0.75,
        "min_pass_rate": 0.90,
        "max_latency_ms": 3000.0,
        "max_cost_usd": 0.05,
        "max_score_drop": 0.05
    }

def test_sla_pass_case(tmp_path, monkeypatch, sample_metrics, default_thresholds):
    monkeypatch.chdir(tmp_path)
    judge = LLMJudge(provider=MockProvider())

    passed = enforce_slas_and_report(
        metrics=sample_metrics,
        sla_thresholds=default_thresholds,
        config_path="configs/test.yaml",
        dataset_path="datasets/test.json",
        sut_provider=MockProvider(),
        judge=judge
    )
    assert passed is True

def test_sla_failure_faithfulness(tmp_path, monkeypatch, sample_metrics, default_thresholds):
    monkeypatch.chdir(tmp_path)
    sample_metrics["avg_faithfulness"] = 0.50  # Below required 0.85
    judge = LLMJudge(provider=MockProvider())

    passed = enforce_slas_and_report(
        metrics=sample_metrics,
        sla_thresholds=default_thresholds,
        config_path="configs/test.yaml",
        dataset_path="datasets/test.json",
        sut_provider=MockProvider(),
        judge=judge
    )
    assert passed is False

def test_sla_failure_latency(tmp_path, monkeypatch, sample_metrics, default_thresholds):
    monkeypatch.chdir(tmp_path)
    sample_metrics["avg_latency_ms"] = 5000.0  # Exceeds max 3000.0 ms
    judge = LLMJudge(provider=MockProvider())

    passed = enforce_slas_and_report(
        metrics=sample_metrics,
        sla_thresholds=default_thresholds,
        config_path="configs/test.yaml",
        dataset_path="datasets/test.json",
        sut_provider=MockProvider(),
        judge=judge
    )
    assert passed is False

def test_sla_score_regression_drop(tmp_path, monkeypatch, sample_metrics, default_thresholds):
    monkeypatch.chdir(tmp_path)
    # Create baseline file with high historical faithfulness score
    baseline_data = {
        "pass_rate_pct": 100.0,
        "avg_latency_ms": 1000.0,
        "total_cost_usd": 0.001,
        "faithfulness": 0.99,
        "answer_relevance": 0.90,
        "correctness": 0.92
    }
    with open("baseline.json", "w") as f:
        json.dump(baseline_data, f)

    # Current faithfulness is 0.90, which drops 0.09 from baseline 0.99 (exceeds max_score_drop 0.05 limit)
    sample_metrics["avg_faithfulness"] = 0.90
    judge = LLMJudge(provider=MockProvider())

    passed = enforce_slas_and_report(
        metrics=sample_metrics,
        sla_thresholds=default_thresholds,
        config_path="configs/test.yaml",
        dataset_path="datasets/test.json",
        sut_provider=MockProvider(),
        judge=judge
    )
    assert passed is False

def test_compute_deltas_pure_function(sample_metrics):
    baseline = {
        "pass_rate_pct": 90.0,
        "avg_latency_ms": 1000.0,
        "total_cost_usd": 0.0005,
        "faithfulness": 0.90,
        "answer_relevance": 0.85,
        "correctness": 0.88,
        "context_precision": 0.80,
        "context_recall": 0.80
    }
    deltas = _compute_deltas(sample_metrics, baseline)
    assert deltas["delta_pass_rate_pct"] == 10.0
    assert deltas["delta_latency_ms"] == 500.0
    assert deltas["delta_faithfulness"] == pytest.approx(0.05)

def test_evaluate_sla_gates_pure_function(sample_metrics, default_thresholds):
    # Passing case
    failures = _evaluate_sla_gates(sample_metrics, default_thresholds)
    assert len(failures) == 0

    # Failing case
    sample_metrics["avg_faithfulness"] = 0.50
    failures = _evaluate_sla_gates(sample_metrics, default_thresholds)
    assert len(failures) == 1
    assert "faithfulness" in failures[0]

