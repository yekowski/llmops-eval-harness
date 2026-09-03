import json
import pytest
from pydantic import ValidationError
from run_eval import enforce_slas_and_report, _compute_deltas, _evaluate_sla_gates
from src.schemas.models import HarnessConfig, ConcurrencyConfig
from src.evaluation.judge import LLMJudge
from src.providers.mock import MockProvider

@pytest.fixture
def sample_metrics():
    return {
        "pass_rate_pct": 100.0,
        "avg_latency_ms": 1500.0,
        "sut_cost_usd": 0.0005,
        "judge_cost_usd": 0.0005,
        "total_cost_usd": 0.001,
        "passed_count": 5,
        "total_count": 5,
        "avg_faithfulness": 0.95,
        "avg_relevance": 0.90,
        "avg_correctness": 0.92,
        "avg_context_precision": 0.88,
        "avg_context_recall": 0.85,
        "has_fallback": False,
        "fallback_generation_count": 0,
        "fallback_retrieval_count": 0,
        "uncertified_judge_count": 0
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
        judge=judge,
        judge_failure_policy="fail"
    )
    assert passed is True

def test_sla_failure_fallback_strict_ci(tmp_path, monkeypatch, sample_metrics, default_thresholds):
    monkeypatch.chdir(tmp_path)
    sample_metrics["has_fallback"] = True
    sample_metrics["fallback_generation_count"] = 1
    judge = LLMJudge(provider=MockProvider())

    passed = enforce_slas_and_report(
        metrics=sample_metrics,
        sla_thresholds=default_thresholds,
        config_path="configs/test.yaml",
        dataset_path="datasets/test.json",
        sut_provider=MockProvider(),
        judge=judge,
        judge_failure_policy="fail"
    )
    assert passed is False

def test_sla_warn_policy_fallback(tmp_path, monkeypatch, sample_metrics, default_thresholds):
    """Verifies that judge_failure_policy='warn' permits fallback runs while setting warning metadata."""
    monkeypatch.chdir(tmp_path)
    sample_metrics["has_fallback"] = True
    sample_metrics["fallback_generation_count"] = 1
    judge = LLMJudge(provider=MockProvider())

    passed = enforce_slas_and_report(
        metrics=sample_metrics,
        sla_thresholds=default_thresholds,
        config_path="configs/test.yaml",
        dataset_path="datasets/test.json",
        sut_provider=MockProvider(),
        judge=judge,
        judge_failure_policy="warn"
    )
    assert passed is True
    assert sample_metrics.get("judge_failure_policy_warning") is True

def test_sla_allow_fallback_dev_mode(tmp_path, monkeypatch, sample_metrics, default_thresholds):
    monkeypatch.chdir(tmp_path)
    sample_metrics["has_fallback"] = True
    sample_metrics["fallback_generation_count"] = 1
    judge = LLMJudge(provider=MockProvider())

    passed = enforce_slas_and_report(
        metrics=sample_metrics,
        sla_thresholds=default_thresholds,
        config_path="configs/test.yaml",
        dataset_path="datasets/test.json",
        sut_provider=MockProvider(),
        judge=judge,
        judge_failure_policy="allow"
    )
    assert passed is True

def test_config_validation_strict_extra_forbid():
    """HarnessConfig must forbid unknown arbitrary top-level keys."""
    valid_cfg = {"judge_failure_policy": "fail", "concurrency": {"max_workers": 4, "requests_per_second": 2.0}}
    HarnessConfig.model_validate(valid_cfg)

    invalid_cfg = {"judge_failure_policy": "fail", "unknown_rogue_key": 123}
    with pytest.raises(ValidationError):
        HarnessConfig.model_validate(invalid_cfg)

def test_config_validation_concurrency_bounds():
    """Concurrency max_workers must be >= 1, requests_per_second > 0."""
    with pytest.raises(ValidationError):
        ConcurrencyConfig(max_workers=0)

    with pytest.raises(ValidationError):
        ConcurrencyConfig(requests_per_second=0.0)

    with pytest.raises(ValidationError):
        ConcurrencyConfig(requests_per_second=-1.5)

def test_config_validation_invalid_judge_policy():
    with pytest.raises(ValidationError):
        HarnessConfig(judge_failure_policy="invalid_mode")  # type: ignore[arg-type]

def test_sla_failure_faithfulness(tmp_path, monkeypatch, sample_metrics, default_thresholds):
    monkeypatch.chdir(tmp_path)
    sample_metrics["avg_faithfulness"] = 0.50
    judge = LLMJudge(provider=MockProvider())

    passed = enforce_slas_and_report(
        metrics=sample_metrics,
        sla_thresholds=default_thresholds,
        config_path="configs/test.yaml",
        dataset_path="datasets/test.json",
        sut_provider=MockProvider(),
        judge=judge,
        judge_failure_policy="fail"
    )
    assert passed is False

def test_sla_failure_latency(tmp_path, monkeypatch, sample_metrics, default_thresholds):
    monkeypatch.chdir(tmp_path)
    sample_metrics["avg_latency_ms"] = 5000.0
    judge = LLMJudge(provider=MockProvider())

    passed = enforce_slas_and_report(
        metrics=sample_metrics,
        sla_thresholds=default_thresholds,
        config_path="configs/test.yaml",
        dataset_path="datasets/test.json",
        sut_provider=MockProvider(),
        judge=judge,
        judge_failure_policy="fail"
    )
    assert passed is False

def test_sla_score_regression_drop(tmp_path, monkeypatch, sample_metrics, default_thresholds):
    monkeypatch.chdir(tmp_path)
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

    sample_metrics["avg_faithfulness"] = 0.90
    judge = LLMJudge(provider=MockProvider())

    passed = enforce_slas_and_report(
        metrics=sample_metrics,
        sla_thresholds=default_thresholds,
        config_path="configs/test.yaml",
        dataset_path="datasets/test.json",
        sut_provider=MockProvider(),
        judge=judge,
        judge_failure_policy="fail"
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
    failures = _evaluate_sla_gates(sample_metrics, default_thresholds, judge_failure_policy="fail")
    assert len(failures) == 0

    sample_metrics["avg_faithfulness"] = 0.50
    failures = _evaluate_sla_gates(sample_metrics, default_thresholds, judge_failure_policy="fail")
    assert len(failures) == 1
    assert "faithfulness" in failures[0]

def test_sla_uncertified_judge_provenance_fails_closed(tmp_path, monkeypatch, sample_metrics, default_thresholds):
    """Verifies that local/mock/unknown judge provenance strictly fails closed when judge_failure_policy='fail'."""
    monkeypatch.chdir(tmp_path)
    sample_metrics["uncertified_judge_count"] = 2
    sample_metrics["has_fallback"] = False
    judge = LLMJudge(provider=MockProvider())

    passed = enforce_slas_and_report(
        metrics=sample_metrics,
        sla_thresholds=default_thresholds,
        config_path="configs/test.yaml",
        dataset_path="datasets/test.json",
        sut_provider=MockProvider(),
        judge=judge,
        judge_failure_policy="fail"
    )
    assert passed is False

