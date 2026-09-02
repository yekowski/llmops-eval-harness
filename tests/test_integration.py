import os
import json
import hashlib
import pytest
import tempfile
import yaml

from run_eval import (
    load_and_validate_config,
    initialize_harness,
    evaluate_dataset,
    enforce_slas_and_report,
)
from src.schemas.models import DatasetEntry
from src.providers.mock import MockProvider
from src.clients.sut import LLMProviderSUT
from src.evaluation.judge import LLMJudge
from src.utils.cache import EvalCache

@pytest.mark.asyncio
async def test_end_to_end_pipeline(tmp_path, monkeypatch):
    """End-to-end test validating config loading, execution, evaluation, SLA enforcement, and history tracking."""
    monkeypatch.chdir(tmp_path)

    # 1. Create a mock dataset where query echo returns expected response
    dataset_file = tmp_path / "dataset.json"
    dataset_content = [
        {
            "query": "Python is a high-level programming language.",
            "context": "Python is a high-level programming language.",
            "expected_answer": "Python is a high-level programming language.",
            "retrieved_contexts": ["Python is a high-level programming language."],
            "ground_truth": "Python is a high-level programming language."
        }
    ]
    with open(dataset_file, "w") as f:
        json.dump(dataset_content, f)

    # 2. Create a mock config YAML with dev allow policy for mock integration test
    config_file = tmp_path / "config.yaml"
    config_content = {
        "dataset_path": str(dataset_file),
        "judge_failure_policy": "allow",
        "sla_thresholds": {
            "min_faithfulness": 0.80,
            "min_relevance": 0.80,
            "min_correctness": 0.80,
            "min_pass_rate": 0.80,
            "max_latency_ms": 5000,
            "max_cost_usd": 1.0,
            "max_score_drop": 0.10
        },
        "sut": {
            "fallback_chain": ["mock"],
            "providers": {
                "mock": {"model": "mock"}
            }
        },
        "judge": {
            "fallback_chain": ["mock"],
            "providers": {
                "mock": {"model": "mock"}
            }
        }
    }
    with open(config_file, "w") as f:
        yaml.dump(config_content, f)

    # 3. Load & validate config
    config, entries, ds_path = load_and_validate_config(str(config_file))
    assert len(entries) == 1
    assert entries[0].query == "Python is a high-level programming language."

    # 4. Initialize harness
    sut, judge, sut_provider, judge_provider = await initialize_harness(config)
    assert sut is not None
    assert judge is not None

    try:
        # 5. Evaluate dataset
        results, metrics = await evaluate_dataset(entries, sut, judge)
        assert len(results) == 1
        assert results[0].passed is True
        assert metrics["passed_count"] == 1
        assert metrics["pass_rate_pct"] == 100.0

        # Baseline file does NOT exist yet
        baseline_file = tmp_path / "baseline.json"
        assert not os.path.exists(baseline_file)

        # 6. Enforce SLAs without --update-baseline: baseline should NOT be created
        sla_passed = enforce_slas_and_report(
            metrics=metrics,
            sla_thresholds=config.get("sla_thresholds", {}),
            config_path=str(config_file),
            dataset_path=ds_path,
            sut_provider=sut_provider,
            judge=judge,
            baseline_path=str(baseline_file),
            update_baseline=False,
            judge_failure_policy="allow"
        )
        assert sla_passed is True
        assert not os.path.exists(baseline_file)  # Protected: not created automatically

        # 7. When run relied on fallback, update_baseline=True refuses to overwrite baseline
        sla_passed_fallback_attempt = enforce_slas_and_report(
            metrics=metrics,
            sla_thresholds=config.get("sla_thresholds", {}),
            config_path=str(config_file),
            dataset_path=ds_path,
            sut_provider=sut_provider,
            judge=judge,
            baseline_path=str(baseline_file),
            update_baseline=True,
            judge_failure_policy="allow"
        )
        assert sla_passed_fallback_attempt is True
        assert not os.path.exists(baseline_file)  # Refused because has_fallback is True

        # 8. For clean non-fallback runs, update_baseline=True successfully writes baseline
        clean_metrics = dict(metrics)
        clean_metrics["has_fallback"] = False
        clean_metrics["fallback_generation_count"] = 0
        clean_metrics["fallback_retrieval_count"] = 0

        sla_passed_clean = enforce_slas_and_report(
            metrics=clean_metrics,
            sla_thresholds=config.get("sla_thresholds", {}),
            config_path=str(config_file),
            dataset_path=ds_path,
            sut_provider=sut_provider,
            judge=judge,
            baseline_path=str(baseline_file),
            update_baseline=True,
            judge_failure_policy="allow"
        )
        assert sla_passed_clean is True
        assert os.path.exists(baseline_file)

        # Verify history was logged
        history_file = tmp_path / "runs/history.jsonl"
        assert os.path.exists(history_file)
        with open(history_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 3
            record = json.loads(lines[0])
            assert record["sla_status"] == "PASSED"
            assert record["aggregate_metrics"]["pass_rate"] == 100.0
    finally:
        if sut_provider:
            await sut_provider.close()
        if judge:
            await judge.close()

@pytest.mark.asyncio
async def test_ci_fallback_rejection_and_baseline_immutability(tmp_path, monkeypatch):
    """Verifies that in strict CI mode (no cloud credentials), fallback evaluation fails the SLA gate and baseline checksum is untouched."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Initial baseline
    baseline_file = tmp_path / "baseline.json"
    initial_baseline = {"pass_rate_pct": 95.0, "avg_latency_ms": 500.0, "total_cost_usd": 0.01, "faithfulness": 0.95, "answer_relevance": 0.90, "correctness": 0.90}
    with open(baseline_file, "w") as f:
        json.dump(initial_baseline, f)

    with open(baseline_file, "rb") as f:
        initial_checksum = hashlib.sha256(f.read()).hexdigest()

    dataset_file = tmp_path / "dataset.json"
    with open(dataset_file, "w") as f:
        json.dump([{"query": "Test", "context": "Test ctx", "expected_answer": "Test ans"}], f)

    # Config configured for cloud Gemini judge without an API key
    config_file = tmp_path / "config.yaml"
    config_content = {
        "dataset_path": str(dataset_file),
        "judge_failure_policy": "fail",
        "sla_thresholds": {
            "min_faithfulness": 0.80,
            "min_relevance": 0.80,
            "min_correctness": 0.80,
            "min_pass_rate": 0.80
        },
        "sut": {"fallback_chain": ["mock"]},
        "judge": {
            "fallback_chain": ["gemini"],
            "providers": {"gemini": {"model": "gemini-3.5-flash"}}
        }
    }
    with open(config_file, "w") as f:
        yaml.dump(config_content, f)

    config, entries, ds_path = load_and_validate_config(str(config_file))
    sut, judge, sut_provider, judge_provider = await initialize_harness(config)

    try:
        results, metrics = await evaluate_dataset(entries, sut, judge)
        # Because no GEMINI_API_KEY was provided, judge degraded to fallback
        assert metrics["has_fallback"] is True

        # In strict CI policy, enforce_slas_and_report must return False
        sla_passed = enforce_slas_and_report(
            metrics=metrics,
            sla_thresholds=config.get("sla_thresholds", {}),
            config_path=str(config_file),
            dataset_path=ds_path,
            sut_provider=sut_provider,
            judge=judge,
            baseline_path=str(baseline_file),
            update_baseline=True,
            judge_failure_policy="fail"
        )
        assert sla_passed is False

        # Verify baseline file checksum is strictly unmodified
        with open(baseline_file, "rb") as f:
            final_checksum = hashlib.sha256(f.read()).hexdigest()
        assert initial_checksum == final_checksum
    finally:
        if sut_provider:
            await sut_provider.close()
        if judge:
            await judge.close()
