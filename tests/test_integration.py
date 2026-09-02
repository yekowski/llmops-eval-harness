import os
import json
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
    """End-to-end smoke test validating config loading, execution, evaluation, SLA enforcement, and history tracking."""
    monkeypatch.chdir(tmp_path)

    # 1. Create a mock dataset
    dataset_file = tmp_path / "dataset.json"
    dataset_content = [
        {
            "query": "What is Python?",
            "context": "Python is a high-level programming language.",
            "expected_answer": "Python is a high-level programming language.",
            "retrieved_contexts": ["Python is a high-level programming language."],
            "ground_truth": "Python is a high-level programming language."
        }
    ]
    with open(dataset_file, "w") as f:
        json.dump(dataset_content, f)

    # 2. Create a mock config YAML
    config_file = tmp_path / "config.yaml"
    config_content = {
        "dataset_path": str(dataset_file),
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
    assert entries[0].query == "What is Python?"

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

        # 6. Enforce SLAs and report
        sla_passed = enforce_slas_and_report(
            metrics=metrics,
            sla_thresholds=config.get("sla_thresholds", {}),
            config_path=str(config_file),
            dataset_path=ds_path,
            sut_provider=sut_provider,
            judge=judge,
            baseline_path=str(tmp_path / "baseline.json")
        )
        assert sla_passed is True

        # Verify baseline was created
        assert os.path.exists(tmp_path / "baseline.json")

        # Verify history was logged
        history_file = tmp_path / "runs/history.jsonl"
        assert os.path.exists(history_file)
        with open(history_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["sla_status"] == "PASSED"
            assert record["aggregate_metrics"]["pass_rate"] == 100.0
    finally:
        if sut_provider:
            await sut_provider.close()
        if judge:
            await judge.close()
