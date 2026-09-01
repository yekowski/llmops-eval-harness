from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import yaml
import argparse
import asyncio
from typing import List, Dict, Tuple, Any, Optional

from src.schemas.models import DatasetEntry, EvaluationResult
from src.clients.base import SystemUnderTest
from src.clients.mock_client import MockRAGClient
from src.clients.sut import LLMProviderSUT
from src.providers.registry import build_provider_from_config
from src.utils.cache import EvalCache
from src.evaluation.judge import LLMJudge
from src.runners.async_runner import run_evaluation
from src.reporters.markdown import generate_markdown_report
from src.reporters.github import write_to_step_summary
from src.utils.tracker import ExperimentTracker, print_history_table


def load_and_validate_config(config_path: str, dataset_path_override: Optional[str] = None) -> Tuple[dict, List[DatasetEntry], str]:
    """Loads SLA configuration and dataset file, validating existence and parsing DatasetEntry objects."""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    default_dataset = "datasets/benchmarks/human_labeled.json"
    if dataset_path_override and (dataset_path_override != default_dataset or "dataset_path" not in config):
        dataset_path = dataset_path_override
    else:
        dataset_path = config.get("dataset_path", dataset_path_override or default_dataset)

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    entries = []
    for item in dataset:
        retrieved_contexts = item.get("retrieved_contexts")
        ground_truth = item.get("ground_truth") or item.get("expected_answer", "")
        expected_answer = item.get("expected_answer") or item.get("ground_truth", "")
        ctx = item.get("context") or item.get("expected_context")
        if not ctx and retrieved_contexts:
            ctx = "\n---\n".join(retrieved_contexts)
        elif not ctx:
            ctx = ""
        entries.append(
            DatasetEntry(
                query=item["query"],
                expected_context=ctx,
                expected_answer=expected_answer,
                retrieved_contexts=retrieved_contexts,
                ground_truth=ground_truth
            )
        )

    return config, entries, dataset_path


async def initialize_harness(config: dict) -> Tuple[SystemUnderTest, LLMJudge, Any, Any]:
    """Instantiates SUT provider router, LLM Judge, and pre-warms circuit breaker states."""
    cache = EvalCache()

    # 1. Initialize SUT Provider fallback chain
    sut_cfg = config.get("sut", {})
    sut_fallback_chain = sut_cfg.get("fallback_chain") if "sut" in config else config.get("fallback_chain")
    sut_providers_cfg = sut_cfg.get("providers") if "sut" in config else config.get("providers", {})
    sut_provider = build_provider_from_config(sut_fallback_chain, sut_providers_cfg)

    if sut_provider:
        sut: SystemUnderTest = LLMProviderSUT(sut_provider)
    else:
        sut = MockRAGClient()

    # 2. Initialize Judge Provider (Decoupled from SUT)
    judge_cfg = config.get("judge", {})
    judge_fallback_chain = judge_cfg.get("fallback_chain")
    judge_providers_cfg = judge_cfg.get("providers", {})
    judge_provider = build_provider_from_config(judge_fallback_chain, judge_providers_cfg)

    judge = LLMJudge(provider=judge_provider, cache=cache)

    # 3. Pre-warm router circuits
    if sut_provider and hasattr(sut_provider, "warmup"):
        await sut_provider.warmup()
    if judge_provider and hasattr(judge_provider, "warmup"):
        await judge_provider.warmup()

    return sut, judge, sut_provider, judge_provider


async def evaluate_dataset(entries: List[DatasetEntry], sut: SystemUnderTest, judge: LLMJudge) -> Tuple[List[EvaluationResult], dict]:
    """Runs concurrent evaluation queries against SUT and aggregates score metrics."""
    print(f"Running evaluation of {len(entries)} entries concurrently against SUT...")
    results = await run_evaluation(entries, sut, judge)

    total_count = len(results)
    passed_count = sum(1 for r in results if r.passed)
    pass_rate_pct = (passed_count / total_count) * 100 if total_count > 0 else 0.0
    avg_latency_ms = (sum(r.latency for r in results) / total_count) * 1000 if total_count > 0 else 0.0
    total_cost_usd = judge.total_cost

    avg_faithfulness = sum(r.faithfulness for r in results) / total_count if total_count > 0 else 0.0
    avg_relevance = sum(r.answer_relevance for r in results) / total_count if total_count > 0 else 0.0
    avg_correctness = sum(r.correctness for r in results) / total_count if total_count > 0 else 0.0

    precisions = [r.context_precision for r in results if r.context_precision is not None]
    recalls = [r.context_recall for r in results if r.context_recall is not None]
    avg_context_precision = (sum(precisions) / len(precisions)) if precisions else None
    avg_context_recall = (sum(recalls) / len(recalls)) if recalls else None

    metrics = {
        "pass_rate_pct": pass_rate_pct,
        "avg_latency_ms": avg_latency_ms,
        "total_cost_usd": total_cost_usd,
        "passed_count": passed_count,
        "total_count": total_count,
        "avg_faithfulness": avg_faithfulness,
        "avg_relevance": avg_relevance,
        "avg_correctness": avg_correctness,
        "avg_context_precision": avg_context_precision,
        "avg_context_recall": avg_context_recall,
    }
    return results, metrics


def enforce_slas_and_report(
    metrics: dict,
    sla_thresholds: dict,
    config_path: str,
    dataset_path: str,
    sut_provider: Any,
    judge: LLMJudge
) -> bool:
    """Computes deltas against baseline, enforces SLA gates, outputs markdown reports, and logs experiment history."""
    baseline_path = "baseline.json"
    baseline = {}
    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, "r") as f:
                baseline = json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not parse baseline.json: {e}")

    pass_rate_pct = metrics["pass_rate_pct"]
    avg_latency_ms = metrics["avg_latency_ms"]
    total_cost_usd = metrics["total_cost_usd"]
    avg_faithfulness = metrics["avg_faithfulness"]
    avg_relevance = metrics["avg_relevance"]
    avg_correctness = metrics["avg_correctness"]
    avg_context_precision = metrics.get("avg_context_precision")
    avg_context_recall = metrics.get("avg_context_recall")

    metrics["delta_pass_rate_pct"] = (pass_rate_pct - baseline["pass_rate_pct"]) if "pass_rate_pct" in baseline else None
    metrics["delta_latency_ms"] = (avg_latency_ms - baseline["avg_latency_ms"]) if "avg_latency_ms" in baseline else None
    metrics["delta_cost_usd"] = (total_cost_usd - baseline["total_cost_usd"]) if "total_cost_usd" in baseline else None
    metrics["delta_faithfulness"] = (avg_faithfulness - baseline["faithfulness"]) if "faithfulness" in baseline else None
    metrics["delta_relevance"] = (avg_relevance - baseline["answer_relevance"]) if "answer_relevance" in baseline else None
    metrics["delta_correctness"] = (avg_correctness - baseline["correctness"]) if "correctness" in baseline else None
    metrics["delta_context_precision"] = (avg_context_precision - baseline["context_precision"]) if ("context_precision" in baseline and avg_context_precision is not None) else None
    metrics["delta_context_recall"] = (avg_context_recall - baseline["context_recall"]) if ("context_recall" in baseline and avg_context_recall is not None) else None

    min_pass_rate = sla_thresholds.get("min_pass_rate", 0.0)
    min_faithfulness = sla_thresholds.get("min_faithfulness", 0.0)
    min_relevance = sla_thresholds.get("min_relevance", 0.0)
    min_correctness = sla_thresholds.get("min_correctness", 0.0)
    min_context_precision = sla_thresholds.get("min_context_precision")
    min_context_recall = sla_thresholds.get("min_context_recall")
    max_latency = sla_thresholds.get("max_latency_ms", float("inf"))
    max_cost = sla_thresholds.get("max_cost_usd", float("inf"))
    max_score_drop = sla_thresholds.get("max_score_drop", float("inf"))

    failures = []
    pass_rate_ratio = metrics["passed_count"] / metrics["total_count"] if metrics["total_count"] > 0 else 0.0

    if pass_rate_ratio < min_pass_rate:
        failures.append(f"pass_rate {pass_rate_ratio:.2f} < required {min_pass_rate:.2f}")
    if avg_faithfulness < min_faithfulness:
        failures.append(f"faithfulness {avg_faithfulness:.2f} < required {min_faithfulness:.2f}")
    if avg_relevance < min_relevance:
        failures.append(f"relevance {avg_relevance:.2f} < required {min_relevance:.2f}")
    if avg_correctness < min_correctness:
        failures.append(f"correctness {avg_correctness:.2f} < required {min_correctness:.2f}")
    if min_context_precision is not None and avg_context_precision is not None:
        if avg_context_precision < min_context_precision:
            failures.append(f"context precision {avg_context_precision:.2f} < required {min_context_precision:.2f}")
    if min_context_recall is not None and avg_context_recall is not None:
        if avg_context_recall < min_context_recall:
            failures.append(f"context recall {avg_context_recall:.2f} < required {min_context_recall:.2f}")
    if avg_latency_ms > max_latency:
        failures.append(f"latency {avg_latency_ms:.2f} ms > required {max_latency:.2f} ms")
    if total_cost_usd > max_cost:
        failures.append(f"cost {total_cost_usd:.6f} > required {max_cost:.4f}")

    if "faithfulness" in baseline and metrics["delta_faithfulness"] is not None:
        if metrics["delta_faithfulness"] < -max_score_drop:
            failures.append(f"Faithfulness dropped by {abs(metrics['delta_faithfulness']):.2f} > allowed {max_score_drop:.2f} limit")
    if "answer_relevance" in baseline and metrics["delta_relevance"] is not None:
        if metrics["delta_relevance"] < -max_score_drop:
            failures.append(f"Relevance dropped by {abs(metrics['delta_relevance']):.2f} > allowed {max_score_drop:.2f} limit")
    if "correctness" in baseline and metrics["delta_correctness"] is not None:
        if metrics["delta_correctness"] < -max_score_drop:
            failures.append(f"Correctness dropped by {abs(metrics['delta_correctness']):.2f} > allowed {max_score_drop:.2f} limit")

    sla_passed = len(failures) == 0
    metrics["sla_passed"] = sla_passed

    markdown_report = generate_markdown_report(metrics, sla_thresholds)
    print("\n" + markdown_report)
    write_to_step_summary(markdown_report)

    tracker = ExperimentTracker()
    if sut_provider:
        if hasattr(sut_provider, "active_provider"):
            sut_provider_name = sut_provider.active_provider.__class__.__name__
        else:
            sut_provider_name = sut_provider.__class__.__name__
    else:
        sut_provider_name = "MockRAGClient"

    if hasattr(judge, "provider") and judge.provider:
        if hasattr(judge.provider, "active_provider"):
            judge_provider_name = judge.provider.active_provider.__class__.__name__
        else:
            judge_provider_name = judge.provider.__class__.__name__
    else:
        judge_provider_name = judge.__class__.__name__

    aggregate_metrics = {
        "faithfulness": avg_faithfulness,
        "relevance": avg_relevance,
        "correctness": avg_correctness,
        "pass_rate": pass_rate_pct,
        "latency": avg_latency_ms,
        "cost": total_cost_usd
    }
    if avg_context_precision is not None:
        aggregate_metrics["context_precision"] = avg_context_precision
    if avg_context_recall is not None:
        aggregate_metrics["context_recall"] = avg_context_recall

    try:
        run_id = tracker.log_run(
            config_path=config_path,
            dataset_path=dataset_path,
            sut_provider=sut_provider_name,
            judge_provider=judge_provider_name,
            aggregate_metrics=aggregate_metrics,
            sla_passed=sla_passed
        )
        print(f"Logged run record {run_id} to runs/history.jsonl")
    except Exception as e:
        print(f"[WARNING] Failed to log run to tracker: {e}")

    if not sla_passed:
        print("❌ SLA check FAILED! Blocking integration.")
        for failure in failures:
            print(f"Reason: {failure}")
        return False
    else:
        new_baseline = {
            "pass_rate_pct": pass_rate_pct,
            "avg_latency_ms": avg_latency_ms,
            "total_cost_usd": total_cost_usd,
            "faithfulness": avg_faithfulness,
            "answer_relevance": avg_relevance,
            "correctness": avg_correctness
        }
        if avg_context_precision is not None:
            new_baseline["context_precision"] = avg_context_precision
        if avg_context_recall is not None:
            new_baseline["context_recall"] = avg_context_recall
        try:
            with open(baseline_path, "w") as f:
                json.dump(new_baseline, f, indent=2)
            print(f"Saved new baseline scores to {baseline_path}")
        except Exception as e:
            print(f"[WARNING] Could not save baseline.json: {e}")

        print("✅ SLA check PASSED! Ready for integration.")
        return True


async def main_async():
    parser = argparse.ArgumentParser(description="LLMOps CI/CD Evaluation Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pr.yaml",
        help="Path to the SLA configuration YAML file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/benchmarks/human_labeled.json",
        help="Path to the evaluation dataset JSON file"
    )
    parser.add_argument(
        "--history",
        type=int,
        nargs="?",
        const=5,
        default=None,
        help="Print a formatted table of the last N runs from the history log (default: 5 if flag present)"
    )
    args = parser.parse_args()

    if args.history is not None:
        print_history_table("runs/history.jsonl", args.history)
        sys.exit(0)

    config, entries, dataset_path = load_and_validate_config(args.config, args.dataset)
    sut, judge, sut_provider, judge_provider = await initialize_harness(config)
    results, metrics = await evaluate_dataset(entries, sut, judge)
    sla_passed = enforce_slas_and_report(
        metrics=metrics,
        sla_thresholds=config.get("sla_thresholds", {}),
        config_path=args.config,
        dataset_path=dataset_path,
        sut_provider=sut_provider,
        judge=judge
    )

    sys.exit(0 if sla_passed else 1)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
