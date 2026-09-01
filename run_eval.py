from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import yaml
import argparse
import asyncio
from src.schemas.models import DatasetEntry
from src.clients.mock_client import MockRAGClient
from src.utils.cache import EvalCache
from src.evaluation.judge import LLMJudge
from src.runners.async_runner import run_evaluation
from src.reporters.markdown import generate_markdown_report
from src.reporters.github import write_to_step_summary

def build_provider_from_config(fallback_chain, providers_config=None):
    """Builds a ProviderRouter from a fallback chain list and provider options dict."""
    if not fallback_chain:
        return None
    from src.providers import (
        GeminiProvider, OpenAIProvider, DeepSeekProvider, GroqProvider,
        QwenProvider, AnthropicProvider, MockProvider, OllamaProvider,
        VLLMProvider, ProviderRouter
    )
    chain_instances = []
    providers_config = providers_config or {}
    for name in fallback_chain:
        name_lower = name.lower()
        prov_opts = providers_config.get(name_lower, {})
        model_name = prov_opts.get("model")
        temperature = prov_opts.get("temperature")
        base_url = prov_opts.get("base_url")
        
        kwargs = {}
        if model_name is not None:
            kwargs["model"] = model_name
        if temperature is not None:
            kwargs["temperature"] = temperature
        if base_url is not None:
            kwargs["base_url"] = base_url
            
        if name_lower == "gemini":
            gemini_kwargs = {}
            if model_name is not None:
                gemini_kwargs["model"] = model_name
            chain_instances.append(GeminiProvider(**gemini_kwargs))
        elif name_lower == "openai":
            chain_instances.append(OpenAIProvider(**kwargs))
        elif name_lower == "deepseek":
            chain_instances.append(DeepSeekProvider(**kwargs))
        elif name_lower == "groq":
            chain_instances.append(GroqProvider(**kwargs))
        elif name_lower == "qwen":
            chain_instances.append(QwenProvider(**kwargs))
        elif name_lower == "anthropic":
            chain_instances.append(AnthropicProvider(**kwargs))
        elif name_lower == "ollama":
            chain_instances.append(OllamaProvider(**kwargs))
        elif name_lower == "vllm":
            chain_instances.append(VLLMProvider(**kwargs))
        elif name_lower == "mock":
            chain_instances.append(MockProvider())
        else:
            raise ValueError(f"Unknown provider name '{name}' in config fallback_chain.")
    return ProviderRouter(chain_instances)

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
        from src.utils.tracker import print_history_table
        print_history_table("runs/history.jsonl", args.history)
        sys.exit(0)

    # 1. Load SLA configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    sla_thresholds = config.get("sla_thresholds", {})

    # 2. Load dataset
    default_dataset = "datasets/benchmarks/human_labeled.json"
    if args.dataset != default_dataset or "dataset_path" not in config:
        dataset_path = args.dataset
    else:
        dataset_path = config.get("dataset_path", args.dataset)

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
        
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Convert dataset entries to Pydantic objects
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

    # 3. Instantiate SUT, Cache, and Judge
    cache = EvalCache()
    
    # 3a. Initialize SUT Provider fallback chain
    sut_cfg = config.get("sut", {})
    sut_fallback_chain = sut_cfg.get("fallback_chain") if "sut" in config else config.get("fallback_chain")
    sut_providers_cfg = sut_cfg.get("providers") if "sut" in config else config.get("providers", {})
    sut_provider = build_provider_from_config(sut_fallback_chain, sut_providers_cfg)

    # Initialize SUT wrapping the fallback provider router if configured
    if sut_provider:
        from src.clients.base import SystemUnderTest
        class LLMProviderSUT(SystemUnderTest):
            def __init__(self, prov):
                self.prov = prov
            async def execute(self, query: str) -> str:
                return await self.prov.generate(query)
        sut = LLMProviderSUT(sut_provider)
    else:
        sut = MockRAGClient()

    # 3b. Initialize Judge Provider (Decoupled from SUT)
    judge_cfg = config.get("judge", {})
    judge_fallback_chain = judge_cfg.get("fallback_chain")
    judge_providers_cfg = judge_cfg.get("providers", {})
    judge_provider = build_provider_from_config(judge_fallback_chain, judge_providers_cfg)

    judge = LLMJudge(provider=judge_provider, cache=cache)

    if sut_provider and hasattr(sut_provider, "warmup"):
        await sut_provider.warmup()
    if judge_provider and hasattr(judge_provider, "warmup"):
        await judge_provider.warmup()

    print(f"Running evaluation of {len(entries)} entries concurrently against SUT...")
    results = await run_evaluation(entries, sut, judge)

    # 4. Aggregate metrics
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

    # Load baseline
    baseline_path = "baseline.json"
    baseline = {}
    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, "r") as f:
                baseline = json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not parse baseline.json: {e}")

    # Compute deltas
    delta_pass_rate = (pass_rate_pct - baseline["pass_rate_pct"]) if "pass_rate_pct" in baseline else None
    delta_latency = (avg_latency_ms - baseline["avg_latency_ms"]) if "avg_latency_ms" in baseline else None
    delta_cost = (total_cost_usd - baseline["total_cost_usd"]) if "total_cost_usd" in baseline else None
    delta_faithfulness = (avg_faithfulness - baseline["faithfulness"]) if "faithfulness" in baseline else None
    delta_relevance = (avg_relevance - baseline["answer_relevance"]) if "answer_relevance" in baseline else None
    delta_correctness = (avg_correctness - baseline["correctness"]) if "correctness" in baseline else None
    delta_context_precision = (avg_context_precision - baseline["context_precision"]) if ("context_precision" in baseline and avg_context_precision is not None) else None
    delta_context_recall = (avg_context_recall - baseline["context_recall"]) if ("context_recall" in baseline and avg_context_recall is not None) else None

    # 5. Check SLAs
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
    pass_rate_ratio = passed_count / total_count if total_count > 0 else 0.0

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

    # Check relative baseline regressions
    if "faithfulness" in baseline and delta_faithfulness is not None:
        if delta_faithfulness < -max_score_drop:
            failures.append(f"Faithfulness dropped by {abs(delta_faithfulness):.2f} > allowed {max_score_drop:.2f} limit")
    if "answer_relevance" in baseline and delta_relevance is not None:
        if delta_relevance < -max_score_drop:
            failures.append(f"Relevance dropped by {abs(delta_relevance):.2f} > allowed {max_score_drop:.2f} limit")
    if "correctness" in baseline and delta_correctness is not None:
        if delta_correctness < -max_score_drop:
            failures.append(f"Correctness dropped by {abs(delta_correctness):.2f} > allowed {max_score_drop:.2f} limit")

    sla_passed = len(failures) == 0

    metrics = {
        "pass_rate_pct": pass_rate_pct,
        "avg_latency_ms": avg_latency_ms,
        "total_cost_usd": total_cost_usd,
        "passed_count": passed_count,
        "total_count": total_count,
        "sla_passed": sla_passed,
        "avg_faithfulness": avg_faithfulness,
        "avg_relevance": avg_relevance,
        "avg_correctness": avg_correctness,
        "avg_context_precision": avg_context_precision,
        "avg_context_recall": avg_context_recall,
        "delta_pass_rate_pct": delta_pass_rate,
        "delta_latency_ms": delta_latency,
        "delta_cost_usd": delta_cost,
        "delta_faithfulness": delta_faithfulness,
        "delta_relevance": delta_relevance,
        "delta_correctness": delta_correctness,
        "delta_context_precision": delta_context_precision,
        "delta_context_recall": delta_context_recall
    }

    # 6. Generate and output Markdown report
    markdown_report = generate_markdown_report(metrics, sla_thresholds)
    print("\n" + markdown_report)

    # 7. Write to GITHUB_STEP_SUMMARY for CI/CD integrations
    write_to_step_summary(markdown_report)

    # 7. Log the run history
    from src.utils.tracker import ExperimentTracker
    tracker = ExperimentTracker()
    
    # Resolve provider names
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
            config_path=args.config,
            dataset_path=dataset_path,
            sut_provider=sut_provider_name,
            judge_provider=judge_provider_name,
            aggregate_metrics=aggregate_metrics,
            sla_passed=sla_passed
        )
        print(f"Logged run record {run_id} to runs/history.jsonl")
    except Exception as e:
        print(f"[WARNING] Failed to log run to tracker: {e}")

    # 8. Exit with non-zero code on SLA failure to fail the PR build
    if not sla_passed:
        print("❌ SLA check FAILED! Blocking integration.")
        for failure in failures:
            print(f"Reason: {failure}")
        sys.exit(1)
    else:
        # Save baseline.json only if SLA checks and regression checks pass
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
        sys.exit(0)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
