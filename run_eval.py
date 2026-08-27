import os
import sys
import json
import yaml
import argparse
import asyncio
from src.schemas.models import DatasetEntry
from src.clients.mock_client import MockRAGClient
from src.cache.prompt_hash import PromptHashCache
from src.evaluation.judge import GeminiJudge
from src.runners.async_runner import run_evaluation
from src.reporters.markdown import generate_markdown_report
from src.reporters.github import write_to_step_summary

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
    args = parser.parse_args()

    # 1. Load SLA configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    sla = config.get("sla", {})
    min_pass_rate = sla.get("min_pass_rate_pct", 90.0)
    max_latency = sla.get("max_latency_ms", 2000.0)
    max_cost = sla.get("max_cost_usd", 0.50)

    # 2. Load dataset
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found at {args.dataset}")
        sys.exit(1)
        
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    # Convert dataset entries to Pydantic objects
    entries = [
        DatasetEntry(
            query=item["query"],
            expected_context=item.get("context") or item.get("expected_context", ""),
            expected_answer=item.get("expected_answer", "")
        )
        for item in dataset
    ]

    # 3. Instantiate SUT, Cache, and Judge
    sut = MockRAGClient()
    cache = PromptHashCache()
    
    # Initialize LLM Provider fallback chain if configured
    provider = None
    fallback_chain = config.get("fallback_chain")
    if fallback_chain:
        from src.providers import GeminiProvider, OpenAIProvider, DeepSeekProvider, GroqProvider, QwenProvider, AnthropicProvider, MockProvider, ProviderRouter
        chain_instances = []
        for name in fallback_chain:
            name_lower = name.lower()
            if name_lower == "gemini":
                chain_instances.append(GeminiProvider())
            elif name_lower == "openai":
                chain_instances.append(OpenAIProvider())
            elif name_lower == "deepseek":
                chain_instances.append(DeepSeekProvider())
            elif name_lower == "groq":
                chain_instances.append(GroqProvider())
            elif name_lower == "qwen":
                chain_instances.append(QwenProvider())
            elif name_lower == "anthropic":
                chain_instances.append(AnthropicProvider())
            elif name_lower == "mock":
                chain_instances.append(MockProvider())
            else:
                raise ValueError(f"Unknown provider name '{name}' in config fallback_chain.")
        provider = ProviderRouter(chain_instances)

    judge = GeminiJudge(provider=provider, cache=cache)

    print(f"Running evaluation of {len(entries)} entries concurrently against SUT...")
    results = await run_evaluation(entries, sut, judge)

    # 4. Aggregate metrics
    total_count = len(results)
    passed_count = sum(1 for r in results if r.passed)
    pass_rate_pct = (passed_count / total_count) * 100 if total_count > 0 else 0.0
    avg_latency_ms = (sum(r.latency for r in results) / total_count) * 1000 if total_count > 0 else 0.0
    total_cost_usd = judge.total_cost

    # 5. Check SLAs
    pass_rate_ok = pass_rate_pct >= min_pass_rate
    latency_ok = avg_latency_ms <= max_latency
    cost_ok = total_cost_usd <= max_cost
    sla_passed = pass_rate_ok and latency_ok and cost_ok

    metrics = {
        "pass_rate_pct": pass_rate_pct,
        "avg_latency_ms": avg_latency_ms,
        "total_cost_usd": total_cost_usd,
        "passed_count": passed_count,
        "total_count": total_count,
        "sla_passed": sla_passed
    }

    # 6. Generate and output Markdown report
    markdown_report = generate_markdown_report(metrics, sla)
    print("\n" + markdown_report)

    # 7. Write to GITHUB_STEP_SUMMARY for CI/CD integrations
    write_to_step_summary(markdown_report)

    # 8. Exit with non-zero code on SLA failure to fail the PR build
    if not sla_passed:
        print("❌ SLA check FAILED! Blocking integration.")
        sys.exit(1)
    else:
        print("✅ SLA check PASSED! Ready for integration.")
        sys.exit(0)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
