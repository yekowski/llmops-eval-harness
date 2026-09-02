import os
import json
import asyncio
from src.utils.cache import EvalCache
from src.evaluation.judge import LLMJudge
from src.evaluation.meta_eval import run_meta_evaluation

async def main():
    dataset_path = "datasets/benchmarks/human_labeled.json"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Initialize cache and judge
    cache = EvalCache()
    judge = LLMJudge(cache=cache)

    print(f"Starting meta-evaluation of LLM Judge against {len(dataset)} human-labeled samples...")
    summary = await run_meta_evaluation(dataset, judge)

    # Print summary table
    print("\n" + "=" * 54)
    print(f"{'LLM Judge Meta-Evaluation Summary':^54}")
    print("=" * 54)
    print(f" Total Samples          : {len(dataset)}")
    print(f" Agreement Rate         : {summary['accuracy']:.2%}")
    print(f" Cohen's Kappa          : {summary['cohens_kappa']:.4f}")

    passed = summary['accuracy'] >= 0.85
    status_str = "PASSED" if passed else "FAILED (Below 85% Threshold)"
    print(f" Status                 : {status_str}")
    print("=" * 54)

    # Print details for drift analysis
    print("\nDrift Analysis Detail:")
    print(f"{'Query Snippet':<30} | {'Human':<6} | {'Judge':<6} | {'Match?':<6}")
    print("-" * 54)
    for detail in summary["details"]:
        query_clip = (detail["query"][:27] + "...") if len(detail["query"]) > 30 else detail["query"]
        human_val = "PASS" if detail["expected_pass_boolean"] else "FAIL"
        judge_val = "PASS" if detail["judge_passed"] else "FAIL"
        match_val = "YES" if detail["agreed"] else "NO"
        print(f"{query_clip:<30} | {human_val:<6} | {judge_val:<6} | {match_val:<6}")
    print("-" * 54)

    # Save raw traces/logs to outputs/raw/ per requirements
    os.makedirs("outputs/raw", exist_ok=True)
    trace_path = "outputs/raw/meta_eval_traces.json"
    with open(trace_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed evaluation traces saved to: {trace_path}")

if __name__ == "__main__":
    asyncio.run(main())
