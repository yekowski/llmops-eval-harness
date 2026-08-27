def generate_markdown_report(metrics: dict, thresholds: dict) -> str:
    """Generates a GitHub-flavored Markdown report comparing results against SLAs."""
    min_pass_rate = thresholds.get("min_pass_rate", 0.0)
    min_faithfulness = thresholds.get("min_faithfulness", 0.0)
    min_relevance = thresholds.get("min_relevance", 0.0)
    min_correctness = thresholds.get("min_correctness", 0.0)
    max_latency = thresholds.get("max_latency_ms", float("inf"))
    max_cost = thresholds.get("max_cost_usd", float("inf"))

    pass_rate_ratio = metrics["passed_count"] / metrics["total_count"] if metrics["total_count"] > 0 else 0.0
    pass_rate_status = "✅ PASS" if pass_rate_ratio >= min_pass_rate else "❌ FAIL"
    latency_status = "✅ PASS" if metrics["avg_latency_ms"] <= max_latency else "❌ FAIL"
    cost_status = "✅ PASS" if metrics["total_cost_usd"] <= max_cost else "❌ FAIL"
    
    faithfulness_status = "✅ PASS" if metrics["avg_faithfulness"] >= min_faithfulness else "❌ FAIL"
    relevance_status = "✅ PASS" if metrics["avg_relevance"] >= min_relevance else "❌ FAIL"
    correctness_status = "✅ PASS" if metrics["avg_correctness"] >= min_correctness else "❌ FAIL"

    overall_status = "✅ PASSED" if metrics["sla_passed"] else "❌ FAILED"
    
    markdown = f"""### LLMOps CI/CD Evaluation Report

| Metric | SLA Threshold | Actual Value | Status |
| :--- | :--- | :--- | :--- |
| **Pass Rate** | >= {min_pass_rate * 100:.1f}% | {pass_rate_ratio * 100:.2f}% ({metrics['passed_count']}/{metrics['total_count']}) | {pass_rate_status} |
| **Average Latency** | <= {max_latency:.1f} ms | {metrics['avg_latency_ms']:.2f} ms | {latency_status} |
| **Total Evaluation Cost** | <= ${max_cost:.4f} | ${metrics['total_cost_usd']:.6f} | {cost_status} |
| **Average Faithfulness** | >= {min_faithfulness:.2f} | {metrics['avg_faithfulness']:.2f} / 1.0 | {faithfulness_status} |
| **Average Relevance** | >= {min_relevance:.2f} | {metrics['avg_relevance']:.2f} / 1.0 | {relevance_status} |
| **Average Correctness** | >= {min_correctness:.2f} | {metrics['avg_correctness']:.2f} / 1.0 | {correctness_status} |

**Overall SLA Status: {overall_status}**
"""
    return markdown
