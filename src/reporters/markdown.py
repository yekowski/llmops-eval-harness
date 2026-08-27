def format_delta(val, format_spec=".2f") -> str:
    if val is None:
        return "-"
    if val > 0.0000001:
        return f"+{val:{format_spec}}"
    elif val < -0.0000001:
        return f"{val:{format_spec}}"
    else:
        if "6f" in format_spec:
            return "+0.000000"
        return "+0.00"

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

    # Deltas
    delta_pass_rate = metrics.get("delta_pass_rate_pct")
    pass_rate_change = format_delta(delta_pass_rate, ".2f") + "%" if delta_pass_rate is not None else "-"
    
    delta_latency = metrics.get("delta_latency_ms")
    latency_change = format_delta(delta_latency, ".2f") + " ms" if delta_latency is not None else "-"
    
    delta_cost = metrics.get("delta_cost_usd")
    cost_change = format_delta(delta_cost, ".6f") if delta_cost is not None else "-"

    faithfulness_change = format_delta(metrics.get("delta_faithfulness"), ".2f")
    relevance_change = format_delta(metrics.get("delta_relevance"), ".2f")
    correctness_change = format_delta(metrics.get("delta_correctness"), ".2f")
    
    markdown = f"""### LLMOps CI/CD Evaluation Report

| Metric | SLA Threshold | Actual Value | Status | Change vs Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Pass Rate** | >= {min_pass_rate * 100:.1f}% | {pass_rate_ratio * 100:.2f}% ({metrics['passed_count']}/{metrics['total_count']}) | {pass_rate_status} | {pass_rate_change} |
| **Average Latency** | <= {max_latency:.1f} ms | {metrics['avg_latency_ms']:.2f} ms | {latency_status} | {latency_change} |
| **Total Evaluation Cost** | <= ${max_cost:.4f} | ${metrics['total_cost_usd']:.6f} | {cost_status} | {cost_change} |
| **Average Faithfulness** | >= {min_faithfulness:.2f} | {metrics['avg_faithfulness']:.2f} / 1.0 | {faithfulness_status} | {faithfulness_change} |
| **Average Relevance** | >= {min_relevance:.2f} | {metrics['avg_relevance']:.2f} / 1.0 | {relevance_status} | {relevance_change} |
| **Average Correctness** | >= {min_correctness:.2f} | {metrics['avg_correctness']:.2f} / 1.0 | {correctness_status} | {correctness_change} |

**Overall SLA Status: {overall_status}**
"""
    return markdown
