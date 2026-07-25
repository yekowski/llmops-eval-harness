def generate_markdown_report(metrics: dict, sla: dict) -> str:
    """Generates a GitHub-flavored Markdown report comparing results against SLAs."""
    pass_rate_status = "✅ PASS" if metrics["pass_rate_pct"] >= sla["min_pass_rate_pct"] else "❌ FAIL"
    latency_status = "✅ PASS" if metrics["avg_latency_ms"] <= sla["max_latency_ms"] else "❌ FAIL"
    cost_status = "✅ PASS" if metrics["total_cost_usd"] <= sla["max_cost_usd"] else "❌ FAIL"
    
    overall_status = "✅ PASSED" if metrics["sla_passed"] else "❌ FAILED"
    
    markdown = f"""### LLMOps CI/CD Evaluation Report

| Metric | SLA Threshold | Actual Value | Status |
| :--- | :--- | :--- | :--- |
| **Pass Rate** | >= {sla['min_pass_rate_pct']}% | {metrics['pass_rate_pct']:.2f}% ({metrics['passed_count']}/{metrics['total_count']}) | {pass_rate_status} |
| **Average Latency** | <= {sla['max_latency_ms']} ms | {metrics['avg_latency_ms']:.2f} ms | {latency_status} |
| **Total Evaluation Cost** | <= ${sla['max_cost_usd']:.4f} | ${metrics['total_cost_usd']:.6f} | {cost_status} |

**Overall SLA Status: {overall_status}**
"""
    return markdown
