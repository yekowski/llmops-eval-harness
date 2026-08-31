# Operational Playbooks & Skills Reference

This guide documents standard operational workflows for local model evaluation, dataset structuring, and experiment history inspection within the LLMOps Evaluation Harness.

---

## 1. Running Local Evaluations with Ollama / vLLM

Local model runtimes (such as Ollama or vLLM) expose OpenAI-compatible HTTP endpoints. You can run evaluations locally without incurring cloud API costs or consuming remote rate limits.

### Configuration Setup

To route evaluation requests to a local runtime, update your configuration YAML (e.g., `configs/pr.yaml` or a local development config) to specify the local provider `base_url` and target `model`:

```yaml
# configs/local_eval.yaml
dataset_path: "data/adversarial_eval.json"

sla_thresholds:
  min_faithfulness: 0.85
  min_relevance: 0.80
  min_correctness: 0.85
  min_pass_rate: 0.90
  max_latency_ms: 3000
  max_cost_usd: 0.05
  max_score_drop: 0.03

fallback_chain:
  - "ollama"
  - "mock"

providers:
  ollama:
    base_url: "http://localhost:11434/v1"
    model: "llama3.2:3b"
    temperature: 0.0
```

### Running Local vLLM Endpoints

For high-throughput local inference via vLLM:

```yaml
providers:
  vllm:
    base_url: "http://localhost:8000/v1"
    model: "mistralai/Mistral-7B-Instruct-v0.2"
    temperature: 0.0
```

### Execution

Execute the evaluation harness targeting your local config:

```bash
PYTHONPATH=. python3 run_eval.py --config configs/local_eval.yaml
```

---

## 2. Structuring RAG Ground-Truth Datasets

To support decoupled retrieval metrics (Context Precision, Context Recall) alongside generation metrics (Faithfulness, Relevance, Correctness), dataset files must include a dedicated `retrieved_contexts` field in addition to ground-truth fields.

### JSON Dataset Schema Example

```json
[
  {
    "id": "eval-001",
    "query": "What is the primary function of the ProviderRouter?",
    "expected_context": "The ProviderRouter manages fallback chains, tracks provider health, and executes circuit breaker cooldowns when rate limits or server errors occur.",
    "expected_answer": "The ProviderRouter handles provider failover, health tracking, and circuit breaking.",
    "retrieved_contexts": [
      "The ProviderRouter manages fallback chains, tracks provider health, and executes circuit breaker cooldowns when rate limits or server errors occur.",
      "Circuit breaker cooldowns default to 60 seconds on HTTP 429 or 5xx responses."
    ]
  }
]
```

### Metrics Separability
- **Retrieval Metrics (Context Precision, Context Recall):** Evaluates how accurately `retrieved_contexts` match `expected_context` independently of LLM generation.
- **Generation Metrics (Faithfulness, Relevance, Correctness):** Evaluates the SUT generated answer against `retrieved_contexts` and `expected_answer`.

---

## 3. Querying Tracking History via CLI

Every completed evaluation run logs an immutable run record to `runs/history.jsonl`. You can query and display recent evaluation history directly from the CLI.

### Quick Reference Command

```bash
PYTHONPATH=. python3 run_eval.py --history
```

### Sample Output

```
=== Evaluation Run History ===
Run ID    Timestamp           Git SHA  SUT            Judge          Pass Rate Latency   Cost      SLA Status 
--------------------------------------------------------------------------------------------------------------
ea70169d  2026-08-28 05:01:38 f6a3ed6  MockProvider   MockProvider   100.0%    0.00ms    $0.0000   PASSED     
fac2fa5c  2026-08-28 05:28:26 640a4fe  MockProvider   MockProvider   100.0%    0.00ms    $0.0002   PASSED     
4e895e9a  2026-08-31 09:20:13 4e895e9  MockProvider   MockProvider   100.0%    0.00ms    $0.0002   PASSED     
===============================
```

### Recorded Fields
- **Run ID:** Short unique identifier for the run.
- **Timestamp:** UTC ISO 8601 timestamp.
- **Git SHA:** Short Git commit hash at run execution.
- **SUT / Judge:** Active provider names used for execution and evaluation.
- **Pass Rate / Latency / Cost:** Aggregated performance and expenditure metrics.
- **SLA Status:** Final gatekeeper verdict (`PASSED` or `FAILED`).
