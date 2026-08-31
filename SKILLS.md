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
- **Retrieval Metrics (Context Precision, Context Recall):** Evaluates how accurately `retrieved_contexts` match `expected_context` / `ground_truth` independently of LLM generation.
- **Generation Metrics (Faithfulness, Relevance, Correctness):** Evaluates the SUT generated answer against `retrieved_contexts` and `expected_answer`.

---

## 3. Executing RAG Retrieval Evaluations

In Tranche 2 RAG evaluation workflows, retrieval performance is measured independently of LLM response generation to isolate retrieval pipeline quality (e.g. vector search, hybrid ranking, chunking strategy).

### Retrieval Metric Definitions

1. **Context Precision (Signal-to-Noise Ratio):**
   - Evaluates whether the chunks present in `retrieved_contexts` are relevant and whether relevant chunks are ranked higher than irrelevant ones.
   - Evaluated by asking the LLM Judge to calculate:
     $$\text{Context Precision} = \frac{\sum_{k=1}^{N} P@k \cdot v_k}{\text{Total Relevant Chunks}}$$
     where $v_k \in \{0, 1\}$ indicates relevance of chunk $k$, and $P@k$ is precision at rank $k$.

2. **Context Recall (Information Completeness):**
   - Evaluates whether all key ground-truth facts in `ground_truth` / `expected_context` were successfully captured across the set of `retrieved_contexts`.
   - Evaluated by analyzing each ground-truth statement and verifying if it is explicitly supported by at least one chunk in `retrieved_contexts`:
     $$\text{Context Recall} = \frac{\text{Number of Ground-Truth Statements Attributable to Retrieved Chunks}}{\text{Total Number of Ground-Truth Statements}}$$

### Modern RAG Test Case Schema

Evaluation entries gracefully support optional `retrieved_contexts` (`List[str]`) and `ground_truth` (`str`) fields. Legacy prompt-response entries without these fields continue to evaluate generation metrics seamlessly.

```json
{
  "id": "rag-eval-042",
  "query": "How does the ProviderRouter handle rate limit (429) errors?",
  "ground_truth": "The ProviderRouter trips a circuit breaker with a 60-second cooldown on 429 rate limit errors to fast-bypass the provider.",
  "expected_answer": "It trips a circuit breaker for 60 seconds to bypass the failing provider without making network roundtrips.",
  "retrieved_contexts": [
    "When a provider returns a 429 (Rate Limit) or 5xx (Server Error), the ProviderRouter trips a circuit breaker with a 60-second cooldown.",
    "Fast-bypass skips network roundtrips for providers currently in a cooldown state.",
    "Unrelated log message: Server started on port 8080."
  ]
}
```

---

## 4. Querying Tracking History via CLI

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

---

## 5. Dashboards & Visualization

The evaluation harness provides a read-only Streamlit dashboard for visualizing historical CI/CD evaluation telemetry, quality metric trends over time, and raw run ledgers.

### Launching the Dashboard

Execute the following command in your terminal:

```bash
streamlit run src/reporters/dashboard.py
```

### Key Features
- **KPI Overview Cards:** Displays the latest run's Pass Rate (with SLA status badge), Average Latency (ms), and Total Evaluation Expenditure ($).
- **Quality Score Trends:** Plotly interactive line charts tracking Faithfulness, Answer Relevance, Correctness, Context Precision, and Context Recall over run history.
- **Latency & Cost Profiling:** Visualizes performance and API cost progression across Git commits.
- **Run Ledger Table:** Interactive dataframe displaying historical run records (`runs/history.jsonl`) with search and filter capabilities.
- **Read-Only Invariant:** Strict separation of concerns — the dashboard parses existing JSONL logs and never modifies state or triggers LLM calls.
