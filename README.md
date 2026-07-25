# LLMOps CI/CD Evaluation Harness

A lightweight, deterministic, and asynchronous Python framework for evaluating LLM/RAG pipelines inside CI/CD pull request workflows.

## Key Features

- **Concurrent Execution Engine**: Asynchronous pipeline test runs using `asyncio.gather` for fast local and CI/CD validation.
- **LLM Judge Evaluator**: Leverages `gemini-3.5-flash` with structured outputs to grade candidate responses against expected ground truths.
- **Prompt Injection Isolation**: Hardened evaluation templates containing strict `<untrusted_rag_output>` XML wrappers and isolation system instructions to defend the judge against prompt injection.
- **API Cost Caching**: Integrates a SHA-256 prompt-hashing cache (`cache/prompt_hash/`) to prevent duplicate LLM judge calls, saving API costs on repetitive local or CI runs.
- **Meta-Evaluation & Drift Protection**: A pre-graded human benchmark dataset (`datasets/benchmarks/human_labeled.json`) evaluated using Observed Agreement Rate and Cohen's Kappa score to verify Judge grading alignment and protect against drift.
- **PR Merge Blocking SLAs**: A configuration-driven SLA runner (`run_eval.py`) validating latency, pass rate, and cost targets, exiting with code `1` on failure to block GitHub Actions merges.

---

## Directory Structure

```
├── .github/workflows/
│   └── eval_pr.yml          # GitHub Actions workflow running evaluation SLAs
├── configs/
│   └── pr.yaml              # SLA Threshold configurations (latency, pass rate, cost)
├── datasets/benchmarks/
│   └── human_labeled.json   # 20 human-labeled golden evaluation samples
├── scripts/
│   ├── run_meta_eval.py     # Runs meta-evaluation drift checks
│   └── validate_concurrency.py # Validates concurrent execution runner speed
├── src/
│   ├── cache/
│   │   └── prompt_hash.py   # SHA-256 caching layer
│   ├── clients/
│   │   ├── base.py          # SystemUnderTest (SUT) ABC interface
│   │   └── mock_client.py   # Mock SUT implementing latency simulation
│   ├── evaluation/
│   │   ├── prompts/
│   │   │   └── judge_templates.py # Hardened judge prompt templates
│   │   ├── judge.py         # Gemini LLM Judge client
│   │   └── meta_eval.py     # Agreement Rate and Cohen's Kappa calculators
│   ├── metrics/
│   │   └── security.py      # Jailbreak and prompt leakage scanners
│   ├── reporters/
│   │   ├── github.py        # Writes report summary to GitHub Actions
│   │   └── markdown.py      # Generates Markdown report tables
│   └── schemas/
│       └── models.py        # Pydantic schemas (DatasetEntry, EvaluationResult)
├── run_eval.py              # Main CLI SLA verification entry point
└── README.md                # This file
```

---

## Local Verification Commands

To run scripts locally, make sure you are in the project root directory and set the `PYTHONPATH`:

### 1. Run Concurrency Checks
Validates that 50 queries execute concurrently in ~1.00 second:
```bash
PYTHONPATH=. python3 scripts/validate_concurrency.py
```

### 2. Run LLM Judge Meta-Evaluation
Runs meta-eval checks of the LLM Judge against the human benchmark dataset, calculating agreement metrics:
```bash
PYTHONPATH=. python3 scripts/run_meta_eval.py
```

### 3. Run the Main SLA Validation CLI
Runs SUT evaluations against the SLA parameters specified in `configs/pr.yaml` (if no `GEMINI_API_KEY` is present, it will gracefully fall back to local rule-based evaluations):
```bash
PYTHONPATH=. python3 run_eval.py --config configs/pr.yaml
```

---

## CI/CD Workflow Integration

On every Pull Request to the `main` branch, the [eval_pr.yml](file://.github/workflows/eval_pr.yml) workflow:
1. Installs Python dependencies (`httpx`, `pydantic`, `pyyaml`).
2. Pulls the `GEMINI_API_KEY` from GitHub secrets.
3. Triggers the SLA check via `run_eval.py`.
4. Writes a clean Markdown summary table directly to the **GitHub Actions PR Summary page**.
5. Exits with code `1` if SLAs are violated, failing the build and blocking PR merge.
