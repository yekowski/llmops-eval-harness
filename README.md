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

## Installation & Environment Setup

### Core Evaluation & CI Harness
```bash
pip install -r requirements.lock
pip install .
```
Or for local development:
```bash
pip install -e ".[dev]"
```

### Telemetry Dashboard (Optional)
The Streamlit and Plotly interactive dashboard dependencies are decoupled into the `dashboard` optional extra:
```bash
pip install ".[dashboard]"
streamlit run src/reporters/dashboard.py
```

---

## CLI & Evaluation Usage

Once installed, invoke the CLI directly:
```bash
llmops-eval --config configs/pr.yaml
```
Or via the backward-compatible entry point:
```bash
python run_eval.py --config configs/pr.yaml
```

To view past evaluation experiments and SLA records:
```bash
llmops-eval --history
```

---

## Governance & GitHub Branch Protection

Code ownership is canonically configured in [`.github/CODEOWNERS`](.github/CODEOWNERS). To ensure these owner reviews and evaluation gates cannot be bypassed, configure branch protection on `main`:

1. In GitHub repository settings, navigate to **Settings** -> **Branches** -> **Branch protection rules**.
2. Add or edit the rule targeting `main`:
   - Enable **"Require a pull request before merging"**.
   - Enable **"Require approvals"** (set to at least `1`).
   - Enable **"Require review from Code Owners"**.
   - Enable **"Require status checks to pass before merging"** and add:
     - `CI Pipeline` (`lint`, `test`)
     - `LLMOps CI/CD PR Evaluation` (`run-evaluation`)
   - Enable **"Do not allow bypassing the above settings"** for administrators.

---

## CI/CD Workflow Integration

On every Pull Request to the `main` branch:
1. [`.github/workflows/ci.yml`](.github/workflows/ci.yml) validates Ruff linting, Mypy type safety, package wheel build, and Pytest coverage across Python 3.10–3.13.
2. [`.github/workflows/eval_pr.yml`](.github/workflows/eval_pr.yml) installs the wheel, runs `llmops-eval --config configs/pr.yaml`, evaluates SLA gates, and posts a Markdown report to the GitHub Actions PR summary.
3. Exits with code `1` if SLAs are violated or uncertified judge provenance is detected in strict CI mode, blocking PR merge.