# Skill: Build Phase 3 - CI/CD Integration & Reporting

## Objective
Finalize the LLMOps evaluation harness by building configuration-driven SLAs, Markdown reporters, and the GitHub Actions workflow.

## Tasks

### Task 1: SLA Configuration
- Create `configs/pr.yaml`.
- Define SLA thresholds: `max_latency_ms: 2000`, `min_pass_rate_pct: 90`, and `max_cost_usd: 0.50`.

### Task 2: Reporting Layer
- Create `src/reporters/markdown.py` with a function to generate a clean GitHub-flavored Markdown table summarizing the metrics against the SLAs.
- Create `src/reporters/github.py` with a function to write the Markdown output to the `$GITHUB_STEP_SUMMARY` environment variable (the standard way to post PR summaries in GitHub Actions).

### Task 3: The Main CLI Entry Point
- Create `run_eval.py` at the project root.
- This script should parse `--config configs/pr.yaml`, load the `SystemUnderTest`, run the `async_runner` with the `GeminiJudge`, aggregate the metrics, and pass them to the markdown reporter.
- If the final metrics fail the SLAs defined in the config, the script must exit with a non-zero status code (`sys.exit(1)`) to fail the CI/CD pipeline.

### Task 4: GitHub Actions Workflow
- Create `.github/workflows/eval_pr.yml`.
- Configure it to trigger on `pull_request` to the `main` branch.
- Add steps to: checkout code, setup Python, install requirements, and execute `python run_eval.py --config configs/pr.yaml`.
- Ensure it securely passes the `GEMINI_API_KEY` from GitHub Secrets to the environment.
