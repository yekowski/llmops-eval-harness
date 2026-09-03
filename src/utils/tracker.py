import os
import json
import uuid
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any

class ExperimentTracker:
    def __init__(self, history_path: str = "runs/history.jsonl"):
        self.history_path = history_path
        self.runs_dir = os.path.dirname(self.history_path)

    def _get_git_commit(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
        except Exception:
            return "unknown"

    def log_run(
        self,
        config_path: str,
        dataset_path: str,
        sut_provider: str,
        judge_provider: str,
        aggregate_metrics: Dict[str, Any],
        sla_passed: bool
    ) -> str:
        """Logs a completed evaluation run to the history file. Returns the run_id."""
        if self.runs_dir:
            os.makedirs(self.runs_dir, exist_ok=True)

        run_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        git_commit = self._get_git_commit()

        record = {
            "run_id": run_id,
            "timestamp": timestamp,
            "git_commit": git_commit,
            "config_path": config_path,
            "dataset_path": dataset_path,
            "sut_provider": sut_provider,
            "judge_provider": judge_provider,
            "aggregate_metrics": aggregate_metrics,
            "sla_status": "PASSED" if sla_passed else "FAILED"
        }

        with open(self.history_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        return run_id


def print_history_table(history_path: str = "runs/history.jsonl", limit: int = 5):
    """Prints a formatted ASCII table of the last N runs from the history log."""
    if not os.path.exists(history_path):
        print(f"No history file found at {history_path}")
        return

    records = []
    with open(history_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    if not records:
        print("No records found in history log.")
        return

    # Get last limit records
    records = records[-limit:]

    # Print headers
    headers = ["Run ID", "Timestamp", "Git SHA", "SUT", "Judge", "Pass Rate", "Latency", "Cost", "SLA Status"]
    col_widths = [10, 20, 9, 15, 15, 10, 10, 10, 11]

    row_format = "".join(f"{{:<{w}}}" for w in col_widths)

    print("\n=== Evaluation Run History ===")
    print(row_format.format(*headers))
    print("-" * sum(col_widths))

    for r in records:
        run_id_short = r.get("run_id", "")[:8]
        timestamp = r.get("timestamp", "")
        if "T" in timestamp:
            timestamp = timestamp.replace("T", " ").split(".")[0]

        git_sha = r.get("git_commit", "unknown")
        sut = r.get("sut_provider", "unknown")
        judge = r.get("judge_provider", "unknown")

        metrics = r.get("aggregate_metrics", {})
        pass_rate = f"{metrics.get('pass_rate', 0.0):.1f}%"
        latency = f"{metrics.get('latency', 0.0):.2f}ms"
        cost = f"${metrics.get('cost', 0.0):.4f}"

        sla = r.get("sla_status", "unknown")

        print(row_format.format(
            run_id_short,
            timestamp,
            git_sha,
            sut,
            judge,
            pass_rate,
            latency,
            cost,
            sla
        ))
    print("===============================\n")
