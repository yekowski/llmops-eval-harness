#!/usr/bin/env python3
"""LLMOps CI/CD Evaluation Harness runner CLI entrypoint wrapper.

This file serves as a backward-compatible entrypoint wrapper around the canonical
package CLI implemented in `src.cli`.
"""
from src.cli import (
    main,
    main_async,
    load_and_validate_config,
    initialize_harness,
    evaluate_dataset,
    _compute_deltas,
    _evaluate_sla_gates,
    enforce_slas_and_report,
)

__all__ = [
    "main",
    "main_async",
    "load_and_validate_config",
    "initialize_harness",
    "evaluate_dataset",
    "_compute_deltas",
    "_evaluate_sla_gates",
    "enforce_slas_and_report",
]

if __name__ == "__main__":
    raise SystemExit(main())
