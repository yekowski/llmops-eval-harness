import os
import sys
import yaml
from typing import Dict, Any, Optional

_PRICING_CACHE: Optional[Dict[str, Any]] = None

def load_pricing_config(config_path: str = "configs/pricing.yaml") -> Dict[str, Any]:
    """Loads model pricing definitions from YAML file with fallback caching."""
    global _PRICING_CACHE
    if _PRICING_CACHE is not None and not os.path.exists(config_path):
        return _PRICING_CACHE

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f) or {}
                _PRICING_CACHE = data.get("models", {})
                return _PRICING_CACHE
        except Exception as e:
            print(f"[WARNING] Could not load pricing config from {config_path}: {e}", file=sys.stderr)

    # Built-in fallback in case config file is missing
    _PRICING_CACHE = {
        "gemini-3.5-flash": {"input_cost_per_1k": 0.000075, "output_cost_per_1k": 0.000300},
        "gemini-1.5-flash": {"input_cost_per_1k": 0.000075, "output_cost_per_1k": 0.000300},
        "gpt-4o": {"input_cost_per_1k": 0.005000, "output_cost_per_1k": 0.015000},
        "gpt-4o-mini": {"input_cost_per_1k": 0.000150, "output_cost_per_1k": 0.000600},
        "claude-3-5-sonnet-20240620": {"input_cost_per_1k": 0.003000, "output_cost_per_1k": 0.015000},
        "deepseek-chat": {"input_cost_per_1k": 0.000140, "output_cost_per_1k": 0.000280},
    }
    return _PRICING_CACHE

def calculate_token_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    pricing_config: Optional[Dict[str, Any]] = None
) -> float:
    """Calculates total evaluation cost in USD based on input and output token counts."""
    pricing = pricing_config or load_pricing_config()
    model_key = (model_name or "").lower().strip()

    rates = pricing.get(model_key)
    if rates is None:
        # Attempt partial match for model variants (e.g., "gemini-3.5-flash" in "models/gemini-3.5-flash")
        for key, val in pricing.items():
            if key in model_key or model_key in key:
                rates = val
                break

    if rates is None:
        print(f"[WARNING] Model '{model_name}' not found in pricing configuration. Defaulting cost to $0.00.", file=sys.stderr)
        return 0.0

    input_rate = rates.get("input_cost_per_1k", 0.0) / 1000.0
    output_rate = rates.get("output_cost_per_1k", 0.0) / 1000.0

    return (input_tokens * input_rate) + (output_tokens * output_rate)
