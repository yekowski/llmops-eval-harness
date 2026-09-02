import os
import sys
import yaml
from typing import Dict, Any, Optional

class PricingError(Exception):
    """Exception raised when pricing information is missing or undefined for a model."""
    pass

_PRICING_CACHE: Dict[str, Dict[str, Any]] = {}

_DEFAULT_BUILTIN_PRICING: Dict[str, Any] = {
    "gemini-3.5-flash": {"input_cost_per_1k": 0.000075, "output_cost_per_1k": 0.000300},
    "gemini-1.5-flash": {"input_cost_per_1k": 0.000075, "output_cost_per_1k": 0.000300},
    "gemini-1.5-pro": {"input_cost_per_1k": 0.001250, "output_cost_per_1k": 0.005000},
    "gpt-4o": {"input_cost_per_1k": 0.005000, "output_cost_per_1k": 0.015000},
    "gpt-4o-mini": {"input_cost_per_1k": 0.000150, "output_cost_per_1k": 0.000600},
    "claude-3-5-sonnet-20240620": {"input_cost_per_1k": 0.003000, "output_cost_per_1k": 0.015000},
    "claude-3-5-haiku": {"input_cost_per_1k": 0.000800, "output_cost_per_1k": 0.004000},
    "deepseek-chat": {"input_cost_per_1k": 0.000140, "output_cost_per_1k": 0.000280},
    "llama-3.3-70b-versatile": {"input_cost_per_1k": 0.000590, "output_cost_per_1k": 0.000790},
    "mock": {"input_cost_per_1k": 0.0, "output_cost_per_1k": 0.0},
    "local": {"input_cost_per_1k": 0.0, "output_cost_per_1k": 0.0},
    "default": {"input_cost_per_1k": 0.0, "output_cost_per_1k": 0.0},
}

def load_pricing_config(config_path: str = "configs/pricing.yaml") -> Dict[str, Any]:
    """Loads model pricing definitions from YAML file with path-isolated caching."""
    if config_path in _PRICING_CACHE:
        return _PRICING_CACHE[config_path]

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f) or {}
                models_dict = data.get("models", {})
                _PRICING_CACHE[config_path] = models_dict
                return models_dict
        except Exception as e:
            print(f"[WARNING] Could not load pricing config from {config_path}: {e}", file=sys.stderr)

    return _DEFAULT_BUILTIN_PRICING

def calculate_token_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    pricing_config: Optional[Dict[str, Any]] = None,
    allow_unknown: bool = False
) -> float:
    """Calculates total evaluation cost in USD based on input and output token counts.

    Raises PricingError if model is unknown and allow_unknown is False.
    """
    pricing = pricing_config if pricing_config is not None else load_pricing_config()
    model_key = (model_name or "").lower().strip()

    # Local / mock models are always zero cost
    if any(loc in model_key for loc in ["mock", "local", "ollama", "vllm", "llama3.2"]):
        return 0.0

    rates = pricing.get(model_key)
    if rates is None:
        # Attempt partial match for model variants (e.g., "gemini-3.5-flash" in "models/gemini-3.5-flash")
        for key, val in pricing.items():
            if len(key) >= 4 and (key in model_key or model_key in key):
                rates = val
                break

    if rates is None:
        if not allow_unknown:
            raise PricingError(f"Model '{model_name}' has no defined pricing rates in pricing configuration.")
        print(f"[WARNING] Model '{model_name}' not found in pricing configuration. Defaulting cost to $0.00.", file=sys.stderr)
        return 0.0

    input_rate = rates.get("input_cost_per_1k", 0.0) / 1000.0
    output_rate = rates.get("output_cost_per_1k", 0.0) / 1000.0

    return (input_tokens * input_rate) + (output_tokens * output_rate)
