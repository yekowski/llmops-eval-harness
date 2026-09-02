import pytest
from src.utils.pricing import load_pricing_config, calculate_token_cost, PricingError

def test_load_pricing_config():
    config = load_pricing_config("configs/pricing.yaml")
    assert "gemini-3.5-flash" in config
    assert "gpt-4o" in config
    assert config["gemini-3.5-flash"]["input_cost_per_1k"] == 0.000075

def test_calculate_token_cost_known_model():
    # 1000 input tokens ($0.000075) + 1000 output tokens ($0.000300) = $0.000375
    cost = calculate_token_cost("gemini-3.5-flash", input_tokens=1000, output_tokens=1000)
    assert cost == pytest.approx(0.000375)

def test_calculate_token_cost_partial_match():
    # Model name with prefix/suffix (e.g., "models/gemini-3.5-flash")
    cost = calculate_token_cost("models/gemini-3.5-flash", input_tokens=1000, output_tokens=1000)
    assert cost == pytest.approx(0.000375)

def test_calculate_token_cost_unknown_model_raises_pricing_error():
    # Unknown model without allow_unknown must raise PricingError
    with pytest.raises(PricingError, match="has no defined pricing rates"):
        calculate_token_cost("unknown-future-model-v99", input_tokens=1000, output_tokens=1000)

def test_calculate_token_cost_unknown_model_allow_unknown_fallback():
    # When allow_unknown is True, defaults to 0.0 with warning
    cost = calculate_token_cost("unknown-future-model-v99", input_tokens=1000, output_tokens=1000, allow_unknown=True)
    assert cost == 0.0
