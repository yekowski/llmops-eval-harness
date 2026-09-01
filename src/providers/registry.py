from typing import List, Dict, Any, Optional, Type
from src.providers.base import LLMProvider
from src.providers.gemini import GeminiProvider
from src.providers.openai import OpenAIProvider
from src.providers.deepseek import DeepSeekProvider
from src.providers.groq import GroqProvider
from src.providers.qwen import QwenProvider
from src.providers.anthropic import AnthropicProvider
from src.providers.mock import MockProvider
from src.providers.ollama import OllamaProvider
from src.providers.vllm import VLLMProvider
from src.providers.router import ProviderRouter

PROVIDER_REGISTRY: Dict[str, Type[LLMProvider]] = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "deepseek": DeepSeekProvider,
    "groq": GroqProvider,
    "qwen": QwenProvider,
    "anthropic": AnthropicProvider,
    "mock": MockProvider,
    "ollama": OllamaProvider,
    "vllm": VLLMProvider,
}

def build_provider(name: str, **kwargs) -> LLMProvider:
    """Instantiates a single provider instance by name from the PROVIDER_REGISTRY."""
    name_lower = name.lower()
    provider_cls = PROVIDER_REGISTRY.get(name_lower)
    if not provider_cls:
        raise ValueError(f"Unknown provider name '{name}' in config fallback_chain.")
    return provider_cls(**kwargs)

def build_provider_from_config(
    fallback_chain: Optional[List[str]],
    providers_config: Optional[Dict[str, Any]] = None
) -> Optional[ProviderRouter]:
    """Builds a ProviderRouter from a fallback chain list and provider options dict."""
    if not fallback_chain:
        return None

    providers_config = providers_config or {}
    chain_instances: List[LLMProvider] = []

    for name in fallback_chain:
        name_lower = name.lower()
        prov_opts = providers_config.get(name_lower, {})

        kwargs = {}
        model_name = prov_opts.get("model")
        temperature = prov_opts.get("temperature")
        base_url = prov_opts.get("base_url")

        if model_name is not None:
            kwargs["model"] = model_name
        if temperature is not None:
            kwargs["temperature"] = temperature
        if base_url is not None:
            kwargs["base_url"] = base_url

        instance = build_provider(name_lower, **kwargs)
        chain_instances.append(instance)

    return ProviderRouter(chain_instances)
