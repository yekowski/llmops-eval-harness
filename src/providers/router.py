import sys
import time
import asyncio
from typing import List, Dict
from src.providers.base import LLMProvider, ProviderResponse, ProviderRateLimitError, ProviderAPIError
from src.utils.helpers import resolve_error_status_code

class ProviderRouter(LLMProvider):
    def __init__(self, providers: List[LLMProvider]):
        if not providers:
            raise ValueError("ProviderRouter requires at least one provider in the fallback chain.")
        self.providers = providers
        self.active_provider = providers[0]
        # In-memory dictionary to track when a provider's cooldown ends
        self._cooldowns: Dict[str, float] = {}
        # Concurrency safety lock for _cooldowns reads/writes
        self._lock = asyncio.Lock()

    async def warmup(self) -> None:
        """Sends a lightweight health check to each provider sequentially to pre-warm the circuit breaker before concurrent runs."""
        import httpx

        for provider in self.providers:
            provider_name = provider.__class__.__name__
            if provider_name == "MockProvider":
                continue
            
            async with self._lock:
                cooldown_end = self._cooldowns.get(provider_name, 0.0)
            if time.time() < cooldown_end:
                continue

            try:
                # For local providers (Ollama/vLLM), use a lightweight HTTP health check
                # instead of a full generation call that would take 30-60s on CPU
                is_local = getattr(provider, "_is_local", False)
                if is_local:
                    base_url = getattr(provider, "base_url", "")
                    # Ollama health check: GET /api/tags (fast, no inference)
                    if "11434" in base_url or "ollama" in provider_name.lower():
                        health_url = base_url.replace("/v1", "/api/tags")
                    else:
                        health_url = base_url.rstrip("/")
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(health_url, timeout=5.0)
                        resp.raise_for_status()
                    continue  # Healthy, skip to next provider

                await provider.generate("ping")
            except (ProviderRateLimitError, ProviderAPIError) as e:
                status_code = resolve_error_status_code(e)

                if status_code in [401, 403]:
                    print(f"[CIRCUIT DISABLED] Provider '{provider_name}' authentication failed. Disabling for entire run.", file=sys.stderr)
                    async with self._lock:
                        self._cooldowns[provider_name] = float('inf')
                else:
                    print(f"[CIRCUIT TRIPPED] Provider '{provider_name}' rate limited/unavailable. Cooling down for 60s.", file=sys.stderr)
                    async with self._lock:
                        self._cooldowns[provider_name] = time.time() + 60.0
            except Exception:
                pass

    async def generate(self, prompt: str, **kwargs) -> ProviderResponse:
        for i, provider in enumerate(self.providers):
            provider_name = provider.__class__.__name__
            
            # Fast Bypass Check
            async with self._lock:
                cooldown_end = self._cooldowns.get(provider_name, 0.0)
            
            if time.time() < cooldown_end:
                print(f"[CIRCUIT OPEN] Skipping '{provider_name}' (in cooldown)", file=sys.stderr)
                continue
                
            self.active_provider = provider
            try:
                # Attempt to generate using the current provider
                from src.providers.base import generation_latency
                start_time = time.perf_counter()
                res = await provider.generate(prompt, **kwargs)
                latency = time.perf_counter() - start_time
                generation_latency.set(latency)
                if not isinstance(res, ProviderResponse):
                    res = ProviderResponse(text=str(res), latency_ms=latency * 1000)
                return res
            except (ProviderRateLimitError, ProviderAPIError) as e:
                # Resolve status code to distinguish transient vs permanent errors
                status_code = resolve_error_status_code(e)

                # Differentiated handling based on error permanence
                if status_code in [401, 403]:
                    print(f"[CIRCUIT DISABLED] Provider '{provider_name}' authentication failed. Disabling for entire run.", file=sys.stderr)
                    async with self._lock:
                        self._cooldowns[provider_name] = float('inf')
                else:
                    # Default to transient cooldown (60 seconds)
                    print(f"[CIRCUIT TRIPPED] Provider '{provider_name}' rate limited/unavailable. Cooling down for 60s.", file=sys.stderr)
                    async with self._lock:
                        self._cooldowns[provider_name] = time.time() + 60.0

                # Format specific failover log messaging
                if i < len(self.providers) - 1:
                    next_provider_name = self.providers[i + 1].__class__.__name__
                    print(
                        f"\n[ROUTER WARNING] Provider '{provider_name}' failed (429/Unavailable). "
                        f"Falling back to '{next_provider_name}'...",
                        file=sys.stderr
                    )
                else:
                    print(
                        f"\n[ROUTER WARNING] Provider '{provider_name}' failed (429/Unavailable). "
                        f"All providers in fallback chain failed.",
                        file=sys.stderr
                    )
                continue
        
        # If all providers fail, raise RuntimeError with explicit messaging
        raise RuntimeError("All providers in fallback chain failed.")

    @property
    def model(self) -> str:
        """Exposes the active provider model dynamically to keep token cost tracking accurate."""
        return getattr(self.active_provider, "model", "mock")
