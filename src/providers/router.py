import sys
import time
import asyncio
import httpx
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
        for provider in self.providers:
            provider_name = provider.__class__.__name__
            if provider_name == "MockProvider":
                continue

            async with self._lock:
                cooldown_end = self._cooldowns.get(provider_name, 0.0)
            if time.time() < cooldown_end:
                continue

            try:
                # Use lightweight provider health check URL if defined
                health_url = provider.health_check_url
                if health_url:
                    client = provider._get_client() if hasattr(provider, "_get_client") else httpx.AsyncClient(timeout=5.0)
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
        overall_start_time = time.perf_counter()

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
                res = await provider.generate(prompt, **kwargs)
                total_latency_ms = (time.perf_counter() - overall_start_time) * 1000.0

                if not isinstance(res, ProviderResponse):
                    res = ProviderResponse(text=str(res))

                res.latency_ms = total_latency_ms
                if not res.provider_name:
                    res.provider_name = provider_name
                if not res.model_name:
                    res.model_name = getattr(provider, "model", "default")
                if not hasattr(res, "execution_mode") or res.execution_mode is None:
                    res.execution_mode = getattr(provider, "execution_mode", "remote")

                return res
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

        raise RuntimeError("All providers in fallback chain failed.")

    @property
    def model(self) -> str:
        """Exposes the active provider model dynamically to keep token cost tracking accurate."""
        return getattr(self.active_provider, "model", "mock")

    @model.setter
    def model(self, value: str) -> None:
        if self.active_provider:
            self.active_provider.model = value

    async def close(self) -> None:
        """Closes all underlying provider clients to prevent open connection leaks."""
        for provider in self.providers:
            if hasattr(provider, "close"):
                try:
                    await provider.close()
                except Exception:
                    pass
