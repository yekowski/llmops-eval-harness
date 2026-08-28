import sys
from typing import List
from src.providers.base import LLMProvider, ProviderRateLimitError, ProviderAPIError

import time
import asyncio
from typing import List, Dict

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

    async def generate(self, prompt: str, **kwargs) -> str:
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
                return res
            except (ProviderRateLimitError, ProviderAPIError) as e:
                # Resolve status code to distinguish transient vs permanent errors
                status_code = getattr(e, "status_code", None)
                if status_code is None:
                    # Parse from error message if status_code wasn't explicitly set
                    msg = str(e).lower()
                    if "401" in msg or "unauthorized" in msg or "api key is required" in msg:
                        status_code = 401
                    elif "403" in msg or "forbidden" in msg:
                        status_code = 403
                    elif "429" in msg or "rate limited" in msg or "rate_limit" in msg:
                        status_code = 429
                    elif "500" in msg or "502" in msg or "503" in msg or "504" in msg:
                        status_code = 500

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
