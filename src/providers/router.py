import sys
from typing import List
from src.providers.base import LLMProvider, ProviderRateLimitError, ProviderAPIError

class ProviderRouter(LLMProvider):
    def __init__(self, providers: List[LLMProvider]):
        if not providers:
            raise ValueError("ProviderRouter requires at least one provider in the fallback chain.")
        self.providers = providers
        self.active_provider = providers[0]

    async def generate(self, prompt: str, **kwargs) -> str:
        for i, provider in enumerate(self.providers):
            self.active_provider = provider
            provider_name = provider.__class__.__name__
            try:
                # Attempt to generate using the current provider
                import time
                from src.providers.base import generation_latency
                start_time = time.perf_counter()
                res = await provider.generate(prompt, **kwargs)
                latency = time.perf_counter() - start_time
                generation_latency.set(latency)
                return res
            except (ProviderRateLimitError, ProviderAPIError) as e:
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
