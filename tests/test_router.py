import asyncio
import pytest
from typing import Literal
from src.providers.base import LLMProvider, ProviderResponse, ProviderRateLimitError, ProviderAPIError
from src.providers.router import ProviderRouter

class FailingProvider(LLMProvider):
    def __init__(self, error_type: str = "429", status_code: int = 429, delay_seconds: float = 0.0):
        self.error_type = error_type
        self.status_code = status_code
        self.delay_seconds = delay_seconds
        self.call_count = 0

    async def generate(self, prompt: str, **kwargs) -> ProviderResponse:
        self.call_count += 1
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        if self.error_type == "429":
            raise ProviderRateLimitError("Rate limit hit", status_code=self.status_code)
        elif self.error_type == "401":
            raise ProviderAPIError("401 Unauthorized API key", status_code=self.status_code)
        else:
            raise ProviderAPIError("500 Internal Error", status_code=self.status_code)

class WorkingProvider(LLMProvider):
    def __init__(self, response_text: str = "Success", delay_seconds: float = 0.0, execution_mode: Literal["remote", "local", "mock"] = "remote", model: str = "mock-model"):
        self.response_text = response_text
        self.delay_seconds = delay_seconds
        self.execution_mode = execution_mode
        self.model = model
        self.call_count = 0

    async def generate(self, prompt: str, **kwargs) -> ProviderResponse:
        self.call_count += 1
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        return ProviderResponse(
            text=self.response_text,
            prompt_tokens=10,
            completion_tokens=5,
            provider_name="WorkingProvider",
            model_name=self.model,
            execution_mode=self.execution_mode
        )

def test_router_fallback_order():
    async def _test():
        p1 = FailingProvider("429", 429)
        p2 = WorkingProvider("Fallback Success")
        router = ProviderRouter([p1, p2])

        resp = await router.generate("test prompt")
        assert resp.text == "Fallback Success"
        assert p1.call_count == 1
        assert p2.call_count == 1
        assert router.active_provider == p2
        assert resp.provider_name == "WorkingProvider"
        assert resp.execution_mode == "remote"
    asyncio.run(_test())

def test_router_total_fallback_latency_measurement():
    """Verifies that ProviderRouter.generate() captures total elapsed time across failed providers and delays."""
    async def _test():
        p1 = FailingProvider("429", 429, delay_seconds=0.05)
        p2 = WorkingProvider("Success after delay", delay_seconds=0.05)
        router = ProviderRouter([p1, p2])

        resp = await router.generate("test prompt")
        assert resp.text == "Success after delay"
        # Total latency must be at least ~100ms (0.05s + 0.05s)
        assert resp.latency_ms >= 90.0
    asyncio.run(_test())

def test_router_circuit_breaker_fast_bypass():
    async def _test():
        p1 = FailingProvider("429", 429)
        p2 = WorkingProvider("Fallback Success")
        router = ProviderRouter([p1, p2])

        # First call trips circuit breaker on p1
        await router.generate("prompt 1")
        assert p1.call_count == 1

        # Second call fast-bypasses p1 without calling generate() again
        resp2 = await router.generate("prompt 2")
        assert resp2.text == "Fallback Success"
        assert p1.call_count == 1  # Fast bypass skipped p1!
        assert p2.call_count == 2
    asyncio.run(_test())

def test_router_permanent_disable_on_401():
    async def _test():
        p1 = FailingProvider("401", 401)
        p2 = WorkingProvider("Working after 401")
        router = ProviderRouter([p1, p2])

        resp = await router.generate("prompt 1")
        assert resp.text == "Working after 401"
        assert router._cooldowns.get("FailingProvider") == float('inf')
    asyncio.run(_test())

def test_router_all_providers_failing_raises():
    async def _test():
        p1 = FailingProvider("429", 429)
        p2 = FailingProvider("500", 500)
        router = ProviderRouter([p1, p2])

        with pytest.raises(RuntimeError, match="All providers in fallback chain failed"):
            await router.generate("prompt")
    asyncio.run(_test())
