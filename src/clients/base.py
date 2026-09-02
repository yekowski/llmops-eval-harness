import time
from abc import ABC, abstractmethod
from src.schemas.models import SUTExecutionResult

class SystemUnderTest(ABC):
    @abstractmethod
    async def execute(self, query: str) -> str:
        """Execute the system under test with the given query and return the answer."""
        pass

    async def execute_detailed(self, query: str) -> SUTExecutionResult:
        """Executes the system under test and returns detailed execution telemetry.
        Defaults to wrapping the basic execute() method for backward compatibility.
        """
        start = time.perf_counter()
        text = await self.execute(query)
        latency_ms = (time.perf_counter() - start) * 1000.0
        prompt_tokens = max(1, len(query) // 4)
        completion_tokens = max(1, len(text) // 4)
        return SUTExecutionResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms
        )
