import asyncio
from src.clients.base import SystemUnderTest

class MockRAGClient(SystemUnderTest):
    async def execute(self, query: str) -> str:
        """Simulates a 1-second network call to the system under test."""
        await asyncio.sleep(1.0)
        return f"Mocked response for query: {query}"
