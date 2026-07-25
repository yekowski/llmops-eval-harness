from abc import ABC, abstractmethod

class SystemUnderTest(ABC):
    @abstractmethod
    async def execute(self, query: str) -> str:
        """Execute the system under test with the given query and return the answer."""
        pass
