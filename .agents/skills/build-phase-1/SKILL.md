---
name: build-phase-1
description: Scaffold the core execution layer for the LLMOps CI/CD evaluation harness
---
# Skill: Scaffold Phase 1 - Core Engine

Your task is to build the foundational execution layer for our evaluation harness. 
Please create the following structure and files:

1. `src/schemas/models.py`: Create a simple Pydantic model for a `DatasetEntry` (query, expected_context, expected_answer) and an `EvaluationResult` (pass/fail boolean, latency, tokens).
2. `src/clients/base.py`: Create an abstract base class `SystemUnderTest` with a single async method `async def execute(self, query: str) -> str`.
3. `src/clients/mock_client.py`: Create a `MockRAGClient` that implements `SystemUnderTest` and uses `asyncio.sleep` to simulate a 1-second network call.
4. `src/runners/async_runner.py`: Write an execution engine that takes a list of `DatasetEntry` objects and a `SystemUnderTest`, then uses `asyncio.gather` to execute them concurrently.
