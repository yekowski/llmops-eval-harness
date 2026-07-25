import asyncio
import time
import sys
from src.schemas.models import DatasetEntry
from src.clients.mock_client import MockRAGClient
from src.runners.async_runner import run_evaluation

async def main():
    print("Generating 50 mock dataset entries...")
    entries = [
        DatasetEntry(
            query=f"Query {i}",
            expected_context=f"Context {i}",
            expected_answer=f"Answer {i}"
        )
        for i in range(50)
    ]

    client = MockRAGClient()

    print("Running 50 queries concurrently against MockRAGClient (each has 1s simulated latency)...")
    start_time = time.perf_counter()
    results = await run_evaluation(entries, client)
    end_time = time.perf_counter()

    total_duration = end_time - start_time
    print(f"Completed execution of 50 queries.")
    print(f"Total time elapsed: {total_duration:.2f} seconds")

    # Assertions to confirm concurrent behavior
    assert len(results) == 50, f"Expected 50 results, got {len(results)}"
    
    # Each successful run should have passed
    assert all(r.passed for r in results), "Expected all evaluation results to have passed=True"
    
    # Verify concurrency: if run serially, 50 queries with 1s sleep would take 50 seconds.
    # Concurrently, it should take just slightly over 1 second.
    print(f"Verifying concurrent execution speed...")
    if total_duration < 2.0:
        print("SUCCESS: Execution completed in under 2.0 seconds, confirming concurrent execution.")
    else:
        print(f"FAILURE: Execution took too long ({total_duration:.2f} seconds). Concurrency check failed.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
