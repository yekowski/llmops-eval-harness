import asyncio
import time
from typing import List, Optional
from src.schemas.models import DatasetEntry, EvaluationResult
from src.clients.base import SystemUnderTest
from src.evaluation.judge import GeminiJudge

async def run_evaluation(
    entries: List[DatasetEntry],
    sut: SystemUnderTest,
    judge: Optional[GeminiJudge] = None
) -> List[EvaluationResult]:
    """Runs concurrent evaluation queries against the system under test (SUT) using asyncio.gather."""
    async def evaluate_single(entry: DatasetEntry) -> EvaluationResult:
        start_time = time.perf_counter()
        try:
            response = await sut.execute(entry.query)
            latency = time.perf_counter() - start_time
            # Simulate basic token estimation for the response
            tokens = len(response.split())
            
            if judge:
                # Grade the SUT response using the Gemini LLM Judge
                judge_res = await judge.evaluate(
                    context=entry.expected_context,
                    expected_answer=entry.expected_answer,
                    generated_answer=response
                )
                passed = judge_res.get("passed", False)
            else:
                # In Phase 1 / Fallback, we count a successful execution as 'passed'
                passed = True
                
            return EvaluationResult(passed=passed, latency=latency, tokens=tokens)
        except Exception:
            latency = time.perf_counter() - start_time
            return EvaluationResult(passed=False, latency=latency, tokens=0)

    # Concurrently execute all evaluation tasks
    results = await asyncio.gather(*(evaluate_single(entry) for entry in entries))
    return list(results)

