import asyncio
import time
import httpx
import random
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
    limit = 5 if judge else len(entries) + 1
    sem = asyncio.Semaphore(limit)
    rate_limit_lock = asyncio.Lock()
    last_request_time = 0.0

    async def evaluate_single(entry: DatasetEntry) -> EvaluationResult:
        nonlocal last_request_time
        async with sem:
            start_time = time.perf_counter()
            try:
                response = await sut.execute(entry.query)
                latency = time.perf_counter() - start_time
                # Simulate basic token estimation for the response
                tokens = len(response.split())
                
                if judge:
                    # Check cache first to avoid rate limiting cache hits
                    is_cached = False
                    judge_res = None
                    if judge.cache:
                        cached = judge.cache.get(entry.expected_answer, entry.expected_context, response)
                        if cached is not None:
                            is_cached = True
                            judge_res = cached
                    
                    if not is_cached:
                        last_exception = None
                        for attempt in range(3):
                            async with rate_limit_lock:
                                now = time.perf_counter()
                                elapsed = now - last_request_time
                                # Ensure at least 4.5 seconds have elapsed since the last API request started
                                if elapsed < 4.5:
                                    await asyncio.sleep(4.5 - elapsed)
                                last_request_time = time.perf_counter()
                            
                            try:
                                # Grade the SUT response using the Gemini LLM Judge
                                judge_res = await judge.evaluate(
                                    context=entry.expected_context,
                                    expected_answer=entry.expected_answer,
                                    generated_answer=response
                                )
                                break  # Succeeded, exit loop
                            except Exception as e:
                                last_exception = e
                                is_429 = False
                                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
                                    is_429 = True
                                
                                if attempt < 2:
                                    delay = (2 ** attempt) + random.uniform(0.5, 2.0)
                                    err_type = "429 rate limit" if is_429 else "error"
                                    print(f"\n[WARNING] LLM Judge hit {err_type}. Retrying in {delay:.2f} seconds (attempt {attempt + 1}/3)...")
                                    await asyncio.sleep(delay)
                                    continue
                                else:
                                    raise
                        
                        if judge_res is None:
                            raise last_exception if last_exception else Exception("LLM Judge call failed after 3 attempts")
                    
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

