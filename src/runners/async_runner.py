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
        from src.providers.base import generation_latency
        async with sem:
            generation_latency.set(0.0)
            start_time = time.perf_counter()
            try:
                response = await sut.execute(entry.query)
                sut_latency = generation_latency.get()
                if sut_latency == 0.0:
                    sut_latency = time.perf_counter() - start_time
                # Simulate basic token estimation for the response
                tokens = len(response.split())
                
                judge_latency = 0.0
                if judge:
                    # Check cache first to avoid rate limiting cache hits
                    is_cached = False
                    judge_res = None
                    if judge.cache:
                        if hasattr(judge, "get_cached_evaluation"):
                            cached = judge.get_cached_evaluation(entry.query, entry.expected_context, response)
                        else:
                            cached = judge.cache.get(entry.expected_answer, entry.expected_context, response)
                        if cached is not None:
                            is_cached = True
                            judge_res = cached
                    
                    if not is_cached:
                        generation_latency.set(0.0)
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
                                judge_start = time.perf_counter()
                                # Grade the SUT response using the Gemini LLM Judge
                                judge_res = await judge.evaluate(
                                    context=entry.expected_context,
                                    expected_answer=entry.expected_answer,
                                    generated_answer=response,
                                    query=entry.query
                                )
                                jl = generation_latency.get()
                                if jl == 0.0:
                                    jl = time.perf_counter() - judge_start
                                judge_latency = jl
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
                    faithfulness = judge_res.get("faithfulness", 0.0)
                    relevance = judge_res.get("answer_relevance", 0.0)
                    correctness = judge_res.get("correctness", 0.0)

                    context_precision = None
                    context_recall = None
                    if entry.retrieved_contexts and hasattr(judge, "evaluate_retrieval"):
                        retrieval_res = await judge.evaluate_retrieval(
                            query=entry.query,
                            retrieved_contexts=entry.retrieved_contexts,
                            ground_truth=entry.ground_truth or entry.expected_answer
                        )
                        context_precision = retrieval_res.get("context_precision")
                        context_recall = retrieval_res.get("context_recall")
                else:
                    # In Phase 1 / Fallback, we count a successful execution as 'passed'
                    passed = True
                    faithfulness = 1.0
                    relevance = 1.0
                    correctness = 1.0
                    context_precision = 1.0 if entry.retrieved_contexts else None
                    context_recall = 1.0 if entry.retrieved_contexts else None
                    
                return EvaluationResult(
                    passed=passed,
                    latency=sut_latency,
                    tokens=tokens,
                    faithfulness=faithfulness,
                    answer_relevance=relevance,
                    correctness=correctness,
                    context_precision=context_precision,
                    context_recall=context_recall,
                    judge_latency=judge_latency
                )
            except Exception:
                sut_latency = generation_latency.get()
                if sut_latency == 0.0:
                    sut_latency = time.perf_counter() - start_time
                return EvaluationResult(passed=False, latency=sut_latency, tokens=0)

    # Concurrently execute all evaluation tasks
    results = await asyncio.gather(*(evaluate_single(entry) for entry in entries))
    return list(results)

