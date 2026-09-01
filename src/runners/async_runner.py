import asyncio
import time
from typing import List, Optional
from src.schemas.models import DatasetEntry, EvaluationResult
from src.clients.base import SystemUnderTest
from src.evaluation.judge import LLMJudge

async def run_evaluation(
    entries: List[DatasetEntry],
    sut: SystemUnderTest,
    judge: Optional[LLMJudge] = None
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
                sut_latency = time.perf_counter() - start_time
                tokens = len(response.split())

                judge_latency = 0.0
                if judge:
                    # Check cache first to avoid rate limiting cache hits
                    judge_res = None
                    if judge.cache and hasattr(judge, "get_cached_evaluation"):
                        judge_res = judge.get_cached_evaluation(entry.query, entry.expected_context, response)

                    if judge_res is None:
                        async with rate_limit_lock:
                            now = time.perf_counter()
                            elapsed = now - last_request_time
                            if elapsed < 1.0:
                                await asyncio.sleep(1.0 - elapsed)
                            last_request_time = time.perf_counter()

                        judge_start = time.perf_counter()
                        judge_res = await judge.evaluate(
                            context=entry.expected_context,
                            expected_answer=entry.expected_answer,
                            generated_answer=response,
                            query=entry.query
                        )
                        judge_latency = time.perf_counter() - judge_start

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
                    # In Phase 1 / Fallback, count successful execution as passed
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
                sut_latency = time.perf_counter() - start_time
                return EvaluationResult(passed=False, latency=sut_latency, tokens=0)

    # Concurrently execute all evaluation tasks
    results = await asyncio.gather(*(evaluate_single(entry) for entry in entries))
    return list(results)
