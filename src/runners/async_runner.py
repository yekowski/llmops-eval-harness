import sys
import time
import hashlib
import asyncio
import traceback
from typing import List, Optional, Dict, Any
from src.schemas.models import DatasetEntry, EvaluationResult, SUTExecutionResult
from src.clients.base import SystemUnderTest
from src.evaluation.judge import LLMJudge
from src.utils.pricing import calculate_token_cost

async def run_evaluation(
    entries: List[DatasetEntry],
    sut: SystemUnderTest,
    judge: Optional[LLMJudge] = None,
    concurrency_config: Optional[Dict[str, Any]] = None
) -> List[EvaluationResult]:
    """Runs concurrent evaluation queries against the system under test (SUT) with validated concurrency and rate controls."""
    cfg = concurrency_config or {}
    max_workers = cfg.get("max_workers", 5 if judge else len(entries) + 1)
    rps = cfg.get("requests_per_second", 1.0)
    interval = (1.0 / rps) if rps > 0 else 0.0

    sem = asyncio.Semaphore(max_workers)
    rate_limit_lock = asyncio.Lock()
    next_available_time = 0.0

    async def _reserve_rate_slot():
        nonlocal next_available_time
        if interval <= 0:
            return
        async with rate_limit_lock:
            now = time.perf_counter()
            scheduled_time = max(now, next_available_time)
            next_available_time = scheduled_time + interval

        delay = scheduled_time - now
        if delay > 0:
            await asyncio.sleep(delay)

    async def evaluate_single(entry: DatasetEntry, idx: int) -> EvaluationResult:
        async with sem:
            start_time = time.perf_counter()
            query_hash = hashlib.sha256(entry.query.encode("utf-8")).hexdigest()[:8]
            try:
                # 1. Execute SUT (Detailed execution with exact tokens)
                if hasattr(sut, "execute_detailed"):
                    sut_res: SUTExecutionResult = await sut.execute_detailed(entry.query)
                else:
                    text = await sut.execute(entry.query)
                    sut_res = SUTExecutionResult(
                        text=text,
                        prompt_tokens=max(1, len(entry.query) // 4),
                        completion_tokens=max(1, len(text) // 4),
                        latency_ms=(time.perf_counter() - start_time) * 1000.0
                    )

                response = sut_res.text
                sut_latency = sut_res.latency_ms / 1000.0 if sut_res.latency_ms > 0 else (time.perf_counter() - start_time)
                sut_tokens = sut_res.prompt_tokens + sut_res.completion_tokens

                # SUT cost calculation: strictly calculate without silently zeroing unknown cloud models
                sut_cost = 0.0
                if hasattr(sut, "provider") and hasattr(sut.provider, "model"):
                    sut_model = sut.provider.model
                    sut_cost = calculate_token_cost(
                        sut_model,
                        sut_res.prompt_tokens,
                        sut_res.completion_tokens
                    )

                judge_latency = 0.0
                judge_mode = "llm"
                retrieval_judge_mode = None
                judge_prompt_tokens = 0
                judge_completion_tokens = 0
                judge_cost = 0.0

                if judge:
                    # 2. Check cache first for generation judge
                    judge_res = None
                    if judge.cache and hasattr(judge, "get_cached_evaluation"):
                        judge_res = judge.get_cached_evaluation(
                            query=entry.query,
                            context=entry.expected_context,
                            expected_answer=entry.expected_answer,
                            generated_answer=response,
                            ground_truth=entry.ground_truth or entry.expected_answer
                        )

                    if judge_res is None:
                        await _reserve_rate_slot()
                        judge_start = time.perf_counter()
                        judge_res = await judge.evaluate(
                            context=entry.expected_context,
                            expected_answer=entry.expected_answer,
                            generated_answer=response,
                            query=entry.query,
                            ground_truth=entry.ground_truth
                        )
                        judge_latency = time.perf_counter() - judge_start

                    passed = judge_res.get("passed", False)
                    faithfulness = judge_res.get("faithfulness", 0.0)
                    relevance = judge_res.get("answer_relevance", 0.0)
                    correctness = judge_res.get("correctness", 0.0)
                    judge_mode = judge_res.get("judge_mode", "llm")
                    judge_prompt_tokens += judge_res.get("judge_prompt_tokens", 0)
                    judge_completion_tokens += judge_res.get("judge_completion_tokens", 0)
                    judge_cost += judge_res.get("judge_cost", 0.0)

                    context_precision = None
                    context_recall = None
                    if entry.retrieved_contexts and hasattr(judge, "evaluate_retrieval"):
                        await _reserve_rate_slot()
                        retrieval_res = await judge.evaluate_retrieval(
                            query=entry.query,
                            retrieved_contexts=entry.retrieved_contexts,
                            ground_truth=entry.ground_truth or entry.expected_answer
                        )
                        context_precision = retrieval_res.get("context_precision")
                        context_recall = retrieval_res.get("context_recall")
                        retrieval_judge_mode = retrieval_res.get("judge_mode", "llm")
                        judge_prompt_tokens += retrieval_res.get("judge_prompt_tokens", 0)
                        judge_completion_tokens += retrieval_res.get("judge_completion_tokens", 0)
                        judge_cost += retrieval_res.get("judge_cost", 0.0)
                else:
                    passed = True
                    faithfulness = 1.0
                    relevance = 1.0
                    correctness = 1.0
                    context_precision = 1.0 if entry.retrieved_contexts else None
                    context_recall = 1.0 if entry.retrieved_contexts else None

                return EvaluationResult(
                    passed=passed,
                    latency=sut_latency,
                    tokens=sut_tokens,
                    sut_prompt_tokens=sut_res.prompt_tokens,
                    sut_completion_tokens=sut_res.completion_tokens,
                    judge_prompt_tokens=judge_prompt_tokens,
                    judge_completion_tokens=judge_completion_tokens,
                    sut_cost=sut_cost,
                    judge_cost=judge_cost,
                    faithfulness=faithfulness,
                    answer_relevance=relevance,
                    correctness=correctness,
                    context_precision=context_precision,
                    context_recall=context_recall,
                    judge_latency=judge_latency,
                    judge_mode=judge_mode,
                    retrieval_judge_mode=retrieval_judge_mode
                )
            except Exception as e:
                sut_latency = time.perf_counter() - start_time
                print(f"[RUNNER ERROR] Failed test case #{idx} [SHA:{query_hash}]: {type(e).__name__}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                return EvaluationResult(
                    passed=False,
                    latency=sut_latency,
                    tokens=0,
                    judge_mode="fallback"
                )

    # Concurrently execute all evaluation tasks
    results = await asyncio.gather(*(evaluate_single(entry, i) for i, entry in enumerate(entries)))
    return list(results)
