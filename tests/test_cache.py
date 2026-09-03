import pytest
import asyncio
from src.utils.cache import EvalCache

def test_cache_hash_determinism(tmp_path):
    cache = EvalCache(cache_dir=str(tmp_path))
    hash1 = cache._compute_hash("ans", "q", "ctx", "exp", "gt", "GeminiProvider", "gemini-3.5-flash", "tmpl1")
    hash2 = cache._compute_hash("ans", "q", "ctx", "exp", "gt", "GeminiProvider", "gemini-3.5-flash", "tmpl1")
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex digest length

def test_cache_input_dimension_sensitivity(tmp_path):
    cache = EvalCache(cache_dir=str(tmp_path))
    h_base = cache._compute_hash("ans", "q", "ctx", "exp", "gt", "GeminiProvider", "model", "tmpl")
    h_diff_ans = cache._compute_hash("ans_diff", "q", "ctx", "exp", "gt", "GeminiProvider", "model", "tmpl")
    h_diff_q = cache._compute_hash("ans", "q_diff", "ctx", "exp", "gt", "GeminiProvider", "model", "tmpl")
    h_diff_ctx = cache._compute_hash("ans", "q", "ctx_diff", "exp", "gt", "GeminiProvider", "model", "tmpl")
    h_diff_exp = cache._compute_hash("ans", "q", "ctx", "exp_diff", "gt", "GeminiProvider", "model", "tmpl")
    h_diff_gt = cache._compute_hash("ans", "q", "ctx", "exp", "gt_diff", "GeminiProvider", "model", "tmpl")
    h_diff_prov = cache._compute_hash("ans", "q", "ctx", "exp", "gt", "OpenAIProvider", "model", "tmpl")
    h_diff_model = cache._compute_hash("ans", "q", "ctx", "exp", "gt", "GeminiProvider", "model_diff", "tmpl")
    h_diff_tmpl = cache._compute_hash("ans", "q", "ctx", "exp", "gt", "GeminiProvider", "model", "tmpl_diff")

    hashes = {h_base, h_diff_ans, h_diff_q, h_diff_ctx, h_diff_exp, h_diff_gt, h_diff_prov, h_diff_model, h_diff_tmpl}
    assert len(hashes) == 9  # All 9 dimension combinations produce distinct hashes

def test_cache_hit_and_miss(tmp_path):
    cache = EvalCache(cache_dir=str(tmp_path))
    ans = "Paris is the capital of France."
    q = "What is the capital of France?"
    ctx = "France's capital city is Paris."
    exp = "Paris"
    gt = "Paris"
    prov = "GeminiProvider"
    model = "gemini-3.5-flash"
    tmpl = "prompt template string"

    # Cache Miss
    miss_res = cache.get(ans, q, ctx, exp, gt, prov, model, tmpl)
    assert miss_res is None

    # Store in Cache
    eval_data = {"faithfulness": 1.0, "correctness": 1.0, "passed": True}
    cache.set(ans, q, ctx, exp, gt, prov, model, tmpl, eval_data)

    # Cache Hit
    hit_res = cache.get(ans, q, ctx, exp, gt, prov, model, tmpl)
    assert hit_res is not None
    assert hit_res["faithfulness"] == 1.0
    assert hit_res["passed"] is True

@pytest.mark.asyncio
async def test_cache_single_flight_lock(tmp_path):
    cache = EvalCache(cache_dir=str(tmp_path))
    key = "test-hash-key"
    lock1 = await cache.get_lock_for_key(key)
    lock2 = await cache.get_lock_for_key(key)
    assert lock1 is lock2

@pytest.mark.asyncio
async def test_cache_concurrent_failover_single_flight(tmp_path):
    """Verifies that under concurrent failover:
    1. Primary provider fails with 429 rate limit.
    2. Secondary remote provider resolves with a delay.
    3. Single-flight lock deduplicates concurrent tasks: secondary provider call_count is exactly 1.
    4. One task receives judge_mode="llm", remaining 9 waiting tasks receive judge_mode="cache".
    5. All tasks return identical scores and certified remote_llm provenance without race conditions or corruption.
    """
    from src.evaluation.judge import LLMJudge
    from src.providers.base import LLMProvider, ProviderResponse, ProviderRateLimitError
    from src.providers.router import ProviderRouter

    class PrimaryFailingRemote(LLMProvider):
        def __init__(self):
            self.model = "gpt-4o"
            self.api_key = "remote-key-1"
            self.provider_name = "OpenAIProvider"
            self.execution_mode = "remote"
            self.call_count = 0

        async def generate(self, prompt, **kwargs):
            self.call_count += 1
            raise ProviderRateLimitError("Rate limit 429", status_code=429)

    class SecondaryWorkingRemote(LLMProvider):
        def __init__(self):
            self.model = "gemini-3.5-flash"
            self.api_key = "remote-key-2"
            self.provider_name = "GeminiProvider"
            self.execution_mode = "remote"
            self.call_count = 0

        async def generate(self, prompt, **kwargs):
            self.call_count += 1
            # Artificial async yield so waiting tasks queue behind single-flight lock
            await asyncio.sleep(0.05)
            return ProviderResponse(
                text='{"faithfulness": 0.95, "answer_relevance": 0.90, "correctness": 0.88, "faithfulness_reasoning": "Accurate", "answer_relevance_reasoning": "Relevant", "correctness_reasoning": "Correct"}',
                prompt_tokens=100,
                completion_tokens=50,
                provider_name="GeminiProvider",
                model_name="gemini-3.5-flash",
                execution_mode="remote"
            )

    p1 = PrimaryFailingRemote()
    p2 = SecondaryWorkingRemote()
    router = ProviderRouter([p1, p2])
    cache = EvalCache(cache_dir=str(tmp_path))
    judge = LLMJudge(provider=router, cache=cache)

    context = "France's capital city is Paris."
    expected = "Paris"
    generated = "The capital of France is Paris."
    query = "What is the capital of France?"

    # Launch 10 simultaneous evaluation requests
    tasks = [
        judge.evaluate(
            context=context,
            expected_answer=expected,
            generated_answer=generated,
            query=query
        )
        for _ in range(10)
    ]
    results = await asyncio.gather(*tasks)

    # 1. Exactly one call to secondary provider
    assert p2.call_count == 1, f"Expected 1 call to secondary provider, got {p2.call_count}"

    # 2. All 10 tasks return identical scores
    faithfulness_scores = [r["faithfulness"] for r in results]
    relevance_scores = [r["answer_relevance"] for r in results]
    correctness_scores = [r["correctness"] for r in results]

    assert all(s == 0.95 for s in faithfulness_scores)
    assert all(s == 0.90 for s in relevance_scores)
    assert all(s == 0.88 for s in correctness_scores)

    # 3. Exactly one task is "llm" (the executing task) and 9 tasks are "cache"
    modes = [r["judge_mode"] for r in results]
    assert modes.count("llm") == 1, f"Expected 1 'llm' mode, got {modes.count('llm')}"
    assert modes.count("cache") == 9, f"Expected 9 'cache' modes, got {modes.count('cache')}"

    # 4. All tasks have verified remote_llm provenance
    provenances = [r["judge_provenance"] for r in results]
    assert all(p == "remote_llm" for p in provenances)

    # 5. Subsequent call hits cache without calling secondary provider again
    res11 = await judge.evaluate(
        context=context,
        expected_answer=expected,
        generated_answer=generated,
        query=query
    )
    assert res11["judge_mode"] == "cache"
    assert res11["judge_provenance"] == "remote_llm"
    assert p2.call_count == 1

