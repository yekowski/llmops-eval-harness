import os
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
