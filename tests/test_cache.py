import pytest
from src.utils.cache import EvalCache

def test_cache_hash_determinism(tmp_path):
    cache = EvalCache(cache_dir=str(tmp_path))
    hash1 = cache._compute_hash("ans", "q", "ctx", "model1", "tmpl1")
    hash2 = cache._compute_hash("ans", "q", "ctx", "model1", "tmpl1")
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex digest length

def test_cache_input_dimension_sensitivity(tmp_path):
    cache = EvalCache(cache_dir=str(tmp_path))
    h_base = cache._compute_hash("ans", "q", "ctx", "model", "tmpl")
    h_diff_ans = cache._compute_hash("ans_diff", "q", "ctx", "model", "tmpl")
    h_diff_q = cache._compute_hash("ans", "q_diff", "ctx", "model", "tmpl")
    h_diff_ctx = cache._compute_hash("ans", "q", "ctx_diff", "model", "tmpl")
    h_diff_model = cache._compute_hash("ans", "q", "ctx", "model_diff", "tmpl")
    h_diff_tmpl = cache._compute_hash("ans", "q", "ctx", "model", "tmpl_diff")

    hashes = {h_base, h_diff_ans, h_diff_q, h_diff_ctx, h_diff_model, h_diff_tmpl}
    assert len(hashes) == 6  # All 6 combinations produce distinct hashes

def test_cache_hit_and_miss(tmp_path):
    cache = EvalCache(cache_dir=str(tmp_path))
    ans = "Paris is the capital of France."
    q = "What is the capital of France?"
    ctx = "France's capital city is Paris."
    model = "gemini-3.5-flash"
    tmpl = "prompt template string"

    # Cache Miss
    miss_res = cache.get(ans, q, ctx, model, tmpl)
    assert miss_res is None

    # Store in Cache
    eval_data = {"faithfulness": 1.0, "correctness": 1.0, "passed": True}
    cache.set(ans, q, ctx, model, tmpl, eval_data)

    # Cache Hit
    hit_res = cache.get(ans, q, ctx, model, tmpl)
    assert hit_res is not None
    assert hit_res["faithfulness"] == 1.0
    assert hit_res["passed"] is True
