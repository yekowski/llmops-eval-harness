import os
import json
import hashlib
import asyncio
import tempfile
from typing import Optional, Dict, Any

class EvalCache:
    def __init__(self, cache_dir: str = ".eval_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def get_lock_for_key(self, key: str) -> asyncio.Lock:
        """Returns or creates a per-key lock for single-flight deduplication."""
        async with self._global_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]

    def _compute_hash(
        self,
        generated_answer: str,
        query: str,
        context: str,
        expected_answer: str,
        ground_truth: str,
        provider_class: str,
        model: str,
        prompt_template: str,
        prompt_template_version: str = "v1"
    ) -> str:
        """Computes SHA-256 hash using canonical JSON formatting across all inputs."""
        payload = {
            "version": prompt_template_version,
            "provider_class": provider_class or "",
            "model": model or "",
            "query": query or "",
            "context": context or "",
            "expected_answer": expected_answer or "",
            "ground_truth": ground_truth or "",
            "generated_answer": generated_answer or "",
            "prompt_template": prompt_template or ""
        }
        canonical_str = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def get(
        self,
        generated_answer: str,
        query: str,
        context: str,
        expected_answer: str,
        ground_truth: str,
        provider_class: str,
        model: str,
        prompt_template: str,
        prompt_template_version: str = "v1"
    ) -> Optional[dict]:
        """Retrieve evaluation result from cache if it exists."""
        h = self._compute_hash(
            generated_answer=generated_answer,
            query=query,
            context=context,
            expected_answer=expected_answer,
            ground_truth=ground_truth,
            provider_class=provider_class,
            model=model,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version
        )
        cache_file = os.path.join(self.cache_dir, f"{h}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                return data
            except Exception:
                return None
        return None

    def set(
        self,
        generated_answer: str,
        query: str,
        context: str,
        expected_answer: str,
        ground_truth: str,
        provider_class: str,
        model: str,
        prompt_template: str,
        result: dict,
        prompt_template_version: str = "v1"
    ) -> None:
        """Store evaluation result atomically in the cache."""
        h = self._compute_hash(
            generated_answer=generated_answer,
            query=query,
            context=context,
            expected_answer=expected_answer,
            ground_truth=ground_truth,
            provider_class=provider_class,
            model=model,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version
        )
        cache_file = os.path.join(self.cache_dir, f"{h}.json")
        try:
            # Atomic write via temporary file in the same directory
            with tempfile.NamedTemporaryFile("w", dir=self.cache_dir, delete=False) as tmp_file:
                json.dump(result, tmp_file, indent=2)
                temp_name = tmp_file.name
            os.replace(temp_name, cache_file)
        except Exception as e:
            print(f"Failed to write cache file atomically: {e}")
