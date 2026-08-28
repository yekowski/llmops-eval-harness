import os
import hashlib
import json
from typing import Optional

class EvalCache:
    def __init__(self, cache_dir: str = ".eval_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _compute_hash(
        self,
        generated_answer: str,
        query: str,
        context: str,
        model: str,
        prompt_template: str
    ) -> str:
        """Compute SHA-256 hash of the concatenated inputs deterministically."""
        ans = generated_answer or ""
        q = query or ""
        ctx = context or ""
        mdl = model or ""
        tmpl = prompt_template or ""
        combined = f"answer:{ans}||question:{q}||context:{ctx}||model:{mdl}||template:{tmpl}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def get(
        self,
        generated_answer: str,
        query: str,
        context: str,
        model: str,
        prompt_template: str
    ) -> Optional[dict]:
        """Retrieve evaluation result from cache if it exists."""
        h = self._compute_hash(generated_answer, query, context, model, prompt_template)
        cache_file = os.path.join(self.cache_dir, f"{h}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                print(f"[CACHE HIT] Loaded evaluation for {h[:8]}")
                return data
            except Exception:
                return None
        return None

    def set(
        self,
        generated_answer: str,
        query: str,
        context: str,
        model: str,
        prompt_template: str,
        result: dict
    ) -> None:
        """Store evaluation result in the cache."""
        h = self._compute_hash(generated_answer, query, context, model, prompt_template)
        cache_file = os.path.join(self.cache_dir, f"{h}.json")
        try:
            with open(cache_file, "w") as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"Failed to write cache file: {e}")
