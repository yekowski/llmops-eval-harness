import os
import hashlib
import json
from typing import Optional

class PromptHashCache:
    def __init__(self, cache_dir: str = "cache/prompt_hash"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _compute_hash(self, prompt: str, context: str, candidate_response: str) -> str:
        """Compute SHA-256 hash of the inputs."""
        # Ensure we handle None values gracefully
        p = prompt or ""
        c = context or ""
        r = candidate_response or ""
        combined = f"prompt:{p}||context:{c}||response:{r}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def get(self, prompt: str, context: str, candidate_response: str) -> Optional[dict]:
        """Retrieve evaluation result from cache if it exists."""
        h = self._compute_hash(prompt, context, candidate_response)
        cache_file = os.path.join(self.cache_dir, f"{h}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def set(self, prompt: str, context: str, candidate_response: str, result: dict) -> None:
        """Store evaluation result in the cache."""
        h = self._compute_hash(prompt, context, candidate_response)
        cache_file = os.path.join(self.cache_dir, f"{h}.json")
        try:
            with open(cache_file, "w") as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"Failed to write cache file: {e}")
