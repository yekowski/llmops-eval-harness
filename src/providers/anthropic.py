import os
import time
import random
import asyncio
import httpx
from typing import Optional
from src.providers.base import LLMProvider, ProviderResponse, ProviderRateLimitError, ProviderAPIError

class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20240620",
        system: Optional[str] = None,
        timeout: float = 30.0,
        client: Optional[httpx.AsyncClient] = None
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.system = system
        self.timeout = timeout
        self._client = client or httpx.AsyncClient(timeout=self.timeout)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def generate(self, prompt: str, **kwargs) -> ProviderResponse:
        if not self.api_key:
            raise ProviderAPIError("Anthropic API key is required but not set.")

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        system_prompt = kwargs.get("system") or self.system
        if system_prompt:
            payload["system"] = system_prompt

        client = self._get_client()
        max_retries = 3
        last_exception = None

        for attempt in range(max_retries):
            start_time = time.perf_counter()
            try:
                response = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                latency_ms = (time.perf_counter() - start_time) * 1000

                text = data["content"][0]["text"]
                usage = data.get("usage", {})
                prompt_tokens = usage.get("input_tokens", max(1, len(prompt) // 4))
                completion_tokens = usage.get("output_tokens", max(1, len(text) // 4))

                return ProviderResponse(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms
                )
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                last_exception = e
                if status_code in [429, 500, 502, 503, 504] and attempt < max_retries - 1:
                    backoff = (2 ** attempt) + random.uniform(0.1, 0.5)
                    await asyncio.sleep(backoff)
                    continue

                if status_code == 429:
                    raise ProviderRateLimitError(f"Anthropic API rate limited: {str(e)}", status_code=status_code)
                else:
                    raise ProviderAPIError(f"Anthropic API HTTP error {status_code}: {str(e)}", status_code=status_code)
            except httpx.RequestError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    backoff = (2 ** attempt) + random.uniform(0.1, 0.5)
                    await asyncio.sleep(backoff)
                    continue
                raise ProviderAPIError(f"Anthropic API request error: {str(e)}")

        raise ProviderAPIError(f"Anthropic API call failed after {max_retries} attempts: {last_exception}")
