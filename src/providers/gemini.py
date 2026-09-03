import os
import time
import random
import asyncio
import httpx
from typing import Optional, Any
from src.providers.base import LLMProvider, ProviderResponse, ProviderRateLimitError, ProviderAPIError

class GeminiProvider(LLMProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.5-flash",
        timeout: float = 30.0,
        client: Optional[httpx.AsyncClient] = None
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self.timeout = timeout
        self._client = client or httpx.AsyncClient(timeout=self.timeout)
        self.execution_mode = "remote"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def generate(self, prompt: str, **kwargs) -> ProviderResponse:
        if not self.api_key:
            raise ProviderAPIError("Gemini API key is required but not set.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        generation_config = {}
        if kwargs.get("json_mode") or kwargs.get("response_format") == "json_object" or kwargs.get("response_format") == {"type": "json_object"}:
            generation_config["responseMimeType"] = "application/json"

        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        if generation_config:
            payload["generationConfig"] = generation_config

        client = self._get_client()
        max_retries = 2
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries):
            start_time = time.perf_counter()
            try:
                response = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                latency_ms = (time.perf_counter() - start_time) * 1000

                text = data["candidates"][0]["content"]["parts"][0]["text"]
                usage = data.get("usageMetadata", {})
                prompt_tokens = usage.get("promptTokenCount", max(1, len(prompt) // 4))
                completion_tokens = usage.get("candidatesTokenCount", max(1, len(text) // 4))

                return ProviderResponse(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    provider_name="GeminiProvider",
                    model_name=self.model,
                    execution_mode="remote"
                )
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                last_exception = e
                if status_code in [429, 500, 502, 503, 504] and attempt < max_retries - 1:
                    backoff = 0.2 * (2 ** attempt) + random.uniform(0.05, 0.15)
                    await asyncio.sleep(backoff)
                    continue

                if status_code == 429:
                    raise ProviderRateLimitError(f"Gemini API rate limited: {str(e)}", status_code=status_code)
                else:
                    raise ProviderAPIError(f"Gemini API HTTP error {status_code}: {str(e)}", status_code=status_code)
            except httpx.RequestError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    backoff = 0.2 * (2 ** attempt) + random.uniform(0.05, 0.15)
                    await asyncio.sleep(backoff)
                    continue
                raise ProviderAPIError(f"Gemini API request error: {str(e)}")

        raise ProviderAPIError(f"Gemini API call failed after {max_retries} attempts: {last_exception}")
