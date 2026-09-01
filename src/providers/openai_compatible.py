import os
import time
import random
import asyncio
import httpx
from typing import Optional
from src.providers.base import LLMProvider, ProviderResponse, ProviderRateLimitError, ProviderAPIError

class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        api_key_env_var: str = "OPENAI_API_KEY",
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        client: Optional[httpx.AsyncClient] = None
    ):
        self.base_url = base_url
        self.api_key = api_key or os.environ.get(api_key_env_var)
        self._is_local = any(loc in base_url for loc in ["localhost", "127.0.0.1", "11434", "8000", "0.0.0.0"])
        if not self.api_key and self._is_local:
            self.api_key = "local"
        self.model = model
        self.api_key_env_var = api_key_env_var
        self.temperature = temperature
        # Local models (CPU inference) need much longer timeouts than cloud APIs
        self.timeout = timeout or (120.0 if self._is_local else 10.0)
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
            raise ProviderAPIError(f"API key is required but not set (checked env var {self.api_key_env_var}).")

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        target_model = self.model
        if "api.groq.com" in self.base_url and target_model in ["llama-3.3-70b-versatile", "llama3-8b-8192"]:
            target_model = "groq/compound"

        payload = {
            "model": target_model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        if "response_format" in kwargs:
            payload["response_format"] = kwargs["response_format"]
        elif kwargs.get("json_mode"):
            payload["response_format"] = {"type": "json_object"}

        temp = kwargs.get("temperature", self.temperature)
        if temp is not None:
            payload["temperature"] = temp

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

                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", max(1, len(prompt) // 4))
                completion_tokens = usage.get("completion_tokens", max(1, len(text) // 4))

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
                    raise ProviderRateLimitError(f"{self.__class__.__name__} rate limited: {str(e)}", status_code=status_code)
                else:
                    raise ProviderAPIError(f"{self.__class__.__name__} HTTP error {status_code}: {str(e)}", status_code=status_code)
            except httpx.RequestError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    backoff = (2 ** attempt) + random.uniform(0.1, 0.5)
                    await asyncio.sleep(backoff)
                    continue
                raise ProviderAPIError(f"{self.__class__.__name__} request error: {str(e)}")

        raise ProviderAPIError(f"{self.__class__.__name__} API call failed after {max_retries} attempts: {last_exception}")
