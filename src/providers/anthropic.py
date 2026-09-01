import os
import httpx
from typing import Optional
from src.providers.base import LLMProvider, ProviderRateLimitError, ProviderAPIError

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

    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            raise ProviderAPIError("Anthropic API key is required but not set.")
            
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Separate the system prompt from the messages array
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

        try:
            client = self._get_client()
            response = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            if status_code == 429:
                raise ProviderRateLimitError(f"Anthropic API rate limited: {str(e)}", status_code=status_code)
            else:
                raise ProviderAPIError(f"Anthropic API HTTP error {status_code}: {str(e)}", status_code=status_code)
        except httpx.RequestError as e:
            raise ProviderAPIError(f"Anthropic API request error: {str(e)}")
