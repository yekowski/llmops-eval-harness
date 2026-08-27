import os
import httpx
from typing import Optional
from src.providers.base import LLMProvider

class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        api_key_env_var: str = "OPENAI_API_KEY"
    ):
        self.api_key = api_key or os.environ.get(api_key_env_var)
        self.base_url = base_url
        self.model = model
        self.api_key_env_var = api_key_env_var

    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            raise ValueError(f"API key is required but not set (checked env var {self.api_key_env_var}).")
            
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "response_format": {
                "type": "json_object"
            }
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
