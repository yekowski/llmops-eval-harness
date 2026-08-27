import os
import httpx
from typing import Optional
from src.providers.base import LLMProvider

class DeepSeekProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            raise ValueError("DeepSeek API key is required but not set.")
            
        url = "https://api.deepseek.com/v1/chat/completions"
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
