import os
import httpx
from typing import Optional
from src.providers.base import LLMProvider

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            raise ValueError("Gemini API key is required but not set.")
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
