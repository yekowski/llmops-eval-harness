import os
import httpx
from typing import Optional
from src.providers.base import LLMProvider, ProviderRateLimitError, ProviderAPIError

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            raise ProviderAPIError("Gemini API key is required but not set.")
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            if status_code == 429:
                raise ProviderRateLimitError(f"Gemini API rate limited: {str(e)}", status_code=status_code)
            else:
                raise ProviderAPIError(f"Gemini API HTTP error {status_code}: {str(e)}", status_code=status_code)
        except httpx.RequestError as e:
            raise ProviderAPIError(f"Gemini API request error: {str(e)}")
