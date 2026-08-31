import os
import httpx
from typing import Optional
from src.providers.base import LLMProvider, ProviderRateLimitError, ProviderAPIError

class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        api_key_env_var: str = "OPENAI_API_KEY",
        temperature: Optional[float] = None
    ):
        self.api_key = api_key or os.environ.get(api_key_env_var)
        self.base_url = base_url
        self.model = model
        self.api_key_env_var = api_key_env_var
        self.temperature = temperature

    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            raise ProviderAPIError(f"API key is required but not set (checked env var {self.api_key_env_var}).")
            
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        target_model = self.model
        if target_model in ["llama-3.3-70b-versatile", "llama3-8b-8192"]:
            target_model = "groq/compound"

        payload = {
            "model": target_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "response_format": {
                "type": "json_object"
            }
        }
        
        # Use temperature if specified in generate call or stored in provider
        temp = kwargs.get("temperature", self.temperature)
        if temp is not None:
            payload["temperature"] = temp

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers, timeout=5.0)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            if status_code == 429:
                raise ProviderRateLimitError(f"{self.__class__.__name__} rate limited: {str(e)}", status_code=status_code)
            else:
                raise ProviderAPIError(f"{self.__class__.__name__} HTTP error {status_code}: {str(e)}", status_code=status_code)
        except httpx.RequestError as e:
            raise ProviderAPIError(f"{self.__class__.__name__} request error: {str(e)}")
