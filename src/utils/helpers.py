from typing import Optional

def strip_markdown_json(text: str) -> str:
    """Strips Markdown code fences (e.g., ```json ... ```) from JSON string output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned

def resolve_error_status_code(exc: Exception) -> Optional[int]:
    """Extracts or infers HTTP status code from provider exception or error message."""
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code

    msg = str(exc).lower()
    if "401" in msg or "unauthorized" in msg or "api key is required" in msg:
        return 401
    elif "403" in msg or "forbidden" in msg:
        return 403
    elif "429" in msg or "rate limited" in msg or "rate_limit" in msg:
        return 429
    elif any(code in msg for code in ["500", "502", "503", "504", "server error", "circuit open", "all providers in fallback chain failed"]):
        return 500
    return None
