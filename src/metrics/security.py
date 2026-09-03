import re
from typing import Dict

JAILBREAK_PATTERNS = [
    r"ignore (all )?prior",
    r"ignore (all )?instructions",
    r"bypass safety",
    r"override rules",
    r"system prompt override",
    r"dan mode",
    r"do anything now",
    r"you must now",
    r"act as",
    r"developer mode",
]

PROMPT_LEAKAGE_PATTERNS = [
    r"my system instructions",
    r"i am programmed to",
    r"under no circumstances",
    r"you are an ai assistant designed",
    r"ignore the instructions",
]

def scan_text(text: str) -> Dict[str, bool]:
    """Scans text for security vulnerabilities: jailbreaks and prompt leakage."""
    if not text:
        return {"jailbreak_detected": False, "leakage_detected": False}

    jailbreak_detected = any(
        re.search(pattern, text, re.IGNORECASE) for pattern in JAILBREAK_PATTERNS
    )

    leakage_detected = any(
        re.search(pattern, text, re.IGNORECASE) for pattern in PROMPT_LEAKAGE_PATTERNS
    )

    return {
        "jailbreak_detected": jailbreak_detected,
        "leakage_detected": leakage_detected
    }
