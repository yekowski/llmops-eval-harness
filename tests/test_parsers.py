import pytest
from src.utils.helpers import strip_markdown_json, resolve_error_status_code
from src.providers.base import ProviderRateLimitError, ProviderAPIError

def test_strip_markdown_json_plain():
    text = '{"faithfulness": 0.9, "correctness": 0.9}'
    assert strip_markdown_json(text) == text

def test_strip_markdown_json_with_json_fences():
    text = '```json\n{"faithfulness": 0.9, "correctness": 0.9}\n```'
    expected = '{"faithfulness": 0.9, "correctness": 0.9}'
    assert strip_markdown_json(text) == expected

def test_strip_markdown_json_with_generic_fences():
    text = '```\n{"faithfulness": 0.85}\n```'
    expected = '{"faithfulness": 0.85}'
    assert strip_markdown_json(text) == expected

def test_strip_markdown_json_with_leading_trailing_whitespace():
    text = '  \n```json\n{"key": "value"}\n```  \n'
    expected = '{"key": "value"}'
    assert strip_markdown_json(text) == expected

def test_resolve_error_status_code_explicit_attribute():
    err = ProviderRateLimitError("Rate limit exceeded", status_code=429)
    assert resolve_error_status_code(err) == 429

def test_resolve_error_status_code_parsed_401():
    err = Exception("401 Unauthorized request or invalid API key")
    assert resolve_error_status_code(err) == 401

def test_resolve_error_status_code_parsed_403():
    err = Exception("403 Forbidden access")
    assert resolve_error_status_code(err) == 403

def test_resolve_error_status_code_parsed_429():
    err = Exception("Rate limited: HTTP 429 Too Many Requests")
    assert resolve_error_status_code(err) == 429

def test_resolve_error_status_code_parsed_500():
    err = Exception("Internal Server Error (500)")
    assert resolve_error_status_code(err) == 500

def test_resolve_error_status_code_circuit_open():
    err = Exception("Circuit open for provider")
    assert resolve_error_status_code(err) == 500

def test_resolve_error_status_code_unknown():
    err = Exception("Random unexpected error")
    assert resolve_error_status_code(err) is None
