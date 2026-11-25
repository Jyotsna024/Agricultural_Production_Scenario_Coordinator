"""
llm_config.py – Gemini REST API integration (no Ollama required)
================================================================
Loads configuration via python-dotenv.
API key is NEVER logged or exposed.
Supports gemini-3.5-flash-lite (default) with automatic fast fallback on 429/503.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

_FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]

_PLACEHOLDER_KEYS = {"", "your_gemini_api_key_here", "your_key_here"}


def get_llm_settings() -> Dict[str, Any]:
    return {
        "provider": _PROVIDER,
        "model": _GEMINI_MODEL,
        "base_url": _BASE_URL,
        "temperature": 0.0,
    }


def is_llm_available() -> bool:
    """Quick local check (no network call): returns True if a non-placeholder API key is set."""
    return bool(_GEMINI_API_KEY) and _GEMINI_API_KEY not in _PLACEHOLDER_KEYS


def get_llm_status() -> Dict[str, Any]:
    """Probe the Gemini API (makes one lightweight generateContent request)."""
    if not is_llm_available():
        return {
            "available": False,
            "provider": _PROVIDER,
            "model": _GEMINI_MODEL,
            "error": "GEMINI_API_KEY is not set. Add it to your .env file.",
        }
    try:
        call_gemini("Reply with the single word: OK", max_output_tokens=256)
        return {
            "available": True,
            "provider": _PROVIDER,
            "model": _GEMINI_MODEL,
            "error": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "provider": _PROVIDER,
            "model": _GEMINI_MODEL,
            "error": str(exc),
        }


def _single_gemini_request(
    model: str,
    prompt: str,
    system_instruction: Optional[str],
    temperature: float,
    max_output_tokens: int,
) -> str:
    url = f"{_BASE_URL}/{model}:generateContent?key={_GEMINI_API_KEY}"
    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    if system_instruction:
        payload["system_instruction"] = {"parts": [{"text": system_instruction}]}

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": _GEMINI_API_KEY,
    }
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        candidates = result.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "")
        return ""


def call_gemini(
    prompt: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
) -> str:
    """
    Call the Gemini generateContent REST endpoint, failing over quickly to
    alternate Flash models on 429/503/404 without long sleeps.
    """
    if not is_llm_available():
        raise EnvironmentError("GEMINI_API_KEY is not set. Add it to your .env file.")

    # Build ordered list of models: configured model first, then fallbacks
    models_to_try: List[str] = [_GEMINI_MODEL] + [
        m for m in _FALLBACK_MODELS if m != _GEMINI_MODEL
    ]

    last_err: Optional[Exception] = None
    for model in models_to_try:
        try:
            return _single_gemini_request(
                model=model,
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        except urllib.error.HTTPError as exc:
            code = exc.code
            body_text = ""
            try:
                body_text = exc.read().decode("utf-8")
            except Exception:
                pass
            if code in (401,) or (code == 400 and "API_KEY_INVALID" in body_text):
                raise RuntimeError(
                    "Gemini API: Invalid API key (401/400). "
                    "Please update GEMINI_API_KEY in your .env file."
                )
            if code == 403:
                raise RuntimeError(
                    "Gemini API: Forbidden (403). The key may not have access to this model."
                )
            # On 429, 503, 404 -> try next fallback model immediately
            last_err = RuntimeError(f"Gemini API HTTP {code} on {model}: {exc.reason}")
            continue
        except Exception as exc:
            last_err = RuntimeError(f"Gemini API request failed on {model}: {exc}")
            continue

    raise last_err or RuntimeError("Gemini API: All models failed.")


def get_llm():
    """Returns a LangChain ChatGoogleGenerativeAI instance."""
    if not is_llm_available():
        raise EnvironmentError("GEMINI_API_KEY is not set.")
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=_GEMINI_MODEL,
        google_api_key=_GEMINI_API_KEY,
        temperature=0.0,
        convert_system_message_to_human=False,
    )


def test_llm_connectivity() -> Dict[str, Any]:
    """Quick end-to-end connectivity test."""
    try:
        reply = call_gemini("Reply with the single word: OK", max_output_tokens=256)
        return {
            "success": "OK" in reply.upper(),
            "provider": _PROVIDER,
            "model": _GEMINI_MODEL,
            "response": reply.strip(),
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "provider": _PROVIDER,
            "model": _GEMINI_MODEL,
            "response": None,
            "error": str(exc),
        }
