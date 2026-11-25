"""
llm_config.py
=============
Load environment variables and initialize the local Ollama LLM safely.
Modular provider design so other providers can be added if needed,
with local Ollama as the default (no cloud API key required).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load .env file from project root (or any parent directory)
load_dotenv()

DEFAULT_PROVIDER = "ollama"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_TEMPERATURE = 0.0


def get_llm_settings() -> Dict[str, Any]:
    """Read current LLM settings from environment variables."""
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower() or DEFAULT_PROVIDER
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/") or DEFAULT_OLLAMA_BASE_URL
    model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL
    try:
        temperature = float(os.getenv("OLLAMA_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
    except ValueError:
        temperature = DEFAULT_TEMPERATURE

    return {
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "temperature": temperature,
    }


def _model_matches(configured_model: str, available_models: List[str]) -> bool:
    """Return True if configured_model matches an installed Ollama model tag."""
    if not configured_model or not available_models:
        return False
    target = configured_model.lower()
    for m in available_models:
        ml = m.lower()
        if ml == target:
            return True
        if ":" not in target and ml == f"{target}:latest":
            return True
        if ml.split(":")[0] == target.split(":")[0] and ":" not in target:
            return True
    return False


def check_ollama_connection(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 3.0,
) -> Dict[str, Any]:
    """
    Check whether the local Ollama service is reachable and whether the
    configured model is installed.

    Returns
    -------
    dict with keys:
        reachable (bool), model_installed (bool), available_models (list[str]),
        base_url (str), model (str), error (str|None)
    """
    settings = get_llm_settings()
    url = (base_url or settings["base_url"]).rstrip("/")
    target_model = model or settings["model"]
    tags_url = f"{url}/api/tags"

    try:
        req = urllib.request.Request(tags_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return {
            "reachable": False,
            "model_installed": False,
            "available_models": [],
            "base_url": url,
            "model": target_model,
            "error": (
                f"Cannot connect to Ollama at {url}. "
                f"Start the service by running 'ollama serve' in a terminal. ({exc.reason})"
            ),
        }
    except Exception as exc:
        return {
            "reachable": False,
            "model_installed": False,
            "available_models": [],
            "base_url": url,
            "model": target_model,
            "error": (
                f"Failed to reach Ollama at {url}. "
                f"Ensure Ollama is running via 'ollama serve'. ({exc})"
            ),
        }

    models_raw = payload.get("models", [])
    available_models = [
        m.get("name", "") for m in models_raw if isinstance(m, dict) and m.get("name")
    ]
    installed = _model_matches(target_model, available_models)

    error = None
    if not installed:
        installed_str = ", ".join(available_models) if available_models else "none"
        error = (
            f"Model '{target_model}' is not installed in Ollama "
            f"(installed: {installed_str}). "
            f"Run 'ollama pull {target_model}' to download it."
        )

    return {
        "reachable": True,
        "model_installed": installed,
        "available_models": available_models,
        "base_url": url,
        "model": target_model,
        "error": error,
    }


def get_llm_status() -> Dict[str, Any]:
    """
    Returns a dict describing the LLM configuration and local service status.
    Keys: available (bool), provider (str), model (str), base_url (str),
          available_models (list[str]), error (str|None)
    """
    settings = get_llm_settings()
    provider = settings["provider"]
    model = settings["model"]
    base_url = settings["base_url"]

    if provider != "ollama":
        return {
            "available": False,
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "available_models": [],
            "error": f"Unsupported LLM_PROVIDER '{provider}'. Set LLM_PROVIDER=ollama.",
        }

    try:
        from langchain_ollama import ChatOllama  # noqa: F401
    except ImportError as exc:
        return {
            "available": False,
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "available_models": [],
            "error": f"langchain-ollama is not installed: {exc}. Run: pip install langchain-ollama",
        }

    conn = check_ollama_connection(base_url=base_url, model=model)
    return {
        "available": bool(conn["reachable"] and conn["model_installed"]),
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "available_models": conn["available_models"],
        "error": conn["error"],
    }


def is_llm_available() -> bool:
    """Returns True if the configured local LLM is ready to use."""
    return get_llm_status()["available"]


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
):
    """
    Return an initialized LangChain chat model (ChatOllama by default).

    Parameters
    ----------
    provider    : override LLM_PROVIDER (defaults to env / 'ollama').
    model       : override OLLAMA_MODEL (defaults to env / 'llama3.2').
    base_url    : override OLLAMA_BASE_URL (defaults to env / 'http://localhost:11434').
    temperature : override OLLAMA_TEMPERATURE (defaults to 0.0 for deterministic extraction).

    Raises
    ------
    ValueError      : if provider is unsupported.
    ImportError     : if langchain-ollama is not installed.
    ConnectionError : if Ollama service is not reachable or model is not installed.
    """
    settings = get_llm_settings()
    resolved_provider = (provider or settings["provider"]).strip().lower()
    resolved_model = model or settings["model"]
    resolved_url = (base_url or settings["base_url"]).rstrip("/")
    resolved_temp = settings["temperature"] if temperature is None else float(temperature)

    if resolved_provider != "ollama":
        raise ValueError(
            f"Unsupported LLM provider '{resolved_provider}'. "
            "Currently configured provider is 'ollama'."
        )

    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama is not installed. Run: pip install langchain-ollama"
        ) from exc

    conn = check_ollama_connection(base_url=resolved_url, model=resolved_model)
    if not conn["reachable"]:
        raise ConnectionError(conn["error"])
    if not conn["model_installed"]:
        raise ValueError(conn["error"])

    return ChatOllama(
        model=resolved_model,
        base_url=resolved_url,
        temperature=resolved_temp,
    )


def test_llm_connectivity(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform a simple end-to-end connectivity test against the local Ollama model.

    Returns
    -------
    dict with keys: success (bool), provider (str), model (str), response (str|None), error (str|None)
    """
    settings = get_llm_settings()
    resolved_model = model or settings["model"]
    try:
        llm = get_llm(model=resolved_model, base_url=base_url)
        reply = llm.invoke("Reply with the single word: OK")
        text = getattr(reply, "content", str(reply)).strip()
        return {
            "success": True,
            "provider": settings["provider"],
            "model": resolved_model,
            "response": text,
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "provider": settings["provider"],
            "model": resolved_model,
            "response": None,
            "error": str(exc),
        }
