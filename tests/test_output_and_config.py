"""
Unit tests for output validation and Gemini LLM configuration.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.agents.llm_config import (
    get_llm_settings,
    get_llm_status,
    is_llm_available,
)
from src.validation.output_validator import validate_explanation_text, validate_output


class TestOutputValidator:
    def test_valid_scenario_output(self):
        sample = {
            "success": True,
            "scenario_name": "Test",
            "state": "Karnataka",
            "district": "Bangalore",
            "crop": "Rice",
            "year": 2017,
            "baseline": {},
            "area_change_percent": 10.0,
            "yield_change_percent": -5.0,
            "baseline_area": 100.0,
            "baseline_yield": 2000.0,
            "baseline_production": 200.0,
            "new_area": 110.0,
            "new_yield": 1900.0,
            "new_production": 110.0 * 1900.0 / 1000.0,
            "production_change": (110.0 * 1900.0 / 1000.0) - 200.0,
            "production_change_percent": 4.5,
            "units": {"area": "1000 ha", "production": "1000 t", "yield": "kg/ha"},
            "validation_errors": [],
            "is_forecast": False,
            "scenario_type": "area_and_yield",
        }
        res = validate_output(sample, explanation="This is a hypothetical scenario analysis.")
        assert res["valid"] is True
        assert len(res["errors"]) == 0

    def test_rejects_is_forecast_true(self):
        sample = {
            "success": True,
            "scenario_name": "Test",
            "state": "Karnataka",
            "district": "Bangalore",
            "crop": "Rice",
            "year": 2017,
            "baseline": {},
            "area_change_percent": 10.0,
            "yield_change_percent": 0.0,
            "baseline_area": 100.0,
            "baseline_yield": 2000.0,
            "baseline_production": 200.0,
            "new_area": 110.0,
            "new_yield": 2000.0,
            "new_production": 220.0,
            "production_change": 20.0,
            "production_change_percent": 10.0,
            "units": {"area": "1000 ha"},
            "validation_errors": [],
            "is_forecast": True,  # INVALID: must be False
            "scenario_type": "area_only",
        }
        res = validate_output(sample)
        assert res["valid"] is False
        assert any("is_forecast" in e for e in res["errors"])

    def test_forecast_language_warning(self):
        exp = "This will definitely produce 500kt and actual future forecast shows growth."
        res = validate_explanation_text(exp)
        assert res["valid"] is True
        assert len(res["warnings"]) > 0


class TestGeminiConfig:
    """Tests for the Gemini LLM configuration (replaces old Ollama tests)."""

    def test_settings_defaults(self):
        settings = get_llm_settings()
        assert "provider" in settings
        assert "model" in settings
        assert settings["temperature"] == 0.0

    def test_settings_has_model_key(self):
        settings = get_llm_settings()
        assert isinstance(settings["model"], str)
        assert len(settings["model"]) > 0

    def test_is_llm_available_with_key(self):
        with patch("src.agents.llm_config._GEMINI_API_KEY", "fake_key_12345"):
            assert is_llm_available() is True

    def test_is_llm_available_without_key(self):
        with patch("src.agents.llm_config._GEMINI_API_KEY", ""):
            assert is_llm_available() is False

    def test_get_llm_status_no_key(self):
        with patch("src.agents.llm_config._GEMINI_API_KEY", ""):
            status = get_llm_status()
            assert status["available"] is False
            assert "GEMINI_API_KEY" in status["error"]

    def test_get_llm_status_with_valid_key_mocked(self):
        """Mock a successful Gemini response to test status=available."""
        import json
        mock_response_body = json.dumps(
            {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]}
        ).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = mock_response_body

        with (
            patch("src.agents.llm_config._GEMINI_API_KEY", "fake_key"),
            patch("urllib.request.urlopen", return_value=mock_resp),
        ):
            status = get_llm_status()
            assert status["available"] is True
            assert status["error"] is None
