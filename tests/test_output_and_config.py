"""
Unit tests for output validation and Ollama LLM configuration.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.agents.llm_config import (
    check_ollama_connection,
    get_llm_settings,
    get_llm_status,
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


class TestOllamaConfig:
    def test_settings_defaults(self):
        settings = get_llm_settings()
        assert settings["provider"] == "ollama"
        assert "localhost" in settings["base_url"]
        assert settings["temperature"] == 0.0

    @patch("urllib.request.urlopen")
    def test_check_ollama_mock(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"models": [{"name": "llama3.2:latest"}]}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        res = check_ollama_connection(base_url="http://localhost:11434", model="llama3.2")
        assert res["reachable"] is True
        assert res["model_installed"] is True
        assert res["available_models"] == ["llama3.2:latest"]
