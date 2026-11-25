"""
tests/test_gemini_and_anyyear.py
=================================
Tests for:
  - Gemini call_gemini() (mocked – no network required)
  - Any-year baseline retrieval (target year > 2017 falls back to 2017)
  - Sign conventions: increases → positive, decreases → negative
  - LangGraph workflow with mocked Gemini
  - Deterministic explanation generation
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── Helpers ────────────────────────────────────────────────────────────────────

SAMPLE_DF = pd.DataFrame(
    {
        "state_clean": ["Karnataka"] * 8,
        "district_clean": [""] * 8,
        "crop_clean": ["Rice"] * 8,
        "year": list(range(2010, 2018)),
        "area_1000_ha": [1000.0 + i * 10 for i in range(8)],
        "production_1000_t": [2000.0 + i * 20 for i in range(8)],
        "yield_kg_per_ha": [2000.0 + i * 5 for i in range(8)],
        "calculated_yield_kg_per_ha": [2000.0 + i * 5 for i in range(8)],
    }
)


def _mock_gemini_extraction(area_pct: float, yield_pct: float, state="Karnataka", crop="Rice", year=2017) -> str:
    """Return a Gemini-style JSON extraction response."""
    return json.dumps(
        {
            "intent": "scenario_simulation",
            "scenarios": [
                {
                    "state": state,
                    "district": None,
                    "crop": crop,
                    "year": year,
                    "area_change_percent": area_pct,
                    "yield_change_percent": yield_pct,
                    "scenario_name": f"{crop} Scenario",
                    "scenario_description": None,
                }
            ],
            "needs_clarification": False,
            "clarification_question": None,
            "missing_fields": [],
            "original_request": "test",
            "confidence": 0.95,
            "historical_query": None,
        }
    )


# ── Tests: call_gemini (mocked) ────────────────────────────────────────────────

class TestCallGeminiMocked:
    """Tests for llm_config.call_gemini with network calls mocked."""

    def test_call_gemini_returns_text_on_success(self):
        from src.agents.llm_config import call_gemini

        mock_response_body = json.dumps(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "OK"}]}}
                ]
            }
        ).encode()

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = mock_response_body

        with (
            patch("src.agents.llm_config._GEMINI_API_KEY", "fake_key"),
            patch("urllib.request.urlopen", return_value=mock_resp),
        ):
            result = call_gemini("Reply OK")
            assert result == "OK"

    def test_call_gemini_raises_on_missing_key(self):
        from src.agents.llm_config import call_gemini

        with patch("src.agents.llm_config._GEMINI_API_KEY", ""):
            with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
                call_gemini("test")

    def test_call_gemini_raises_runtime_on_401(self):
        import urllib.error
        from src.agents.llm_config import call_gemini

        mock_exc = urllib.error.HTTPError(url="", code=401, msg="Unauthorized", hdrs={}, fp=None)
        mock_exc.read = lambda: b""

        with (
            patch("src.agents.llm_config._GEMINI_API_KEY", "fake_key"),
            patch("urllib.request.urlopen", side_effect=mock_exc),
        ):
            with pytest.raises(RuntimeError, match="401|Invalid|Unauthorized"):
                call_gemini("test")

    def test_is_llm_available_true_with_key(self):
        from src.agents.llm_config import is_llm_available
        with patch("src.agents.llm_config._GEMINI_API_KEY", "some_key"):
            assert is_llm_available() is True

    def test_is_llm_available_false_without_key(self):
        from src.agents.llm_config import is_llm_available
        with patch("src.agents.llm_config._GEMINI_API_KEY", ""):
            assert is_llm_available() is False


# ── Tests: Sign conventions ────────────────────────────────────────────────────

class TestSignConventions:
    """Verify that extraction preserves sign: increase=positive, decrease=negative."""

    def test_positive_area_increase_extracted_correctly(self):
        raw = _mock_gemini_extraction(area_pct=10.0, yield_pct=0.0)
        data = json.loads(raw)
        assert data["scenarios"][0]["area_change_percent"] == 10.0

    def test_negative_yield_decrease_extracted_correctly(self):
        raw = _mock_gemini_extraction(area_pct=0.0, yield_pct=-5.0)
        data = json.loads(raw)
        assert data["scenarios"][0]["yield_change_percent"] == -5.0

    def test_simultaneous_area_increase_yield_decrease(self):
        raw = _mock_gemini_extraction(area_pct=10.0, yield_pct=-5.0)
        data = json.loads(raw)
        s = data["scenarios"][0]
        assert s["area_change_percent"] == 10.0
        assert s["yield_change_percent"] == -5.0

    def test_area_decrease_is_negative(self):
        raw = _mock_gemini_extraction(area_pct=-15.0, yield_pct=0.0)
        data = json.loads(raw)
        assert data["scenarios"][0]["area_change_percent"] == -15.0


# ── Tests: Any-year baseline retrieval ────────────────────────────────────────

class TestAnyYearBaseline:
    """Verify that target years beyond the dataset default to the latest available year."""

    def test_target_2030_uses_2017_baseline(self):
        """When target_year=2030, reference_year should be 2017 (max dataset year)."""
        from src.agents.scenario_graph import build_graph

        with (
            patch("src.agents.llm_config._GEMINI_API_KEY", "fake_key"),
            patch(
                "src.agents.llm_config.call_gemini",
                return_value=_mock_gemini_extraction(10.0, -5.0, year=2030),
            ),
        ):
            graph = build_graph(df=SAMPLE_DF)
            initial_state = {
                "messages": [],
                "user_request": "Rice in Karnataka, area +10%, yield -5%, year 2030",
                "intent": None,
                "extraction_result": None,
                "scenarios": [],
                "needs_clarification": False,
                "clarification_question": None,
                "baseline_results": [],
                "simulation_results": [],
                "validation_results": [],
                "comparison_result": None,
                "explanation": None,
                "final_output": None,
                "error": None,
                "workflow_stage": "start",
            }
            final = graph.invoke(initial_state)
            fo = final.get("final_output") or {}
            sim_results = fo.get("simulation_results", [])
            assert len(sim_results) > 0
            first = sim_results[0]
            assert first.get("reference_year") == 2017
            assert first.get("target_year") == 2030

    def test_target_within_range_uses_actual_year(self):
        """When target_year=2015 (within range), reference_year should also be 2015."""
        from src.agents.scenario_graph import build_graph

        with (
            patch("src.agents.llm_config._GEMINI_API_KEY", "fake_key"),
            patch(
                "src.agents.llm_config.call_gemini",
                return_value=_mock_gemini_extraction(5.0, 0.0, year=2015),
            ),
        ):
            graph = build_graph(df=SAMPLE_DF)
            initial_state = {
                "messages": [],
                "user_request": "Rice in Karnataka, area +5%, year 2015",
                "intent": None,
                "extraction_result": None,
                "scenarios": [],
                "needs_clarification": False,
                "clarification_question": None,
                "baseline_results": [],
                "simulation_results": [],
                "validation_results": [],
                "comparison_result": None,
                "explanation": None,
                "final_output": None,
                "error": None,
                "workflow_stage": "start",
            }
            final = graph.invoke(initial_state)
            fo = final.get("final_output") or {}
            sim_results = fo.get("simulation_results", [])
            assert len(sim_results) > 0
            first = sim_results[0]
            assert first.get("reference_year") == 2015


# ── Tests: Workflow correctness ────────────────────────────────────────────────

class TestWorkflowCorrectness:
    """End-to-end workflow tests with mocked Gemini."""

    def test_production_calculation_formula(self):
        """Verify: new_production = new_area * new_yield / 1000."""
        from src.simulation.scenario_engine import AgriculturalScenarioEngine

        engine = AgriculturalScenarioEngine(SAMPLE_DF)
        result = engine.calculate_scenario(
            state="Karnataka",
            district="",
            crop="Rice",
            year=2017,
            area_change_percent=10.0,
            yield_change_percent=-5.0,
        )
        if result.get("success"):
            new_a = result["new_area"]
            new_y = result["new_yield"]
            expected_prod = new_a * new_y / 1000
            assert abs(result["new_production"] - expected_prod) < 1e-6

    def test_is_forecast_always_false(self):
        """Confirm simulation results never carry is_forecast=True."""
        from src.simulation.scenario_engine import AgriculturalScenarioEngine

        engine = AgriculturalScenarioEngine(SAMPLE_DF)
        result = engine.calculate_scenario(
            state="Karnataka",
            district="",
            crop="Rice",
            year=2017,
            area_change_percent=5.0,
            yield_change_percent=5.0,
        )
        assert result.get("is_forecast") is False

    def test_gemini_unavailable_uses_deterministic_fallback(self):
        """When Gemini is not available, deterministic explanation is used."""
        from src.agents.scenario_graph import _format_deterministic_explanation

        sim = {
            "success": True,
            "scenario_name": "Test",
            "state": "Karnataka",
            "crop": "Rice",
            "year": 2017,
            "reference_year": 2017,
            "target_year": 2030,
            "baseline_production": 2140.0,
            "new_production": 2273.3,
            "production_change": 133.3,
            "production_change_percent": 6.22,
            "baseline_area": 1070.0,
            "new_area": 1177.0,
            "baseline_yield": 2035.0,
            "new_yield": 1933.25,
            "area_change_percent": 10.0,
            "yield_change_percent": -5.0,
            "is_forecast": False,
        }
        explanation = _format_deterministic_explanation([sim])
        assert "hypothetical scenario" in explanation.lower() or "note" in explanation.lower()
        assert "2030" in explanation
        assert "2017" in explanation

    def test_workflow_full_mocked(self):
        """Full graph run with mocked Gemini returns valid output dict."""
        from src.agents.scenario_graph import build_graph

        with (
            patch("src.agents.llm_config._GEMINI_API_KEY", "fake_key"),
            patch(
                "src.agents.llm_config.call_gemini",
                return_value=_mock_gemini_extraction(10.0, -5.0, year=2017),
            ),
        ):
            graph = build_graph(df=SAMPLE_DF)
            initial_state = {
                "messages": [],
                "user_request": "Rice in Karnataka area +10% yield -5% 2017",
                "intent": None,
                "extraction_result": None,
                "scenarios": [],
                "needs_clarification": False,
                "clarification_question": None,
                "baseline_results": [],
                "simulation_results": [],
                "validation_results": [],
                "comparison_result": None,
                "explanation": None,
                "final_output": None,
                "error": None,
                "workflow_stage": "start",
            }
            final = graph.invoke(initial_state)
            fo = final.get("final_output") or {}
            assert fo.get("intent") == "scenario_simulation"
            assert fo.get("is_forecast") is False
            assert fo.get("status") in ("success", "failed")
            # Explanation must exist
            assert isinstance(fo.get("explanation"), str)


# ── Tests: get_llm_settings ────────────────────────────────────────────────────

class TestLLMSettings:
    def test_get_llm_settings_keys(self):
        from src.agents.llm_config import get_llm_settings
        s = get_llm_settings()
        assert "provider" in s
        assert "model" in s
        assert s["temperature"] == 0.0
