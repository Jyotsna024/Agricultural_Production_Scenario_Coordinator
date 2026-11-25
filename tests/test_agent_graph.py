"""
Unit and integration tests for LangGraph workflow and AgricultureAgent
using mocked Gemini API calls (runs fast and reliably without requiring a real API key).
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from src.agents.agent import AgricultureAgent
from src.agents.schemas import ExtractionResult, ScenarioAssumption
from src.analysis.historical_analysis import load_cleaned_data


def make_extraction_json(
    scenario_crop="Rice",
    scenario_state="Karnataka",
    year=2017,
    area_change=10.0,
    yield_change=-5.0,
) -> str:
    """Return a Gemini-style extraction JSON string."""
    return json.dumps({
        "intent": "scenario_simulation",
        "scenarios": [
            {
                "state": scenario_state,
                "district": None,
                "crop": scenario_crop,
                "year": year,
                "area_change_percent": area_change,
                "yield_change_percent": yield_change,
                "scenario_name": "Hypothetical Scenario",
                "scenario_description": None,
            }
        ],
        "needs_clarification": False,
        "clarification_question": None,
        "missing_fields": [],
        "original_request": "Test request",
        "confidence": 0.95,
        "historical_query": None,
    })


@pytest.fixture(scope="module")
def cleaned_df():
    return load_cleaned_data()


class TestAgentWithMockedGemini:
    def test_scenario_simulation_flow(self, cleaned_df):
        """Full graph run with mocked Gemini returns a successful scenario result."""
        with (
            patch("src.agents.llm_config._GEMINI_API_KEY", "fake_key"),
            patch(
                "src.agents.llm_config.call_gemini",
                side_effect=lambda prompt, **kw: (
                    "Hypothetical analysis: production is calculated to increase slightly."
                    if "explain" in prompt.lower() or "baseline" in prompt.lower()
                    else make_extraction_json()
                ),
            ),
        ):
            agent = AgricultureAgent(df=cleaned_df)
            res = agent.run(
                "What happens if rice cultivation area in Karnataka "
                "increases by 10% and yield decreases by 5% in 2017?"
            )
            assert res["status"] == "success"
            assert res["intent"] == "scenario_simulation"
            assert res["is_forecast"] is False
            assert len(res["simulation_results"]) == 1
            sim = res["simulation_results"][0]
            assert sim["success"] is True
            assert sim["crop"].lower() == "rice"
            assert sim["state"].lower() == "karnataka"
            assert sim["new_production"] > 0
            assert sim["baseline_production"] > 0

    def test_missing_field_clarification_flow(self, cleaned_df):
        """When Gemini returns clarification_needed, agent surfaces the question."""
        clarification_json = json.dumps({
            "intent": "clarification_needed",
            "scenarios": [],
            "needs_clarification": True,
            "clarification_question": "Which crop would you like to analyse for Karnataka in 2017?",
            "missing_fields": ["crop"],
            "original_request": "Increase area by 10% in Karnataka 2017",
            "confidence": 0.8,
            "historical_query": None,
        })

        with (
            patch("src.agents.llm_config._GEMINI_API_KEY", "fake_key"),
            patch("src.agents.llm_config.call_gemini", return_value=clarification_json),
        ):
            agent = AgricultureAgent(df=cleaned_df)
            res = agent.run("Increase area by 10% in Karnataka 2017")
            assert res["needs_clarification"] is True
            assert "crop" in res["clarification_question"].lower()

    def test_manual_scenario_execution(self, cleaned_df):
        """Direct Python path — no LLM needed at all."""
        agent = AgricultureAgent(df=cleaned_df)
        res = agent.run_manual_scenario(
            state="Karnataka",
            crop="Rice",
            year=2017,
            area_change_percent=10.0,
            yield_change_percent=-5.0,
            district="",
            scenario_name="Direct Test",
        )
        assert res["status"] == "success"
        assert res["simulation_results"][0]["success"] is True
        assert res["is_forecast"] is False
        assert res["simulation_results"][0]["new_production"] > 0

    def test_sign_convention_increase_gives_positive(self, cleaned_df):
        """Area increase +10% should result in new_area > baseline_area."""
        agent = AgricultureAgent(df=cleaned_df)
        res = agent.run_manual_scenario(
            state="Karnataka",
            crop="Rice",
            year=2017,
            area_change_percent=10.0,
            yield_change_percent=0.0,
        )
        sim = res["simulation_results"][0]
        if sim["success"]:
            assert sim["new_area"] > sim["baseline_area"]

    def test_sign_convention_decrease_gives_smaller_area(self, cleaned_df):
        """Area decrease -10% should result in new_area < baseline_area."""
        agent = AgricultureAgent(df=cleaned_df)
        res = agent.run_manual_scenario(
            state="Karnataka",
            crop="Rice",
            year=2017,
            area_change_percent=-10.0,
            yield_change_percent=0.0,
        )
        sim = res["simulation_results"][0]
        if sim["success"]:
            assert sim["new_area"] < sim["baseline_area"]

    def test_production_formula_correctness(self, cleaned_df):
        """new_production = new_area * new_yield / 1000."""
        agent = AgricultureAgent(df=cleaned_df)
        res = agent.run_manual_scenario(
            state="Karnataka",
            crop="Rice",
            year=2017,
            area_change_percent=10.0,
            yield_change_percent=-5.0,
        )
        sim = res["simulation_results"][0]
        if sim["success"]:
            expected = sim["new_area"] * sim["new_yield"] / 1000
            assert abs(sim["new_production"] - expected) < 1e-4
