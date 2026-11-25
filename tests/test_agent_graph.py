"""
Unit and integration tests for LangGraph workflow and AgricultureAgent
using mocked LLM outputs (runs fast and reliably without requiring live Ollama).
"""

import json
from unittest.mock import MagicMock
import pytest
from langchain_core.messages import AIMessage

from src.agents.agent import AgricultureAgent
from src.agents.schemas import ExtractionResult, ScenarioAssumption
from src.analysis.historical_analysis import load_cleaned_data


def make_mock_llm(scenario_crop="Rice", scenario_state="Karnataka", year=2017):
    """Return a mocked LLM that outputs valid extraction JSON."""
    mock = MagicMock()
    extraction_json = json.dumps({
        "intent": "scenario_simulation",
        "scenarios": [
            {
                "state": scenario_state,
                "district": None,
                "crop": scenario_crop,
                "year": year,
                "area_change_percent": 10.0,
                "yield_change_percent": -5.0,
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

    def invoke_side_effect(messages, **kwargs):
        # If asked for explanation or plain text
        for m in messages:
            content = getattr(m, "content", "")
            if "explain" in content.lower():
                return AIMessage(content="Hypothetical analysis: production is calculated to increase slightly.")
        return AIMessage(content=extraction_json)

    mock.invoke.side_effect = invoke_side_effect
    return mock


@pytest.fixture(scope="module")
def cleaned_df():
    return load_cleaned_data()


class TestAgentWithMockedLLM:
    def test_scenario_simulation_flow(self, cleaned_df):
        mock_llm = make_mock_llm()
        agent = AgricultureAgent(df=cleaned_df, llm_factory=lambda: mock_llm)

        res = agent.run("What happens if rice cultivation area in Karnataka increases by 10% and yield decreases by 5% in 2017?")
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
        # Mock LLM that indicates missing crop
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=json.dumps({
            "intent": "clarification_needed",
            "scenarios": [],
            "needs_clarification": True,
            "clarification_question": "Which crop would you like to analyse for Karnataka in 2017?",
            "missing_fields": ["crop"],
            "original_request": "Increase area by 10% in Karnataka 2017",
            "confidence": 0.8,
            "historical_query": None,
        }))

        agent = AgricultureAgent(df=cleaned_df, llm_factory=lambda: mock_llm)
        res = agent.run("Increase area by 10% in Karnataka 2017")
        assert res["needs_clarification"] is True
        assert "crop" in res["clarification_question"].lower()

    def test_manual_scenario_execution(self, cleaned_df):
        # Direct Python path - no LLM needed at all
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
