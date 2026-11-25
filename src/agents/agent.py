"""
PS 137 - Agricultural Production Scenario Coordinator
======================================================
src/agents/agent.py
High-level AgricultureAgent facade around the LangGraph workflow,
Member 2 historical analysis, and Member 3 scenario simulation engine.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from langchain_core.messages import HumanMessage

from src.agents.scenario_graph import _format_deterministic_explanation, build_graph
from src.analysis.historical_analysis import (
    analyse_crop_wise,
    analyse_state_wise,
    analyse_year_wise,
    filter_data,
    load_cleaned_data,
)
from src.simulation.scenario_comparison import ScenarioComparison
from src.simulation.scenario_engine import AgriculturalScenarioEngine
from src.validation.output_validator import validate_output


class AgricultureAgent:
    """
    Coordinator agent for natural-language agricultural scenario analysis
    powered by Gemini REST API + LangGraph + deterministic Python engines.
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        llm_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.df = df if df is not None else load_cleaned_data()
        self.engine = AgriculturalScenarioEngine(self.df)
        self.llm_factory = llm_factory
        self.graph = build_graph(df=self.df, llm_factory=llm_factory)

    def run(self, user_request: str) -> Dict[str, Any]:
        """Execute the full LangGraph workflow for a natural-language prompt."""
        initial_state = {
            "messages": [HumanMessage(content=user_request)],
            "user_request": user_request,
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
        final_state = self.graph.invoke(initial_state)
        return final_state.get("final_output") or {
            "status": "error",
            "intent": final_state.get("intent", "invalid"),
            "user_request": user_request,
            "needs_clarification": bool(final_state.get("needs_clarification")),
            "clarification_question": final_state.get("clarification_question"),
            "extracted_assumptions": final_state.get("scenarios", []),
            "baseline_results": final_state.get("baseline_results", []),
            "simulation_results": final_state.get("simulation_results", []),
            "validation_results": final_state.get("validation_results", []),
            "comparison_result": final_state.get("comparison_result"),
            "historical_data": None,
            "explanation": final_state.get("explanation", "Workflow did not produce output."),
            "error": final_state.get("error"),
            "is_forecast": False,
        }

    def run_manual_scenario(
        self,
        state: str,
        crop: str,
        year: int,
        area_change_percent: float = 0.0,
        yield_change_percent: float = 0.0,
        district: str = "",
        scenario_name: str = "Manual Scenario",
    ) -> Dict[str, Any]:
        """Run a deterministic scenario directly without requiring the LLM."""
        res = self.engine.calculate_scenario(
            state=state,
            district=district,
            crop=crop,
            year=year,
            area_change_percent=area_change_percent,
            yield_change_percent=yield_change_percent,
            scenario_name=scenario_name,
        )
        val = validate_output(res)
        explanation = _format_deterministic_explanation([res])
        return {
            "status": "success" if res.get("success") else "failed",
            "intent": "scenario_simulation",
            "user_request": f"Manual scenario: {scenario_name}",
            "needs_clarification": False,
            "clarification_question": None,
            "extracted_assumptions": [
                {
                    "state": state,
                    "district": district or None,
                    "crop": crop,
                    "year": year,
                    "area_change_percent": area_change_percent,
                    "yield_change_percent": yield_change_percent,
                    "scenario_name": scenario_name,
                }
            ],
            "baseline_results": [res.get("baseline")] if res.get("baseline") else [],
            "simulation_results": [res],
            "validation_results": [val],
            "comparison_result": None,
            "historical_data": None,
            "explanation": explanation,
            "error": None if res.get("success") else "; ".join(res.get("validation_errors", [])),
            "is_forecast": False,
        }

    def run_historical_analysis(
        self,
        state: Optional[str] = None,
        crop: Optional[str] = None,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run deterministic historical trend analysis directly."""
        filtered = filter_data(
            self.df,
            states=[state] if state else None,
            crops=[crop] if crop else None,
            years=[year] if year else None,
        )
        if filtered.empty:
            return {
                "found": False,
                "record_count": 0,
                "year_wise": [],
                "state_wise": [],
                "crop_wise": [],
            }
        return {
            "found": True,
            "record_count": int(len(filtered)),
            "year_wise": analyse_year_wise(filtered).to_dict(orient="records"),
            "state_wise": analyse_state_wise(filtered).head(10).to_dict(orient="records"),
            "crop_wise": analyse_crop_wise(filtered).head(10).to_dict(orient="records"),
        }
