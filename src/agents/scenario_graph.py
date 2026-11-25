"""
PS 137 - Agricultural Production Scenario Coordinator
======================================================
src/agents/scenario_graph.py
LangGraph workflow orchestrating:
  1. Natural-language understanding & structured extraction (local Ollama)
  2. Extraction & required-field validation
  3. Conditional routing:
       - clarification_needed / invalid -> return clarification or error
       - historical_analysis -> Member 2 lookup -> Ollama explanation
       - scenario_simulation / comparison -> Member 2 baseline -> Member 3 engine
         -> output validation -> (optional comparison) -> Ollama explanation
  4. Final structured output assembly
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from src.agents.llm_config import get_llm
from src.agents.prompts import (
    CLARIFICATION_SYSTEM_PROMPT,
    EXPLANATION_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    HISTORICAL_ANALYSIS_SYSTEM_PROMPT,
    format_explanation_prompt,
    format_extraction_prompt,
    format_historical_prompt,
)
from src.agents.schemas import ExtractionResult, ScenarioAssumption
from src.agents.state import AgentState
from src.analysis.historical_analysis import (
    analyse_crop_wise,
    analyse_state_wise,
    analyse_year_wise,
    filter_data,
    get_aggregated_baseline,
    get_historical_baseline,
    load_cleaned_data,
)
from src.simulation.scenario_comparison import ScenarioComparison
from src.simulation.scenario_engine import AgriculturalScenarioEngine
from src.validation.output_validator import validate_output


def _extract_json_from_text(raw_text: str) -> dict:
    """Parse JSON from an LLM response, stripping optional markdown fences."""
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            text = brace_match.group(0)
    return json.loads(text)


def _invoke_extraction(llm: Any, user_request: str) -> ExtractionResult:
    """
    Use ChatOllama to extract structured ExtractionResult.
    Tries structured output first, then falls back to JSON parsing for
    broader compatibility across local Ollama models.
    """
    messages = [
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=format_extraction_prompt(user_request)),
    ]

    # Try structured output if supported
    if hasattr(llm, "with_structured_output"):
        try:
            structured_llm = llm.with_structured_output(ExtractionResult)
            res = structured_llm.invoke(messages)
            if isinstance(res, ExtractionResult):
                if not res.original_request:
                    res.original_request = user_request
                return res
            if isinstance(res, dict):
                res.setdefault("original_request", user_request)
                return ExtractionResult.model_validate(res)
        except Exception:
            pass

    # Fallback: raw text -> JSON -> Pydantic validation
    raw_msg = llm.invoke(messages)
    raw_text = getattr(raw_msg, "content", str(raw_msg))
    parsed = _extract_json_from_text(raw_text)
    parsed.setdefault("original_request", user_request)
    return ExtractionResult.model_validate(parsed)


def _format_deterministic_explanation(
    sim_results: List[dict],
    comparison_result: Optional[dict] = None,
) -> str:
    """Plain-language fallback summary built strictly from verified Python numbers."""
    if not sim_results:
        return "No scenario results were produced."

    lines = [
        "Note: This is a hypothetical scenario analysis, not an actual forecast or prediction.",
        "",
    ]
    for r in sim_results:
        name = r.get("scenario_name", "Scenario")
        if not r.get("success"):
            errs = "; ".join(r.get("validation_errors", ["Unknown error"]))
            lines.append(f"- {name}: Could not be computed ({errs}).")
            continue
        state = r.get("state")
        district = r.get("district")
        crop = r.get("crop")
        year = r.get("year")
        b_prod = r.get("baseline_production", 0.0)
        n_prod = r.get("new_production", 0.0)
        diff = r.get("production_change", 0.0)
        pct = r.get("production_change_percent", 0.0)
        lines.append(
            f"- {name} ({crop} in {district}, {state}, {year}): "
            f"Under {r.get('area_change_percent', 0.0):+.1f}% area change and "
            f"{r.get('yield_change_percent', 0.0):+.1f}% yield change, "
            f"hypothetical production moves from {b_prod:,.2f} to {n_prod:,.2f} "
            f"thousand tonnes ({diff:+,.2f} thousand tonnes, {pct:+.2f}%)."
        )

    if comparison_result and comparison_result.get("best_production_scenario"):
        best = comparison_result["best_production_scenario"]
        lines.append(
            f"\nHighest hypothetical production: {best.get('scenario_name')} "
            f"({best.get('new_production', 0.0):,.2f} thousand tonnes)."
        )
    return "\n".join(lines)


def build_graph(
    df: Optional[pd.DataFrame] = None,
    llm_factory: Optional[Callable[[], Any]] = None,
):
    """
    Build and compile the LangGraph StateGraph for the Agricultural
    Production Scenario Coordinator.

    Parameters
    ----------
    df          : optional pre-loaded cleaned DataFrame (loads default if None).
    llm_factory : optional callable returning a LangChain chat model (defaults to get_llm).
    """
    _df = df if df is not None else load_cleaned_data()
    _get_llm = llm_factory if llm_factory is not None else get_llm
    _engine = AgriculturalScenarioEngine(_df)

    # ── Node 1: Understand & Extract ──────────────────────────────────────────
    def understand_and_extract(state: AgentState) -> dict:
        user_request = state.get("user_request", "").strip()
        if not user_request:
            return {
                "intent": "clarification_needed",
                "needs_clarification": True,
                "clarification_question": "Please enter an agricultural scenario or historical question to analyse.",
                "scenarios": [],
                "workflow_stage": "understand_and_extract",
            }

        try:
            llm = _get_llm()
            extraction = _invoke_extraction(llm, user_request)
            ext_dict = extraction.model_dump()
            return {
                "intent": extraction.intent,
                "extraction_result": ext_dict,
                "scenarios": ext_dict.get("scenarios", []),
                "needs_clarification": extraction.needs_clarification,
                "clarification_question": extraction.clarification_question,
                "error": None,
                "workflow_stage": "understand_and_extract",
            }
        except Exception as exc:
            return {
                "intent": "invalid",
                "needs_clarification": False,
                "error": f"LLM extraction failed: {exc}",
                "scenarios": [],
                "workflow_stage": "understand_and_extract",
            }

    # ── Node 2: Validate Extraction ───────────────────────────────────────────
    def validate_extraction(state: AgentState) -> dict:
        if state.get("error"):
            return {"workflow_stage": "validate_extraction"}

        intent = state.get("intent", "invalid")
        scenarios = state.get("scenarios") or []
        ext = state.get("extraction_result") or {}

        if intent in ("clarification_needed", "invalid"):
            question = state.get("clarification_question") or (
                "Could you please specify the crop, state, year (2010-2017), "
                "and any area or yield percentage changes you want to analyse?"
            )
            return {
                "needs_clarification": intent == "clarification_needed",
                "clarification_question": question if intent == "clarification_needed" else None,
                "workflow_stage": "validate_extraction",
            }

        if intent in ("scenario_simulation", "comparison"):
            if not scenarios:
                return {
                    "intent": "clarification_needed",
                    "needs_clarification": True,
                    "clarification_question": (
                        "I understood this as a scenario request, but could not identify "
                        "the crop, state, and year. Please specify e.g. 'Rice in Karnataka for 2017'."
                    ),
                    "workflow_stage": "validate_extraction",
                }

            missing: List[str] = []
            for sc in scenarios:
                if not sc.get("crop"):
                    missing.append("crop")
                if not sc.get("state"):
                    missing.append("state")
                if not sc.get("year"):
                    missing.append("year (between 2010 and 2017)")

            if missing:
                unique_missing = list(dict.fromkeys(missing))
                question = ext.get("clarification_question")
                if not question:
                    try:
                        llm = _get_llm()
                        msg = llm.invoke([
                            SystemMessage(content=CLARIFICATION_SYSTEM_PROMPT),
                            HumanMessage(
                                content=(
                                    f"User request: {state.get('user_request')}\n"
                                    f"Missing fields: {', '.join(unique_missing)}"
                                )
                            ),
                        ])
                        question = getattr(msg, "content", str(msg)).strip()
                    except Exception:
                        question = (
                            f"To run this scenario using historical records, could you please "
                            f"specify the missing {', '.join(unique_missing)}?"
                        )
                return {
                    "intent": "clarification_needed",
                    "needs_clarification": True,
                    "clarification_question": question,
                    "workflow_stage": "validate_extraction",
                }

        return {
            "needs_clarification": False,
            "workflow_stage": "validate_extraction",
        }

    # ── Conditional Router ────────────────────────────────────────────────────
    def route_after_validation(state: AgentState) -> str:
        if state.get("error"):
            return "handle_clarification_or_invalid"
        if state.get("needs_clarification") or state.get("intent") in ("clarification_needed", "invalid"):
            return "handle_clarification_or_invalid"
        if state.get("intent") == "historical_analysis":
            return "retrieve_historical_data"
        return "retrieve_baseline"

    # ── Node 3a: Clarification / Invalid Handler ──────────────────────────────
    def handle_clarification_or_invalid(state: AgentState) -> dict:
        intent = state.get("intent", "invalid")
        question = state.get("clarification_question")
        error = state.get("error")
        message = error or question or (
            "I can help with agricultural historical analysis and what-if crop "
            "production scenarios (2010-2017). Please specify a crop, state, and year."
        )
        final_output = {
            "status": "clarification_needed" if state.get("needs_clarification") else "error",
            "intent": intent,
            "user_request": state.get("user_request", ""),
            "needs_clarification": bool(state.get("needs_clarification")),
            "clarification_question": question,
            "extracted_assumptions": state.get("scenarios", []),
            "baseline_results": [],
            "simulation_results": [],
            "validation_results": [],
            "comparison_result": None,
            "historical_data": None,
            "explanation": message,
            "error": error,
            "is_forecast": False,
        }
        return {
            "explanation": message,
            "final_output": final_output,
            "messages": [AIMessage(content=message)],
            "workflow_stage": "handle_clarification_or_invalid",
        }

    # ── Node 3b: Historical Analysis Retrieval ────────────────────────────────
    def retrieve_historical_data(state: AgentState) -> dict:
        scenarios = state.get("scenarios") or []
        first = scenarios[0] if scenarios else {}
        st = first.get("state")
        cr = first.get("crop")
        yr = first.get("year")
        dist = first.get("district")

        filtered = filter_data(
            _df,
            states=[st] if st else None,
            districts=[dist] if dist else None,
            crops=[cr] if cr else None,
            years=[int(yr)] if yr else None,
        )

        if filtered.empty:
            hist_data = {
                "found": False,
                "filters": {"state": st, "district": dist, "crop": cr, "year": yr},
                "record_count": 0,
                "year_wise": [],
                "state_wise": [],
                "crop_wise": [],
                "warning": "No historical records matched the specified filters.",
            }
        else:
            hist_data = {
                "found": True,
                "filters": {"state": st, "district": dist, "crop": cr, "year": yr},
                "record_count": int(len(filtered)),
                "year_wise": analyse_year_wise(filtered).to_dict(orient="records"),
                "state_wise": analyse_state_wise(filtered).head(10).to_dict(orient="records"),
                "crop_wise": analyse_crop_wise(filtered).head(10).to_dict(orient="records"),
                "warning": None,
            }

        return {
            "baseline_results": [hist_data],
            "workflow_stage": "retrieve_historical_data",
        }

    # ── Node 4b: Historical Explanation ───────────────────────────────────────
    def generate_historical_explanation(state: AgentState) -> dict:
        hist_list = state.get("baseline_results") or [{}]
        hist_data = hist_list[0]
        user_query = state.get("user_request", "")

        if not hist_data.get("found"):
            explanation = hist_data.get("warning") or "No matching historical records were found."
            return {
                "explanation": explanation,
                "workflow_stage": "generate_historical_explanation",
            }

        try:
            llm = _get_llm()
            msg = llm.invoke([
                SystemMessage(content=HISTORICAL_ANALYSIS_SYSTEM_PROMPT),
                HumanMessage(content=format_historical_prompt(hist_data, user_query)),
            ])
            explanation = getattr(msg, "content", str(msg)).strip()
        except Exception as exc:
            yw = hist_data.get("year_wise", [])
            explanation = (
                f"Found {hist_data.get('record_count', 0)} historical records across "
                f"{len(yw)} year(s). (Ollama explanation unavailable: {exc})"
            )

        return {
            "explanation": explanation,
            "workflow_stage": "generate_historical_explanation",
        }

    # ── Node 3c: Retrieve Baseline (Member 2) ─────────────────────────────────
    def retrieve_baseline(state: AgentState) -> dict:
        scenarios = state.get("scenarios") or []
        baselines: List[dict] = []
        for sc in scenarios:
            st = sc.get("state") or ""
            dist = sc.get("district") or ""
            cr = sc.get("crop") or ""
            yr = int(sc.get("year") or 0)
            if isinstance(dist, str) and dist.strip():
                b = get_historical_baseline(_df, state=st, district=dist.strip(), crop=cr, year=yr)
            else:
                b = get_aggregated_baseline(_df, crop=cr, year=yr, state=st or None, district=None)
            baselines.append(b)
        return {
            "baseline_results": baselines,
            "workflow_stage": "retrieve_baseline",
        }

    # ── Node 4c: Execute Deterministic Simulation (Member 3) ──────────────────
    def execute_simulation(state: AgentState) -> dict:
        scenarios = state.get("scenarios") or []
        sim_results = _engine.calculate_multiple_scenarios(scenarios)
        return {
            "simulation_results": sim_results,
            "workflow_stage": "execute_simulation",
        }

    # ── Node 5c: Validate Results ─────────────────────────────────────────────
    def validate_results(state: AgentState) -> dict:
        sim_results = state.get("simulation_results") or []
        val_results = [validate_output(res) for res in sim_results]
        return {
            "validation_results": val_results,
            "workflow_stage": "validate_results",
        }

    # ── Node 6c: Compare Scenarios (if >= 2) ──────────────────────────────────
    def compare_scenarios(state: AgentState) -> dict:
        sim_results = state.get("simulation_results") or []
        if len(sim_results) < 2:
            return {
                "comparison_result": None,
                "workflow_stage": "compare_scenarios",
            }
        successful = [r for r in sim_results if r.get("success") and r.get("baseline")]
        baseline = successful[0]["baseline"] if successful else {}
        comp = ScenarioComparison(baseline_dict=baseline, scenarios=sim_results).compare()
        return {
            "comparison_result": comp,
            "workflow_stage": "compare_scenarios",
        }

    # ── Node 7c: Generate Explanation (Ollama) ────────────────────────────────
    def generate_explanation(state: AgentState) -> dict:
        sim_results = state.get("simulation_results") or []
        assumptions = state.get("scenarios") or []
        comparison_result = state.get("comparison_result")
        successful = [r for r in sim_results if r.get("success") and r.get("baseline")]
        baseline = successful[0]["baseline"] if successful else {}

        try:
            llm = _get_llm()
            msg = llm.invoke([
                SystemMessage(content=EXPLANATION_SYSTEM_PROMPT),
                HumanMessage(
                    content=format_explanation_prompt(
                        baseline=baseline,
                        results=sim_results,
                        assumptions=assumptions,
                    )
                ),
            ])
            explanation = getattr(msg, "content", str(msg)).strip()
        except Exception:
            explanation = _format_deterministic_explanation(sim_results, comparison_result)

        return {
            "explanation": explanation,
            "workflow_stage": "generate_explanation",
        }

    # ── Node 8: Assemble Final Output ─────────────────────────────────────────
    def assemble_output(state: AgentState) -> dict:
        intent = state.get("intent", "scenario_simulation")
        sim_results = state.get("simulation_results") or []
        val_results = state.get("validation_results") or []
        explanation = state.get("explanation") or ""

        if intent == "historical_analysis":
            hist_list = state.get("baseline_results") or []
            hist_data = hist_list[0] if hist_list else None
            status = "success" if (hist_data and hist_data.get("found")) else "not_found"
        else:
            hist_data = None
            any_success = any(r.get("success") for r in sim_results)
            status = "success" if any_success else "failed"

        final_output = {
            "status": status,
            "intent": intent,
            "user_request": state.get("user_request", ""),
            "needs_clarification": False,
            "clarification_question": None,
            "extracted_assumptions": state.get("scenarios", []),
            "baseline_results": state.get("baseline_results", []),
            "simulation_results": sim_results,
            "validation_results": val_results,
            "comparison_result": state.get("comparison_result"),
            "historical_data": hist_data,
            "explanation": explanation,
            "error": state.get("error"),
            "is_forecast": False,
        }
        return {
            "final_output": final_output,
            "messages": [AIMessage(content=explanation)],
            "workflow_stage": "assemble_output",
        }

    # ── Wire Graph ────────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    graph.add_node("understand_and_extract", understand_and_extract)
    graph.add_node("validate_extraction", validate_extraction)
    graph.add_node("handle_clarification_or_invalid", handle_clarification_or_invalid)
    graph.add_node("retrieve_historical_data", retrieve_historical_data)
    graph.add_node("generate_historical_explanation", generate_historical_explanation)
    graph.add_node("retrieve_baseline", retrieve_baseline)
    graph.add_node("execute_simulation", execute_simulation)
    graph.add_node("validate_results", validate_results)
    graph.add_node("compare_scenarios", compare_scenarios)
    graph.add_node("generate_explanation", generate_explanation)
    graph.add_node("assemble_output", assemble_output)

    graph.set_entry_point("understand_and_extract")
    graph.add_edge("understand_and_extract", "validate_extraction")
    graph.add_conditional_edges(
        "validate_extraction",
        route_after_validation,
        {
            "handle_clarification_or_invalid": "handle_clarification_or_invalid",
            "retrieve_historical_data": "retrieve_historical_data",
            "retrieve_baseline": "retrieve_baseline",
        },
    )

    graph.add_edge("handle_clarification_or_invalid", END)

    graph.add_edge("retrieve_historical_data", "generate_historical_explanation")
    graph.add_edge("generate_historical_explanation", "assemble_output")

    graph.add_edge("retrieve_baseline", "execute_simulation")
    graph.add_edge("execute_simulation", "validate_results")
    graph.add_edge("validate_results", "compare_scenarios")
    graph.add_edge("compare_scenarios", "generate_explanation")
    graph.add_edge("generate_explanation", "assemble_output")

    graph.add_edge("assemble_output", END)

    return graph.compile()
