"""
PS 137 - Agricultural Production Scenario Coordinator
======================================================
src/agents/scenario_graph.py
LangGraph workflow orchestrating:
  1. Natural-language understanding & structured extraction (Gemini REST API)
  2. Extraction & required-field validation
  3. Conditional routing:
       - clarification_needed / invalid -> return clarification or error
       - historical_analysis -> Member 2 lookup -> Gemini explanation
       - scenario_simulation / comparison -> Member 2 baseline -> Member 3 engine
         -> output validation -> (optional comparison) -> Gemini explanation
  4. Final structured output assembly
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

import src.agents.llm_config as llm_config
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


def _rule_based_fallback_extraction(
    user_request: str,
    known_states: Optional[List[str]] = None,
    known_crops: Optional[List[str]] = None,
) -> dict:
    """Deterministic regex fallback when Gemini is unavailable or rate-limited."""
    text = user_request.strip()
    lower = text.lower()

    found_state = None
    for st_name in sorted(known_states or [], key=len, reverse=True):
        if re.search(rf"\b{re.escape(st_name.lower())}\b", lower):
            found_state = st_name
            break

    found_crop = None
    for cr_name in sorted(known_crops or [], key=len, reverse=True):
        if re.search(rf"\b{re.escape(cr_name.lower())}\b", lower):
            found_crop = cr_name
            break

    year_match = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    found_year = int(year_match.group(1)) if year_match else None

    def _extract_pct(keyword: str) -> float:
        # e.g., "area increases by 10%", "yield decreases by 5%", "area +10%", "yield -5%"
        m1 = re.search(
            rf"{keyword}[^0-9+\-]*(increase[sd]?|expand[sd]?|rise[sd]?|grow[sn]?|up|\+)?\s*(decrease[sd]?|drop[sd]?|fall[sn]?|reduce[sd]?|decline[sd]?|down|-)?\s*(?:by\s+)?([+-]?\d+(?:\.\d+)?)\s*%",
            lower,
        )
        if m1:
            val = float(m1.group(3))
            if m1.group(2) and val > 0:
                val = -val
            return val
        return 0.0

    area_pct = _extract_pct("area")
    yield_pct = _extract_pct("yield")

    if found_state and found_crop and (abs(area_pct) > 1e-9 or abs(yield_pct) > 1e-9):
        return {
            "intent": "scenario_simulation",
            "scenarios": [
                {
                    "state": found_state,
                    "district": None,
                    "crop": found_crop,
                    "year": found_year,
                    "area_change_percent": area_pct,
                    "yield_change_percent": yield_pct,
                    "scenario_name": f"{found_crop} in {found_state}",
                    "scenario_description": "Extracted via deterministic fallback",
                }
            ],
            "needs_clarification": False,
            "clarification_question": None,
            "missing_fields": [],
            "original_request": user_request,
            "confidence": 0.7,
            "historical_query": None,
        }

    return {
        "intent": "clarification_needed",
        "scenarios": [],
        "needs_clarification": True,
        "clarification_question": (
            "I could not reach the AI service. "
            "Please use the Manual Scenario mode in the sidebar."
        ),
        "missing_fields": [],
        "original_request": user_request,
        "confidence": 0.0,
        "historical_query": None,
    }


def _invoke_extraction_gemini(
    user_request: str,
    known_states: Optional[List[str]] = None,
    known_crops: Optional[List[str]] = None,
) -> ExtractionResult:
    """
    Use Gemini REST API to extract structured ExtractionResult.
    Falls back to deterministic rule-based extraction if Gemini is unavailable.
    """
    full_prompt = (
        EXTRACTION_SYSTEM_PROMPT
        + "\n\n"
        + format_extraction_prompt(user_request)
    )

    try:
        raw_text = llm_config.call_gemini(
            prompt=full_prompt,
            temperature=0.0,
            max_output_tokens=1024,
        )
        parsed = _extract_json_from_text(raw_text)
    except Exception:
        parsed = _rule_based_fallback_extraction(
            user_request, known_states=known_states, known_crops=known_crops
        )

    parsed.setdefault("original_request", user_request)

    # Robustness guard: if percentage changes are present, ensure correct intent
    scenarios = parsed.get("scenarios") or []
    has_changes = any(
        abs(float(s.get("area_change_percent") or 0.0)) > 1e-9
        or abs(float(s.get("yield_change_percent") or 0.0)) > 1e-9
        for s in scenarios if isinstance(s, dict)
    )
    if has_changes and parsed.get("intent") == "historical_analysis":
        parsed["intent"] = "comparison" if len(scenarios) > 1 else "scenario_simulation"

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
        state = r.get("state", "")
        crop = r.get("crop", "")
        year = r.get("year", "")
        ref_year = r.get("reference_year", year)
        target_year = r.get("target_year", year)
        b_prod = r.get("baseline_production")
        n_prod = r.get("new_production")
        pct = r.get("production_change_percent")
        area_chg = r.get("area_change_percent", 0.0)
        yield_chg = r.get("yield_change_percent", 0.0)
        b_area = r.get("baseline_area")
        n_area = r.get("new_area")
        b_yield = r.get("baseline_yield")
        n_yield = r.get("new_yield")

        lines.append(f"**{name}** — {crop} in {state}")
        if ref_year and target_year and ref_year != target_year:
            lines.append(f"  Reference baseline year: {ref_year} (target year: {target_year})")
        lines.append(f"  Baseline production: {b_prod:.2f} thousand tonnes" if b_prod is not None else "  Baseline: N/A")
        lines.append(f"  Scenario production: {n_prod:.2f} thousand tonnes" if n_prod is not None else "  Scenario: N/A")
        if pct is not None:
            direction = "increase" if pct >= 0 else "decrease"
            lines.append(f"  Change: {pct:+.2f}% ({direction})")
        lines.append(f"  Assumptions: Area {area_chg:+.1f}%, Yield {yield_chg:+.1f}%")
        lines.append("")

    if comparison_result:
        lines.append("**Scenario Comparison Summary**")
        scenarios_cmp = comparison_result.get("scenarios", [])
        if scenarios_cmp:
            prods = [(s.get("scenario_name", "?"), s.get("new_production")) for s in scenarios_cmp if s.get("new_production") is not None]
            if prods:
                best = max(prods, key=lambda x: x[1])
                lines.append(f"  Highest production: {best[0]} at {best[1]:.2f} thousand tonnes")

    return "\n".join(lines)


def build_graph(
    df: Optional[pd.DataFrame] = None,
    llm_factory: Optional[Callable[[], Any]] = None,
) -> Any:
    """
    Build and compile the LangGraph state machine for the agricultural scenario agent.

    Parameters
    ----------
    df          : Pre-loaded cleaned DataFrame (loaded from CSV if None).
    llm_factory : Unused (kept for API compatibility). Gemini REST API is used directly.

    Returns
    -------
    CompiledGraph ready to .invoke(initial_state).
    """
    # Load data once
    if df is None:
        df = load_cleaned_data()

    engine = AgriculturalScenarioEngine(df)
    known_states = sorted(df["state_clean"].dropna().unique().tolist()) if "state_clean" in df.columns else []
    known_crops = sorted(df["crop_clean"].dropna().unique().tolist()) if "crop_clean" in df.columns else []

    # ── Node 1: Understand and Extract ────────────────────────────────────────
    def understand_and_extract(state: AgentState) -> dict:
        user_request = state.get("user_request", "")
        try:
            extraction = _invoke_extraction_gemini(
                user_request, known_states=known_states, known_crops=known_crops
            )
        except Exception as exc:
            return {
                "intent": "invalid",
                "extraction_result": None,
                "scenarios": [],
                "needs_clarification": False,
                "clarification_question": None,
                "error": f"Extraction failed: {exc}",
                "workflow_stage": "understand_and_extract",
            }
        return {
            "intent": extraction.intent,
            "extraction_result": extraction.model_dump(),
            "scenarios": [s.model_dump() for s in extraction.scenarios],
            "needs_clarification": extraction.needs_clarification,
            "clarification_question": extraction.clarification_question,
            "workflow_stage": "understand_and_extract",
        }

    # ── Node 2: Validate Extraction ────────────────────────────────────────────
    def validate_extraction(state: AgentState) -> dict:
        intent = state.get("intent", "invalid")
        scenarios = state.get("scenarios", [])
        needs_clarification = state.get("needs_clarification", False)

        # If already flagged for clarification, pass through
        if needs_clarification or intent in ("clarification_needed", "invalid"):
            return {"workflow_stage": "validate_extraction"}

        # For simulation/comparison: check required fields
        if intent in ("scenario_simulation", "comparison"):
            missing_global = []
            for s in scenarios:
                for field in ("state", "crop"):
                    if not s.get(field):
                        missing_global.append(field)
            if missing_global:
                q = f"To run a scenario, I need the following: {', '.join(set(missing_global))}. Please specify."
                return {
                    "intent": "clarification_needed",
                    "needs_clarification": True,
                    "clarification_question": q,
                    "workflow_stage": "validate_extraction",
                }

        return {"workflow_stage": "validate_extraction"}

    # ── Routing ────────────────────────────────────────────────────────────────
    def route_after_validation(state: AgentState) -> str:
        intent = state.get("intent", "invalid")
        needs_clarification = state.get("needs_clarification", False)
        if needs_clarification or intent in ("clarification_needed", "invalid"):
            return "handle_clarification_or_invalid"
        if intent == "historical_analysis":
            return "retrieve_historical_data"
        return "retrieve_baseline"

    # ── Node 3: Handle Clarification / Invalid ─────────────────────────────────
    def handle_clarification_or_invalid(state: AgentState) -> dict:
        intent = state.get("intent", "invalid")
        q = state.get("clarification_question") or (
            "Your request could not be understood. Please describe the crop, state, "
            "and the percentage change in area or yield you want to explore."
            if intent == "invalid"
            else "Could you provide more details about your farming scenario?"
        )
        final_output = {
            "status": "clarification_needed" if intent == "clarification_needed" else "invalid",
            "intent": intent,
            "user_request": state.get("user_request", ""),
            "needs_clarification": True,
            "clarification_question": q,
            "extracted_assumptions": state.get("scenarios", []),
            "baseline_results": [],
            "simulation_results": [],
            "validation_results": [],
            "comparison_result": None,
            "historical_data": None,
            "explanation": q,
            "error": state.get("error"),
            "is_forecast": False,
        }
        return {
            "final_output": final_output,
            "messages": [AIMessage(content=q)],
            "workflow_stage": "handle_clarification_or_invalid",
        }

    # ── Node 4: Retrieve Historical Data ──────────────────────────────────────
    def retrieve_historical_data(state: AgentState) -> dict:
        extraction = state.get("extraction_result") or {}
        scenarios = state.get("scenarios", [])
        s = scenarios[0] if scenarios else {}

        state_name = s.get("state")
        crop_name = s.get("crop")
        year = s.get("year")

        filtered = filter_data(
            df,
            states=[state_name] if state_name else None,
            crops=[crop_name] if crop_name else None,
            years=[year] if year else None,
        )
        if filtered.empty:
            hist_data = {"found": False, "record_count": 0}
        else:
            hist_data = {
                "found": True,
                "record_count": int(len(filtered)),
                "year_wise": analyse_year_wise(filtered).head(10).to_dict(orient="records"),
                "state_wise": analyse_state_wise(filtered).head(10).to_dict(orient="records"),
                "crop_wise": analyse_crop_wise(filtered).head(10).to_dict(orient="records"),
            }
        return {
            "baseline_results": [hist_data],
            "workflow_stage": "retrieve_historical_data",
        }

    # ── Node 5: Generate Historical Explanation ────────────────────────────────
    def generate_historical_explanation(state: AgentState) -> dict:
        hist_list = state.get("baseline_results") or []
        hist_data = hist_list[0] if hist_list else {}
        user_request = state.get("user_request", "")

        if not hist_data.get("found"):
            explanation = "No historical data was found for your query. Please check the state, crop, and year."
        else:
            try:
                explanation = llm_config.call_gemini(
                    prompt=format_historical_prompt(hist_data, user_request),
                    system_instruction=HISTORICAL_ANALYSIS_SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=1024,
                )
            except Exception:
                # Deterministic fallback
                recs = hist_data.get("year_wise", [])
                if recs:
                    latest = recs[-1]
                    explanation = (
                        f"Historical data found ({hist_data.get('record_count',0)} records). "
                        f"Most recent year: {latest.get('year', 'N/A')}, "
                        f"Production: {latest.get('production_1000_t', 'N/A'):.2f} thousand tonnes."
                    )
                else:
                    explanation = "Historical data was found but could not be summarized."

        return {
            "explanation": explanation,
            "workflow_stage": "generate_historical_explanation",
        }

    # ── Node 6: Retrieve Baseline for Scenario ────────────────────────────────
    def retrieve_baseline(state: AgentState) -> dict:
        scenarios = state.get("scenarios", [])
        baseline_results = []

        for s in scenarios:
            state_name = s.get("state") or ""
            crop_name = s.get("crop") or ""
            year = s.get("year")
            district = (s.get("district") or "").strip()

            # Determine reference year: use the year from the scenario if within dataset range,
            # otherwise use the nearest available year (max = 2017).
            dataset_years = sorted(df["year"].unique().tolist()) if not df.empty else list(range(2010, 2018))
            max_year = max(dataset_years)
            min_year = min(dataset_years)

            if year is not None and min_year <= year <= max_year:
                reference_year = year
            else:
                reference_year = max_year  # default to latest

            # Try district-level first if district is specified, otherwise state aggregate
            if district:
                baseline = get_historical_baseline(
                    df,
                    state=state_name,
                    crop=crop_name,
                    year=reference_year,
                    district=district,
                )
            else:
                baseline = {"found": False}

            if not baseline.get("found"):
                baseline = get_aggregated_baseline(
                    df, state=state_name, crop=crop_name, year=reference_year
                )

            # Annotate with target vs reference year info
            baseline["reference_year"] = reference_year
            baseline["target_year"] = year if year is not None else reference_year
            baseline["using_reference_year_note"] = (
                f"Using {reference_year} as the reference baseline "
                f"(target year {year} is {'beyond' if year and year > max_year else 'within'} the available dataset range 2010–2017)."
                if year and year != reference_year
                else f"Using {reference_year} data as baseline."
            )

            baseline_results.append(baseline)

        return {
            "baseline_results": baseline_results,
            "workflow_stage": "retrieve_baseline",
        }

    # ── Node 7: Execute Simulation ────────────────────────────────────────────
    def execute_simulation(state: AgentState) -> dict:
        scenarios = state.get("scenarios", [])
        baseline_results = state.get("baseline_results", [])
        sim_results = []

        for i, s in enumerate(scenarios):
            state_name = s.get("state") or ""
            crop_name = s.get("crop") or ""
            district = (s.get("district") or "").strip()
            area_chg = float(s.get("area_change_percent") or 0.0)
            yield_chg = float(s.get("yield_change_percent") or 0.0)
            scenario_name = s.get("scenario_name", f"Scenario {i+1}")

            baseline = baseline_results[i] if i < len(baseline_results) else {}
            reference_year = baseline.get("reference_year")
            target_year = baseline.get("target_year", reference_year)

            res = engine.calculate_scenario(
                state=state_name,
                district=district,
                crop=crop_name,
                year=reference_year,
                area_change_percent=area_chg,
                yield_change_percent=yield_chg,
                scenario_name=scenario_name,
            )
            # Inject any-year metadata
            res["reference_year"] = reference_year
            res["target_year"] = target_year
            res["using_reference_year_note"] = baseline.get("using_reference_year_note", "")
            res["is_forecast"] = False

            sim_results.append(res)

        return {
            "simulation_results": sim_results,
            "workflow_stage": "execute_simulation",
        }

    # ── Node 8: Validate Results ──────────────────────────────────────────────
    def validate_results(state: AgentState) -> dict:
        sim_results = state.get("simulation_results", [])
        val_results = [validate_output(r) for r in sim_results]
        return {
            "validation_results": val_results,
            "workflow_stage": "validate_results",
        }

    # ── Node 9: Compare Scenarios ─────────────────────────────────────────────
    def compare_scenarios(state: AgentState) -> dict:
        sim_results = state.get("simulation_results", [])
        intent = state.get("intent", "scenario_simulation")
        comparison_result = None

        if intent == "comparison" and len(sim_results) >= 2:
            successful = [r for r in sim_results if r.get("success")]
            if len(successful) >= 2:
                try:
                    cmp_obj = ScenarioComparison(successful)
                    comparison_result = cmp_obj.compare()
                except Exception:
                    comparison_result = None

        return {
            "comparison_result": comparison_result,
            "workflow_stage": "compare_scenarios",
        }

    # ── Node 10: Generate Explanation ─────────────────────────────────────────
    def generate_explanation(state: AgentState) -> dict:
        sim_results = state.get("simulation_results", [])
        scenarios = state.get("scenarios", [])
        baseline_results = state.get("baseline_results", [])
        comparison_result = state.get("comparison_result")

        baseline = baseline_results[0] if baseline_results else {}
        assumptions = scenarios

        if not llm_config.is_llm_available():
            explanation = _format_deterministic_explanation(sim_results, comparison_result)
        else:
            try:
                explanation = llm_config.call_gemini(
                    prompt=format_explanation_prompt(
                        baseline=baseline,
                        results=sim_results,
                        assumptions=assumptions,
                    ),
                    system_instruction=EXPLANATION_SYSTEM_PROMPT,
                    temperature=0.3,
                    max_output_tokens=1024,
                )
            except Exception:
                explanation = _format_deterministic_explanation(sim_results, comparison_result)

        return {
            "explanation": explanation,
            "workflow_stage": "generate_explanation",
        }

    # ── Node 11: Assemble Final Output ─────────────────────────────────────────
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

    # ── Wire Graph ─────────────────────────────────────────────────────────────
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
