"""
PS 137 - Agricultural Production Scenario Coordinator
======================================================
src/agents/tools.py
LangChain @tool wrappers around Member 2 (historical_analysis) and
Member 3 (simulation) modules.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
from langchain_core.tools import tool

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

_CACHED_DF: Optional[pd.DataFrame] = None


def get_dataset() -> pd.DataFrame:
    """Load and cache the cleaned agricultural dataset."""
    global _CACHED_DF
    if _CACHED_DF is None:
        _CACHED_DF = load_cleaned_data()
    return _CACHED_DF


def set_dataset(df: pd.DataFrame) -> None:
    """Override cached dataset (useful for testing)."""
    global _CACHED_DF
    _CACHED_DF = df


@tool
def lookup_historical_baseline(
    state: str,
    district: str,
    crop: str,
    year: int,
) -> dict:
    """
    Retrieve the historical baseline record for a specific
    (state, district, crop, year) from the cleaned dataset.
    """
    df = get_dataset()
    return get_historical_baseline(
        df=df,
        state=state,
        district=district,
        crop=crop,
        year=year,
    )


@tool
def lookup_aggregated_baseline(
    crop: str,
    year: int,
    state: Optional[str] = None,
    district: Optional[str] = None,
) -> dict:
    """
    Retrieve aggregated historical statistics (total area, total production,
    weighted yield) for a crop and year, optionally filtered by state or district.
    """
    df = get_dataset()
    return get_aggregated_baseline(
        df=df,
        crop=crop,
        year=year,
        state=state,
        district=district,
    )


@tool
def run_scenario_simulation(
    state: str,
    crop: str,
    year: int,
    area_change_percent: float = 0.0,
    yield_change_percent: float = 0.0,
    district: str = "",
    scenario_name: str = "Scenario",
) -> dict:
    """
    Run a deterministic hypothetical scenario simulation using Member 3's
    AgriculturalScenarioEngine. Never invents data or uses LLM math.
    """
    df = get_dataset()
    engine = AgriculturalScenarioEngine(df)
    return engine.calculate_scenario(
        state=state,
        district=district,
        crop=crop,
        year=year,
        area_change_percent=area_change_percent,
        yield_change_percent=yield_change_percent,
        scenario_name=scenario_name,
    )


@tool
def compare_multiple_scenarios(scenarios: List[dict]) -> dict:
    """
    Run and compare multiple scenario definitions against their historical baseline.
    Each item in `scenarios` should contain state, crop, year, area_change_percent,
    yield_change_percent, and optionally district and scenario_name.
    """
    df = get_dataset()
    engine = AgriculturalScenarioEngine(df)
    results = engine.calculate_multiple_scenarios(scenarios)
    successful = [r for r in results if r.get("success") and r.get("baseline")]
    baseline = successful[0]["baseline"] if successful else {}
    comparator = ScenarioComparison(baseline_dict=baseline, scenarios=results)
    return comparator.compare()


@tool
def get_historical_trends(
    state: Optional[str] = None,
    crop: Optional[str] = None,
    year: Optional[int] = None,
) -> dict:
    """
    Retrieve historical year-wise, state-wise, and crop-wise trends
    filtered optionally by state, crop, and/or year.
    """
    df = get_dataset()
    filtered = filter_data(
        df,
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
            "warning": "No matching historical records found for the given filters.",
        }

    year_df = analyse_year_wise(filtered)
    state_df = analyse_state_wise(filtered).head(10)
    crop_df = analyse_crop_wise(filtered).head(10)

    return {
        "found": True,
        "record_count": int(len(filtered)),
        "year_wise": year_df.to_dict(orient="records"),
        "state_wise": state_df.to_dict(orient="records"),
        "crop_wise": crop_df.to_dict(orient="records"),
        "warning": None,
    }


ALL_TOOLS = [
    lookup_historical_baseline,
    lookup_aggregated_baseline,
    run_scenario_simulation,
    compare_multiple_scenarios,
    get_historical_trends,
]
