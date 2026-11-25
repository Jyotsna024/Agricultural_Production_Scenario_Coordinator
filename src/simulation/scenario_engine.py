"""
PS 137 – Agricultural Production Scenario Coordinator
======================================================
src/simulation/scenario_engine.py
Member 3 – Scenario Simulation Engine

PURPOSE
-------
Applies user-specified percentage changes to a historical baseline
(area and/or yield) and computes the resulting production figures.

UNITS (inherited from Member 1 / Member 2)
  area_1000_ha      – Thousand hectares
  production_1000_t – Thousand tonnes
  yield_kg_per_ha   – Kilograms per hectare

FORMULAS
  new_area              = original_area * (1 + area_change_percent / 100)
  new_yield             = original_yield * (1 + yield_change_percent / 100)
  new_production        = new_area * new_yield / 1000   # 1000-ha * kg/ha / 1000 → 1000-t
  production_change     = new_production - baseline_production
  production_change_pct = (production_change / baseline_production) * 100

RULES
  - Never fabricate baseline data. If the historical record is not found,
    success=False is returned with an explanation.
  - All numeric outputs are floats; None when unavailable.
  - Validation is embedded via scenario_validator.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Allow running from project root with sys.path.insert(0, '.')
_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR  = _THIS_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_SRC_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR.parent))

from src.analysis.historical_analysis import (
    get_historical_baseline,
    get_aggregated_baseline,
)
from src.simulation.scenario_validator import (
    validate_scenario_inputs,
    validate_scenario_result,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _determine_scenario_type(area_pct: float, yield_pct: float) -> str:
    """Classify the scenario by which levers are active."""
    has_area  = abs(area_pct)  > 1e-9
    has_yield = abs(yield_pct) > 1e-9
    if has_area and has_yield:
        return "area_and_yield"
    if has_area:
        return "area_only"
    if has_yield:
        return "yield_only"
    return "baseline"   # no change applied


def _build_units_dict() -> dict:
    return {
        "area"      : "Thousand hectares (1000 ha)",
        "production": "Thousand tonnes (1000 t)",
        "yield"     : "Kilograms per hectare (kg/ha)",
    }


def _apply_formulas(
    baseline_area: float,
    baseline_yield: float,
    baseline_production: float,
    area_change_percent: float,
    yield_change_percent: float,
) -> tuple[float, float, float, float, float]:
    """
    Apply the scenario formulas and return
    (new_area, new_yield, new_production, production_change, production_change_percent).
    """
    new_area  = baseline_area  * (1 + area_change_percent  / 100.0)
    new_yield = baseline_yield * (1 + yield_change_percent / 100.0)
    new_production = new_area * new_yield / 1000.0   # 1000-ha × kg/ha / 1000 → 1000-t

    production_change = new_production - baseline_production
    if baseline_production != 0:
        production_change_percent = (production_change / baseline_production) * 100.0
    else:
        production_change_percent = float("nan")

    return new_area, new_yield, new_production, production_change, production_change_percent


# ─────────────────────────────────────────────────────────────────────────────
# Standalone function (no df needed)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_scenario_standalone(
    baseline_production: float,
    baseline_area: float,
    baseline_yield: float,
    area_change_percent: float,
    yield_change_percent: float,
    scenario_name: str = "Scenario",
) -> dict:
    """
    Calculate scenario outcomes when you already have the baseline values.

    Use this when you do not need to look up data from the DataFrame (e.g.,
    for unit tests, API calls where the caller supplies the baseline, etc.).

    Parameters
    ----------
    baseline_production  : float – baseline production in 1000 t.
    baseline_area        : float – baseline area in 1000 ha.
    baseline_yield       : float – baseline yield in kg/ha.
    area_change_percent  : float – % change to apply to area.
    yield_change_percent : float – % change to apply to yield.
    scenario_name        : str   – label for this scenario.

    Returns
    -------
    dict
        Full scenario result dict (same schema as AgriculturalScenarioEngine.calculate_scenario).
    """
    # Lightweight validation (no state/crop needed here)
    errors: List[str] = []
    warnings_list: List[str] = []
    for label, val in [
        ("baseline_production", baseline_production),
        ("baseline_area", baseline_area),
        ("baseline_yield", baseline_yield),
    ]:
        try:
            fval = float(val)
            if fval < 0:
                warnings_list.append(f"'{label}' is negative ({fval}). Verify this is correct.")
        except (TypeError, ValueError):
            errors.append(f"'{label}' must be numeric, got {val!r}.")

    # Validate change percents
    pct_validation = validate_scenario_inputs(
        state="__standalone__",
        district="__standalone__",
        crop="__standalone__",
        year=2000,
        area_change_percent=area_change_percent,
        yield_change_percent=yield_change_percent,
    )
    errors.extend(pct_validation["errors"])
    warnings_list.extend(pct_validation["warnings"])

    if errors:
        return {
            "success": False,
            "scenario_name": scenario_name,
            "state": None,
            "district": None,
            "crop": None,
            "year": None,
            "baseline": None,
            "area_change_percent": area_change_percent,
            "yield_change_percent": yield_change_percent,
            "baseline_area": baseline_area,
            "baseline_yield": baseline_yield,
            "baseline_production": baseline_production,
            "new_area": None,
            "new_yield": None,
            "new_production": None,
            "production_change": None,
            "production_change_percent": None,
            "units": _build_units_dict(),
            "validation_errors": errors,
            "is_forecast": False,
            "scenario_type": _determine_scenario_type(area_change_percent, yield_change_percent),
        }

    new_area, new_yield, new_prod, prod_change, prod_change_pct = _apply_formulas(
        float(baseline_area),
        float(baseline_yield),
        float(baseline_production),
        float(area_change_percent),
        float(yield_change_percent),
    )

    result = {
        "success": True,
        "scenario_name": scenario_name,
        "state": None,
        "district": None,
        "crop": None,
        "year": None,
        "baseline": None,
        "area_change_percent": float(area_change_percent),
        "yield_change_percent": float(yield_change_percent),
        "baseline_area": float(baseline_area),
        "baseline_yield": float(baseline_yield),
        "baseline_production": float(baseline_production),
        "new_area": new_area,
        "new_yield": new_yield,
        "new_production": new_prod,
        "production_change": prod_change,
        "production_change_percent": prod_change_pct,
        "units": _build_units_dict(),
        "validation_errors": [],
        "is_forecast": False,
        "scenario_type": _determine_scenario_type(area_change_percent, yield_change_percent),
    }

    if warnings_list:
        result["validation_warnings"] = warnings_list

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main engine class
# ─────────────────────────────────────────────────────────────────────────────

class AgriculturalScenarioEngine:
    """
    Scenario engine that looks up historical baselines from the cleaned
    DataFrame and applies percentage change scenarios.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned agriculture dataset loaded via
        ``historical_analysis.load_cleaned_data()``.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"AgriculturalScenarioEngine expects a pd.DataFrame, "
                f"got {type(df).__name__!r}."
            )
        self._df = df

    # ── Single scenario ───────────────────────────────────────────────────────

    def calculate_scenario(
        self,
        state: str,
        district: str,
        crop: str,
        year: int,
        area_change_percent: float = 0.0,
        yield_change_percent: float = 0.0,
        scenario_name: str = "Scenario",
    ) -> dict:
        """
        Calculate a scenario outcome based on the historical baseline for
        (state, district, crop, year) with the given percentage changes.

        Parameters
        ----------
        state               : State name (case-insensitive).
        district            : District name (case-insensitive).
        crop                : Crop name (case-insensitive).
        year                : Calendar year.
        area_change_percent : % change to apply to baseline area.
        yield_change_percent: % change to apply to baseline yield.
        scenario_name       : Human-readable label for this scenario.

        Returns
        -------
        dict with keys:
            success, scenario_name, state, district, crop, year,
            baseline, area_change_percent, yield_change_percent,
            baseline_area, baseline_yield, baseline_production,
            new_area, new_yield, new_production,
            production_change, production_change_percent,
            units, validation_errors, is_forecast, scenario_type
        """
        # ── Validate inputs ───────────────────────────────────────────────────
        val = validate_scenario_inputs(
            state=state,
            district=district,
            crop=crop,
            year=year,
            area_change_percent=area_change_percent,
            yield_change_percent=yield_change_percent,
        )

        def _failure(reason: str, baseline: Optional[dict] = None) -> dict:
            return {
                "success": False,
                "scenario_name": scenario_name,
                "state": state,
                "district": district,
                "crop": crop,
                "year": year,
                "baseline": baseline,
                "area_change_percent": area_change_percent,
                "yield_change_percent": yield_change_percent,
                "baseline_area": None,
                "baseline_yield": None,
                "baseline_production": None,
                "new_area": None,
                "new_yield": None,
                "new_production": None,
                "production_change": None,
                "production_change_percent": None,
                "units": _build_units_dict(),
                "validation_errors": val["errors"] + ([reason] if reason else []),
                "is_forecast": False,
                "scenario_type": _determine_scenario_type(
                    area_change_percent, yield_change_percent
                ),
            }

        if not val["valid"]:
            return _failure("")

        # ── Fetch historical baseline ──────────────────────────────────────────
        has_district = isinstance(district, str) and bool(district.strip())
        if has_district:
            baseline = get_historical_baseline(
                df=self._df,
                state=state,
                district=district.strip(),
                crop=crop,
                year=year,
            )
        else:
            agg = get_aggregated_baseline(
                df=self._df,
                crop=crop,
                year=year,
                state=state.strip() if isinstance(state, str) and state.strip() else None,
                district=None,
            )
            # Normalize aggregated keys to match get_historical_baseline schema
            baseline = {
                "found": agg["found"],
                "match_count": agg.get("record_count", 0),
                "state": agg.get("state") or state,
                "district": "All Districts (State Aggregate)" if agg.get("level") == "state" else "All Districts (National Aggregate)",
                "crop": agg.get("crop") or crop,
                "year": agg.get("year") or year,
                "area_1000_ha": agg.get("total_area_1000_ha"),
                "production_1000_t": agg.get("total_production_1000_t"),
                "yield_kg_per_ha": agg.get("weighted_yield_kg_per_ha"),
                "calculated_yield_kg_per_ha": agg.get("weighted_yield_kg_per_ha"),
                "record_status": f"aggregated_{agg.get('level', 'state')}",
                "has_production_data": agg.get("total_production_1000_t") is not None,
                "units": agg.get("units", _build_units_dict()),
                "source_records": [],
                "warning": agg.get("warning"),
            }

        if not baseline["found"]:
            return _failure(
                baseline.get("warning") or
                f"Historical baseline not found for "
                f"state='{state}', district='{district}', crop='{crop}', year={year}.",
                baseline=baseline,
            )

        # ── Extract baseline values ───────────────────────────────────────────
        b_area  = baseline["area_1000_ha"]
        b_prod  = baseline["production_1000_t"]

        # Prefer calculated_yield_kg_per_ha if recorded yield is unavailable
        b_yield = (
            baseline["yield_kg_per_ha"]
            if baseline["yield_kg_per_ha"] is not None
            else baseline["calculated_yield_kg_per_ha"]
        )

        # If yield is still missing, derive it from area and production
        if b_yield is None and b_area and b_area > 0 and b_prod is not None:
            b_yield = (b_prod * 1_000) / b_area   # kg/ha

        # Validate that we have usable baseline numbers
        if b_area is None or b_area <= 0:
            return _failure(
                f"Baseline area is missing or zero ({b_area}). Cannot compute scenario.",
                baseline=baseline,
            )
        if b_yield is None or b_yield <= 0:
            return _failure(
                f"Baseline yield is missing or zero ({b_yield}). Cannot compute scenario.",
                baseline=baseline,
            )
        if b_prod is None:
            # Derive production from area × yield
            b_prod = b_area * b_yield / 1_000.0

        # ── Apply formulas ────────────────────────────────────────────────────
        new_area, new_yield, new_prod, prod_change, prod_change_pct = _apply_formulas(
            baseline_area=b_area,
            baseline_yield=b_yield,
            baseline_production=b_prod,
            area_change_percent=float(area_change_percent),
            yield_change_percent=float(yield_change_percent),
        )

        result = {
            "success": True,
            "scenario_name": scenario_name,
            "state": baseline["state"],
            "district": baseline["district"],
            "crop": baseline["crop"],
            "year": baseline["year"],
            "baseline": baseline,
            "area_change_percent": float(area_change_percent),
            "yield_change_percent": float(yield_change_percent),
            "baseline_area": b_area,
            "baseline_yield": b_yield,
            "baseline_production": b_prod,
            "new_area": new_area,
            "new_yield": new_yield,
            "new_production": new_prod,
            "production_change": prod_change,
            "production_change_percent": prod_change_pct,
            "units": _build_units_dict(),
            "validation_errors": [],
            "is_forecast": False,
            "scenario_type": _determine_scenario_type(area_change_percent, yield_change_percent),
        }

        # Attach warnings if any
        if val["warnings"]:
            result["validation_warnings"] = val["warnings"]
        if baseline.get("warning"):
            result.setdefault("validation_warnings", [])
            result["validation_warnings"].append(baseline["warning"])

        return result

    # ── Multiple scenarios ────────────────────────────────────────────────────

    def calculate_multiple_scenarios(self, scenarios: List[dict]) -> List[dict]:
        """
        Calculate a list of scenarios in sequence.

        Parameters
        ----------
        scenarios : list of dict, each with keys:
            state, district, crop, year,
            area_change_percent, yield_change_percent, scenario_name

        Returns
        -------
        list of dict (same schema as calculate_scenario)
        """
        results = []
        for i, sc in enumerate(scenarios):
            try:
                result = self.calculate_scenario(
                    state=sc.get("state", ""),
                    district=sc.get("district", ""),
                    crop=sc.get("crop", ""),
                    year=sc.get("year", 0),
                    area_change_percent=sc.get("area_change_percent", 0.0),
                    yield_change_percent=sc.get("yield_change_percent", 0.0),
                    scenario_name=sc.get("scenario_name", f"Scenario_{i + 1}"),
                )
            except Exception as exc:
                result = {
                    "success": False,
                    "scenario_name": sc.get("scenario_name", f"Scenario_{i + 1}"),
                    "validation_errors": [f"Unexpected error: {exc}"],
                    "is_forecast": False,
                    "scenario_type": "unknown",
                    "units": _build_units_dict(),
                }
            results.append(result)
        return results
