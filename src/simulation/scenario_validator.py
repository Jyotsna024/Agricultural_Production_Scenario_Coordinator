"""
PS 137 – Agricultural Production Scenario Coordinator
======================================================
src/simulation/scenario_validator.py
Member 3 – Scenario Simulation Engine: Input & Result Validation

PURPOSE
-------
Validates scenario inputs before calculation and validates scenario
result dicts after calculation for internal consistency and data quality.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

# ─── Constants ────────────────────────────────────────────────────────────────
AREA_CHANGE_MIN = -99.0      # exclusive lower bound (would eliminate all area)
AREA_CHANGE_MAX = 1000.0     # inclusive upper bound
YIELD_CHANGE_MIN = -99.0     # exclusive lower bound
YIELD_CHANGE_MAX = 1000.0    # inclusive upper bound

# Required keys in a scenario result dict
REQUIRED_RESULT_KEYS = [
    "success",
    "scenario_name",
    "state",
    "district",
    "crop",
    "year",
    "baseline",
    "area_change_percent",
    "yield_change_percent",
    "baseline_area",
    "baseline_yield",
    "baseline_production",
    "new_area",
    "new_yield",
    "new_production",
    "production_change",
    "production_change_percent",
    "units",
    "validation_errors",
    "is_forecast",
    "scenario_type",
]


# ─── Public API ───────────────────────────────────────────────────────────────

def validate_scenario_inputs(
    state: Any,
    district: Any,
    crop: Any,
    year: Any,
    area_change_percent: Any,
    yield_change_percent: Any,
) -> dict:
    """
    Validate the raw inputs for a scenario calculation.

    Parameters
    ----------
    state               : must be a non-empty string.
    district            : string (may be empty/None — warns but does not block).
    crop                : must be a non-empty string.
    year                : must be an integer (or int-convertible).
    area_change_percent : float in the open-open range (-99, 1000].
    yield_change_percent: float in the open-open range (-99, 1000].

    Returns
    -------
    dict
        {
            valid    : bool,
            errors   : list[str],   # fatal — block the calculation
            warnings : list[str],   # non-fatal — proceed but flag
        }
    """
    errors: List[str] = []
    warnings: List[str] = []

    # ── state ─────────────────────────────────────────────────────────────────
    if not isinstance(state, str) or not state.strip():
        errors.append("'state' must be a non-empty string.")

    # ── district ──────────────────────────────────────────────────────────────
    if district is None or (isinstance(district, str) and not district.strip()):
        warnings.append(
            "'district' is empty or None. "
            "Scenario will apply at state/national level if baseline allows."
        )

    # ── crop ──────────────────────────────────────────────────────────────────
    if not isinstance(crop, str) or not crop.strip():
        errors.append("'crop' must be a non-empty string.")

    # ── year ──────────────────────────────────────────────────────────────────
    try:
        year_int = int(year)
        if year_int < 1900 or year_int > 2100:
            warnings.append(
                f"'year' value {year_int} is unusual (expected 1900–2100). "
                "Proceeding, but verify this is correct."
            )
    except (TypeError, ValueError):
        errors.append(
            f"'year' must be an integer, got {type(year).__name__!r}: {year!r}."
        )

    # ── area_change_percent ───────────────────────────────────────────────────
    try:
        acp = float(area_change_percent)
        if math.isnan(acp) or math.isinf(acp):
            errors.append("'area_change_percent' must be a finite number.")
        elif acp <= AREA_CHANGE_MIN:
            errors.append(
                f"'area_change_percent' must be greater than {AREA_CHANGE_MIN} "
                f"(would eliminate all cultivated area). Got {acp}."
            )
        elif acp > AREA_CHANGE_MAX:
            errors.append(
                f"'area_change_percent' must be at most {AREA_CHANGE_MAX}. Got {acp}."
            )
        else:
            # Resulting area check (needs a positive baseline, but we validate direction)
            new_area_factor = 1 + acp / 100
            if new_area_factor <= 0:
                errors.append(
                    f"'area_change_percent' of {acp} would produce a non-positive area."
                )
    except (TypeError, ValueError):
        errors.append(
            f"'area_change_percent' must be numeric, got {type(area_change_percent).__name__!r}."
        )

    # ── yield_change_percent ──────────────────────────────────────────────────
    try:
        ycp = float(yield_change_percent)
        if math.isnan(ycp) or math.isinf(ycp):
            errors.append("'yield_change_percent' must be a finite number.")
        elif ycp <= YIELD_CHANGE_MIN:
            errors.append(
                f"'yield_change_percent' must be greater than {YIELD_CHANGE_MIN} "
                f"(would eliminate all yield). Got {ycp}."
            )
        elif ycp > YIELD_CHANGE_MAX:
            errors.append(
                f"'yield_change_percent' must be at most {YIELD_CHANGE_MAX}. Got {ycp}."
            )
        else:
            new_yield_factor = 1 + ycp / 100
            if new_yield_factor <= 0:
                errors.append(
                    f"'yield_change_percent' of {ycp} would produce a non-positive yield."
                )
    except (TypeError, ValueError):
        errors.append(
            f"'yield_change_percent' must be numeric, got {type(yield_change_percent).__name__!r}."
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def validate_scenario_result(result_dict: dict) -> dict:
    """
    Validate a scenario result dictionary produced by the scenario engine.

    Checks:
      1. All required keys are present.
      2. No NaN or infinite values in numeric fields.
      3. Production formula is internally consistent:
             new_production ≈ new_area * new_yield / 1000

    Parameters
    ----------
    result_dict : dict
        Output from AgriculturalScenarioEngine.calculate_scenario() or
        calculate_scenario_standalone().

    Returns
    -------
    dict
        {
            valid    : bool,
            errors   : list[str],
            warnings : list[str],
        }
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(result_dict, dict):
        return {
            "valid": False,
            "errors": ["result_dict is not a dict."],
            "warnings": [],
        }

    # ── 1. Required keys ──────────────────────────────────────────────────────
    missing_keys = [k for k in REQUIRED_RESULT_KEYS if k not in result_dict]
    if missing_keys:
        errors.append(f"Missing required keys: {missing_keys}.")

    # ── 2. NaN / Inf check on numeric fields ──────────────────────────────────
    numeric_fields = [
        "baseline_area", "baseline_yield", "baseline_production",
        "new_area", "new_yield", "new_production",
        "production_change", "production_change_percent",
        "area_change_percent", "yield_change_percent",
    ]
    for field in numeric_fields:
        val = result_dict.get(field)
        if val is None:
            # None is allowed when baseline was not found (success=False)
            continue
        try:
            fval = float(val)
            if math.isnan(fval):
                warnings.append(f"Field '{field}' is NaN.")
            elif math.isinf(fval):
                errors.append(f"Field '{field}' is infinite.")
        except (TypeError, ValueError):
            errors.append(f"Field '{field}' is not numeric: {val!r}.")

    # ── 3. Formula consistency ─────────────────────────────────────────────────
    if result_dict.get("success"):
        na = result_dict.get("new_area")
        ny = result_dict.get("new_yield")
        np_ = result_dict.get("new_production")
        if None not in (na, ny, np_):
            try:
                expected_prod = float(na) * float(ny) / 1000.0
                actual_prod = float(np_)
                # Allow small floating-point tolerance
                if abs(expected_prod) > 1e-9:
                    relative_error = abs(actual_prod - expected_prod) / abs(expected_prod)
                    if relative_error > 1e-6:
                        errors.append(
                            f"Production formula mismatch: "
                            f"new_area * new_yield / 1000 = {expected_prod:.6f} "
                            f"but new_production = {actual_prod:.6f} "
                            f"(relative error {relative_error:.2e})."
                        )
            except (TypeError, ValueError):
                warnings.append(
                    "Could not verify production formula due to non-numeric values."
                )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
