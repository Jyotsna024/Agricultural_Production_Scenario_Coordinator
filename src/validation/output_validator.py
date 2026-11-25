"""
PS 137 - Agricultural Production Scenario Coordinator
======================================================
src/validation/output_validator.py
Member 4 - Final Output Validation

Validates final scenario results and AI-generated explanations to ensure:
1. No NaN or infinite numeric values.
2. Formula consistency (new_production == new_area * new_yield / 1000).
3. Units are explicitly present.
4. is_forecast is False (hypothetical scenario, never a forecast).
5. Explanation text does not present scenario outputs as actual future forecasts.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

from src.simulation.scenario_validator import validate_scenario_result

_FORECAST_CLAIM_PATTERNS = [
    re.compile(r"\bwill\s+definitely\s+(?:produce|reach|be)\b", re.IGNORECASE),
    re.compile(r"\bguaranteed\s+(?:to\s+produce|production|yield)\b", re.IGNORECASE),
    re.compile(r"\bactual\s+future\s+(?:forecast|prediction)\b", re.IGNORECASE),
    re.compile(r"\bthis\s+(?:is\s+a|serves\s+as\s+a)\s+(?:forecast|prediction)\b", re.IGNORECASE),
]


def validate_explanation_text(explanation: Optional[str]) -> Dict[str, Any]:
    """
    Check that an explanation string does not claim the hypothetical scenario
    is an actual forecast or guaranteed future prediction.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not explanation or not isinstance(explanation, str) or not explanation.strip():
        warnings.append("Explanation text is empty.")
        return {"valid": True, "errors": errors, "warnings": warnings}

    for pattern in _FORECAST_CLAIM_PATTERNS:
        if pattern.search(explanation):
            warnings.append(
                "Explanation contains wording that could sound like a forecast. "
                "Scenario outputs are hypothetical analyses, not forecasts."
            )
            break

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def validate_output(result_dict: Dict[str, Any], explanation: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate a scenario result dict (and optional explanation text).

    Returns
    -------
    dict with keys: valid (bool), errors (list[str]), warnings (list[str])
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(result_dict, dict):
        return {
            "valid": False,
            "errors": ["Output result must be a dict."],
            "warnings": [],
        }

    # Delegate core structural, NaN/Inf, and formula checks to Member 3 validator
    sim_check = validate_scenario_result(result_dict)
    errors.extend(sim_check.get("errors", []))
    warnings.extend(sim_check.get("warnings", []))

    # Ensure is_forecast is False
    if result_dict.get("is_forecast") is not False:
        errors.append(
            "Output 'is_forecast' must be False. "
            "Hypothetical scenarios must not be marked as forecasts."
        )

    # Ensure units dictionary is present and non-empty
    units = result_dict.get("units")
    if not isinstance(units, dict) or not units:
        errors.append("Output 'units' metadata is missing or empty.")

    # Check numeric fields explicitly for NaN/Inf on top-level numbers
    for key in (
        "baseline_area",
        "baseline_yield",
        "baseline_production",
        "new_area",
        "new_yield",
        "new_production",
        "production_change",
        "production_change_percent",
    ):
        val = result_dict.get(key)
        if val is not None:
            try:
                fval = float(val)
                if math.isnan(fval) or math.isinf(fval):
                    errors.append(f"Field '{key}' contains invalid float value: {val}.")
            except (TypeError, ValueError):
                errors.append(f"Field '{key}' is not numeric: {val!r}.")

    if explanation is not None:
        exp_check = validate_explanation_text(explanation)
        errors.extend(exp_check["errors"])
        warnings.extend(exp_check["warnings"])

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
