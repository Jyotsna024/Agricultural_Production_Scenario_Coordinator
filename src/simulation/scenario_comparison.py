"""
PS 137 – Agricultural Production Scenario Coordinator
======================================================
src/simulation/scenario_comparison.py
Member 3 – Scenario Comparison

PURPOSE
-------
Compares multiple scenario results against a shared baseline, ranks them
by production outcome, and assembles a tidy comparison table.

RULES
  - Never modify the original baseline or scenario dicts.
  - Comparisons are purely arithmetic; no forecasting logic here.
  - All numeric values in the comparison table are floats (or None).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# ScenarioComparison class
# ─────────────────────────────────────────────────────────────────────────────

class ScenarioComparison:
    """
    Compare multiple scenario results against a common baseline.

    Parameters
    ----------
    baseline_dict : dict
        The baseline information dict.  Can be the ``baseline`` sub-dict
        from a scenario result (output of get_historical_baseline /
        get_aggregated_baseline) or a minimal dict with at least
        ``production_1000_t`` / ``total_production_1000_t`` and
        ``area_1000_ha`` / ``total_area_1000_ha`` fields.
    scenarios : list[dict]
        List of scenario result dicts produced by
        ``AgriculturalScenarioEngine.calculate_scenario()`` or
        ``calculate_scenario_standalone()``.
    """

    def __init__(self, baseline_dict: dict, scenarios: List[dict]) -> None:
        if not isinstance(baseline_dict, dict):
            raise TypeError(
                f"baseline_dict must be a dict, got {type(baseline_dict).__name__!r}."
            )
        if not isinstance(scenarios, list):
            raise TypeError(
                f"scenarios must be a list, got {type(scenarios).__name__!r}."
            )
        self._baseline  = baseline_dict
        self._scenarios = scenarios

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_baseline_production(baseline_dict: dict) -> Optional[float]:
        """
        Extract baseline production from either a historical or aggregated
        baseline dict (they use different key names).
        """
        for key in ("production_1000_t", "total_production_1000_t", "baseline_production"):
            val = baseline_dict.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _get_baseline_area(baseline_dict: dict) -> Optional[float]:
        for key in ("area_1000_ha", "total_area_1000_ha", "baseline_area"):
            val = baseline_dict.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _safe_float(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    def compare(self) -> dict:
        """
        Build the full comparison result.

        Returns
        -------
        dict with keys:
            baseline               : the original baseline_dict
            scenarios              : the original scenarios list
            comparison_table       : list[dict] – one row per scenario
            best_production_scenario  : dict – scenario with highest new_production
            worst_production_scenario : dict – scenario with lowest new_production
            summary                : dict – aggregated statistics
        """
        baseline_prod = self._get_baseline_production(self._baseline)
        baseline_area = self._get_baseline_area(self._baseline)

        # ── Build comparison table ─────────────────────────────────────────────
        table: List[dict] = []
        for sc in self._scenarios:
            row = self._build_row(sc, baseline_prod)
            table.append(row)

        # ── Rank by production ────────────────────────────────────────────────
        valid_rows = [r for r in table if r["production"] is not None]

        best  = None
        worst = None
        if valid_rows:
            best  = max(valid_rows, key=lambda r: r["production"])
            worst = min(valid_rows, key=lambda r: r["production"])

        # Resolve back to full scenario dict for best/worst
        best_sc  = self._find_scenario_by_name(best["scenario_name"])  if best  else None
        worst_sc = self._find_scenario_by_name(worst["scenario_name"]) if worst else None

        # ── Summary ───────────────────────────────────────────────────────────
        productions = [r["production"] for r in valid_rows]
        changes     = [r["production_change_percent"] for r in valid_rows
                       if r["production_change_percent"] is not None]

        summary = {
            "total_scenarios": len(self._scenarios),
            "successful_scenarios": len(valid_rows),
            "baseline_production_1000_t": baseline_prod,
            "baseline_area_1000_ha": baseline_area,
            "max_production_1000_t": max(productions) if productions else None,
            "min_production_1000_t": min(productions) if productions else None,
            "max_production_change_percent": max(changes) if changes else None,
            "min_production_change_percent": min(changes) if changes else None,
            "avg_production_1000_t": (
                sum(productions) / len(productions) if productions else None
            ),
        }

        return {
            "baseline": self._baseline,
            "scenarios": self._scenarios,
            "comparison_table": table,
            "best_production_scenario": best_sc,
            "worst_production_scenario": worst_sc,
            "summary": summary,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_row(self, sc: dict, baseline_prod: Optional[float]) -> dict:
        """Build a single comparison_table row from a scenario result dict."""
        name             = sc.get("scenario_name", "Unknown")
        area_chg         = self._safe_float(sc.get("area_change_percent"))
        yield_chg        = self._safe_float(sc.get("yield_change_percent"))
        new_prod         = self._safe_float(sc.get("new_production"))
        prod_change      = self._safe_float(sc.get("production_change"))
        prod_change_pct  = self._safe_float(sc.get("production_change_percent"))
        scenario_type    = sc.get("scenario_type", "unknown")
        success          = sc.get("success", False)

        return {
            "scenario_name"            : name,
            "area_change_percent"      : area_chg,
            "yield_change_percent"     : yield_chg,
            "production"               : new_prod,
            "production_change"        : prod_change,
            "production_change_percent": prod_change_pct,
            "scenario_type"            : scenario_type,
            "success"                  : success,
        }

    def _find_scenario_by_name(self, name: str) -> Optional[dict]:
        """Return the first scenario dict whose scenario_name matches."""
        for sc in self._scenarios:
            if sc.get("scenario_name") == name:
                return sc
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Standalone helper function
# ─────────────────────────────────────────────────────────────────────────────

def create_comparison_dataframe(comparison_result: dict) -> pd.DataFrame:
    """
    Convert the ``comparison_table`` list inside a ``compare()`` result
    into a tidy pandas DataFrame.

    Parameters
    ----------
    comparison_result : dict
        Output of ``ScenarioComparison.compare()``.

    Returns
    -------
    pd.DataFrame
        One row per scenario, columns matching the comparison_table schema:
        scenario_name, area_change_percent, yield_change_percent,
        production, production_change, production_change_percent,
        scenario_type, success.

        Sorted by production descending (NaN rows appended at end).
    """
    table = comparison_result.get("comparison_table", [])
    if not table:
        return pd.DataFrame(columns=[
            "scenario_name",
            "area_change_percent",
            "yield_change_percent",
            "production",
            "production_change",
            "production_change_percent",
            "scenario_type",
            "success",
        ])

    df = pd.DataFrame(table)

    # Ensure all expected columns exist (fill missing ones with None)
    expected_cols = [
        "scenario_name",
        "area_change_percent",
        "yield_change_percent",
        "production",
        "production_change",
        "production_change_percent",
        "scenario_type",
        "success",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    # Sort by production descending, NaN last
    df = df.sort_values("production", ascending=False, na_position="last")
    df = df.reset_index(drop=True)

    return df[expected_cols]
