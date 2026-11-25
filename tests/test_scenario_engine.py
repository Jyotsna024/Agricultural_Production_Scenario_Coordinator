"""
Unit tests for Member 3 scenario engine, validator, and comparison modules.
"""

import pytest
import pandas as pd

from src.simulation.scenario_engine import AgriculturalScenarioEngine, calculate_scenario_standalone
from src.simulation.scenario_validator import validate_scenario_inputs, validate_scenario_result
from src.simulation.scenario_comparison import ScenarioComparison, create_comparison_dataframe
from src.analysis.historical_analysis import load_cleaned_data


class TestScenarioValidator:
    def test_valid_inputs(self):
        res = validate_scenario_inputs("Karnataka", "Bangalore", "Rice", 2015, 10.0, -5.0)
        assert res["valid"] is True
        assert len(res["errors"]) == 0

    def test_invalid_area_drop(self):
        res = validate_scenario_inputs("Karnataka", "", "Rice", 2015, -100.0, 0.0)
        assert res["valid"] is False
        assert any("area_change_percent" in e for e in res["errors"])

    def test_invalid_yield_drop(self):
        res = validate_scenario_inputs("Karnataka", "", "Rice", 2015, 0.0, -105.0)
        assert res["valid"] is False
        assert any("yield_change_percent" in e for e in res["errors"])

    def test_missing_crop(self):
        res = validate_scenario_inputs("Karnataka", "", "", 2015, 10.0, 0.0)
        assert res["valid"] is False


class TestScenarioEngineStandalone:
    def test_formula_correctness(self):
        # 100 1000 ha, 2000 kg/ha -> 100 * 2000 / 1000 = 200 1000 t
        res = calculate_scenario_standalone(
            baseline_production=200.0,
            baseline_area=100.0,
            baseline_yield=2000.0,
            area_change_percent=10.0,   # 110
            yield_change_percent=-10.0, # 1800
            scenario_name="Test",
        )
        assert res["success"] is True
        assert res["is_forecast"] is False
        assert pytest.approx(res["new_area"], 0.01) == 110.0
        assert pytest.approx(res["new_yield"], 0.01) == 1800.0
        # 110 * 1800 / 1000 = 198.0
        assert pytest.approx(res["new_production"], 0.01) == 198.0
        assert pytest.approx(res["production_change"], 0.01) == -2.0
        assert pytest.approx(res["production_change_percent"], 0.01) == -1.0


class TestScenarioComparison:
    def test_compare_multiple(self):
        baseline = {"total_production_1000_t": 200.0, "total_area_1000_ha": 100.0}
        sc1 = calculate_scenario_standalone(200.0, 100.0, 2000.0, 10.0, 0.0, "Scenario A")
        sc2 = calculate_scenario_standalone(200.0, 100.0, 2000.0, -10.0, 0.0, "Scenario B")
        comp = ScenarioComparison(baseline, [sc1, sc2]).compare()

        assert comp["summary"]["total_scenarios"] == 2
        assert comp["best_production_scenario"]["scenario_name"] == "Scenario A"
        assert comp["worst_production_scenario"]["scenario_name"] == "Scenario B"

        df = create_comparison_dataframe(comp)
        assert len(df) == 2
        assert df.iloc[0]["scenario_name"] == "Scenario A"
