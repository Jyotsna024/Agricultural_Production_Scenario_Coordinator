"""
Unit tests for Plotly visualization functions and Streamlit AppTest integration.
"""

from unittest.mock import MagicMock, patch
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.visualization.scenario_charts import (
    plot_area_yield_comparison,
    plot_baseline_vs_scenario,
    plot_historical_trend,
    plot_scenario_comparison,
)


class TestVisualizationCharts:
    def test_plot_baseline_vs_scenario(self):
        fig = plot_baseline_vs_scenario(2907.51, 3038.35, crop="Rice")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_plot_area_yield_comparison(self):
        fig = plot_area_yield_comparison(992.95, 1092.25, 2928.15, 2781.75, crop="Rice")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_plot_scenario_comparison(self):
        df = pd.DataFrame([
            {"scenario_name": "Scenario A", "production": 3100.0},
            {"scenario_name": "Scenario B", "production": 2750.0},
        ])
        fig = plot_scenario_comparison(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_plot_scenario_comparison_empty(self):
        fig = plot_scenario_comparison(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_plot_historical_trend(self):
        records = [
            {"year": 2015, "total_production_1000_t": 2500.0},
            {"year": 2016, "total_production_1000_t": 2700.0},
            {"year": 2017, "total_production_1000_t": 2907.51},
        ]
        fig = plot_historical_trend(records, crop="Rice")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1


class TestStreamlitAppIntegration:
    def test_streamlit_apptest_renders(self):
        """Verify the Streamlit app renders without exceptions."""
        from streamlit.testing.v1 import AppTest

        # Mock LLM status so app shows correctly even without a real API key
        with (
            patch("src.agents.llm_config._GEMINI_API_KEY", ""),
            patch("src.agents.llm_config.is_llm_available", return_value=False),
        ):
            at = AppTest.from_file("app/app.py", default_timeout=30)
            at.run()
            assert not at.exception
            # Should have a text area for the prompt
            assert len(at.text_area) >= 1
            # Should have an Analyse button
            assert any("Analyse" in b.label for b in at.button)

    def test_streamlit_apptest_manual_scenario_button_exists(self):
        """Verify the Manual Scenario button exists in the sidebar."""
        from streamlit.testing.v1 import AppTest

        with (
            patch("src.agents.llm_config._GEMINI_API_KEY", ""),
            patch("src.agents.llm_config.is_llm_available", return_value=False),
        ):
            at = AppTest.from_file("app/app.py", default_timeout=30)
            at.run()
            assert not at.exception
            # Should have a manual scenario run button
            manual_btns = [b for b in at.button if "Manual" in b.label]
            assert len(manual_btns) >= 1
