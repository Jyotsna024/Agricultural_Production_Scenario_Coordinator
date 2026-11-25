"""
PS 137 - Agricultural Production Scenario Coordinator
======================================================
src/visualization/scenario_charts.py
Plotly chart generators for scenario comparison, baseline vs scenario,
area and yield changes, and historical trends.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go


def plot_baseline_vs_scenario(
    baseline_prod: float,
    scenario_prod: float,
    title: str = "Production: Baseline vs Hypothetical Scenario",
    crop: str = "Crop",
    units: str = "Thousand Tonnes (1000 t)",
) -> go.Figure:
    """Create a bar chart comparing baseline vs scenario production."""
    fig = go.Figure(
        data=[
            go.Bar(
                name="Historical Baseline",
                x=["Historical Baseline"],
                y=[baseline_prod],
                marker_color="#2b5c8f",
                text=[f"{baseline_prod:,.1f}"],
                textposition="auto",
            ),
            go.Bar(
                name="Hypothetical Scenario",
                x=["Hypothetical Scenario"],
                y=[scenario_prod],
                marker_color="#2ca02c" if scenario_prod >= baseline_prod else "#d62728",
                text=[f"{scenario_prod:,.1f}"],
                textposition="auto",
            ),
        ]
    )
    fig.update_layout(
        title=f"{title} ({crop})",
        yaxis_title=units,
        xaxis_title="",
        template="plotly_white",
        showlegend=False,
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_area_yield_comparison(
    baseline_area: float,
    new_area: float,
    baseline_yield: float,
    new_yield: float,
    crop: str = "Crop",
) -> go.Figure:
    """Create side-by-side grouped bars for Area and Yield lever changes."""
    fig = go.Figure(
        data=[
            go.Bar(
                name="Historical Baseline",
                x=["Cultivated Area (1000 ha)", "Crop Yield (kg/ha)"],
                y=[baseline_area, baseline_yield],
                marker_color="#4575b4",
                text=[f"{baseline_area:,.1f}", f"{baseline_yield:,.1f}"],
                textposition="auto",
            ),
            go.Bar(
                name="Hypothetical Scenario",
                x=["Cultivated Area (1000 ha)", "Crop Yield (kg/ha)"],
                y=[new_area, new_yield],
                marker_color="#fdae61",
                text=[f"{new_area:,.1f}", f"{new_yield:,.1f}"],
                textposition="auto",
            ),
        ]
    )
    fig.update_layout(
        title=f"Assumption Levers: Area and Yield ({crop})",
        barmode="group",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_scenario_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Plot multi-scenario production comparison bars from a DataFrame."""
    if comparison_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No scenario data to compare", template="plotly_white")
        return fig

    df = comparison_df.copy()
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["scenario_name"],
                y=df["production"],
                text=[f"{v:,.1f}" if pd.notna(v) else "N/A" for v in df["production"]],
                textposition="auto",
                marker_color="#1f77b4",
            )
        ]
    )
    fig.update_layout(
        title="Hypothetical Production Across Scenarios",
        xaxis_title="Scenario Name",
        yaxis_title="Hypothetical Production (1000 t)",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_historical_trend(year_wise_records: List[Dict[str, Any]], crop: str = "Crop") -> go.Figure:
    """Create line + marker chart of historical production over years."""
    if not year_wise_records:
        fig = go.Figure()
        fig.update_layout(title="No historical trend records found", template="plotly_white")
        return fig

    df = pd.DataFrame(year_wise_records)
    prod_col = "total_production_1000_t" if "total_production_1000_t" in df.columns else "production_1000_t"
    year_col = "year"

    fig = go.Figure()
    if prod_col in df.columns and year_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[year_col],
                y=df[prod_col],
                mode="lines+markers",
                name="Historical Production",
                line=dict(color="#2ca02c", width=3),
                marker=dict(size=8),
            )
        )
    fig.update_layout(
        title=f"Historical Recorded Production Trend ({crop})",
        xaxis_title="Year",
        yaxis_title="Production (1000 t)",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig
