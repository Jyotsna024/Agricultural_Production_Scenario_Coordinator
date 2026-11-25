"""
PS 137 - Agricultural Production Scenario Coordinator
======================================================
app/app.py
Streamlit Web Dashboard:
  - Clean, prompt-based single-page user interface
  - Powered by local Ollama + LangGraph + deterministic Python engines
  - Instant manual scenario mode fallback when Ollama is unavailable
  - Strict distinction: Hypothetical Scenarios vs Forecasts
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.agents.agent import AgricultureAgent
from src.agents.llm_config import (
    check_ollama_connection,
    get_llm_settings,
    get_llm_status,
)
from src.analysis.historical_analysis import load_cleaned_data
from src.simulation.scenario_comparison import create_comparison_dataframe
from src.visualization.scenario_charts import (
    plot_area_yield_comparison,
    plot_baseline_vs_scenario,
    plot_historical_trend,
    plot_scenario_comparison,
)

# Page configuration
st.set_page_config(
    page_title="Agricultural Production Scenario Coordinator",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #2b5c8f;
    }
    .disclaimer-banner {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        border: 1px solid #ffeeba;
        font-size: 0.92rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_cached_agent() -> AgricultureAgent:
    """Load and cache dataset and AgricultureAgent."""
    df = load_cleaned_data()
    return AgricultureAgent(df=df)


# Initialize session state
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = (
        "What happens if rice cultivation area in Karnataka increases by 10% "
        "and yield decreases by 5% in 2017?"
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Status")
    status = get_llm_status()
    settings = get_llm_settings()

    if status["available"]:
        st.success(f"🟢 **Ollama Connected**\n\nModel: `{status['model']}`")
    else:
        st.warning(
            f"🟡 **Ollama Offline / Standby**\n\n"
            f"{status.get('error', 'Service not reached')}\n\n"
            "Tip: Run `ollama serve` in a terminal."
        )

    st.markdown("---")
    st.subheader("Configuration")
    st.caption(f"**Provider:** `{settings['provider']}`")
    st.caption(f"**Base URL:** `{settings['base_url']}`")
    st.caption(f"**Target Model:** `{settings['model']}`")

    if status.get("available_models"):
        st.caption(f"**Installed Models:** {', '.join(status['available_models'])}")

    st.markdown("---")
    st.subheader("Manual Scenario Override")
    with st.expander("Run Direct Deterministic Scenario"):
        m_state = st.text_input("State", value="Karnataka")
        m_crop = st.text_input("Crop", value="Rice")
        m_year = st.number_input("Year", min_value=2010, max_value=2017, value=2017)
        m_area_chg = st.slider("Area Change (%)", min_value=-50.0, max_value=50.0, value=10.0, step=1.0)
        m_yield_chg = st.slider("Yield Change (%)", min_value=-50.0, max_value=50.0, value=-5.0, step=1.0)
        m_district = st.text_input("District (optional, leave blank for state)", value="")
        if st.button("Run Manual Calculation", use_container_width=True):
            agent = get_cached_agent()
            with st.spinner("Computing deterministic scenario..."):
                st.session_state.last_result = agent.run_manual_scenario(
                    state=m_state,
                    crop=m_crop,
                    year=int(m_year),
                    area_change_percent=float(m_area_chg),
                    yield_change_percent=float(m_yield_chg),
                    district=m_district,
                    scenario_name="Manual User Scenario",
                )
            st.rerun()

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🌾 Agricultural Production Scenario Coordinator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Problem Statement 137 • Natural-language what-if scenario exploration grounded in verified historical data (2010–2017)'
    '</div>',
    unsafe_allow_html=True,
)

# ── Prompt Input Section ──────────────────────────────────────────────────────
user_prompt = st.text_area(
    "Ask an agricultural question or enter a what-if scenario assumption in plain English:",
    value=st.session_state.prompt_input,
    height=100,
    key="user_prompt_widget",
)

# Example Prompt Quick Buttons
st.markdown("**Example prompts:**")
col_e1, col_e2, col_e3, col_e4 = st.columns(4)

if col_e1.button("🌾 Rice +10% Area, -5% Yield (Karnataka 2017)"):
    st.session_state.prompt_input = "What happens if rice cultivation area in Karnataka increases by 10% and yield decreases by 5% in 2017?"
    st.rerun()

if col_e2.button("🌽 Maize Area Expansion +15% (2015)"):
    st.session_state.prompt_input = "Suppose maize area in Karnataka increases by 15% in 2015 while yield remains unchanged. Calculate the new production."
    st.rerun()

if col_e3.button("📊 Compare Drought vs Tech in Wheat (2016)"):
    st.session_state.prompt_input = "Compare two scenarios for wheat in Madhya Pradesh in 2016: Scenario A with yield drop of 12%, and Scenario B with area growth of 8%."
    st.rerun()

if col_e4.button("📈 Historical Rice Production Trend"):
    st.session_state.prompt_input = "Show me the historical production trend of rice in Karnataka from 2010 to 2017."
    st.rerun()

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    analyse_clicked = st.button("🚀 Analyse", type="primary", use_container_width=True)

# ── Execution Trigger ─────────────────────────────────────────────────────────
if analyse_clicked and user_prompt.strip():
    agent = get_cached_agent()
    progress_bar = st.progress(10, text="Understanding prompt via local Ollama...")
    try:
        progress_bar.progress(35, text="Extracting parameters and validating fields...")
        progress_bar.progress(60, text="Retrieving historical baseline & executing Python scenario engine...")
        result = agent.run(user_prompt.strip())
        progress_bar.progress(90, text="Validating outputs & generating explanation...")
        st.session_state.last_result = result
        progress_bar.progress(100, text="Completed!")
        progress_bar.empty()
    except Exception as exc:
        progress_bar.empty()
        st.error(f"Error during analysis: {exc}")

# ── Results Display ───────────────────────────────────────────────────────────
res = st.session_state.last_result

if res:
    st.markdown("---")
    intent = res.get("intent", "scenario_simulation")
    status = res.get("status", "unknown")

    # Clarification Needed or Error
    if res.get("needs_clarification"):
        st.info(f"ℹ️ **Clarification Needed:**\n\n{res.get('clarification_question')}")

    elif status == "failed" or res.get("error"):
        st.error(f"⚠️ **Could not complete scenario:** {res.get('error') or res.get('explanation')}")

    # Historical Analysis Display
    elif intent == "historical_analysis":
        st.subheader("📊 Historical Trend Analysis")
        hist = res.get("historical_data") or {}
        if hist.get("found"):
            yw = hist.get("year_wise", [])
            first_crop = (res.get("extracted_assumptions") or [{}])[0].get("crop", "Crop")
            st.plotly_chart(plot_historical_trend(yw, crop=first_crop), use_container_width=True)

            with st.expander("View Historical Data Table"):
                st.dataframe(pd.DataFrame(yw), use_container_width=True)
        else:
            st.warning(hist.get("warning", "No matching historical records found."))

        st.markdown("### 💡 Analysis Summary")
        st.write(res.get("explanation", ""))

    # Scenario Simulation / Comparison Display
    else:
        sim_results = res.get("simulation_results") or []
        successful_sims = [s for s in sim_results if s.get("success")]

        if successful_sims:
            # Highlight Primary Scenario Metrics
            primary = successful_sims[0]
            b_prod = primary.get("baseline_production") or 0.0
            n_prod = primary.get("new_production") or 0.0
            diff = primary.get("production_change") or 0.0
            pct = primary.get("production_change_percent") or 0.0
            crop_name = primary.get("crop") or "Crop"
            state_name = primary.get("state") or "Region"
            year_val = primary.get("year") or ""

            st.markdown(f"### 🎯 Results for **{crop_name}** in **{state_name}** ({year_val})")

            # 4 Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Historical Baseline Production", f"{b_prod:,.2f} kt", help="Recorded in 1000 metric tonnes")
            m2.metric("Hypothetical Scenario Production", f"{n_prod:,.2f} kt", delta=f"{diff:+,.2f} kt")
            m3.metric("Absolute Difference", f"{diff:+,.2f} kt")
            m4.metric("Percentage Change", f"{pct:+.2f}%")

            # Mandatory Disclaimer Banner
            st.markdown(
                '<div class="disclaimer-banner">'
                '⚠️ <strong>Important:</strong> This is a <strong>hypothetical what-if scenario</strong> '
                'based on arithmetic simulation of user assumptions, <strong>NOT a future forecast or prediction</strong>.'
                '</div>',
                unsafe_allow_html=True,
            )

            # Visualizations
            c_chart1, c_chart2 = st.columns(2)
            with c_chart1:
                st.plotly_chart(
                    plot_baseline_vs_scenario(
                        baseline_prod=b_prod,
                        scenario_prod=n_prod,
                        crop=crop_name,
                    ),
                    use_container_width=True,
                )
            with c_chart2:
                st.plotly_chart(
                    plot_area_yield_comparison(
                        baseline_area=primary.get("baseline_area") or 0.0,
                        new_area=primary.get("new_area") or 0.0,
                        baseline_yield=primary.get("baseline_yield") or 0.0,
                        new_yield=primary.get("new_yield") or 0.0,
                        crop=crop_name,
                    ),
                    use_container_width=True,
                )

            # Multi-scenario comparison table if applicable
            comp_res = res.get("comparison_result")
            if comp_res:
                st.markdown("### ⚖️ Multi-Scenario Comparison")
                comp_df = create_comparison_dataframe(comp_res)
                st.plotly_chart(plot_scenario_comparison(comp_df), use_container_width=True)
                st.dataframe(comp_df, use_container_width=True)

            # AI Explanation
            st.markdown("### 🤖 Scenario Explanation")
            st.info(res.get("explanation", ""))

            # Technical Details Expander
            with st.expander("🔍 View Technical Details & Formula Audit"):
                st.markdown(
                    """
                    **Deterministic Formula Applied:**
                    $$\\text{New Area} = \\text{Baseline Area} \\times \\left(1 + \\frac{\\Delta\\text{Area}\\%}{100}\\right)$$
                    $$\\text{New Yield} = \\text{Baseline Yield} \\times \\left(1 + \\frac{\\Delta\\text{Yield}\\%}{100}\\right)$$
                    $$\\text{New Production (1000 t)} = \\frac{\\text{New Area (1000 ha)} \\times \\text{New Yield (kg/ha)}}{1000}$$
                    """
                )
                st.json(
                    {
                        "extracted_assumptions": res.get("extracted_assumptions"),
                        "baseline_record": primary.get("baseline"),
                        "simulation_record": primary,
                        "validation_record": res.get("validation_results"),
                    }
                )
        else:
            st.error("Scenario simulation could not produce valid results. Check state/crop/year spelling.")
