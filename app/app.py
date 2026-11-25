"""
PS 137 - Agricultural Production Scenario Coordinator
======================================================
app/app.py
Streamlit Dashboard — "Your Farm, Your Scenario 🌾"
Farmer-friendly interface powered by Gemini + LangGraph + deterministic Python engines.

Features:
  - Natural language prompt input (any hypothetical scenario)
  - Any target year (not just 2010-2017)
  - Manual scenario controls (always available)
  - Clear distinction: Historical data vs Hypothetical scenarios
  - Disclaimer: not a forecast
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
from src.agents.llm_config import get_llm_settings, get_llm_status, is_llm_available
from src.analysis.historical_analysis import load_cleaned_data
from src.simulation.scenario_comparison import create_comparison_dataframe
from src.visualization.scenario_charts import (
    plot_area_yield_comparison,
    plot_baseline_vs_scenario,
    plot_historical_trend,
    plot_scenario_comparison,
)

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Your Farm, Your Scenario 🌾",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom styling ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1e5c1a;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid #2b8a3e;
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
    .example-btn { margin: 0.2rem 0; }
    .stTextArea textarea { font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state initialisation ───────────────────────────────────────────────
def _init_session():
    defaults = {
        "result": None,
        "prompt_text": "",
        "run_triggered": False,
        "agent": None,
        "df": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()


# ── Load data once ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading agricultural dataset…")
def _load_df():
    return load_cleaned_data()


@st.cache_resource(show_spinner="Initialising AI agent…")
def _load_agent(_df):
    return AgricultureAgent(df=_df)


df = _load_df()
agent = _load_agent(df)

AVAILABLE_STATES = sorted(df["state_clean"].dropna().unique().tolist())
AVAILABLE_CROPS = sorted(df["crop_clean"].dropna().unique().tolist())
DATASET_YEARS = sorted(df["year"].dropna().unique().tolist())
MIN_YEAR = min(DATASET_YEARS)
MAX_YEAR = max(DATASET_YEARS)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # LLM status (instant local check — no API call on every rerun)
    settings = get_llm_settings()
    if is_llm_available():
        st.success(f"✅ AI Ready: {settings['model']}")
        if st.button("🔌 Test AI Connection", use_container_width=True):
            with st.spinner("Testing Gemini API…"):
                llm_status = get_llm_status()
            if llm_status.get("available"):
                st.success(f"✅ Connected to {llm_status['model']}")
            else:
                st.warning(f"⚠️ {llm_status.get('error', 'Connection failed')}")
    else:
        st.error("❌ AI unavailable: GEMINI_API_KEY is not set.")
        st.info(
            "👉 Add your Gemini API key to `.env`:\n"
            "```\nGEMINI_API_KEY=your_key_here\nGEMINI_MODEL=gemini-3.5-flash-lite\n```\n"
            "Get a free key at https://aistudio.google.com"
        )

    st.divider()
    st.subheader("🔧 Manual Scenario Mode")
    st.caption("Use these controls to run a scenario directly — no AI needed.")

    manual_state = st.selectbox("State", AVAILABLE_STATES, index=AVAILABLE_STATES.index("Karnataka") if "Karnataka" in AVAILABLE_STATES else 0)
    manual_crop = st.selectbox("Crop", AVAILABLE_CROPS, index=AVAILABLE_CROPS.index("Rice") if "Rice" in AVAILABLE_CROPS else 0)
    manual_ref_year = st.selectbox(
        "Reference Year (baseline)",
        DATASET_YEARS,
        index=len(DATASET_YEARS) - 1,
        help=f"Historical baseline year. Data is available from {MIN_YEAR} to {MAX_YEAR}.",
    )
    manual_target_year = st.number_input(
        "Target Year (your hypothetical year)",
        min_value=2010,
        max_value=2100,
        value=2025,
        step=1,
        help="Any year — 2025, 2030, etc. The calculation uses the reference year's data as the baseline.",
    )
    manual_area_change = st.slider("Area change (%)", -80.0, 100.0, 0.0, 1.0)
    manual_yield_change = st.slider("Yield change (%)", -80.0, 100.0, 0.0, 1.0)

    run_manual = st.button("▶ Run Manual Scenario", use_container_width=True)

    st.divider()
    st.caption(f"Dataset: {MIN_YEAR}–{MAX_YEAR} · {len(AVAILABLE_STATES)} states · {len(AVAILABLE_CROPS)} crops")

# ── Main page header ───────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🌾 Your Farm, Your Scenario</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Understand how changes in your farming plans may affect crop production. '
    'Describe any hypothetical situation and get a clear, plain-language analysis.</div>',
    unsafe_allow_html=True,
)

# ── Example prompts ────────────────────────────────────────────────────────────
st.markdown("**Try one of these examples:**")
col1, col2 = st.columns(2)
examples = [
    "What happens if rice area in Karnataka increases by 10% and yield decreases by 5% in 2030?",
    "If wheat cultivation in Punjab expands by 15% with no change in yield, what is the impact?",
    "Show historical wheat production trends in Punjab from 2010 to 2017.",
    "Compare: rice in Karnataka with area +10% vs area -10%, both in 2025.",
]
with col1:
    if st.button(examples[0], use_container_width=True, key="ex0"):
        st.session_state["prompt_text"] = examples[0]
    if st.button(examples[2], use_container_width=True, key="ex2"):
        st.session_state["prompt_text"] = examples[2]
with col2:
    if st.button(examples[1], use_container_width=True, key="ex1"):
        st.session_state["prompt_text"] = examples[1]
    if st.button(examples[3], use_container_width=True, key="ex3"):
        st.session_state["prompt_text"] = examples[3]

# ── Prompt input ───────────────────────────────────────────────────────────────
prompt_text = st.text_area(
    "📝 Tell us what you are planning for your farm:",
    value=st.session_state.get("prompt_text", ""),
    height=100,
    placeholder="e.g. What if rice area in Karnataka increases by 10% and yield decreases by 5% in 2030?",
    key="main_prompt",
)

analyse_col, _ = st.columns([1, 3])
with analyse_col:
    run_ai = st.button("🔍 Analyse My Scenario", use_container_width=True, type="primary")


# ── Run scenarios ──────────────────────────────────────────────────────────────
result = None

if run_ai and prompt_text.strip():
    if not is_llm_available():
        st.error(
            "❌ Gemini API key is not configured. "
            "Please add `GEMINI_API_KEY=your_key` to the `.env` file and restart the app. "
            "Meanwhile, use **Manual Scenario Mode** in the sidebar."
        )
    else:
        with st.spinner("🤔 Analysing your scenario with AI…"):
            try:
                result = agent.run(prompt_text.strip())
                st.session_state["result"] = result
            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")

elif run_manual:
    with st.spinner("⚙️ Running manual scenario…"):
        try:
            result = agent.run_manual_scenario(
                state=manual_state,
                crop=manual_crop,
                year=manual_ref_year,
                area_change_percent=manual_area_change,
                yield_change_percent=manual_yield_change,
                scenario_name=f"{manual_crop} in {manual_state} ({manual_target_year})",
            )
            # Inject target year info
            for r in result.get("simulation_results", []):
                r["target_year"] = manual_target_year
                r["reference_year"] = manual_ref_year
                r["using_reference_year_note"] = (
                    f"Using {manual_ref_year} data as baseline for target year {manual_target_year}."
                )
                r["is_forecast"] = False
            st.session_state["result"] = result
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Use cached result if no new run
if result is None and st.session_state.get("result"):
    result = st.session_state["result"]


# ── Display results ────────────────────────────────────────────────────────────
def _fmt(val, unit="", decimals=2):
    if val is None:
        return "N/A"
    return f"{val:,.{decimals}f} {unit}".strip()


if result:
    intent = result.get("intent", "")
    status = result.get("status", "")

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    if intent in ("scenario_simulation", "comparison"):
        st.markdown(
            '<div class="disclaimer-banner">⚠️ <strong>Important:</strong> This is a mathematical '
            "what-if calculation based on your assumptions and available historical data. "
            "It is <strong>not a guarantee of future harvest or income</strong>. "
            "Real outcomes depend on weather, soil conditions, government policies, and many other factors.</div>",
            unsafe_allow_html=True,
        )

    # ── Clarification needed ───────────────────────────────────────────────────
    if result.get("needs_clarification") or status == "clarification_needed":
        st.info(f"💬 {result.get('clarification_question', 'Please provide more details.')}")
        st.stop()

    # ── Historical analysis results ────────────────────────────────────────────
    if intent == "historical_analysis":
        st.subheader("📊 Historical Data Analysis")
        hist_data = result.get("historical_data") or (result.get("baseline_results") or [{}])[0]
        if not hist_data or not hist_data.get("found"):
            st.warning("No historical data found for your query. Try a different state, crop, or year range.")
        else:
            st.metric("Records Found", hist_data.get("record_count", 0))
            year_wise = hist_data.get("year_wise", [])
            if year_wise:
                trend_df = pd.DataFrame(year_wise)
                st.markdown("#### Year-wise Production Trend")
                fig = plot_historical_trend(trend_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(trend_df.style.format(precision=2), use_container_width=True)

        st.subheader("🤖 AI Explanation")
        st.write(result.get("explanation", "No explanation generated."))

    # ── Scenario / comparison results ──────────────────────────────────────────
    elif intent in ("scenario_simulation", "comparison"):
        sim_results = result.get("simulation_results", [])
        assumptions = result.get("extracted_assumptions", [])

        for i, (sim, assump) in enumerate(zip(sim_results, assumptions + [{}] * max(0, len(sim_results) - len(assumptions)))):
            if not sim.get("success"):
                errs = "; ".join(sim.get("validation_errors", ["Unknown error"]))
                st.error(f"❌ Scenario '{sim.get('scenario_name', i+1)}' failed: {errs}")
                continue

            ref_year = sim.get("reference_year")
            tgt_year = sim.get("target_year", ref_year)
            note = sim.get("using_reference_year_note", "")

            st.subheader(f"📋 {sim.get('scenario_name', f'Scenario {i+1}')}")
            if note:
                st.caption(f"ℹ️ {note}")

            # Key metrics
            c1, c2, c3, c4 = st.columns(4)
            b_prod = sim.get("baseline_production")
            n_prod = sim.get("new_production")
            diff = sim.get("production_change")
            pct = sim.get("production_change_percent")

            with c1:
                st.metric(
                    f"📦 Baseline Production ({ref_year})",
                    _fmt(b_prod, "k-tonnes"),
                )
            with c2:
                st.metric(
                    f"🌱 Scenario Production ({tgt_year})",
                    _fmt(n_prod, "k-tonnes"),
                    delta=f"{diff:+.2f} k-tonnes ({pct:+.1f}%)" if diff is not None and pct is not None else None,
                )
            with c3:
                area_chg = sim.get("area_change_percent", 0.0)
                st.metric(
                    "🗺️ Area Change",
                    f"{area_chg:+.1f}%",
                    delta_color="normal",
                )
            with c4:
                yield_chg = sim.get("yield_change_percent", 0.0)
                st.metric(
                    "🌿 Yield Change",
                    f"{yield_chg:+.1f}%",
                    delta_color="normal",
                )

            # What changed table
            with st.expander("📐 Detailed Numbers", expanded=False):
                detail = {
                    "Parameter": ["Area (k-ha)", "Yield (kg/ha)", "Production (k-tonnes)"],
                    f"Baseline ({ref_year})": [
                        _fmt(sim.get("baseline_area")),
                        _fmt(sim.get("baseline_yield")),
                        _fmt(b_prod),
                    ],
                    f"Scenario ({tgt_year})": [
                        _fmt(sim.get("new_area")),
                        _fmt(sim.get("new_yield")),
                        _fmt(n_prod),
                    ],
                    "Change": [
                        f"{area_chg:+.1f}%",
                        f"{yield_chg:+.1f}%",
                        f"{pct:+.1f}%" if pct is not None else "N/A",
                    ],
                }
                st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)

            # Charts
            try:
                fig_bvs = plot_baseline_vs_scenario(sim)
                if fig_bvs:
                    st.plotly_chart(fig_bvs, use_container_width=True)
            except Exception:
                pass

            try:
                fig_ay = plot_area_yield_comparison(sim)
                if fig_ay:
                    st.plotly_chart(fig_ay, use_container_width=True)
            except Exception:
                pass

        # Comparison chart (if multiple scenarios)
        if intent == "comparison" and result.get("comparison_result"):
            st.subheader("🔀 Scenario Comparison")
            cmp = result["comparison_result"]
            cmp_scenarios = cmp.get("scenarios", [])
            if cmp_scenarios:
                try:
                    cmp_df = create_comparison_dataframe(cmp_scenarios)
                    fig_cmp = plot_scenario_comparison(cmp_df)
                    if fig_cmp:
                        st.plotly_chart(fig_cmp, use_container_width=True)
                    st.dataframe(cmp_df.style.format(precision=2), use_container_width=True)
                except Exception:
                    pass

        # AI explanation
        explanation = result.get("explanation", "")
        if explanation:
            st.subheader("🤖 AI Explanation")
            st.write(explanation)

    # ── Error / fallback ───────────────────────────────────────────────────────
    elif status in ("error", "failed", "invalid"):
        st.error(f"❌ Could not process request. {result.get('error', '')}")
        explanation = result.get("explanation", "")
        if explanation:
            st.write(explanation)

    # ── Assumptions used ───────────────────────────────────────────────────────
    assumptions = result.get("extracted_assumptions", [])
    if assumptions and intent in ("scenario_simulation", "comparison"):
        with st.expander("🔍 Assumptions extracted by AI", expanded=False):
            for a in assumptions:
                st.json(a)

else:
    # Landing state
    st.info(
        "👆 Enter your farming scenario above and click **Analyse My Scenario**, "
        "or use the **Manual Scenario Mode** in the sidebar for a quick calculation."
    )
    st.markdown(
        """
        **What you can ask:**
        - *What if rice area in Karnataka increases by 10% and yield decreases by 5% in 2030?*
        - *If wheat in Punjab expands by 20%, what happens to production?*
        - *Compare two scenarios: maize in Maharashtra with area +10% vs area -10%.*
        - *Show historical sugarcane production trends in Maharashtra.*

        **Available data:** Years 2010–2017 · 20 States · 29 Crops
        """
    )
