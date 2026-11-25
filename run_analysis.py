"""
PS 137 – Agricultural Production Scenario Coordinator
======================================================
run_analysis.py – Member 2 Analysis Runner

PURPOSE
-------
Runs the complete Member 2 historical analysis pipeline:
  1. Load Member 1's cleaned dataset.
  2. Generate dataset overview.
  3. Year-wise analysis.
  4. State-wise analysis.
  5. Crop-wise analysis.
  6. Generate charts.
  7. Generate historical insights.
  8. Save all outputs to reports/member2/.

HOW TO RUN
----------
  python run_analysis.py

REQUIREMENTS
------------
  - Member 1's pipeline must have been run first (python main.py).
  - The file data/processed/cleaned_agriculture_dataset.csv must exist
    and contain actual records.
"""

import sys
import warnings
from pathlib import Path

# ── Ensure project root is on the path ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REPORTS_DIR = PROJECT_ROOT / "reports" / "member2"
PLOTS_DIR   = REPORTS_DIR / "plots"


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def main() -> None:
    section("PS 137 – Member 2 Historical Analysis Runner")
    print(f"  Project root : {PROJECT_ROOT}")

    # ── 0. Import member 2 module ─────────────────────────────────────────────
    try:
        from src.analysis.historical_analysis import (
            load_cleaned_data,
            validate_dataset,
            generate_dataset_overview,
            analyse_year_wise,
            analyse_state_wise,
            analyse_crop_wise,
            analyse_crop_trends,
            identify_crop_changes,
            generate_historical_insights,
            save_all_outputs,
            plot_yearly_production,
            plot_yearly_area,
            plot_yearly_weighted_yield,
            plot_yoy_pct_change,
            plot_state_production,
            plot_top10_states,
            plot_state_heatmap,
            plot_crop_production,
            plot_crop_area,
            plot_crop_weighted_yield,
            plot_crop_trends,
            _setup_matplotlib,
            AnalysisPolicy,
        )
    except ImportError as e:
        print(f"[ERROR] Cannot import historical_analysis module: {e}")
        sys.exit(1)

    # Configure matplotlib
    _setup_matplotlib()

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    section("STEP 1 – Loading cleaned dataset")
    try:
        df = load_cleaned_data()
        print(f"  Loaded : {len(df):,} rows × {df.shape[1]} columns")
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}")
        print("\n  Ensure Member 1's pipeline has been run successfully:")
        print("    python main.py")
        sys.exit(1)

    # ── 2. Dataset overview ───────────────────────────────────────────────────
    section("STEP 2 – Dataset overview")
    overview_text = generate_dataset_overview(df)
    print(overview_text[:1500])  # Print first part to console
    if len(overview_text) > 1500:
        print("  ... (truncated – see dataset_overview.txt for full report)")

    # ── 3. Year-wise analysis ─────────────────────────────────────────────────
    section("STEP 3 – Year-wise analysis")
    policy = AnalysisPolicy(exclude_zero_area=True, exclude_zero_production=False)
    year_df = analyse_year_wise(df, policy=policy)
    print(f"\n  Year-wise summary ({len(year_df)} years):")
    print(year_df[["year", "total_production_1000_t", "total_area_1000_ha",
                   "weighted_yield_kg_per_ha", "yoy_production_pct_change"]].to_string(index=False))

    # ── 4. State-wise analysis ────────────────────────────────────────────────
    section("STEP 4 – State-wise analysis")
    state_df = analyse_state_wise(df, policy=policy)
    print(f"\n  State-wise summary ({len(state_df)} states):")
    print(state_df[["state", "total_production_1000_t", "total_area_1000_ha",
                   "weighted_yield_kg_per_ha", "production_share_pct"]].to_string(index=False))

    # ── 5. Crop-wise analysis ─────────────────────────────────────────────────
    section("STEP 5 – Crop-wise analysis")
    crop_df = analyse_crop_wise(df, policy=policy)
    print(f"\n  Crop-wise summary ({len(crop_df)} crops):")
    print(crop_df[["crop", "total_production_1000_t", "total_area_1000_ha",
                   "weighted_yield_kg_per_ha", "production_share_pct"]].to_string(index=False))

    crop_changes = identify_crop_changes(df)
    if crop_changes.get("largest_increase"):
        name, abs_ch, pct_ch = crop_changes["largest_increase"]
        print(f"\n  Largest crop increase : {name} (+{abs_ch:,.2f} thousand tonnes)")
    if crop_changes.get("largest_decrease"):
        name, abs_ch, pct_ch = crop_changes["largest_decrease"]
        print(f"  Largest crop decrease : {name} ({abs_ch:,.2f} thousand tonnes)")

    # ── 6. Historical insights ────────────────────────────────────────────────
    section("STEP 6 – Historical insights")
    insights_text = generate_historical_insights(df, year_df, state_df, crop_df)
    print(insights_text[:2000])
    if len(insights_text) > 2000:
        print("  ... (see historical_insights.txt for full report)")

    # ── 7. Save tabular outputs ───────────────────────────────────────────────
    section("STEP 7 – Saving tabular reports")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    saved_files = save_all_outputs(
        overview_text=overview_text,
        year_df=year_df,
        state_df=state_df,
        crop_df=crop_df,
        insights_text=insights_text,
        reports_dir=REPORTS_DIR,
    )

    # ── 8. Generate charts ────────────────────────────────────────────────────
    section("STEP 8 – Generating charts")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    chart_results = {}

    def try_plot(name: str, fn, *args, **kwargs):
        try:
            path = fn(*args, **kwargs)
            if path:
                chart_results[name] = path
                print(f"  [OK] Saved : {path.name}")
        except Exception as e:
            print(f"  [WARN] Chart '{name}' failed: {e}")
            chart_results[name] = None

    try_plot("yearly_production",    plot_yearly_production,    year_df,  PLOTS_DIR)
    try_plot("yearly_area",          plot_yearly_area,          year_df,  PLOTS_DIR)
    try_plot("yearly_yield",         plot_yearly_weighted_yield, year_df, PLOTS_DIR)
    try_plot("yoy_change",           plot_yoy_pct_change,       year_df,  PLOTS_DIR)
    try_plot("state_production",     plot_state_production,     state_df, PLOTS_DIR)
    try_plot("top10_states",         plot_top10_states,         state_df, PLOTS_DIR)
    try_plot("state_heatmap",        plot_state_heatmap,        df,       PLOTS_DIR)
    try_plot("crop_production",      plot_crop_production,      crop_df,  PLOTS_DIR)
    try_plot("crop_area",            plot_crop_area,            crop_df,  PLOTS_DIR)
    try_plot("crop_yield",           plot_crop_weighted_yield,  crop_df,  PLOTS_DIR)
    try_plot("crop_trends",          plot_crop_trends,          df,       PLOTS_DIR)

    charts_ok    = sum(1 for v in chart_results.values() if v is not None)
    charts_failed = sum(1 for v in chart_results.values() if v is None)

    # ── 9. Summary ────────────────────────────────────────────────────────────
    section("ANALYSIS COMPLETE – EXECUTION SUMMARY")
    print(f"\n  Dataset:")
    print(f"    Records loaded           : {len(df):,}")
    print(f"    States                   : {df['state_clean'].nunique() if 'state_clean' in df.columns else df['state'].nunique()}")
    print(f"    Districts                : {df['district_clean'].nunique() if 'district_clean' in df.columns else df['district'].nunique()}")
    print(f"    Crops                    : {df['crop_clean'].nunique() if 'crop_clean' in df.columns else df['crop'].nunique()}")
    print(f"    Year range               : {year_df['year'].min()} – {year_df['year'].max()}")

    print(f"\n  Analysis outputs saved to : {REPORTS_DIR}")
    for name, path in saved_files.items():
        print(f"    {path.name}")

    print(f"\n  Charts saved to : {PLOTS_DIR}")
    print(f"    Generated successfully : {charts_ok}")
    print(f"    Failed (see warnings)  : {charts_failed}")
    if charts_failed > 0:
        for name, v in chart_results.items():
            if v is None:
                print(f"      SKIPPED: {name}")

    print(f"\n  Commands to use baseline retrieval (for Member 3):")
    print(f"    from src.analysis.historical_analysis import load_cleaned_data, get_historical_baseline")
    print(f"    df = load_cleaned_data()")
    print(f"    result = get_historical_baseline(df, 'Karnataka', 'Mysore', 'Rice', 2015)")
    print()


if __name__ == "__main__":
    main()
