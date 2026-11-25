"""
PS 137 – Agricultural Production Scenario Coordinator
======================================================
src/analysis/historical_analysis.py
Member 2 – Historical Analysis and Trend Identification

PURPOSE
-------
Provides comprehensive historical agricultural analysis using the
cleaned dataset produced by Member 1's pipeline.

Responsibilities:
  - Load and validate the cleaned dataset.
  - Perform year-wise, state-wise, and crop-wise trend analysis.
  - Generate charts and analytical tables.
  - Provide a baseline retrieval interface for Member 3.

RULES
-----
  - Never fabricate data. All results come from the cleaned CSV.
  - Never modify the cleaned dataset.
  - Use weighted yield, not simple averages, when aggregating.
  - Document exclusions explicitly.
  - Do not call historical trends forecasts.

UNITS (inherited from Member 1)
  - area_1000_ha        : Thousand hectares
  - production_1000_t   : Thousand tonnes
  - yield_kg_per_ha     : Kilograms per hectare
  - Weighted yield formula: (sum_production_1000_t * 1000) / sum_area_1000_ha

HOW TO RUN
----------
  python run_analysis.py
  python src/analysis/historical_analysis.py  # (runs standalone demo)
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ── Locate project root ───────────────────────────────────────────────────────
# This file is at:  <project_root>/src/analysis/historical_analysis.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEANED_CSV = PROJECT_ROOT / "data" / "processed" / "cleaned_agriculture_dataset.csv"
REPORTS_DIR = PROJECT_ROOT / "reports" / "member2"
PLOTS_DIR   = REPORTS_DIR / "plots"

# Required columns for analysis
REQUIRED_COLS = [
    "state", "district", "year", "crop",
    "area_1000_ha", "production_1000_t",
]
# Clean columns (may exist after preprocessing)
PREFERRED_TEXT_COLS = {
    "state"    : "state_clean",
    "district" : "district_clean",
    "crop"     : "crop_clean",
}

# ─────────────────────────────────────────────────────────────────────────────
# Analysis policy
# ─────────────────────────────────────────────────────────────────────────────
class AnalysisPolicy:
    """
    Configurable rules for handling records with data quality issues.

    Parameters
    ----------
    use_complete_only : bool
        If True, weighted-yield aggregations use only 'complete_positive'
        records (area > 0 AND production > 0).
        If False, all records with area > 0 and non-NaN production are used.
    exclude_zero_area : bool
        If True, records where area_1000_ha == 0 are excluded from
        area/production aggregations.  They are always excluded from yield.
    exclude_zero_production : bool
        If True, records where production_1000_t == 0 are excluded from
        production aggregations.
    min_area_threshold : float
        Records where area < this threshold (in 1000 ha) are excluded from
        yield calculations (avoids rounding artefacts).
    """
    def __init__(
        self,
        use_complete_only: bool = False,
        exclude_zero_area: bool = True,
        exclude_zero_production: bool = False,
        min_area_threshold: float = 0.0,
    ):
        self.use_complete_only       = use_complete_only
        self.exclude_zero_area       = exclude_zero_area
        self.exclude_zero_production = exclude_zero_production
        self.min_area_threshold      = min_area_threshold

    def __repr__(self) -> str:
        return (
            f"AnalysisPolicy("
            f"use_complete_only={self.use_complete_only}, "
            f"exclude_zero_area={self.exclude_zero_area}, "
            f"exclude_zero_production={self.exclude_zero_production}, "
            f"min_area_threshold={self.min_area_threshold})"
        )


DEFAULT_POLICY = AnalysisPolicy()


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def _section(title: str) -> None:
    """Print a visible section header."""
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def _resolve_text_col(df: pd.DataFrame, base: str) -> str:
    """
    Return the '_clean' version of a text column if it exists in df,
    otherwise fall back to the original column name.
    """
    preferred = PREFERRED_TEXT_COLS.get(base, base + "_clean")
    return preferred if preferred in df.columns else base


def _weighted_yield(production_sum: float, area_sum: float) -> float:
    """
    Compute weighted yield in kg/ha.

    Formula:
        yield_kg_per_ha = (sum_production_1000_t * 1000) / sum_area_1000_ha

    Both inputs are in thousands, so the ×1000 converts the
    production-to-area ratio from tonnes/ha to kg/ha.

    Returns NaN if area_sum <= 0.
    """
    if area_sum > 0:
        return (production_sum * 1_000) / area_sum
    return float("nan")


def _apply_policy_filter(df: pd.DataFrame, policy: AnalysisPolicy) -> Tuple[pd.DataFrame, dict]:
    """
    Apply the analysis policy to df and return the filtered DataFrame
    plus an exclusion summary dict.
    """
    original_count = len(df)
    excluded = {}

    working = df.copy()

    if policy.use_complete_only and "record_status" in working.columns:
        before = len(working)
        working = working[working["record_status"] == "complete_positive"]
        excluded["non_complete_positive"] = before - len(working)

    if policy.exclude_zero_area:
        before = len(working)
        working = working[working["area_1000_ha"].fillna(0) > 0]
        excluded["zero_area"] = before - len(working)

    if policy.exclude_zero_production:
        before = len(working)
        working = working[
            working["production_1000_t"].fillna(0) > 0
        ]
        excluded["zero_production"] = before - len(working)

    if policy.min_area_threshold > 0:
        before = len(working)
        working = working[
            working["area_1000_ha"].fillna(0) >= policy.min_area_threshold
        ]
        excluded["below_min_area"] = before - len(working)

    total_excluded = original_count - len(working)
    excluded["total_excluded"] = total_excluded
    excluded["total_used"]     = len(working)
    excluded["original_count"] = original_count

    return working, excluded


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading and validation
# ─────────────────────────────────────────────────────────────────────────────
def load_cleaned_data(csv_path: Union[str, Path, None] = None) -> pd.DataFrame:
    """
    Load the cleaned agriculture CSV produced by Member 1.

    Parameters
    ----------
    csv_path : str or Path, optional
        Override the default path.  Defaults to
        data/processed/cleaned_agriculture_dataset.csv.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame with the year column cast to int where possible.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(csv_path) if csv_path else CLEANED_CSV

    if not path.exists():
        raise FileNotFoundError(
            f"[Member 2] Cleaned CSV not found at:\n  {path}\n"
            "Run Member 1's pipeline first:\n"
            "  python main.py"
        )

    df = pd.read_csv(path, low_memory=False)

    if len(df) == 0:
        raise ValueError(
            f"[Member 2] Cleaned CSV at {path} is empty.\n"
            "Run Member 1's pipeline first:  python main.py"
        )

    # Validate required columns
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"[Member 2] Required columns missing from cleaned CSV: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Parse year column
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    # Try to store as Int64 (nullable integer) to preserve NaN
    try:
        df["year"] = df["year"].astype("Int64")
    except Exception:
        pass  # keep as float if conversion fails

    # Ensure numeric types for measurement columns
    for col in ["area_1000_ha", "production_1000_t", "yield_kg_per_ha"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Perform data validation checks and return a summary dictionary.

    Checks performed:
      - Required column presence
      - Data type verification
      - Missing value counts
      - Duplicate row count
      - Invalid/non-positive area and production
      - Year range
      - record_status distribution (if column exists)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
        Validation summary (suitable for printing or saving to a report).
    """
    summary = {}

    # Basic shape
    summary["total_records"]  = len(df)
    summary["total_columns"]  = len(df.columns)

    # Required columns
    summary["required_columns_present"] = [
        c for c in REQUIRED_COLS if c in df.columns
    ]
    summary["required_columns_missing"] = [
        c for c in REQUIRED_COLS if c not in df.columns
    ]

    # Year range
    if "year" in df.columns:
        years = df["year"].dropna()
        summary["year_min"]    = int(years.min()) if len(years) > 0 else None
        summary["year_max"]    = int(years.max()) if len(years) > 0 else None
        summary["year_values"] = sorted([int(y) for y in years.unique()])

    # Unique counts
    state_col    = _resolve_text_col(df, "state")
    district_col = _resolve_text_col(df, "district")
    crop_col     = _resolve_text_col(df, "crop")

    summary["unique_states"]    = int(df[state_col].nunique())    if state_col    in df.columns else None
    summary["unique_districts"] = int(df[district_col].nunique()) if district_col in df.columns else None
    summary["unique_crops"]     = int(df[crop_col].nunique())     if crop_col     in df.columns else None

    # Missing values
    summary["missing_values"] = df.isnull().sum()[df.isnull().sum() > 0].to_dict()

    # Duplicates
    summary["exact_duplicates"] = int(df.duplicated().sum())

    # Composite key duplicates
    key_cols = [c for c in ["state", "district", "year", "crop"] if c in df.columns]
    if len(key_cols) == 4:
        summary["composite_key_duplicates"] = int(df.duplicated(subset=key_cols).sum())

    # Invalid numeric records
    for col in ["area_1000_ha", "production_1000_t", "yield_kg_per_ha"]:
        if col in df.columns:
            summary[f"negative_{col}"]  = int((df[col] < 0).sum())
            summary[f"zero_{col}"]      = int((df[col] == 0).sum())
            summary[f"missing_{col}"]   = int(df[col].isna().sum())

    # record_status
    if "record_status" in df.columns:
        summary["record_status_distribution"] = (
            df["record_status"].value_counts().to_dict()
        )

    # Unique names
    if state_col in df.columns:
        summary["unique_state_names"] = sorted(df[state_col].dropna().unique().tolist())
    if crop_col in df.columns:
        summary["unique_crop_names"] = sorted(df[crop_col].dropna().unique().tolist())

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset overview
# ─────────────────────────────────────────────────────────────────────────────
def generate_dataset_overview(df: pd.DataFrame) -> str:
    """
    Generate a formatted text overview of the dataset.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    str
        Multi-line text report (save to dataset_overview.txt).
    """
    summary = validate_dataset(df)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "PS 137 – Agricultural Production Scenario Coordinator",
        "MEMBER 2 – DATASET OVERVIEW REPORT",
        f"Generated : {ts}",
        "=" * 65,
        "",
        "1. BASIC DIMENSIONS",
        f"   Total records  : {summary['total_records']:,}",
        f"   Total columns  : {summary['total_columns']}",
        f"   States         : {summary.get('unique_states', 'N/A')}",
        f"   Districts      : {summary.get('unique_districts', 'N/A')}",
        f"   Crops          : {summary.get('unique_crops', 'N/A')}",
        f"   Year range     : {summary.get('year_min', 'N/A')} – {summary.get('year_max', 'N/A')}",
        f"   Years available: {summary.get('year_values', 'N/A')}",
        "",
        "2. DATA QUALITY",
        f"   Exact duplicates         : {summary.get('exact_duplicates', 'N/A'):,}"
            if isinstance(summary.get('exact_duplicates'), int) else
            f"   Exact duplicates         : {summary.get('exact_duplicates', 'N/A')}",
        f"   Composite key duplicates : {summary.get('composite_key_duplicates', 'N/A')}",
        "",
        "3. MISSING VALUES",
    ]

    mv = summary.get("missing_values", {})
    if mv:
        for col, cnt in mv.items():
            pct = cnt / summary["total_records"] * 100
            lines.append(f"   {col:40s} : {cnt:>8,}  ({pct:.2f}%)")
    else:
        lines.append("   None")

    lines += ["", "4. NUMERIC COLUMN SUMMARY"]
    for col in ["area_1000_ha", "production_1000_t", "yield_kg_per_ha"]:
        if col in df.columns:
            series = df[col].dropna()
            lines.append(f"   {col}")
            lines.append(f"     Count   : {len(series):,}")
            lines.append(f"     Min     : {series.min():.4f}")
            lines.append(f"     Max     : {series.max():.4f}")
            lines.append(f"     Mean    : {series.mean():.4f}")
            lines.append(f"     Median  : {series.median():.4f}")
            lines.append(f"     Missing : {summary.get(f'missing_{col}', 'N/A')}")

    lines += ["", "5. RECORD STATUS DISTRIBUTION"]
    rsd = summary.get("record_status_distribution", {})
    if rsd:
        total = summary["total_records"]
        for status, cnt in sorted(rsd.items(), key=lambda x: -x[1]):
            pct = cnt / total * 100
            lines.append(f"   {status:45s} : {cnt:>8,}  ({pct:.1f}%)")
    else:
        lines.append("   record_status column not found.")

    lines += ["", "6. UNIQUE STATE NAMES"]
    for s in summary.get("unique_state_names", []):
        lines.append(f"   {s}")

    lines += ["", "7. UNIQUE CROP NAMES"]
    for c in summary.get("unique_crop_names", []):
        lines.append(f"   {c}")

    lines += [
        "",
        "=" * 65,
        "END OF OVERVIEW REPORT",
        "",
        "COLUMN SCHEMA",
        "  Original columns (from Member 1):",
        "    state               – State name (raw)",
        "    district            – District name (raw)",
        "    year                – Calendar year (integer)",
        "    crop                – Crop name (raw)",
        "    area_1000_ha        – Area in thousand hectares",
        "    production_1000_t   – Production in thousand tonnes",
        "    yield_kg_per_ha     – Yield in kg/ha (original recorded value)",
        "  Derived columns (added by Member 1 preprocessing):",
        "    state_clean         – Title-cased, corrected state name",
        "    district_clean      – Title-cased district name",
        "    crop_clean          – Title-cased crop name",
        "    calculated_yield_kg_per_ha – Yield from formula",
        "    yield_discrepancy   – True if recorded vs calculated differ > 10 kg/ha",
        "    has_production_data – True if production_1000_t is not NaN",
        "    is_zero_activity    – True if area=0 AND production=0",
        "    is_small_area_rounding – True if 0 < area < 0.1",
        "    record_status       – Quality classification of each record",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Year-wise historical analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyse_year_wise(
    df: pd.DataFrame,
    policy: AnalysisPolicy = DEFAULT_POLICY,
) -> pd.DataFrame:
    """
    Compute year-wise aggregated statistics.

    Returns a DataFrame with one row per year containing:
      - year
      - total_production_1000_t
      - total_area_1000_ha
      - weighted_yield_kg_per_ha   (= total_production*1000 / total_area)
      - record_count
      - yoy_production_change      (year-over-year absolute change)
      - yoy_production_pct_change  (year-over-year percentage change)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset from Member 1.
    policy : AnalysisPolicy
        Rules for which records to include.

    Returns
    -------
    pd.DataFrame
        Year-wise summary sorted by year.
    """
    working, exclusions = _apply_policy_filter(df, policy)

    # Only keep rows where year, area, and production are present
    agg_df = working.dropna(subset=["year", "area_1000_ha", "production_1000_t"])
    n_excluded_null = len(working) - len(agg_df)

    print(f"\n  [Year-wise] Records used         : {len(agg_df):,}")
    print(f"  [Year-wise] Excluded (null y/a/p): {n_excluded_null:,}")
    print(f"  [Year-wise] Policy exclusions    : {exclusions.get('total_excluded', 0):,}")

    grouped = agg_df.groupby("year", as_index=False).agg(
        total_production_1000_t=("production_1000_t", "sum"),
        total_area_1000_ha=("area_1000_ha", "sum"),
        record_count=("production_1000_t", "count"),
    )
    grouped = grouped.sort_values("year").reset_index(drop=True)

    # Weighted yield
    grouped["weighted_yield_kg_per_ha"] = grouped.apply(
        lambda r: _weighted_yield(r["total_production_1000_t"], r["total_area_1000_ha"]),
        axis=1,
    )

    # Year-over-year changes
    grouped["yoy_production_change"] = grouped["total_production_1000_t"].diff()
    grouped["yoy_production_pct_change"] = grouped["total_production_1000_t"].pct_change() * 100

    # Handle pct change when previous value is zero (avoid inf)
    inf_mask = grouped["yoy_production_pct_change"].isin([float("inf"), float("-inf")])
    grouped.loc[inf_mask, "yoy_production_pct_change"] = float("nan")

    return grouped


# ─────────────────────────────────────────────────────────────────────────────
# 4. State-wise historical analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyse_state_wise(
    df: pd.DataFrame,
    policy: AnalysisPolicy = DEFAULT_POLICY,
) -> pd.DataFrame:
    """
    Compute state-wise aggregated statistics.

    Returns a DataFrame with one row per state containing:
      - state (resolved to state_clean if available)
      - total_production_1000_t
      - total_area_1000_ha
      - weighted_yield_kg_per_ha
      - record_count
      - production_share_pct  (share of national total production)

    Parameters
    ----------
    df : pd.DataFrame
    policy : AnalysisPolicy

    Returns
    -------
    pd.DataFrame
        State-wise summary sorted by total production descending.
    """
    state_col = _resolve_text_col(df, "state")
    working, exclusions = _apply_policy_filter(df, policy)
    agg_df = working.dropna(subset=[state_col, "area_1000_ha", "production_1000_t"])

    print(f"\n  [State-wise] Records used : {len(agg_df):,}")
    print(f"  [State-wise] Policy excl  : {exclusions.get('total_excluded', 0):,}")

    grouped = agg_df.groupby(state_col, as_index=False).agg(
        total_production_1000_t=(  "production_1000_t", "sum"),
        total_area_1000_ha     =(  "area_1000_ha",      "sum"),
        record_count           =(  "production_1000_t", "count"),
    )
    grouped.rename(columns={state_col: "state"}, inplace=True)

    grouped["weighted_yield_kg_per_ha"] = grouped.apply(
        lambda r: _weighted_yield(r["total_production_1000_t"], r["total_area_1000_ha"]),
        axis=1,
    )

    total_prod = grouped["total_production_1000_t"].sum()
    grouped["production_share_pct"] = (
        grouped["total_production_1000_t"] / total_prod * 100
        if total_prod > 0 else float("nan")
    )

    return grouped.sort_values("total_production_1000_t", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Crop-wise historical analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyse_crop_wise(
    df: pd.DataFrame,
    policy: AnalysisPolicy = DEFAULT_POLICY,
) -> pd.DataFrame:
    """
    Compute crop-wise aggregated statistics.

    Returns a DataFrame with one row per crop containing:
      - crop (resolved to crop_clean if available)
      - total_production_1000_t
      - total_area_1000_ha
      - weighted_yield_kg_per_ha
      - record_count
      - production_share_pct

    Parameters
    ----------
    df : pd.DataFrame
    policy : AnalysisPolicy

    Returns
    -------
    pd.DataFrame
        Crop-wise summary sorted by total production descending.
    """
    crop_col = _resolve_text_col(df, "crop")
    working, exclusions = _apply_policy_filter(df, policy)
    agg_df = working.dropna(subset=[crop_col, "area_1000_ha", "production_1000_t"])

    print(f"\n  [Crop-wise] Records used : {len(agg_df):,}")
    print(f"  [Crop-wise] Policy excl  : {exclusions.get('total_excluded', 0):,}")

    grouped = agg_df.groupby(crop_col, as_index=False).agg(
        total_production_1000_t=("production_1000_t", "sum"),
        total_area_1000_ha     =("area_1000_ha",      "sum"),
        record_count           =("production_1000_t", "count"),
    )
    grouped.rename(columns={crop_col: "crop"}, inplace=True)

    grouped["weighted_yield_kg_per_ha"] = grouped.apply(
        lambda r: _weighted_yield(r["total_production_1000_t"], r["total_area_1000_ha"]),
        axis=1,
    )

    total_prod = grouped["total_production_1000_t"].sum()
    grouped["production_share_pct"] = (
        grouped["total_production_1000_t"] / total_prod * 100
        if total_prod > 0 else float("nan")
    )

    return grouped.sort_values("total_production_1000_t", ascending=False).reset_index(drop=True)


def analyse_crop_trends(
    df: pd.DataFrame,
    policy: AnalysisPolicy = DEFAULT_POLICY,
) -> pd.DataFrame:
    """
    Compute crop-wise production by year (pivot: crops × years).

    Also identifies crops with the largest observed production
    increase and decrease across the available time period.

    Returns
    -------
    pd.DataFrame
        Pivot table with crops as index and years as columns,
        values = total_production_1000_t.
    """
    crop_col = _resolve_text_col(df, "crop")
    working, _ = _apply_policy_filter(df, policy)
    agg_df = working.dropna(subset=[crop_col, "year", "production_1000_t"])

    pivot = agg_df.pivot_table(
        values="production_1000_t",
        index=crop_col,
        columns="year",
        aggfunc="sum",
    )
    return pivot


def identify_crop_changes(
    df: pd.DataFrame,
    policy: AnalysisPolicy = DEFAULT_POLICY,
) -> dict:
    """
    Identify crops with the largest observed production change.

    Compares first-available year vs last-available year for each crop.

    Returns
    -------
    dict with keys:
        largest_increase  : (crop_name, absolute_change, pct_change)
        largest_decrease  : (crop_name, absolute_change, pct_change)
        all_changes       : pd.DataFrame with per-crop changes
    """
    crop_col = _resolve_text_col(df, "crop")
    working, _ = _apply_policy_filter(df, policy)
    agg_df = working.dropna(subset=[crop_col, "year", "production_1000_t"])

    yearly_crop = agg_df.groupby([crop_col, "year"])["production_1000_t"].sum().reset_index()
    yearly_crop.columns = ["crop", "year", "production"]

    changes = []
    for crop, grp in yearly_crop.groupby("crop"):
        grp_sorted = grp.sort_values("year")
        if len(grp_sorted) < 2:
            continue
        first_val = grp_sorted["production"].iloc[0]
        last_val  = grp_sorted["production"].iloc[-1]
        abs_change = last_val - first_val
        if first_val > 0:
            pct_change = (abs_change / first_val) * 100
        elif last_val > 0:
            pct_change = float("inf")  # grew from zero
        else:
            pct_change = float("nan")
        changes.append({
            "crop"           : crop,
            "first_year_prod": first_val,
            "last_year_prod" : last_val,
            "abs_change"     : abs_change,
            "pct_change"     : pct_change,
        })

    if not changes:
        return {"largest_increase": None, "largest_decrease": None, "all_changes": pd.DataFrame()}

    changes_df = pd.DataFrame(changes)

    # Replace inf with nan for sorting
    sortable = changes_df.copy()
    sortable["pct_change"] = sortable["pct_change"].replace([float("inf"), float("-inf")], float("nan"))

    inc_row = sortable.loc[sortable["abs_change"].idxmax()]
    dec_row = sortable.loc[sortable["abs_change"].idxmin()]

    return {
        "largest_increase" : (inc_row["crop"], inc_row["abs_change"], inc_row["pct_change"]),
        "largest_decrease" : (dec_row["crop"], dec_row["abs_change"], dec_row["pct_change"]),
        "all_changes"      : changes_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Filtering functions
# ─────────────────────────────────────────────────────────────────────────────
def filter_data(
    df: pd.DataFrame,
    states: Optional[List[str]] = None,
    districts: Optional[List[str]] = None,
    crops: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    case_insensitive: bool = True,
) -> pd.DataFrame:
    """
    Filter the dataset by one or more criteria.

    Uses _clean columns (state_clean, district_clean, crop_clean)
    when available to handle capitalisation differences.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.
    states : list of str, optional
        State names to include. Partial or exact matching.
    districts : list of str, optional
        District names to include.
    crops : list of str, optional
        Crop names to include.
    years : list of int, optional
        Years to include.
    case_insensitive : bool
        If True, comparisons are case-insensitive.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame. If empty, prints a warning.
    """
    result = df.copy()

    def _match(series: pd.Series, values: List[str]) -> pd.Series:
        if case_insensitive:
            lower_vals = [v.lower() for v in values]
            return series.str.lower().isin(lower_vals)
        return series.isin(values)

    if states is not None:
        col = _resolve_text_col(result, "state")
        if col in result.columns:
            result = result[_match(result[col].fillna(""), states)]
        else:
            warnings.warn("State column not found; state filter skipped.")

    if districts is not None:
        col = _resolve_text_col(result, "district")
        if col in result.columns:
            result = result[_match(result[col].fillna(""), districts)]
        else:
            warnings.warn("District column not found; district filter skipped.")

    if crops is not None:
        col = _resolve_text_col(result, "crop")
        if col in result.columns:
            result = result[_match(result[col].fillna(""), crops)]
        else:
            warnings.warn("Crop column not found; crop filter skipped.")

    if years is not None:
        if "year" in result.columns:
            result = result[result["year"].isin(years)]
        else:
            warnings.warn("Year column not found; year filter skipped.")

    if len(result) == 0:
        print(
            f"\n  [WARNING] No records match the requested filter:\n"
            f"    states={states}, districts={districts}, "
            f"crops={crops}, years={years}"
        )

    return result.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Historical baseline retrieval for Member 3
# ─────────────────────────────────────────────────────────────────────────────
def get_historical_baseline(
    df: pd.DataFrame,
    state: str,
    district: str,
    crop: str,
    year: int,
) -> dict:
    """
    Retrieve the actual historical record for a given
    (state, district, crop, year) combination.

    This is Member 3's primary integration point.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset loaded via load_cleaned_data().
    state : str
        State name (case-insensitive; matched against state_clean if available).
    district : str
        District name (case-insensitive).
    crop : str
        Crop name (case-insensitive).
    year : int
        Calendar year.

    Returns
    -------
    dict with keys:
        found          : bool – True if exactly one record matched.
        match_count    : int  – number of matching records found.
        state          : str  – matched state name (from clean column).
        district       : str  – matched district name.
        crop           : str  – matched crop name.
        year           : int  – year of the record.
        area_1000_ha   : float or None – area in thousand hectares.
        production_1000_t : float or None – production in thousand tonnes.
        yield_kg_per_ha   : float or None – original recorded yield.
        calculated_yield_kg_per_ha : float or None – formula-derived yield.
        record_status  : str or None – quality classification.
        has_production_data : bool or None.
        units          : dict – unit descriptions.
        source_records : list of dicts – all matching raw records.
        warning        : str or None – explanation if 0 or >1 records found.

    Notes
    -----
    - Returns found=False if no match exists. Never fabricates values.
    - If multiple records match, returns found=False with all matches in
      source_records (caller must decide how to handle duplicates).
    - Use get_aggregated_baseline() when geographic aggregation is intended.
    """
    state_col    = _resolve_text_col(df, "state")
    district_col = _resolve_text_col(df, "district")
    crop_col     = _resolve_text_col(df, "crop")

    matched = df.copy()

    # Case-insensitive matching
    if state_col in matched.columns:
        matched = matched[matched[state_col].str.lower().fillna("") == state.lower()]
    if district_col in matched.columns:
        matched = matched[matched[district_col].str.lower().fillna("") == district.lower()]
    if crop_col in matched.columns:
        matched = matched[matched[crop_col].str.lower().fillna("") == crop.lower()]
    if "year" in matched.columns:
        matched = matched[matched["year"] == year]

    units = {
        "area"       : "Thousand hectares (1000 ha)",
        "production" : "Thousand tonnes (1000 t)",
        "yield"      : "Kilograms per hectare (kg/ha)",
    }

    source_records = matched.to_dict(orient="records")
    n = len(matched)

    base = {
        "found"      : False,
        "match_count": n,
        "state"      : state,
        "district"   : district,
        "crop"       : crop,
        "year"       : year,
        "area_1000_ha"              : None,
        "production_1000_t"         : None,
        "yield_kg_per_ha"           : None,
        "calculated_yield_kg_per_ha": None,
        "record_status"             : None,
        "has_production_data"       : None,
        "units"                     : units,
        "source_records"            : source_records,
        "warning"                   : None,
    }

    if n == 0:
        base["warning"] = (
            f"No historical record found for: "
            f"state='{state}', district='{district}', "
            f"crop='{crop}', year={year}. "
            "Do not fabricate a baseline."
        )
        return base

    if n > 1:
        base["match_count"] = n
        base["warning"] = (
            f"{n} records found for state='{state}', district='{district}', "
            f"crop='{crop}', year={year}. "
            "Possible duplicate. All records returned in source_records. "
            "Caller should decide how to handle duplicates."
        )
        return base

    # Exactly one match – return values
    row = matched.iloc[0]
    base["found"]   = True
    base["state"]   = str(row.get(state_col,    state))
    base["district"]= str(row.get(district_col, district))
    base["crop"]    = str(row.get(crop_col,     crop))
    base["year"]    = int(row["year"]) if pd.notna(row.get("year")) else year

    def _safe_float(val):
        return float(val) if pd.notna(val) else None

    base["area_1000_ha"]               = _safe_float(row.get("area_1000_ha"))
    base["production_1000_t"]          = _safe_float(row.get("production_1000_t"))
    base["yield_kg_per_ha"]            = _safe_float(row.get("yield_kg_per_ha"))
    base["calculated_yield_kg_per_ha"] = _safe_float(row.get("calculated_yield_kg_per_ha"))
    base["record_status"]              = str(row["record_status"]) if "record_status" in row.index and pd.notna(row.get("record_status")) else None
    base["has_production_data"]        = bool(row["has_production_data"]) if "has_production_data" in row.index else None

    return base


def get_aggregated_baseline(
    df: pd.DataFrame,
    crop: str,
    year: int,
    state: Optional[str] = None,
    district: Optional[str] = None,
) -> dict:
    """
    Retrieve aggregated historical statistics for a given crop and year,
    optionally filtered to a state or district.

    Use this when Member 3 wants a geographic aggregate (e.g., all districts
    of a state) rather than a single district record.

    Parameters
    ----------
    df : pd.DataFrame
    crop : str – crop name (case-insensitive).
    year : int – calendar year.
    state : str, optional – restrict to this state.
    district : str, optional – restrict to this district (within state).

    Returns
    -------
    dict with keys:
        found                   : bool
        level                   : "district" | "state" | "national"
        state                   : str or None
        district                : str or None
        crop                    : str
        year                    : int
        record_count            : int
        total_area_1000_ha      : float or None
        total_production_1000_t : float or None
        weighted_yield_kg_per_ha: float or None
        units                   : dict
        source_rows_count       : int
        warning                 : str or None

    Notes
    -----
    This function AGGREGATES by summing area and production.
    It does NOT mix aggregated and district-level data.
    Yield is computed as weighted average, not simple mean.
    """
    crop_col     = _resolve_text_col(df, "crop")
    state_col    = _resolve_text_col(df, "state")
    district_col = _resolve_text_col(df, "district")

    matched = df.copy()
    matched = matched[matched[crop_col].str.lower().fillna("") == crop.lower()]
    matched = matched[matched["year"] == year]

    level = "national"
    if state is not None:
        matched = matched[matched[state_col].str.lower().fillna("") == state.lower()]
        level = "state"
    if district is not None:
        matched = matched[matched[district_col].str.lower().fillna("") == district.lower()]
        level = "district"

    units = {
        "area"       : "Thousand hectares (1000 ha)",
        "production" : "Thousand tonnes (1000 t)",
        "yield"      : "Kilograms per hectare (kg/ha)",
    }

    result = {
        "found"                   : False,
        "level"                   : level,
        "state"                   : state,
        "district"                : district,
        "crop"                    : crop,
        "year"                    : year,
        "record_count"            : 0,
        "total_area_1000_ha"      : None,
        "total_production_1000_t" : None,
        "weighted_yield_kg_per_ha": None,
        "units"                   : units,
        "source_rows_count"       : len(matched),
        "warning"                 : None,
    }

    if len(matched) == 0:
        result["warning"] = (
            f"No records found for crop='{crop}', year={year}"
            + (f", state='{state}'" if state else "")
            + (f", district='{district}'" if district else "")
        )
        return result

    # Only aggregate rows where area and production are both valid
    agg_df = matched.dropna(subset=["area_1000_ha", "production_1000_t"])
    agg_df = agg_df[agg_df["area_1000_ha"] > 0]

    total_area = agg_df["area_1000_ha"].sum()
    total_prod = agg_df["production_1000_t"].sum()

    result["found"]                   = True
    result["record_count"]            = len(agg_df)
    result["total_area_1000_ha"]      = float(total_area) if total_area > 0 else None
    result["total_production_1000_t"] = float(total_prod)
    result["weighted_yield_kg_per_ha"]= _weighted_yield(total_prod, total_area) if total_area > 0 else None

    if len(matched) > len(agg_df):
        n_excluded = len(matched) - len(agg_df)
        result["warning"] = (
            f"{n_excluded} records excluded from aggregation due to "
            "missing or zero area/production."
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 8. Historical insights
# ─────────────────────────────────────────────────────────────────────────────
def generate_historical_insights(
    df: pd.DataFrame,
    year_df: pd.DataFrame,
    state_df: pd.DataFrame,
    crop_df: pd.DataFrame,
) -> str:
    """
    Generate factual analytical insights from actual dataset aggregations.

    All insights are derived from real data. Hypothetical or future scenarios
    are explicitly excluded.

    Parameters
    ----------
    df         : raw cleaned DataFrame
    year_df    : output of analyse_year_wise()
    state_df   : output of analyse_state_wise()
    crop_df    : output of analyse_crop_wise()

    Returns
    -------
    str : formatted insights text report
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    crop_changes = identify_crop_changes(df)

    lines = [
        "PS 137 – Agricultural Production Scenario Coordinator",
        "MEMBER 2 – HISTORICAL INSIGHTS REPORT",
        f"Generated : {ts}",
        "=" * 65,
        "",
        "IMPORTANT DISCLAIMER",
        "  All observations below are based on HISTORICAL RECORDED DATA.",
        "  These are descriptive statistics and observed patterns.",
        "  They are NOT predictions, forecasts, or causal explanations.",
        "  Correlation observed in this data does not imply causation.",
        "",
        "=" * 65,
        "YEAR-WISE INSIGHTS",
        "=" * 65,
    ]

    if len(year_df) > 0:
        max_prod_row = year_df.loc[year_df["total_production_1000_t"].idxmax()]
        min_prod_row = year_df.loc[year_df["total_production_1000_t"].idxmin()]
        max_yield_row = year_df.loc[year_df["weighted_yield_kg_per_ha"].idxmax()]

        lines += [
            f"  Year with highest total production : {int(max_prod_row['year'])}",
            f"    Production : {max_prod_row['total_production_1000_t']:,.2f} thousand tonnes",
            f"    Area       : {max_prod_row['total_area_1000_ha']:,.2f} thousand hectares",
            f"    Yield      : {max_prod_row['weighted_yield_kg_per_ha']:,.2f} kg/ha",
            "",
            f"  Year with lowest total production  : {int(min_prod_row['year'])}",
            f"    Production : {min_prod_row['total_production_1000_t']:,.2f} thousand tonnes",
            "",
            f"  Year with highest weighted yield   : {int(max_yield_row['year'])}",
            f"    Yield      : {max_yield_row['weighted_yield_kg_per_ha']:,.2f} kg/ha",
            "",
        ]

        # Year-over-year changes
        yoy = year_df.dropna(subset=["yoy_production_pct_change"])
        if len(yoy) > 0:
            max_increase = yoy.loc[yoy["yoy_production_pct_change"].idxmax()]
            max_decrease = yoy.loc[yoy["yoy_production_pct_change"].idxmin()]
            lines += [
                f"  Largest year-over-year increase : {int(max_increase['year'])}",
                f"    Change : +{max_increase['yoy_production_pct_change']:.2f}%",
                f"  Largest year-over-year decrease : {int(max_decrease['year'])}",
                f"    Change : {max_decrease['yoy_production_pct_change']:.2f}%",
                "",
            ]

    lines += [
        "=" * 65,
        "STATE-WISE INSIGHTS",
        "=" * 65,
    ]
    if len(state_df) > 0:
        top5 = state_df.head(5)
        lines.append(f"  Top 5 states by total production (across all available years):")
        for _, row in top5.iterrows():
            lines.append(
                f"    {row['state']:30s}  "
                f"{row['total_production_1000_t']:>12,.2f} 000t  "
                f"({row['production_share_pct']:.1f}% of total)"
            )
        lines.append("")

        # Highest yield state (from states with enough records)
        reliable = state_df[state_df["record_count"] >= 10]
        if len(reliable) > 0:
            top_yield_state = reliable.loc[reliable["weighted_yield_kg_per_ha"].idxmax()]
            lines += [
                f"  Highest weighted yield state (≥10 records):",
                f"    {top_yield_state['state']} : {top_yield_state['weighted_yield_kg_per_ha']:,.2f} kg/ha",
                "",
            ]

    lines += [
        "=" * 65,
        "CROP-WISE INSIGHTS",
        "=" * 65,
    ]
    if len(crop_df) > 0:
        top_crop = crop_df.iloc[0]
        lines += [
            f"  Highest production crop (cumulative, all years):",
            f"    {top_crop['crop']} : {top_crop['total_production_1000_t']:,.2f} thousand tonnes",
            f"    Share of total production : {top_crop['production_share_pct']:.1f}%",
            "",
            "  Top 5 crops by total production:",
        ]
        for _, row in crop_df.head(5).iterrows():
            lines.append(
                f"    {row['crop']:30s}  "
                f"{row['total_production_1000_t']:>12,.2f} 000t  "
                f"({row['production_share_pct']:.1f}%)"
            )
        lines.append("")

        # Yield comparison
        reliable_crops = crop_df[crop_df["record_count"] >= 5]
        if len(reliable_crops) > 0:
            high_yield_crop = reliable_crops.loc[reliable_crops["weighted_yield_kg_per_ha"].idxmax()]
            lines += [
                f"  Highest weighted yield crop (≥5 records):",
                f"    {high_yield_crop['crop']} : {high_yield_crop['weighted_yield_kg_per_ha']:,.2f} kg/ha",
                "",
            ]

    lines += [
        "=" * 65,
        "PRODUCTION CHANGE BY CROP",
        "=" * 65,
    ]
    if crop_changes.get("largest_increase"):
        name, abs_ch, pct_ch = crop_changes["largest_increase"]
        lines.append(f"  Largest absolute production INCREASE:")
        lines.append(f"    Crop       : {name}")
        lines.append(f"    Change     : +{abs_ch:,.2f} thousand tonnes")
        pct_str = f"{pct_ch:.1f}%" if pd.notna(pct_ch) and not np.isinf(pct_ch) else "N/A"
        lines.append(f"    % Change   : +{pct_str}")
        lines.append("")
    if crop_changes.get("largest_decrease"):
        name, abs_ch, pct_ch = crop_changes["largest_decrease"]
        lines.append(f"  Largest absolute production DECREASE:")
        lines.append(f"    Crop       : {name}")
        lines.append(f"    Change     : {abs_ch:,.2f} thousand tonnes")
        pct_str = f"{pct_ch:.1f}%" if pd.notna(pct_ch) and not np.isinf(pct_ch) else "N/A"
        lines.append(f"    % Change   : {pct_str}")
        lines.append("")

    lines += [
        "=" * 65,
        "GENERAL NOTES",
        "=" * 65,
        "  - All production values are in THOUSAND TONNES (multiply by 1000 for tonnes).",
        "  - All area values are in THOUSAND HECTARES.",
        "  - Yield is in kg/ha (already a ratio; not scaled by thousands).",
        "  - Weighted yield = (Σ production × 1000) / Σ area, using 1000-unit values.",
        "  - Records with zero area are excluded from yield calculations.",
        "  - Some records have missing production values; these are excluded from",
        "    production totals but NOT from record counts.",
        "",
        "=" * 65,
        "END OF INSIGHTS REPORT",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Chart generation
# ─────────────────────────────────────────────────────────────────────────────
def _setup_matplotlib() -> None:
    """Configure matplotlib for consistent, non-interactive chart generation."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "figure.dpi"       : 150,
            "axes.titlesize"   : 14,
            "axes.labelsize"   : 12,
            "xtick.labelsize"  : 10,
            "ytick.labelsize"  : 10,
            "legend.fontsize"  : 10,
            "figure.titlesize" : 16,
        })
    except ImportError:
        warnings.warn("matplotlib not found; charts cannot be generated.")


def plot_yearly_production(year_df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot total production by year as a line chart."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        year_df["year"],
        year_df["total_production_1000_t"],
        marker="o", linewidth=2, color="#2196F3", markersize=7,
    )
    ax.fill_between(
        year_df["year"],
        year_df["total_production_1000_t"],
        alpha=0.15, color="#2196F3",
    )
    ax.set_title("Total Agricultural Production by Year\n(India, all crops and states)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Production (Thousand Tonnes)")
    ax.set_xticks(year_df["year"])
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    out = output_dir / "yearly_production_trend.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_yearly_area(year_df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot total cultivated area by year."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        year_df["year"],
        year_df["total_area_1000_ha"],
        marker="s", linewidth=2, color="#4CAF50", markersize=7,
    )
    ax.fill_between(
        year_df["year"],
        year_df["total_area_1000_ha"],
        alpha=0.15, color="#4CAF50",
    )
    ax.set_title("Total Cultivated Area by Year\n(India, all crops and states)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Area (Thousand Hectares)")
    ax.set_xticks(year_df["year"])
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    out = output_dir / "yearly_area_trend.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_yearly_weighted_yield(year_df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot weighted yield by year."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        year_df["year"],
        year_df["weighted_yield_kg_per_ha"],
        marker="D", linewidth=2, color="#FF9800", markersize=7,
    )
    ax.set_title("Weighted Agricultural Yield by Year\n(India, all crops and states)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Weighted Yield (kg/ha)")
    ax.set_xticks(year_df["year"])
    ax.grid(axis="y", alpha=0.4)
    ax.annotate(
        "Weighted yield = (Σ production × 1000) / Σ area",
        xy=(0.02, 0.95), xycoords="axes fraction",
        fontsize=9, color="gray",
    )
    fig.tight_layout()
    out = output_dir / "yearly_weighted_yield_trend.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_yoy_pct_change(year_df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot year-over-year percentage change in production."""
    import matplotlib.pyplot as plt

    yoy = year_df.dropna(subset=["yoy_production_pct_change"])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in yoy["yoy_production_pct_change"]]
    bars = ax.bar(yoy["year"], yoy["yoy_production_pct_change"], color=colors, edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Year-over-Year Production Change (%)\n(Compared to previous year)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Production Change (%)")
    ax.set_xticks(yoy["year"])
    ax.grid(axis="y", alpha=0.4)

    for bar, val in zip(bars, yoy["yoy_production_pct_change"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.3 if val >= 0 else -1.2),
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    out = output_dir / "yearly_production_yoy_change.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_state_production(state_df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot state-wise production bar chart (all states)."""
    import matplotlib.pyplot as plt

    sorted_df = state_df.sort_values("total_production_1000_t", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_df) * 0.4)))
    bars = ax.barh(
        sorted_df["state"],
        sorted_df["total_production_1000_t"],
        color="#2196F3", edgecolor="white",
    )
    ax.set_title("Total Agricultural Production by State\n(All years combined, all crops)")
    ax.set_xlabel("Total Production (Thousand Tonnes)")
    ax.set_ylabel("State")
    ax.grid(axis="x", alpha=0.4)
    fig.tight_layout()
    out = output_dir / "state_production_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_top10_states(state_df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot top 10 states by production."""
    import matplotlib.pyplot as plt

    top10 = state_df.head(10).sort_values("total_production_1000_t", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(
        top10["state"],
        top10["total_production_1000_t"],
        color="#1565C0", edgecolor="white",
    )
    ax.set_title("Top 10 States by Total Agricultural Production\n(All years combined)")
    ax.set_xlabel("Total Production (Thousand Tonnes)")
    ax.set_ylabel("State")
    for bar, val in zip(bars, top10["total_production_1000_t"]):
        ax.text(
            bar.get_width() + 100,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}",
            va="center", fontsize=9,
        )
    ax.grid(axis="x", alpha=0.4)
    fig.tight_layout()
    out = output_dir / "top_10_states.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_state_heatmap(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """
    Plot a state × year production heatmap.
    Limited to top 15 states by total production.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        state_col = _resolve_text_col(df, "state")
        working = df.dropna(subset=[state_col, "year", "production_1000_t"])

        # Top 15 states
        top_states = (
            working.groupby(state_col)["production_1000_t"]
            .sum()
            .nlargest(15)
            .index.tolist()
        )
        filtered = working[working[state_col].isin(top_states)]

        pivot = filtered.pivot_table(
            values="production_1000_t",
            index=state_col,
            columns="year",
            aggfunc="sum",
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="YlOrRd",
            annot=True,
            fmt=".0f",
            linewidths=0.5,
            cbar_kws={"label": "Production (Thousand Tonnes)"},
        )
        ax.set_title(
            "State-wise Production Heatmap (Top 15 States × Year)\n"
            "Values in Thousand Tonnes"
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("State")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        fig.tight_layout()
        out = output_dir / "state_year_heatmap.png"
        fig.savefig(out)
        plt.close(fig)
        return out

    except ImportError:
        warnings.warn("seaborn not installed; state heatmap skipped.")
        return None


def plot_crop_production(crop_df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot crop-wise production bar chart (all crops)."""
    import matplotlib.pyplot as plt

    sorted_df = crop_df.sort_values("total_production_1000_t", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_df) * 0.4)))
    ax.barh(
        sorted_df["crop"],
        sorted_df["total_production_1000_t"],
        color="#E91E63", edgecolor="white",
    )
    ax.set_title("Total Agricultural Production by Crop\n(All years and states combined)")
    ax.set_xlabel("Total Production (Thousand Tonnes)")
    ax.set_ylabel("Crop")
    ax.grid(axis="x", alpha=0.4)
    fig.tight_layout()
    out = output_dir / "crop_production_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_crop_area(crop_df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot crop-wise cultivated area comparison."""
    import matplotlib.pyplot as plt

    sorted_df = crop_df.sort_values("total_area_1000_ha", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_df) * 0.4)))
    ax.barh(
        sorted_df["crop"],
        sorted_df["total_area_1000_ha"],
        color="#9C27B0", edgecolor="white",
    )
    ax.set_title("Total Cultivated Area by Crop\n(All years and states combined)")
    ax.set_xlabel("Total Area (Thousand Hectares)")
    ax.set_ylabel("Crop")
    ax.grid(axis="x", alpha=0.4)
    fig.tight_layout()
    out = output_dir / "crop_area_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_crop_weighted_yield(crop_df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot crop-wise weighted yield."""
    import matplotlib.pyplot as plt

    sorted_df = crop_df.dropna(subset=["weighted_yield_kg_per_ha"]).sort_values(
        "weighted_yield_kg_per_ha", ascending=True
    )
    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_df) * 0.4)))
    ax.barh(
        sorted_df["crop"],
        sorted_df["weighted_yield_kg_per_ha"],
        color="#FF5722", edgecolor="white",
    )
    ax.set_title("Weighted Yield by Crop (kg/ha)\n(All years and states combined)")
    ax.set_xlabel("Weighted Yield (kg/ha)")
    ax.set_ylabel("Crop")
    ax.grid(axis="x", alpha=0.4)
    ax.annotate(
        "Weighted yield = (Σ production × 1000) / Σ area",
        xy=(0.02, 0.98), xycoords="axes fraction",
        fontsize=8, color="gray", va="top",
    )
    fig.tight_layout()
    out = output_dir / "crop_weighted_yield.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_crop_trends(df: pd.DataFrame, output_dir: Path, top_n: int = 8) -> Path:
    """
    Plot crop-wise production trends over years for top N crops.
    """
    import matplotlib.pyplot as plt

    crop_col = _resolve_text_col(df, "crop")
    working  = df.dropna(subset=[crop_col, "year", "production_1000_t"])

    # Identify top N crops
    top_crops = (
        working.groupby(crop_col)["production_1000_t"]
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )
    filtered = working[working[crop_col].isin(top_crops)]

    yearly = filtered.groupby([crop_col, "year"])["production_1000_t"].sum().reset_index()
    yearly.columns = ["crop", "year", "production"]

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10.colors
    for i, crop in enumerate(top_crops):
        crop_data = yearly[yearly["crop"] == crop].sort_values("year")
        ax.plot(
            crop_data["year"],
            crop_data["production"],
            marker="o", linewidth=2,
            label=crop,
            color=colors[i % len(colors)],
        )

    ax.set_title(f"Production Trends for Top {top_n} Crops by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Production (Thousand Tonnes)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    ax.grid(alpha=0.4)
    ax.set_xticks(sorted(working["year"].unique()))
    fig.tight_layout()
    out = output_dir / "crop_production_trends.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 10. Save outputs
# ─────────────────────────────────────────────────────────────────────────────
def save_all_outputs(
    overview_text  : str,
    year_df        : pd.DataFrame,
    state_df       : pd.DataFrame,
    crop_df        : pd.DataFrame,
    insights_text  : str,
    reports_dir    : Path = REPORTS_DIR,
) -> dict:
    """
    Save all analysis outputs to the reports/member2/ directory.

    Returns a dict of saved file paths.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    # Overview
    p = reports_dir / "dataset_overview.txt"
    p.write_text(overview_text, encoding="utf-8")
    saved["dataset_overview"] = p
    print(f"  Saved : {p}")

    # Year-wise CSV
    p = reports_dir / "year_wise_analysis.csv"
    year_df.to_csv(p, index=False)
    saved["year_wise_analysis"] = p
    print(f"  Saved : {p}")

    # State-wise CSV
    p = reports_dir / "state_wise_analysis.csv"
    state_df.to_csv(p, index=False)
    saved["state_wise_analysis"] = p
    print(f"  Saved : {p}")

    # Crop-wise CSV
    p = reports_dir / "crop_wise_analysis.csv"
    crop_df.to_csv(p, index=False)
    saved["crop_wise_analysis"] = p
    print(f"  Saved : {p}")

    # Insights
    p = reports_dir / "historical_insights.txt"
    p.write_text(insights_text, encoding="utf-8")
    saved["historical_insights"] = p
    print(f"  Saved : {p}")

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Standalone demo (runs when file is executed directly)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _section("PS 137 – Member 2 Historical Analysis (standalone demo)")
    try:
        df = load_cleaned_data()
        print(f"  Dataset loaded: {len(df):,} rows × {df.shape[1]} columns")
        summary = validate_dataset(df)
        print(f"  States   : {summary.get('unique_states')}")
        print(f"  Crops    : {summary.get('unique_crops')}")
        print(f"  Years    : {summary.get('year_values')}")
        print("\n  Run 'python run_analysis.py' for the full analysis.")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
