"""
PS 137 – Agricultural Production Scenario Coordinator
======================================================
src/preprocessing/preprocess_agriculture.py
Member 1 – Data Cleaning and Preprocessing

PURPOSE
-------
Reads the raw Parquet file, cleans and standardises the data,
adds derived quality columns, and saves the final processed CSV.

IMPORTANT RULES
---------------
- The raw Parquet file is NEVER modified.
- No data is fabricated.
- Missing values are kept as NaN unless a clear reason exists to change them.
- Zero production records are NOT deleted – they are flagged.
- All cleaning decisions are documented in reports/.

HOW TO RUN
----------
From the project root:
    python src/preprocessing/preprocess_agriculture.py
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
PARQUET_PATH   = PROJECT_ROOT / "data" / "raw"  / "crop_statistics.parquet"
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
OUTPUT_CSV     = PROCESSED_DIR / "cleaned_agriculture_dataset.csv"
REPORTS_DIR    = PROJECT_ROOT / "reports"
AUDIT_REPORT   = REPORTS_DIR / "data_quality_audit.txt"
CLEANING_REPORT= REPORTS_DIR / "cleaning_summary_report.txt"
VALIDATION_RPT = REPORTS_DIR / "final_validation_report.txt"


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def log(msg: str) -> None:
    print(f"  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Load raw dataset
# ─────────────────────────────────────────────────────────────────────────────
def load_raw_data() -> pd.DataFrame:
    """
    Load the raw Parquet file into a DataFrame.
    Exits with a helpful message if the file does not exist.
    """
    section("STEP 1 – Loading raw dataset")
    if not PARQUET_PATH.exists():
        print(f"[ERROR] Raw Parquet file not found at:\n  {PARQUET_PATH}")
        print("  Run data collection first:")
        print("    python src/data_collection/collect_dataset.py")
        sys.exit(1)

    df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
    log(f"Loaded  : {PARQUET_PATH.name}")
    log(f"Shape   : {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
    log(f"Columns : {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Data quality audit (writes report file)
# ─────────────────────────────────────────────────────────────────────────────
def run_quality_audit(df: pd.DataFrame) -> dict:
    """
    Examine the raw data for quality issues.
    Returns a dictionary of findings.
    Writes the full audit to reports/data_quality_audit.txt.
    """
    section("STEP 2 – Data quality audit")
    findings = {}

    # ── 2a. Missing values ───────────────────────────────────────────────────
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    findings["missing_values"] = missing[missing > 0].to_dict()
    findings["missing_pct"]    = missing_pct[missing > 0].to_dict()

    # ── 2b. Exact duplicates ─────────────────────────────────────────────────
    n_exact_dupes = df.duplicated().sum()
    findings["exact_duplicates"] = int(n_exact_dupes)

    # ── 2c. Composite-key duplicates (state + district + year + crop) ────────
    key_cols = ["state", "district", "year", "crop"]
    available_keys = [c for c in key_cols if c in df.columns]
    if len(available_keys) == len(key_cols):
        n_key_dupes = df.duplicated(subset=key_cols).sum()
    else:
        n_key_dupes = "N/A (some key columns missing)"
    findings["composite_key_duplicates"] = n_key_dupes

    # ── 2d. Negative values ──────────────────────────────────────────────────
    num_cols = ["area_1000_ha", "production_1000_t", "yield_kg_per_ha"]
    neg_counts = {}
    for col in num_cols:
        if col in df.columns:
            neg_counts[col] = int((df[col] < 0).sum())
    findings["negative_values"] = neg_counts

    # ── 2e. Zero-value analysis ──────────────────────────────────────────────
    zero_counts = {}
    for col in num_cols:
        if col in df.columns:
            zero_counts[col] = int((df[col] == 0).sum())
    findings["zero_values"] = zero_counts

    # ── 2f. Unusually high yield ─────────────────────────────────────────────
    HIGH_YIELD_THRESHOLD = 15_000  # kg/ha – above this warrants investigation
    if "yield_kg_per_ha" in df.columns:
        n_high_yield = int((df["yield_kg_per_ha"] > HIGH_YIELD_THRESHOLD).sum())
        findings["high_yield_records"] = n_high_yield
        findings["high_yield_threshold_kg_ha"] = HIGH_YIELD_THRESHOLD

    # ── 2g. Invalid year values ──────────────────────────────────────────────
    if "year" in df.columns:
        valid_years = df["year"].between(2000, 2030)
        findings["invalid_year_records"] = int((~valid_years).sum())
        findings["year_range_found"] = {
            "min": int(df["year"].min()),
            "max": int(df["year"].max())
        }

    # ── 2h. Unique counts ────────────────────────────────────────────────────
    for col in ["state", "district", "crop", "year"]:
        if col in df.columns:
            findings[f"unique_{col}s"] = int(df[col].nunique())

    # ── Print summary ────────────────────────────────────────────────────────
    log(f"Missing values per column  : {findings.get('missing_values', {})}")
    log(f"Exact duplicates           : {findings['exact_duplicates']}")
    log(f"Composite-key duplicates   : {findings['composite_key_duplicates']}")
    log(f"Negative values            : {findings['negative_values']}")
    log(f"Zero values                : {findings['zero_values']}")
    log(f"High yield records (>{HIGH_YIELD_THRESHOLD} kg/ha): {findings.get('high_yield_records', 0)}")
    log(f"Invalid year records       : {findings.get('invalid_year_records', 0)}")

    # ── Write audit report ───────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_audit_report(df, findings)

    return findings


def _write_audit_report(df: pd.DataFrame, findings: dict) -> None:
    """Write the detailed audit report to a text file."""
    lines = [
        "PS 137 – Agricultural Production Scenario Coordinator",
        "DATA QUALITY AUDIT REPORT",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
        "",
        "1. DATASET SHAPE",
        f"   Rows    : {df.shape[0]:,}",
        f"   Columns : {df.shape[1]}",
        f"   Columns : {list(df.columns)}",
        "",
        "2. MISSING VALUES",
    ]
    mv = findings.get("missing_values", {})
    if mv:
        for col, cnt in mv.items():
            pct = findings["missing_pct"].get(col, 0)
            lines.append(f"   {col:35s} : {cnt:>8,}  ({pct:.2f}%)")
    else:
        lines.append("   None found.")

    lines += [
        "",
        "3. DUPLICATE RECORDS",
        f"   Exact duplicates            : {findings.get('exact_duplicates', 0):,}",
        f"   Composite-key duplicates    : {findings.get('composite_key_duplicates', 'N/A')}",
        "",
        "4. NEGATIVE VALUES",
    ]
    for col, cnt in findings.get("negative_values", {}).items():
        lines.append(f"   {col:35s} : {cnt:>8,}")

    lines += [
        "",
        "5. ZERO VALUES",
    ]
    for col, cnt in findings.get("zero_values", {}).items():
        lines.append(f"   {col:35s} : {cnt:>8,}")

    thr = findings.get("high_yield_threshold_kg_ha", 15000)
    lines += [
        "",
        "6. HIGH YIELD RECORDS",
        f"   Threshold                   : {thr:,} kg/ha",
        f"   Records above threshold     : {findings.get('high_yield_records', 0):,}",
        "   NOTE: These are flagged for investigation, not auto-removed.",
        "",
        "7. YEAR RANGE",
        f"   Min year  : {findings.get('year_range_found', {}).get('min', 'N/A')}",
        f"   Max year  : {findings.get('year_range_found', {}).get('max', 'N/A')}",
        f"   Invalid   : {findings.get('invalid_year_records', 0):,}",
        "",
        "8. UNIQUE VALUE COUNTS",
        f"   States    : {findings.get('unique_states', 'N/A')}",
        f"   Districts : {findings.get('unique_districts', 'N/A')}",
        f"   Crops     : {findings.get('unique_crops', 'N/A')}",
        f"   Years     : {findings.get('unique_years', 'N/A')}",
        "",
        "9. SAMPLE STATE NAMES (top 10)",
    ]

    if "state" in df.columns:
        for s in sorted(df["state"].dropna().unique())[:10]:
            lines.append(f"   {s}")

    lines += [
        "",
        "10. SAMPLE CROP NAMES (top 20)",
    ]
    if "crop" in df.columns:
        for c in sorted(df["crop"].dropna().unique())[:20]:
            lines.append(f"   {c}")

    lines += [
        "",
        "=" * 65,
        "END OF AUDIT REPORT",
    ]

    with open(AUDIT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log(f"Audit report saved : {AUDIT_REPORT}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Clean the data
# ─────────────────────────────────────────────────────────────────────────────
def clean_data(df_raw: pd.DataFrame, findings: dict) -> pd.DataFrame:
    """
    Apply all cleaning steps to a COPY of the raw DataFrame.
    Returns the cleaned DataFrame with new quality columns added.
    """
    section("STEP 3 – Cleaning and preprocessing")

    # ── 3a. Work on a copy (never modify raw) ────────────────────────────────
    df = df_raw.copy()
    log("Working on a copy of the raw DataFrame (raw is untouched).")

    # ── 3b. Remove exact duplicates ──────────────────────────────────────────
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    log(f"Exact duplicates removed : {n_removed:,}  (kept : {len(df):,})")

    # ── 3c. Text standardisation – create clean columns ──────────────────────
    # Map of known old name → new standardised name (applied case-insensitively)
    STATE_CORRECTIONS = {
        "orissa"      : "Odisha",
        "uttaranchal" : "Uttarakhand",
        "pondicherry" : "Puducherry",
    }

    def standardise_text(series: pd.Series, corrections: dict = None) -> pd.Series:
        """Strip whitespace, apply Title Case, then apply corrections."""
        cleaned = series.astype(str).str.strip().str.title()
        cleaned = cleaned.replace("Nan", np.nan)   # restore NaN after str ops
        if corrections:
            # Apply corrections where the lower-cased cleaned value matches a key
            corrected = cleaned.str.lower().map(
                lambda x: corrections.get(x, x) if pd.notna(x) else np.nan
            )
            # For values not in corrections, use the Title Case version
            cleaned = cleaned.where(
                ~cleaned.str.lower().isin(corrections), other=corrected
            )
        return cleaned

    if "state" in df.columns:
        df["state_clean"] = standardise_text(df["state"], STATE_CORRECTIONS)
    if "district" in df.columns:
        df["district_clean"] = standardise_text(df["district"])
    if "crop" in df.columns:
        df["crop_clean"] = standardise_text(df["crop"])

    log("Created: state_clean, district_clean, crop_clean")

    # ── 3d. Convert numeric columns if needed ─────────────────────────────────
    for col in ["area_1000_ha", "production_1000_t", "yield_kg_per_ha"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # ── 3e. Yield recalculation (new column, original not overwritten) ────────
    if "area_1000_ha" in df.columns and "production_1000_t" in df.columns:
        # Only calculate where area > 0 and production is not NaN
        mask_valid = (df["area_1000_ha"] > 0) & df["production_1000_t"].notna()
        df["calculated_yield_kg_per_ha"] = np.nan
        df.loc[mask_valid, "calculated_yield_kg_per_ha"] = (
            (df.loc[mask_valid, "production_1000_t"] * 1_000)
            / df.loc[mask_valid, "area_1000_ha"]
        ).round(2)
        log("Created: calculated_yield_kg_per_ha")

    # ── 3f. Yield discrepancy flag ────────────────────────────────────────────
    TOLERANCE = 10  # kg/ha
    if "yield_kg_per_ha" in df.columns and "calculated_yield_kg_per_ha" in df.columns:
        both_known = df["yield_kg_per_ha"].notna() & df["calculated_yield_kg_per_ha"].notna()
        df["yield_discrepancy"] = False
        df.loc[both_known, "yield_discrepancy"] = (
            (df.loc[both_known, "yield_kg_per_ha"]
             - df.loc[both_known, "calculated_yield_kg_per_ha"]).abs()
            > TOLERANCE
        )
        n_disc = df["yield_discrepancy"].sum()
        log(f"Yield discrepancy records (>{TOLERANCE} kg/ha difference): {n_disc:,}")

    # ── 3g. Derived boolean columns ───────────────────────────────────────────
    # has_production_data: True if production is a real number (not NaN)
    if "production_1000_t" in df.columns:
        df["has_production_data"] = df["production_1000_t"].notna()

    # is_zero_activity: area AND production are both 0 (not NaN, literally 0)
    if "area_1000_ha" in df.columns and "production_1000_t" in df.columns:
        df["is_zero_activity"] = (
            (df["area_1000_ha"]     == 0) &
            (df["production_1000_t"]== 0)
        )

    # is_small_area_rounding: area is > 0 but very small (rounding artefact)
    if "area_1000_ha" in df.columns:
        df["is_small_area_rounding"] = (
            (df["area_1000_ha"] > 0) &
            (df["area_1000_ha"] < 0.1)
        )

    log("Created: has_production_data, is_zero_activity, is_small_area_rounding")

    # ── 3h. record_status column ──────────────────────────────────────────────
    df["record_status"] = _assign_record_status(df)
    log("Created: record_status")
    log(f"\n  record_status breakdown:\n{df['record_status'].value_counts().to_string()}")

    return df


def _assign_record_status(df: pd.DataFrame) -> pd.Series:
    """
    Assign a human-readable status to every record.

    Rules (checked in this order):
      1. small_area_rounding    : area > 0 AND area < 0.1
      2. zero_activity          : area == 0 AND production == 0 (not NaN)
      3. missing_prod_yield_zero_area : area == 0 AND production is NaN
      4. missing_prod_yield_with_area : area > 0 AND production is NaN
      5. zero_prod_with_positive_area : area > 0 AND production == 0
      6. complete_positive       : area > 0 AND production > 0
      7. unknown                 : anything else
    """
    area = df.get("area_1000_ha",      pd.Series(dtype=float))
    prod = df.get("production_1000_t", pd.Series(dtype=float))

    status = pd.Series("unknown", index=df.index)

    # Apply from lowest to highest priority (last write wins)
    # 6. complete_positive
    status.loc[(area > 0) & (prod > 0)] = "complete_positive"

    # 5. zero_prod_with_positive_area
    status.loc[(area > 0) & (prod == 0)] = "zero_prod_with_positive_area"

    # 4. missing_prod_yield_with_area
    status.loc[(area > 0) & prod.isna()] = "missing_prod_yield_with_area"

    # 3. missing_prod_yield_zero_area
    status.loc[(area == 0) & prod.isna()] = "missing_prod_yield_zero_area"

    # 2. zero_activity
    status.loc[(area == 0) & (prod == 0)] = "zero_activity"

    # 1. small_area_rounding (overrides complete_positive if area is tiny)
    status.loc[(area > 0) & (area < 0.1)] = "small_area_rounding"

    return status


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Save cleaned CSV
# ─────────────────────────────────────────────────────────────────────────────
def save_cleaned_csv(df: pd.DataFrame) -> None:
    """
    Save the cleaned DataFrame as a CSV.
    The original Parquet file remains untouched.
    """
    section("STEP 4 – Saving cleaned dataset")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    size_kb = OUTPUT_CSV.stat().st_size / 1024
    log(f"Saved to  : {OUTPUT_CSV}")
    log(f"File size : {size_kb:.1f} KB")
    log(f"Rows      : {len(df):,}")
    log(f"Columns   : {len(df.columns)}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Write cleaning summary report
# ─────────────────────────────────────────────────────────────────────────────
def write_cleaning_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """Write a human-readable cleaning summary."""
    section("STEP 5 – Writing cleaning summary report")
    lines = [
        "PS 137 – Agricultural Production Scenario Coordinator",
        "CLEANING SUMMARY REPORT",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
        "",
        "BEFORE CLEANING",
        f"  Rows    : {len(df_raw):,}",
        f"  Columns : {df_raw.shape[1]}",
        "",
        "AFTER CLEANING",
        f"  Rows    : {len(df_clean):,}",
        f"  Columns : {df_clean.shape[1]}",
        "",
        "CHANGES MADE",
        f"  Rows removed (exact duplicates) : {len(df_raw) - len(df_clean):,}",
        "  Original columns                : preserved unchanged",
        "  New columns added               : state_clean, district_clean, crop_clean,",
        "                                    calculated_yield_kg_per_ha, yield_discrepancy,",
        "                                    has_production_data, is_zero_activity,",
        "                                    is_small_area_rounding, record_status",
        "",
        "RECORD STATUS BREAKDOWN",
    ]
    if "record_status" in df_clean.columns:
        for status, count in df_clean["record_status"].value_counts().items():
            pct = count / len(df_clean) * 100
            lines.append(f"  {status:45s} : {count:>8,}  ({pct:.1f}%)")

    lines += [
        "",
        "OUTPUT FILE",
        f"  {OUTPUT_CSV}",
        "",
        "=" * 65,
        "END OF CLEANING SUMMARY",
    ]

    with open(CLEANING_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Cleaning report saved : {CLEANING_REPORT}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – Final validation report
# ─────────────────────────────────────────────────────────────────────────────
def write_validation_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """
    Load the saved CSV, re-verify it, and write the final validation report.
    This confirms the file is readable and meets all quality criteria.
    """
    section("STEP 6 – Final validation")

    # Re-load from disk to prove the CSV was saved correctly
    df_check = pd.read_csv(OUTPUT_CSV, low_memory=False)

    expected_original_cols = [
        "state", "district", "year", "crop",
        "area_1000_ha", "production_1000_t", "yield_kg_per_ha"
    ]
    expected_new_cols = [
        "state_clean", "district_clean", "crop_clean",
        "calculated_yield_kg_per_ha", "yield_discrepancy",
        "has_production_data", "is_zero_activity",
        "is_small_area_rounding", "record_status"
    ]

    checks = []

    # CSV loads successfully
    checks.append(("CSV loads successfully", True))

    # Row count
    checks.append(("Row count matches cleaned DataFrame",
                   len(df_check) == len(df_clean)))

    # Original columns present
    orig_ok = all(c in df_check.columns for c in expected_original_cols)
    checks.append(("All original columns present", orig_ok))

    # New columns present
    new_ok = all(c in df_check.columns for c in expected_new_cols)
    checks.append(("All derived columns present", new_ok))

    # No negative area or production
    neg_area = int((df_check["area_1000_ha"] < 0).sum()) if "area_1000_ha" in df_check.columns else 0
    neg_prod = int((df_check["production_1000_t"] < 0).sum()) if "production_1000_t" in df_check.columns else 0
    checks.append(("No negative area values",        neg_area == 0))
    checks.append(("No negative production values",  neg_prod == 0))

    # Raw Parquet unchanged
    raw_df2 = pd.read_parquet(PARQUET_PATH)
    checks.append(("Raw Parquet row count unchanged",
                   len(raw_df2) == len(df_raw)))

    lines = [
        "PS 137 – Agricultural Production Scenario Coordinator",
        "FINAL VALIDATION REPORT",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
        "",
        "VALIDATION CHECKS",
    ]

    all_pass = True
    for check_name, result in checks:
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        lines.append(f"  [{status}] {check_name}")

    lines += [
        "",
        "DATASET STATISTICS (cleaned CSV)",
        f"  Rows    : {len(df_check):,}",
        f"  Columns : {df_check.shape[1]}",
        "",
        "COVERAGE",
        f"  States    : {df_check['state_clean'].nunique() if 'state_clean' in df_check.columns else 'N/A'}",
        f"  Districts : {df_check['district_clean'].nunique() if 'district_clean' in df_check.columns else 'N/A'}",
        f"  Crops     : {df_check['crop_clean'].nunique() if 'crop_clean' in df_check.columns else 'N/A'}",
    ]

    if "year" in df_check.columns:
        lines.append(f"  Years     : {sorted(df_check['year'].dropna().unique().tolist())}")

    lines += [
        "",
        "MISSING VALUES IN CLEANED CSV",
    ]
    mv = df_check.isnull().sum()
    mv = mv[mv > 0]
    if mv.empty:
        lines.append("  None")
    else:
        for col, cnt in mv.items():
            pct = cnt / len(df_check) * 100
            lines.append(f"  {col:40s} : {cnt:>8,}  ({pct:.2f}%)")

    lines += [
        "",
        "RECORD STATUS BREAKDOWN",
    ]
    if "record_status" in df_check.columns:
        for status, count in df_check["record_status"].value_counts().items():
            pct = count / len(df_check) * 100
            lines.append(f"  {status:45s} : {count:>8,}  ({pct:.1f}%)")

    overall = "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED – review above"
    lines += [
        "",
        "=" * 65,
        f"OVERALL RESULT : {overall}",
        "=" * 65,
    ]

    with open(VALIDATION_RPT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log(f"Validation report saved : {VALIDATION_RPT}")
    log(f"Overall result          : {overall}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    section("PS 137 – DATA PREPROCESSING  (Member 1)")
    print(f"  Project root : {PROJECT_ROOT}")

    df_raw   = load_raw_data()
    findings = run_quality_audit(df_raw)
    df_clean = clean_data(df_raw, findings)
    save_cleaned_csv(df_clean)
    write_cleaning_report(df_raw, df_clean)
    write_validation_report(df_raw, df_clean)

    section("PREPROCESSING COMPLETE")
    print(f"  Cleaned CSV : {OUTPUT_CSV}")
    print(f"  Reports     : {REPORTS_DIR}\n")


if __name__ == "__main__":
    main()
