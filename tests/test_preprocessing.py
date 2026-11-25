"""
PS 137 – Agricultural Production Scenario Coordinator
======================================================
tests/test_preprocessing.py
Member 1 – Automated Tests

PURPOSE
-------
Tests the preprocessing pipeline to verify:
  - Dataset loading
  - Expected columns
  - Missing-value handling
  - Duplicate detection
  - Derived columns
  - Yield calculation
  - Zero-area handling
  - Negative-value detection

HOW TO RUN (from the project root)
------------------------------------
    pytest tests/ -v
    pytest tests/ -v --tb=short
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Add project root to path so imports work ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PARQUET_PATH = PROJECT_ROOT / "data" / "raw"  / "crop_statistics.parquet"
CSV_PATH     = PROJECT_ROOT / "data" / "processed" / "cleaned_agriculture_dataset.csv"

EXPECTED_ORIGINAL_COLS = [
    "state", "district", "year", "crop",
    "area_1000_ha", "production_1000_t", "yield_kg_per_ha"
]
EXPECTED_NEW_COLS = [
    "state_clean", "district_clean", "crop_clean",
    "has_production_data", "is_zero_activity",
    "is_small_area_rounding", "record_status"
]


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Load the raw Parquet file once for the whole test session."""
    if not PARQUET_PATH.exists():
        pytest.skip(
            f"Raw Parquet not found at {PARQUET_PATH}. "
            "Run 'python src/data_collection/collect_dataset.py' first."
        )
    return pd.read_parquet(PARQUET_PATH)


@pytest.fixture(scope="session")
def clean_df() -> pd.DataFrame:
    """Load the cleaned CSV file once for the whole test session."""
    if not CSV_PATH.exists():
        pytest.skip(
            f"Cleaned CSV not found at {CSV_PATH}. "
            "Run 'python src/preprocessing/preprocess_agriculture.py' first."
        )
    return pd.read_csv(CSV_PATH, low_memory=False)


# ─────────────────────────────────────────────────────────────────────────────
# Group 1 – Dataset loading tests
# ─────────────────────────────────────────────────────────────────────────────
class TestDatasetLoading:

    def test_raw_parquet_loads(self, raw_df):
        """The raw Parquet file must load without errors and have rows."""
        assert isinstance(raw_df, pd.DataFrame), "Expected a Pandas DataFrame."
        assert len(raw_df) > 0, "Raw dataset should not be empty."

    def test_cleaned_csv_loads(self, clean_df):
        """The cleaned CSV must load without errors and have rows."""
        assert isinstance(clean_df, pd.DataFrame), "Expected a Pandas DataFrame."
        assert len(clean_df) > 0, "Cleaned dataset should not be empty."

    def test_raw_has_reasonable_row_count(self, raw_df):
        """Raw dataset should have at least 10,000 rows (sanity check)."""
        assert len(raw_df) >= 10_000, (
            f"Raw dataset has only {len(raw_df):,} rows. "
            "Expected at least 10,000."
        )

    def test_cleaned_row_count_close_to_raw(self, raw_df, clean_df):
        """
        Cleaned dataset should not lose more than 5% of records
        (we only remove confirmed exact duplicates).
        """
        pct_kept = len(clean_df) / len(raw_df) * 100
        assert pct_kept >= 95, (
            f"Cleaned dataset has {pct_kept:.1f}% of raw rows. "
            "More than 5% were removed – investigate."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Group 2 – Column tests
# ─────────────────────────────────────────────────────────────────────────────
class TestColumns:

    def test_original_columns_present_in_raw(self, raw_df):
        """All expected original columns must exist in the raw dataset."""
        for col in EXPECTED_ORIGINAL_COLS:
            assert col in raw_df.columns, (
                f"Expected column '{col}' not found in raw dataset. "
                f"Actual columns: {list(raw_df.columns)}"
            )

    def test_original_columns_preserved_in_cleaned(self, clean_df):
        """Original columns must still be present in the cleaned dataset."""
        for col in EXPECTED_ORIGINAL_COLS:
            assert col in clean_df.columns, (
                f"Original column '{col}' is missing from cleaned dataset."
            )

    def test_new_derived_columns_present(self, clean_df):
        """All derived quality columns must be present in the cleaned dataset."""
        for col in EXPECTED_NEW_COLS:
            assert col in clean_df.columns, (
                f"Derived column '{col}' is missing from cleaned dataset."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Group 3 – Missing value tests
# ─────────────────────────────────────────────────────────────────────────────
class TestMissingValues:

    def test_missing_production_not_replaced_with_zero(self, clean_df):
        """
        Records where production was originally NaN must remain NaN,
        not be replaced with 0.
        """
        # has_production_data = False means production is NaN in cleaned data
        if "has_production_data" in clean_df.columns:
            no_prod = clean_df[~clean_df["has_production_data"]]
            # In those rows, production_1000_t should be NaN, not 0
            if len(no_prod) > 0:
                assert no_prod["production_1000_t"].isna().all(), (
                    "Records with has_production_data=False should have NaN "
                    "production, but found zeros."
                )

    def test_has_production_data_matches_production_col(self, clean_df):
        """
        has_production_data must be True exactly when production_1000_t
        is not NaN.
        """
        if "has_production_data" in clean_df.columns:
            expected = clean_df["production_1000_t"].notna()
            actual   = clean_df["has_production_data"]
            mismatches = (expected != actual).sum()
            assert mismatches == 0, (
                f"{mismatches} rows have a mismatch between "
                "has_production_data and whether production_1000_t is NaN."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Group 4 – Duplicate tests
# ─────────────────────────────────────────────────────────────────────────────
class TestDuplicates:

    def test_no_exact_duplicates_in_cleaned(self, clean_df):
        """Exact duplicate rows must have been removed."""
        n_dupes = clean_df.duplicated().sum()
        assert n_dupes == 0, (
            f"Found {n_dupes:,} exact duplicate rows in the cleaned dataset."
        )

    def test_composite_key_duplicates_reported(self, clean_df):
        """
        This is a reporting test, not a strict pass/fail.
        We just verify the check can be run without crashing.
        """
        key_cols = [c for c in ["state", "district", "year", "crop"]
                    if c in clean_df.columns]
        if len(key_cols) == 4:
            n_dupes = clean_df.duplicated(subset=key_cols).sum()
            # Print for information; do NOT fail the test just for this
            print(f"\n  INFO: Composite-key duplicates in cleaned CSV: {n_dupes:,}")


# ─────────────────────────────────────────────────────────────────────────────
# Group 5 – Derived column correctness
# ─────────────────────────────────────────────────────────────────────────────
class TestDerivedColumns:

    def test_is_zero_activity_correct(self, clean_df):
        """
        is_zero_activity must be True only when both area AND production are 0.
        """
        if "is_zero_activity" in clean_df.columns:
            flagged = clean_df[clean_df["is_zero_activity"]]
            if len(flagged) > 0:
                assert (flagged["area_1000_ha"]      == 0).all(), \
                    "is_zero_activity=True but area is not 0 for some rows."
                assert (flagged["production_1000_t"] == 0).all(), \
                    "is_zero_activity=True but production is not 0 for some rows."

    def test_is_small_area_rounding_correct(self, clean_df):
        """
        is_small_area_rounding must only flag rows where 0 < area < 0.1.
        """
        if "is_small_area_rounding" in clean_df.columns:
            flagged = clean_df[clean_df["is_small_area_rounding"]]
            if len(flagged) > 0:
                assert (flagged["area_1000_ha"] > 0).all(), \
                    "is_small_area_rounding=True but area is 0 for some rows."
                assert (flagged["area_1000_ha"] < 0.1).all(), \
                    "is_small_area_rounding=True but area >= 0.1 for some rows."

    def test_record_status_valid_values(self, clean_df):
        """record_status must only contain the defined category names."""
        VALID_STATUSES = {
            "complete_positive",
            "zero_activity",
            "missing_prod_yield_with_area",
            "missing_prod_yield_zero_area",
            "zero_prod_with_positive_area",
            "small_area_rounding",
            "unknown",
        }
        if "record_status" in clean_df.columns:
            actual = set(clean_df["record_status"].unique())
            invalid = actual - VALID_STATUSES
            assert not invalid, (
                f"record_status contains unexpected values: {invalid}"
            )

    def test_state_clean_is_title_case(self, clean_df):
        """state_clean values should be Title Case (not all-caps or all-lower)."""
        if "state_clean" in clean_df.columns:
            sample = clean_df["state_clean"].dropna().head(50)
            for val in sample:
                assert val == val.title() or val in {"NaN"}, (
                    f"state_clean value '{val}' is not in Title Case."
                )


# ─────────────────────────────────────────────────────────────────────────────
# Group 6 – Yield calculation tests
# ─────────────────────────────────────────────────────────────────────────────
class TestYieldCalculation:

    def test_calculated_yield_formula(self, clean_df):
        """
        For rows where area > 0 and production is not NaN,
        calculated_yield_kg_per_ha must equal (production * 1000) / area
        within a small tolerance.
        """
        if "calculated_yield_kg_per_ha" not in clean_df.columns:
            pytest.skip("calculated_yield_kg_per_ha column not found.")

        mask = (
            clean_df["area_1000_ha"] > 0
        ) & clean_df["production_1000_t"].notna()

        subset = clean_df[mask].copy()
        expected = (subset["production_1000_t"] * 1_000) / subset["area_1000_ha"]
        actual   = subset["calculated_yield_kg_per_ha"]

        diff = (expected - actual).abs()
        max_diff = diff.max()
        assert max_diff < 1.0, (
            f"Calculated yield differs from formula by up to {max_diff:.4f} kg/ha. "
            "Expected < 1 kg/ha tolerance."
        )

    def test_yield_not_calculated_when_area_zero(self, clean_df):
        """
        calculated_yield_kg_per_ha must be NaN when area == 0
        (division by zero is not allowed).
        """
        if "calculated_yield_kg_per_ha" not in clean_df.columns:
            pytest.skip("calculated_yield_kg_per_ha column not found.")

        zero_area = clean_df[clean_df["area_1000_ha"] == 0]
        if len(zero_area) > 0:
            assert zero_area["calculated_yield_kg_per_ha"].isna().all(), (
                "calculated_yield_kg_per_ha should be NaN when area == 0."
            )

    def test_original_yield_not_modified(self, raw_df, clean_df):
        """
        The original yield_kg_per_ha values must not change between
        the raw dataset and the cleaned dataset.
        """
        raw_yield   = raw_df["yield_kg_per_ha"].reset_index(drop=True)
        clean_yield = clean_df["yield_kg_per_ha"].reset_index(drop=True)

        # Compare only rows that exist in both (after duplicate removal)
        # We compare the value distribution, not row-for-row order
        assert raw_yield.sum() == pytest.approx(clean_yield.sum(), rel=0.001), (
            "Sum of yield_kg_per_ha changed between raw and cleaned data. "
            "Original yield values may have been overwritten."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Group 7 – Negative value tests
# ─────────────────────────────────────────────────────────────────────────────
class TestNegativeValues:

    def test_no_negative_area(self, clean_df):
        """area_1000_ha must not contain negative values."""
        n_neg = (clean_df["area_1000_ha"] < 0).sum()
        assert n_neg == 0, (
            f"Found {n_neg:,} negative area values in the cleaned dataset."
        )

    def test_no_negative_production(self, clean_df):
        """production_1000_t must not contain negative values."""
        n_neg = (clean_df["production_1000_t"] < 0).sum()
        assert n_neg == 0, (
            f"Found {n_neg:,} negative production values in the cleaned dataset."
        )

    def test_no_negative_yield(self, clean_df):
        """yield_kg_per_ha must not contain negative values."""
        n_neg = (clean_df["yield_kg_per_ha"] < 0).sum()
        assert n_neg == 0, (
            f"Found {n_neg:,} negative yield values in the cleaned dataset."
        )
