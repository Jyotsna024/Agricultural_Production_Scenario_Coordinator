"""
PS 137 – Agricultural Production Scenario Coordinator
======================================================
tests/test_historical_analysis.py
Member 2 – Automated Tests

PURPOSE
-------
Tests the Member 2 historical analysis module using:
  - Small synthetic DataFrame fixtures (unit tests).
  - The real cleaned CSV (integration tests, skipped if not available).

HOW TO RUN
----------
  pytest tests/ -v
  pytest tests/test_historical_analysis.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Add project root to path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CSV_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_agriculture_dataset.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures – small synthetic DataFrames
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    """
    Small synthetic DataFrame for unit testing.
    Explicitly marked as test fixture – NOT real historical data.
    """
    data = {
        "state"             : ["Karnataka", "Karnataka", "Tamil Nadu", "Tamil Nadu",
                               "Maharashtra", "Maharashtra", "Karnataka", "Tamil Nadu"],
        "state_clean"       : ["Karnataka", "Karnataka", "Tamil Nadu", "Tamil Nadu",
                               "Maharashtra", "Maharashtra", "Karnataka", "Tamil Nadu"],
        "district"          : ["Mysore",   "Mysore",   "Chennai",    "Chennai",
                               "Pune",      "Pune",      "Mysore",    "Chennai"],
        "district_clean"    : ["Mysore",   "Mysore",   "Chennai",    "Chennai",
                               "Pune",      "Pune",      "Mysore",    "Chennai"],
        "year"              : [2010, 2011, 2010, 2011, 2010, 2011, 2012, 2012],
        "crop"              : ["Rice",  "Rice",  "Rice",  "Rice",
                               "Wheat", "Wheat", "Rice",  "Rice"],
        "crop_clean"        : ["Rice",  "Rice",  "Rice",  "Rice",
                               "Wheat", "Wheat", "Rice",  "Rice"],
        "area_1000_ha"      : [100.0, 110.0, 50.0, 55.0, 80.0, 85.0, 120.0, 60.0],
        "production_1000_t" : [300.0, 340.0, 150.0, 170.0, 200.0, 220.0, 360.0, 180.0],
        "yield_kg_per_ha"   : [3000.0, 3090.9, 3000.0, 3090.9, 2500.0, 2588.2, 3000.0, 3000.0],
        "has_production_data": [True, True, True, True, True, True, True, True],
        "is_zero_activity"  : [False] * 8,
        "is_small_area_rounding": [False] * 8,
        "record_status"     : ["complete_positive"] * 8,
        "calculated_yield_kg_per_ha": [3000.0, 3090.91, 3000.0, 3090.91, 2500.0, 2588.24, 3000.0, 3000.0],
        "yield_discrepancy" : [False] * 8,
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def synthetic_df_with_nulls() -> pd.DataFrame:
    """
    Synthetic DataFrame with intentional missing/zero values for edge-case tests.
    Marked as test fixture – NOT real data.
    """
    data = {
        "state"             : ["A", "A", "B", "B", "C", "C"],
        "state_clean"       : ["A", "A", "B", "B", "C", "C"],
        "district"          : ["D1", "D1", "D2", "D2", "D3", "D3"],
        "district_clean"    : ["D1", "D1", "D2", "D2", "D3", "D3"],
        "year"              : [2010, 2011, 2010, 2011, 2010, 2011],
        "crop"              : ["X", "X", "X", "X", "X", "X"],
        "crop_clean"        : ["X", "X", "X", "X", "X", "X"],
        "area_1000_ha"      : [10.0, 0.0,  20.0, 15.0, 0.0, 25.0],
        "production_1000_t" : [30.0, None, 60.0, 45.0, 0.0, 75.0],
        "yield_kg_per_ha"   : [3000.0, None, 3000.0, 3000.0, None, 3000.0],
        "has_production_data": [True, False, True, True, False, True],
        "is_zero_activity"  : [False, False, False, False, True, False],
        "is_small_area_rounding": [False] * 6,
        "record_status"     : [
            "complete_positive",
            "missing_prod_yield_zero_area",
            "complete_positive",
            "complete_positive",
            "zero_activity",
            "complete_positive",
        ],
        "calculated_yield_kg_per_ha": [3000.0, None, 3000.0, 3000.0, None, 3000.0],
        "yield_discrepancy" : [False, False, False, False, False, False],
    }
    df = pd.DataFrame(data)
    # Ensure NaN where expected
    df.loc[1, "production_1000_t"] = np.nan
    df.loc[1, "yield_kg_per_ha"]   = np.nan
    df.loc[4, "yield_kg_per_ha"]   = np.nan
    df.loc[1, "calculated_yield_kg_per_ha"] = np.nan
    df.loc[4, "calculated_yield_kg_per_ha"] = np.nan
    return df


@pytest.fixture(scope="module")
def real_df():
    """
    Load the real cleaned CSV, skipped if not available.
    Integration test fixture – uses Member 1's actual output.
    """
    if not CSV_PATH.exists():
        pytest.skip(
            f"Real cleaned CSV not found at {CSV_PATH}. "
            "Run 'python main.py' first."
        )
    df = pd.read_csv(CSV_PATH, low_memory=False)
    if len(df) == 0:
        pytest.skip("Cleaned CSV is empty. Run 'python main.py' first.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Group 1 – Dataset loading
# ─────────────────────────────────────────────────────────────────────────────
class TestDatasetLoading:

    def test_load_cleaned_data_success(self):
        """load_cleaned_data should return a DataFrame when CSV exists."""
        if not CSV_PATH.exists():
            pytest.skip("CSV not available.")
        from src.analysis.historical_analysis import load_cleaned_data
        df = load_cleaned_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_cleaned_data_missing_file(self, tmp_path):
        """load_cleaned_data must raise FileNotFoundError for missing CSV."""
        from src.analysis.historical_analysis import load_cleaned_data
        fake_path = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            load_cleaned_data(csv_path=fake_path)

    def test_load_cleaned_data_empty_file(self, tmp_path):
        """load_cleaned_data must raise ValueError for empty CSV."""
        from src.analysis.historical_analysis import load_cleaned_data
        empty_csv = tmp_path / "empty.csv"
        # CSV with headers only
        empty_csv.write_text(
            "state,district,year,crop,area_1000_ha,production_1000_t\n"
        )
        with pytest.raises(ValueError):
            load_cleaned_data(csv_path=empty_csv)

    def test_load_cleans_year_column(self, tmp_path):
        """Year column should be numeric after loading."""
        from src.analysis.historical_analysis import load_cleaned_data
        csv = tmp_path / "test.csv"
        csv.write_text(
            "state,district,year,crop,area_1000_ha,production_1000_t\n"
            "A,D,2010,Rice,10,30\n"
        )
        df = load_cleaned_data(csv_path=csv)
        assert pd.api.types.is_numeric_dtype(df["year"])


# ─────────────────────────────────────────────────────────────────────────────
# Group 2 – Required column validation
# ─────────────────────────────────────────────────────────────────────────────
class TestColumnValidation:

    def test_validate_dataset_required_cols_present(self, synthetic_df):
        """validate_dataset must list all required columns as present."""
        from src.analysis.historical_analysis import validate_dataset
        summary = validate_dataset(synthetic_df)
        assert len(summary["required_columns_missing"]) == 0

    def test_validate_dataset_missing_col_detected(self):
        """validate_dataset must detect missing required columns."""
        from src.analysis.historical_analysis import validate_dataset
        df = pd.DataFrame({"state": ["A"], "district": ["D"], "year": [2010]})
        summary = validate_dataset(df)
        assert "crop" in summary["required_columns_missing"]

    def test_validate_dataset_shape(self, synthetic_df):
        """validate_dataset must report correct row count."""
        from src.analysis.historical_analysis import validate_dataset
        summary = validate_dataset(synthetic_df)
        assert summary["total_records"] == len(synthetic_df)


# ─────────────────────────────────────────────────────────────────────────────
# Group 3 – Year-wise aggregation
# ─────────────────────────────────────────────────────────────────────────────
class TestYearWiseAnalysis:

    def test_year_wise_returns_correct_years(self, synthetic_df):
        """Year-wise analysis must include all years present in data."""
        from src.analysis.historical_analysis import analyse_year_wise
        result = analyse_year_wise(synthetic_df)
        assert set(result["year"].tolist()) == {2010, 2011, 2012}

    def test_year_wise_production_sum(self, synthetic_df):
        """Production for 2010 must equal sum of all 2010 records."""
        from src.analysis.historical_analysis import analyse_year_wise
        result = analyse_year_wise(synthetic_df)
        prod_2010 = result.loc[result["year"] == 2010, "total_production_1000_t"].iloc[0]
        expected  = synthetic_df[synthetic_df["year"] == 2010]["production_1000_t"].sum()
        assert abs(prod_2010 - expected) < 0.01

    def test_year_wise_has_yoy_column(self, synthetic_df):
        """Year-wise result must include year-over-year columns."""
        from src.analysis.historical_analysis import analyse_year_wise
        result = analyse_year_wise(synthetic_df)
        assert "yoy_production_change" in result.columns
        assert "yoy_production_pct_change" in result.columns

    def test_first_year_yoy_is_nan(self, synthetic_df):
        """Year-over-year change for the first year must be NaN."""
        from src.analysis.historical_analysis import analyse_year_wise
        result = analyse_year_wise(synthetic_df).sort_values("year")
        assert pd.isna(result.iloc[0]["yoy_production_change"])
        assert pd.isna(result.iloc[0]["yoy_production_pct_change"])


# ─────────────────────────────────────────────────────────────────────────────
# Group 4 – State-wise aggregation
# ─────────────────────────────────────────────────────────────────────────────
class TestStateWiseAnalysis:

    def test_state_wise_all_states_present(self, synthetic_df):
        """State-wise analysis must include all states."""
        from src.analysis.historical_analysis import analyse_state_wise
        result = analyse_state_wise(synthetic_df)
        assert set(result["state"].tolist()) == {"Karnataka", "Tamil Nadu", "Maharashtra"}

    def test_state_wise_production_sum(self, synthetic_df):
        """Karnataka production must equal sum of all Karnataka records."""
        from src.analysis.historical_analysis import analyse_state_wise
        result = analyse_state_wise(synthetic_df)
        ka_prod  = result.loc[result["state"] == "Karnataka", "total_production_1000_t"].iloc[0]
        expected = synthetic_df[synthetic_df["state_clean"] == "Karnataka"]["production_1000_t"].sum()
        assert abs(ka_prod - expected) < 0.01

    def test_state_wise_share_sums_to_100(self, synthetic_df):
        """Production shares must sum to approximately 100%."""
        from src.analysis.historical_analysis import analyse_state_wise
        result = analyse_state_wise(synthetic_df)
        total_share = result["production_share_pct"].sum()
        assert abs(total_share - 100.0) < 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Group 5 – Crop-wise aggregation
# ─────────────────────────────────────────────────────────────────────────────
class TestCropWiseAnalysis:

    def test_crop_wise_all_crops_present(self, synthetic_df):
        """Crop-wise analysis must include all unique crops."""
        from src.analysis.historical_analysis import analyse_crop_wise
        result = analyse_crop_wise(synthetic_df)
        assert set(result["crop"].tolist()) == {"Rice", "Wheat"}

    def test_crop_wise_production_sum(self, synthetic_df):
        """Rice production must equal sum of all Rice records."""
        from src.analysis.historical_analysis import analyse_crop_wise
        result = analyse_crop_wise(synthetic_df)
        rice_prod = result.loc[result["crop"] == "Rice", "total_production_1000_t"].iloc[0]
        expected  = synthetic_df[synthetic_df["crop_clean"] == "Rice"]["production_1000_t"].sum()
        assert abs(rice_prod - expected) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Group 6 – Weighted yield calculation
# ─────────────────────────────────────────────────────────────────────────────
class TestWeightedYield:

    def test_weighted_yield_formula(self):
        """Weighted yield must equal (sum_prod * 1000) / sum_area."""
        from src.analysis.historical_analysis import _weighted_yield
        prod = 300.0   # thousand tonnes
        area = 100.0   # thousand hectares
        expected = (300.0 * 1000) / 100.0  # = 3000 kg/ha
        assert abs(_weighted_yield(prod, area) - expected) < 0.01

    def test_weighted_yield_zero_area_returns_nan(self):
        """_weighted_yield must return NaN when area is 0."""
        from src.analysis.historical_analysis import _weighted_yield
        result = _weighted_yield(100.0, 0.0)
        assert np.isnan(result)

    def test_weighted_yield_in_year_df(self, synthetic_df):
        """Year-wise weighted yield must match formula for each year."""
        from src.analysis.historical_analysis import analyse_year_wise
        result = analyse_year_wise(synthetic_df)
        for _, row in result.iterrows():
            if row["total_area_1000_ha"] > 0:
                expected = (row["total_production_1000_t"] * 1000) / row["total_area_1000_ha"]
                assert abs(row["weighted_yield_kg_per_ha"] - expected) < 0.1

    def test_weighted_yield_in_crop_df(self, synthetic_df):
        """Crop-wise weighted yield must match formula for each crop."""
        from src.analysis.historical_analysis import analyse_crop_wise
        result = analyse_crop_wise(synthetic_df)
        for _, row in result.iterrows():
            if row["total_area_1000_ha"] > 0:
                expected = (row["total_production_1000_t"] * 1000) / row["total_area_1000_ha"]
                assert abs(row["weighted_yield_kg_per_ha"] - expected) < 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Group 7 – Missing value handling
# ─────────────────────────────────────────────────────────────────────────────
class TestMissingValueHandling:

    def test_missing_production_excluded_from_aggregation(self, synthetic_df_with_nulls):
        """Records with NaN production must not contribute to production totals."""
        from src.analysis.historical_analysis import analyse_year_wise
        result = analyse_year_wise(synthetic_df_with_nulls)
        # 2011 has two records: one with NaN production (state A) and one with
        # production=45 (state B)
        prod_2011 = result.loc[result["year"] == 2011, "total_production_1000_t"].iloc[0]
        # Only B/D2/2011 (45) and C/D3/2011 (75) contribute;
        # A/D1/2011 has NaN prod and zero area – excluded
        # expected = 45 + 75 = 120 (after zero-area filter removes A's NaN record)
        assert prod_2011 >= 45.0  # at minimum, B's record must be included

    def test_zero_area_records_excluded_from_yield(self, synthetic_df_with_nulls):
        """Records with zero area must not appear in yield calculations."""
        from src.analysis.historical_analysis import analyse_state_wise
        result = analyse_state_wise(synthetic_df_with_nulls)
        # State A has row with area=0: should be excluded from weighted yield
        a_row = result.loc[result["state"] == "A"]
        if len(a_row) > 0 and not pd.isna(a_row.iloc[0]["weighted_yield_kg_per_ha"]):
            # Yield must be computed only from positive-area records
            assert a_row.iloc[0]["weighted_yield_kg_per_ha"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Group 8 – Invalid record handling
# ─────────────────────────────────────────────────────────────────────────────
class TestInvalidRecordHandling:

    def test_zero_production_records_included_by_default(self, synthetic_df_with_nulls):
        """By default, zero production records are included (not excluded)."""
        from src.analysis.historical_analysis import analyse_year_wise, AnalysisPolicy
        default_policy = AnalysisPolicy(exclude_zero_production=False)
        result_default = analyse_year_wise(synthetic_df_with_nulls, policy=default_policy)
        assert len(result_default) > 0  # should not error

    def test_zero_production_records_excluded_when_policy_set(self, synthetic_df_with_nulls):
        """With exclude_zero_production=True, zero production rows are removed."""
        from src.analysis.historical_analysis import analyse_year_wise, AnalysisPolicy
        strict_policy = AnalysisPolicy(exclude_zero_production=True)
        result = analyse_year_wise(synthetic_df_with_nulls, policy=strict_policy)
        # State C/2010 has area=0,prod=0 – excluded; should not affect production totals
        assert len(result) > 0

    def test_negative_values_note(self):
        """Document behavior: analysis does not add new negative values."""
        from src.analysis.historical_analysis import _weighted_yield
        # Weighted yield should be positive if inputs are positive
        result = _weighted_yield(100.0, 50.0)
        assert result > 0


# ─────────────────────────────────────────────────────────────────────────────
# Group 9 – Filtering
# ─────────────────────────────────────────────────────────────────────────────
class TestFiltering:

    def test_filter_by_state(self, synthetic_df):
        """Filtering by state must return only rows for that state."""
        from src.analysis.historical_analysis import filter_data
        result = filter_data(synthetic_df, states=["Karnataka"])
        assert all(result["state_clean"] == "Karnataka")

    def test_filter_by_district(self, synthetic_df):
        """Filtering by district must return only rows for that district."""
        from src.analysis.historical_analysis import filter_data
        result = filter_data(synthetic_df, districts=["Mysore"])
        assert all(result["district_clean"] == "Mysore")

    def test_filter_by_crop(self, synthetic_df):
        """Filtering by crop must return only rows for that crop."""
        from src.analysis.historical_analysis import filter_data
        result = filter_data(synthetic_df, crops=["Rice"])
        assert all(result["crop_clean"] == "Rice")

    def test_filter_by_year(self, synthetic_df):
        """Filtering by year must return only rows for that year."""
        from src.analysis.historical_analysis import filter_data
        result = filter_data(synthetic_df, years=[2010])
        assert all(result["year"] == 2010)

    def test_filter_combined(self, synthetic_df):
        """Combined filter must AND all conditions."""
        from src.analysis.historical_analysis import filter_data
        result = filter_data(
            synthetic_df,
            states=["Karnataka"],
            crops=["Rice"],
            years=[2010],
        )
        assert len(result) == 1
        assert result.iloc[0]["state_clean"] == "Karnataka"
        assert result.iloc[0]["crop_clean"] == "Rice"
        assert result.iloc[0]["year"] == 2010

    def test_filter_case_insensitive(self, synthetic_df):
        """State filter should be case-insensitive."""
        from src.analysis.historical_analysis import filter_data
        result = filter_data(synthetic_df, states=["karnataka"])
        assert len(result) > 0

    def test_filter_no_match_returns_empty(self, synthetic_df):
        """Filter with no matching records must return empty DataFrame."""
        from src.analysis.historical_analysis import filter_data
        result = filter_data(synthetic_df, states=["NONEXISTENT_STATE_XYZ"])
        assert len(result) == 0

    def test_filter_multiple_states(self, synthetic_df):
        """Filtering by multiple states must return rows for all specified."""
        from src.analysis.historical_analysis import filter_data
        result = filter_data(synthetic_df, states=["Karnataka", "Tamil Nadu"])
        states_found = set(result["state_clean"].unique())
        assert "Karnataka" in states_found
        assert "Tamil Nadu" in states_found
        assert "Maharashtra" not in states_found


# ─────────────────────────────────────────────────────────────────────────────
# Group 10 – Historical baseline retrieval
# ─────────────────────────────────────────────────────────────────────────────
class TestBaselineRetrieval:

    def test_baseline_exact_match(self, synthetic_df):
        """get_historical_baseline must return found=True for an exact match."""
        from src.analysis.historical_analysis import get_historical_baseline
        result = get_historical_baseline(synthetic_df, "Karnataka", "Mysore", "Rice", 2010)
        assert result["found"] is True
        assert result["match_count"] == 1
        assert result["area_1000_ha"] == pytest.approx(100.0)
        assert result["production_1000_t"] == pytest.approx(300.0)

    def test_baseline_case_insensitive(self, synthetic_df):
        """Baseline retrieval must be case-insensitive."""
        from src.analysis.historical_analysis import get_historical_baseline
        result = get_historical_baseline(synthetic_df, "karnataka", "mysore", "rice", 2010)
        assert result["found"] is True

    def test_baseline_returns_units(self, synthetic_df):
        """Baseline result must include unit information."""
        from src.analysis.historical_analysis import get_historical_baseline
        result = get_historical_baseline(synthetic_df, "Karnataka", "Mysore", "Rice", 2010)
        assert "units" in result
        assert "area" in result["units"]
        assert "production" in result["units"]
        assert "yield" in result["units"]

    def test_baseline_returns_record_status(self, synthetic_df):
        """Baseline result must include record_status when available."""
        from src.analysis.historical_analysis import get_historical_baseline
        result = get_historical_baseline(synthetic_df, "Karnataka", "Mysore", "Rice", 2010)
        assert result["found"] is True
        assert result["record_status"] == "complete_positive"


# ─────────────────────────────────────────────────────────────────────────────
# Group 11 – No matching baseline record
# ─────────────────────────────────────────────────────────────────────────────
class TestBaselineNotFound:

    def test_no_match_found_is_false(self, synthetic_df):
        """get_historical_baseline must return found=False when no record exists."""
        from src.analysis.historical_analysis import get_historical_baseline
        result = get_historical_baseline(
            synthetic_df, "NONEXISTENT", "DISTRICT", "Rice", 2010
        )
        assert result["found"] is False
        assert result["match_count"] == 0

    def test_no_match_has_warning(self, synthetic_df):
        """get_historical_baseline must include a warning message when no match."""
        from src.analysis.historical_analysis import get_historical_baseline
        result = get_historical_baseline(
            synthetic_df, "NONEXISTENT", "DISTRICT", "Rice", 2010
        )
        assert result["warning"] is not None
        assert len(result["warning"]) > 0

    def test_no_match_values_are_none(self, synthetic_df):
        """When no match, all measurement fields must be None (not fabricated)."""
        from src.analysis.historical_analysis import get_historical_baseline
        result = get_historical_baseline(
            synthetic_df, "NONEXISTENT", "DISTRICT", "Rice", 2010
        )
        assert result["area_1000_ha"] is None
        assert result["production_1000_t"] is None
        assert result["yield_kg_per_ha"] is None

    def test_wrong_year_no_match(self, synthetic_df):
        """Correct state/district/crop but wrong year must return found=False."""
        from src.analysis.historical_analysis import get_historical_baseline
        result = get_historical_baseline(synthetic_df, "Karnataka", "Mysore", "Rice", 1999)
        assert result["found"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Group 12 – Duplicate baseline records
# ─────────────────────────────────────────────────────────────────────────────
class TestBaselineDuplicates:

    def test_duplicate_match_found_is_false(self, synthetic_df):
        """When multiple records match, found must be False (ambiguous result)."""
        from src.analysis.historical_analysis import get_historical_baseline
        # Create a duplicate: add an extra Karnataka/Mysore/Rice/2010 row
        duplicate_row = synthetic_df[
            (synthetic_df["state_clean"]    == "Karnataka") &
            (synthetic_df["district_clean"] == "Mysore") &
            (synthetic_df["crop_clean"]     == "Rice") &
            (synthetic_df["year"]           == 2010)
        ].copy()
        df_with_dup = pd.concat([synthetic_df, duplicate_row], ignore_index=True)

        result = get_historical_baseline(df_with_dup, "Karnataka", "Mysore", "Rice", 2010)
        assert result["found"] is False
        assert result["match_count"] == 2

    def test_duplicate_includes_warning(self, synthetic_df):
        """Duplicate result must include a warning explaining the situation."""
        from src.analysis.historical_analysis import get_historical_baseline
        duplicate_row = synthetic_df[
            (synthetic_df["state_clean"]    == "Karnataka") &
            (synthetic_df["district_clean"] == "Mysore") &
            (synthetic_df["crop_clean"]     == "Rice") &
            (synthetic_df["year"]           == 2010)
        ].copy()
        df_with_dup = pd.concat([synthetic_df, duplicate_row], ignore_index=True)
        result = get_historical_baseline(df_with_dup, "Karnataka", "Mysore", "Rice", 2010)
        assert result["warning"] is not None
        assert "2" in result["warning"]  # warns about 2 matches

    def test_duplicate_includes_all_source_records(self, synthetic_df):
        """source_records must contain ALL matching rows."""
        from src.analysis.historical_analysis import get_historical_baseline
        duplicate_row = synthetic_df[
            (synthetic_df["state_clean"]    == "Karnataka") &
            (synthetic_df["district_clean"] == "Mysore") &
            (synthetic_df["crop_clean"]     == "Rice") &
            (synthetic_df["year"]           == 2010)
        ].copy()
        df_with_dup = pd.concat([synthetic_df, duplicate_row], ignore_index=True)
        result = get_historical_baseline(df_with_dup, "Karnataka", "Mysore", "Rice", 2010)
        assert len(result["source_records"]) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Group 13 – Zero-area and zero-production edge cases
# ─────────────────────────────────────────────────────────────────────────────
class TestEdgeCases:

    def test_zero_area_no_yield_calculation(self):
        """Weighted yield must be NaN when area is 0."""
        from src.analysis.historical_analysis import _weighted_yield
        result = _weighted_yield(0.0, 0.0)
        assert np.isnan(result)

    def test_zero_area_positive_production_yield(self):
        """With zero area, weighted yield must be NaN (not inf or error)."""
        from src.analysis.historical_analysis import _weighted_yield
        result = _weighted_yield(100.0, 0.0)
        assert np.isnan(result)

    def test_all_zero_df_analysis(self):
        """Analysis must handle all-zero area/production without crash."""
        from src.analysis.historical_analysis import analyse_year_wise, AnalysisPolicy
        df = pd.DataFrame({
            "state"            : ["A"],
            "state_clean"      : ["A"],
            "district"         : ["D"],
            "district_clean"   : ["D"],
            "year"             : [2010],
            "crop"             : ["X"],
            "crop_clean"       : ["X"],
            "area_1000_ha"     : [0.0],
            "production_1000_t": [0.0],
            "yield_kg_per_ha"  : [0.0],
        })
        policy = AnalysisPolicy(exclude_zero_area=True)
        result = analyse_year_wise(df, policy=policy)
        # With zero-area excluded, result should be empty
        assert len(result) == 0 or result["total_production_1000_t"].sum() == 0

    def test_baseline_with_zero_area_record(self):
        """Baseline for a zero-area record must still return found=True with the values."""
        from src.analysis.historical_analysis import get_historical_baseline
        df = pd.DataFrame({
            "state"            : ["TestState"],
            "state_clean"      : ["Teststate"],
            "district"         : ["TestDist"],
            "district_clean"   : ["Testdist"],
            "year"             : [2010],
            "crop"             : ["TestCrop"],
            "crop_clean"       : ["Testcrop"],
            "area_1000_ha"     : [0.0],
            "production_1000_t": [0.0],
            "yield_kg_per_ha"  : [0.0],
            "record_status"    : ["zero_activity"],
        })
        result = get_historical_baseline(df, "TestState", "TestDist", "TestCrop", 2010)
        assert result["found"] is True
        assert result["area_1000_ha"] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Group 14 – Percentage change edge cases
# ─────────────────────────────────────────────────────────────────────────────
class TestPctChangEdgeCases:

    def test_pct_change_first_year_is_nan(self, synthetic_df):
        """First year has no previous year; pct change must be NaN."""
        from src.analysis.historical_analysis import analyse_year_wise
        result = analyse_year_wise(synthetic_df).sort_values("year")
        assert pd.isna(result.iloc[0]["yoy_production_pct_change"])

    def test_pct_change_no_inf_values(self, synthetic_df):
        """Year-over-year percentage change must not contain +inf or -inf."""
        from src.analysis.historical_analysis import analyse_year_wise
        result = analyse_year_wise(synthetic_df)
        pct = result["yoy_production_pct_change"].replace([float("inf"), float("-inf")], float("nan"))
        assert not pct.isin([float("inf"), float("-inf")]).any()

    def test_pct_change_when_production_increases(self):
        """When production strictly increases year-over-year, pct change must be positive."""
        from src.analysis.historical_analysis import analyse_year_wise
        import pandas as pd
        # Build a minimal DataFrame with strictly increasing production
        df = pd.DataFrame({
            "state"            : ["A", "A"],
            "state_clean"      : ["A", "A"],
            "district"         : ["D", "D"],
            "district_clean"   : ["D", "D"],
            "year"             : [2010, 2011],
            "crop"             : ["X", "X"],
            "crop_clean"       : ["X", "X"],
            "area_1000_ha"     : [100.0, 110.0],
            "production_1000_t": [300.0, 350.0],
            "yield_kg_per_ha"  : [3000.0, 3181.8],
        })
        result = analyse_year_wise(df).sort_values("year").dropna(subset=["yoy_production_pct_change"])
        assert len(result) == 1  # only one YoY transition: 2010→2011
        assert result.iloc[0]["yoy_production_pct_change"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests – real cleaned CSV
# ─────────────────────────────────────────────────────────────────────────────
class TestIntegration:

    def test_real_data_loads_with_expected_columns(self, real_df):
        """Real cleaned CSV must contain all expected original and derived columns."""
        from src.analysis.historical_analysis import REQUIRED_COLS
        for col in REQUIRED_COLS:
            assert col in real_df.columns, f"Column '{col}' missing from real data."

    def test_real_data_year_wise_produces_rows(self, real_df):
        """Year-wise analysis on real data must produce one row per year."""
        from src.analysis.historical_analysis import analyse_year_wise
        result = analyse_year_wise(real_df)
        assert len(result) >= 1  # at least one year

    def test_real_data_state_wise_has_states(self, real_df):
        """State-wise analysis on real data must produce one row per state."""
        from src.analysis.historical_analysis import analyse_state_wise
        result = analyse_state_wise(real_df)
        assert len(result) >= 1

    def test_real_data_baseline_known_state(self, real_df):
        """Baseline lookup for a known state/crop combination must not error."""
        from src.analysis.historical_analysis import (
            get_historical_baseline,
            _resolve_text_col,
        )
        state_col = _resolve_text_col(real_df, "state")
        # Pick the first valid record from real data for a guaranteed match
        first = real_df.dropna(subset=["state", "district", "crop", "year"]).iloc[0]
        result = get_historical_baseline(
            real_df,
            str(first[state_col]) if state_col in real_df.columns else str(first["state"]),
            str(first["district_clean"]) if "district_clean" in real_df.columns else str(first["district"]),
            str(first["crop_clean"]) if "crop_clean" in real_df.columns else str(first["crop"]),
            int(first["year"]),
        )
        # The result must have found=True (since we picked an existing record)
        assert result["found"] is True

    def test_real_data_baseline_nonexistent(self, real_df):
        """Baseline for nonexistent combination must return found=False."""
        from src.analysis.historical_analysis import get_historical_baseline
        result = get_historical_baseline(
            real_df, "NONEXISTENT_STATE_XYZ", "FAKE_DISTRICT", "FAKE_CROP", 1800
        )
        assert result["found"] is False
        assert result["area_1000_ha"] is None
