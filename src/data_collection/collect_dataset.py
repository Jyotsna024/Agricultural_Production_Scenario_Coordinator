"""
PS 137 – Agricultural Production Scenario Coordinator
======================================================
src/data_collection/collect_dataset.py
Member 1 – Dataset Collection

PURPOSE
-------
Downloads the India Public Agriculture dataset from Hugging Face,
saves it as a Parquet file in data/raw/, and records metadata in
data/raw/sources.json.

IMPORTANT RULES
---------------
- The raw Parquet file is NEVER modified after saving.
- If the file already exists, the download is skipped.
- No data is fabricated; we only use what the dataset provides.

HOW TO RUN
----------
From the project root (PS137_Agricultural_Scenario_Coordinator/):
    python src/data_collection/collect_dataset.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Locate project root regardless of where this script is called from ──────
# This file is at:  <project_root>/src/data_collection/collect_dataset.py
# Going up two levels gives us the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PARQUET_PATH  = RAW_DIR / "crop_statistics.parquet"
SOURCES_PATH  = RAW_DIR / "sources.json"

DATASET_ID    = "therajasekhar/india-public-agriculture"
DATASET_URL   = f"https://huggingface.co/datasets/{DATASET_ID}"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: pretty section header
# ─────────────────────────────────────────────────────────────────────────────
def section(title: str) -> None:
    """Print a visible section header to the console."""
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Check whether the raw file already exists
# ─────────────────────────────────────────────────────────────────────────────
def raw_file_exists() -> bool:
    """Return True if the Parquet file has already been downloaded."""
    return PARQUET_PATH.exists() and PARQUET_PATH.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Download from Hugging Face and save as Parquet
# ─────────────────────────────────────────────────────────────────────────────
def download_dataset() -> pd.DataFrame:
    """
    Download the Hugging Face dataset and return it as a Pandas DataFrame.
    The 'datasets' library handles the connection and caching automatically.
    """
    section("STEP 1 – Downloading dataset from Hugging Face")
    print(f"Dataset ID : {DATASET_ID}")
    print(f"URL        : {DATASET_URL}")
    print("Please wait – this may take a few minutes on the first run …\n")

    try:
        # Import here so the rest of the script works even if 'datasets'
        # is not installed yet (the error message will tell the user).
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("[ERROR] The 'datasets' library is not installed.")
        print("        Run:  pip install datasets")
        sys.exit(1)

    try:
        # load_dataset returns a DatasetDict; 'train' is the default split.
        hf_dataset = load_dataset(DATASET_ID)
        print(f"Available splits : {list(hf_dataset.keys())}")

        # Use the first available split (usually 'train').
        split_name = list(hf_dataset.keys())[0]
        df = hf_dataset[split_name].to_pandas()
        print(f"Using split      : '{split_name}'")
        return df

    except Exception as err:
        print(f"[ERROR] Failed to download dataset.\nReason: {err}")
        print("\nTroubleshooting tips:")
        print("  1. Check your internet connection.")
        print("  2. Verify the dataset ID on https://huggingface.co/datasets/therajasekhar/india-public-agriculture")
        print("  3. Make sure you have run:  pip install datasets pyarrow")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Save the DataFrame as Parquet (raw, unmodified)
# ─────────────────────────────────────────────────────────────────────────────
def save_parquet(df: pd.DataFrame) -> None:
    """
    Save the DataFrame to data/raw/crop_statistics.parquet.
    Parquet is a column-oriented format that preserves data types
    and is much faster to read than CSV for large datasets.
    """
    section("STEP 2 – Saving raw dataset as Parquet")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df.to_parquet(PARQUET_PATH, index=False, engine="pyarrow")
    size_mb = PARQUET_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved to   : {PARQUET_PATH}")
    print(f"File size  : {size_mb:.2f} MB")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Display basic dataset information
# ─────────────────────────────────────────────────────────────────────────────
def display_info(df: pd.DataFrame) -> None:
    """Print shape, column names, data types, and first 5 rows."""
    section("STEP 3 – Dataset overview")

    print(f"\nShape         : {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
    print(f"\nColumn names  : {list(df.columns)}")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    # Quick check for expected columns
    expected_cols = {
        "state", "district", "year", "crop",
        "area_1000_ha", "production_1000_t", "yield_kg_per_ha"
    }
    actual_cols = set(df.columns)
    missing_expected = expected_cols - actual_cols
    extra_cols       = actual_cols - expected_cols

    print("\nColumn verification:")
    if missing_expected:
        print(f"  [WARNING] Expected columns NOT found  : {missing_expected}")
    else:
        print("  [OK] All expected columns are present.")
    if extra_cols:
        print(f"  [INFO]    Extra columns in dataset     : {extra_cols}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Update sources.json with actual collection details
# ─────────────────────────────────────────────────────────────────────────────
def update_sources_json(df: pd.DataFrame) -> None:
    """
    Write (or update) data/raw/sources.json with the actual dataset
    metadata discovered after download.
    """
    section("STEP 4 – Recording collection metadata")

    # Gather actual stats from the downloaded DataFrame
    year_col = "year" if "year" in df.columns else None
    year_min = int(df[year_col].min()) if year_col else "unknown"
    year_max = int(df[year_col].max()) if year_col else "unknown"

    metadata = {
        "dataset_name"       : "India Public Agriculture Dataset",
        "source"             : "Hugging Face",
        "huggingface_id"     : DATASET_ID,
        "url"                : DATASET_URL,
        "collection_date"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_saved_as"      : "data/raw/crop_statistics.parquet",
        "actual_rows"        : int(df.shape[0]),
        "actual_columns"     : int(df.shape[1]),
        "actual_column_names": list(df.columns),
        "year_range"         : f"{year_min} to {year_max}",
        "file_size_bytes"    : int(PARQUET_PATH.stat().st_size),
        "description"        : (
            "Historical agricultural production records for India "
            "covering states, districts, crops, area, production, and yield."
        ),
        "units": {
            "area_1000_ha"      : "Thousand hectares",
            "production_1000_t" : "Thousand tonnes",
            "yield_kg_per_ha"   : "Kilograms per hectare"
        },
        "license"      : "Public – verify on Hugging Face dataset card",
        "collected_by" : "Member 1",
        "project_id"   : "PS 137"
    }

    with open(SOURCES_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"Metadata saved to : {SOURCES_PATH}")
    print(f"Rows collected    : {int(df.shape[0]):,}")
    print(f"Columns           : {int(df.shape[1])}")
    print(f"Year range        : {year_min} – {year_max}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Orchestrate the data collection pipeline:
      1. Check if already downloaded.
      2. Download from Hugging Face (if needed).
      3. Save as Parquet.
      4. Display dataset info.
      5. Update sources.json.
    """
    section("PS 137 – DATA COLLECTION  (Member 1)")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Target file  : {PARQUET_PATH}")

    if raw_file_exists():
        # ── Skip download; just load and inspect ────────────────────────────
        print("\n[INFO] Raw Parquet file already exists – skipping download.")
        print("       Loading existing file for inspection …")
        df = pd.read_parquet(PARQUET_PATH)
    else:
        # ── Full download pipeline ───────────────────────────────────────────
        df = download_dataset()
        save_parquet(df)

    display_info(df)
    update_sources_json(df)

    section("COLLECTION COMPLETE")
    print("Next step: run the EDA notebook or the preprocessing script.")
    print(f"  Parquet : {PARQUET_PATH}")
    print(f"  Sources : {SOURCES_PATH}\n")


if __name__ == "__main__":
    main()
