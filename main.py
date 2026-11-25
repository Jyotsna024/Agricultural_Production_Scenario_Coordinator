"""
PS 137 – Agricultural Production Scenario Coordinator
======================================================
main.py – Member 1 Data Pipeline Runner

PURPOSE
-------
Runs the complete Member 1 data pipeline in the correct order:
  1. Dataset collection   (src/data_collection/collect_dataset.py)
  2. Preprocessing        (src/preprocessing/preprocess_agriculture.py)

HOW TO RUN (from the project root folder)
------------------------------------------
  python main.py
  python main.py --collect-only
  python main.py --preprocess-only
  python main.py --validate-only

WHAT IT DOES
------------
  collect-only     : Downloads and saves raw dataset only.
  preprocess-only  : Runs cleaning on an already-downloaded dataset.
  validate-only    : Loads the cleaned CSV and prints a quick summary.
  (no flag)        : Runs all three steps in order.
"""

import argparse
import sys
from pathlib import Path

# ── Ensure the project root is on the Python path so imports work ────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_collection() -> None:
    """Step 1 – Download dataset from Hugging Face."""
    print("\n" + "=" * 60)
    print("  PHASE 2 – DATASET COLLECTION")
    print("=" * 60)
    from src.data_collection.collect_dataset import main as collect_main
    collect_main()


def run_preprocessing() -> None:
    """Step 2 – Clean and preprocess the raw dataset."""
    print("\n" + "=" * 60)
    print("  PHASE 5 – DATA PREPROCESSING")
    print("=" * 60)
    from src.preprocessing.preprocess_agriculture import main as preprocess_main
    preprocess_main()


def run_validation() -> None:
    """Step 3 – Quick validation of the cleaned CSV."""
    import pandas as pd

    print("\n" + "=" * 60)
    print("  PHASE 6 – QUICK VALIDATION")
    print("=" * 60)

    csv_path = PROJECT_ROOT / "data" / "processed" / "cleaned_agriculture_dataset.csv"

    if not csv_path.exists():
        print(f"[ERROR] Cleaned CSV not found at:\n  {csv_path}")
        print("  Run preprocessing first:  python main.py --preprocess-only")
        sys.exit(1)

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"\nLoaded cleaned CSV : {csv_path.name}")
    print(f"Rows               : {df.shape[0]:,}")
    print(f"Columns            : {df.shape[1]}")
    print(f"\nColumn list:")
    for col in df.columns:
        print(f"  - {col}")

    if "record_status" in df.columns:
        print(f"\nRecord status breakdown:")
        print(df["record_status"].value_counts().to_string())

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("\nMissing values : None")
    else:
        print(f"\nMissing values:")
        print(missing.to_string())

    print("\n[OK] Quick validation complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PS 137 – Member 1 Data Pipeline Runner"
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Run dataset collection only."
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Run preprocessing only (requires raw dataset)."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run quick validation only (requires cleaned CSV)."
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  PS 137 – Agricultural Production Scenario Coordinator")
    print("  Member 1 – Data Pipeline")
    print("=" * 60)
    print(f"  Project root : {PROJECT_ROOT}")

    if args.collect_only:
        run_collection()
    elif args.preprocess_only:
        run_preprocessing()
    elif args.validate_only:
        run_validation()
    else:
        # Run all phases in order
        run_collection()
        run_preprocessing()
        run_validation()

    print("\n" + "=" * 60)
    print("  Pipeline finished. Check data/ and reports/ for outputs.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
