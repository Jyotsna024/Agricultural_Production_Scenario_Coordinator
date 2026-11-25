# PS 137 – Preprocessing Documentation

## Overview

This document describes every step Member 1 performed to transform the raw
downloaded dataset into the final cleaned CSV used by all team members.

**Input file:**  `data/raw/crop_statistics.parquet`
**Output file:** `data/processed/cleaned_agriculture_dataset.csv`

---

## 1. Data Collection Process

### How it was done
The dataset was downloaded from Hugging Face using the `datasets` Python library:

```python
from datasets import load_dataset
hf_dataset = load_dataset("therajasekhar/india-public-agriculture")
df = hf_dataset["train"].to_pandas()
df.to_parquet("data/raw/crop_statistics.parquet", index=False)
```

### Key principles
- The raw Parquet file is saved **once** and **never modified**.
- All cleaning is done on a **copy** of the DataFrame.
- If the Parquet file already exists, the download is skipped.
- Metadata (rows, columns, year range) is recorded in `data/raw/sources.json`.

---

## 2. Exploratory Data Analysis (EDA)

EDA was performed in `notebooks/01_dataset_exploration.ipynb`.

### What was examined
1. Dataset shape (rows × columns)
2. First five records
3. Column names and data types
4. Descriptive statistics (mean, std, min, max, quartiles)
5. Missing values and their percentages
6. Duplicate records
7. Unique states, districts, and crops
8. Year coverage
9. Zero-value analysis (area = 0, production = 0)
10. Negative-value analysis
11. Min/max for area, production, and yield
12. State-wise, crop-wise, and year-wise production summaries
13. Visualisations (bar charts, heatmaps, histograms)

> Full EDA results are in the notebook. Key findings are summarised in
> `reports/data_quality_audit.txt`.

---

## 3. Data Quality Audit

Performed by `src/preprocessing/preprocess_agriculture.py` and recorded in
`reports/data_quality_audit.txt`.

### Checks performed
| Check | What was looked for |
|-------|-------------------|
| Missing values | NaN in any column |
| Exact duplicates | Identical rows |
| Composite key duplicates | Same (state, district, year, crop) repeated |
| Invalid data types | Non-numeric in numeric columns |
| Negative values | area < 0, production < 0, yield < 0 |
| Zero area | area_1000_ha == 0 |
| Zero production | production_1000_t == 0 with area > 0 |
| Unusually high yield | yield > plausible maximum (investigated, not auto-removed) |
| State name issues | Inconsistent capitalisation or old names |
| District name issues | Whitespace, inconsistent capitalisation |
| Crop name issues | Inconsistent capitalisation or spelling |
| Invalid years | Outside expected range 2010–2017 |

---

## 4. Cleaning Techniques

### A – Preserve raw data
The raw Parquet file is **never touched**. All work is on a separate DataFrame:
```python
df_clean = df_raw.copy()
```

### B – Missing value strategy
| Situation | Action |
|-----------|--------|
| production_1000_t is NaN | Keep as NaN; flag in `has_production_data` column |
| yield_kg_per_ha is NaN | Keep as NaN; do not calculate if area is 0 |
| state / district / crop is NaN | Keep as NaN; flag |
| year is NaN | Investigate; keep unless clearly invalid |

**Rule:** We do NOT replace missing production or yield with 0.
Zero means "nothing was produced"; NaN means "we don't know."
These are very different and must not be confused.

### C – Duplicate handling
- **Exact duplicates:** Identified and listed; removed only if confirmed redundant.
- **Composite key duplicates:** (state, district, year, crop) combinations that
  appear more than once are flagged and investigated before any removal.

### D – Text standardisation
Three new columns are created (originals are kept unchanged):

| New column | Based on | Transformation |
|------------|----------|----------------|
| `state_clean` | `state` | Strip whitespace, Title Case |
| `district_clean` | `district` | Strip whitespace, Title Case |
| `crop_clean` | `crop` | Strip whitespace, Title Case |

**Known name corrections applied (if present in data):**
- "Orissa" → "Odisha"
- "Uttaranchal" → "Uttarakhand"
- "Pondicherry" → "Puducherry"

---

## 5. Numerical Validation

| Column | Check | Action if problem found |
|--------|-------|------------------------|
| `area_1000_ha` | Must be ≥ 0 | Flag as negative; investigate |
| `production_1000_t` | Must be ≥ 0 | Flag as negative; investigate |
| `yield_kg_per_ha` | Must be ≥ 0 | Flag as negative; investigate |
| `yield_kg_per_ha` | Must be plausible | Values > 15,000 kg/ha are flagged |

**Zero-area records are NOT deleted.** They may represent genuine
non-cultivation (fallow land, seasonal gaps). They are flagged instead.

---

## 6. Yield Consistency Check

The expected yield formula:
```
calculated_yield = (production_1000_t × 1000) / area_1000_ha
```

A new column `calculated_yield_kg_per_ha` is added.
The original `yield_kg_per_ha` column is **never overwritten**.

A `yield_discrepancy` flag is set to True when:
```
|recorded_yield – calculated_yield| > 10 kg/ha
```
(10 kg/ha tolerance accounts for rounding in the original data.)

---

## 7. Derived Columns

| Column | Type | Description |
|--------|------|-------------|
| `has_production_data` | Boolean | True if production_1000_t is not NaN |
| `is_zero_activity` | Boolean | True if area = 0 AND production = 0 |
| `is_small_area_rounding` | Boolean | True if area > 0 but area < 0.1 (possible rounding artefact) |
| `record_status` | Category | Classification of each record (see below) |
| `state_clean` | Text | Standardised state name |
| `district_clean` | Text | Standardised district name |
| `crop_clean` | Text | Standardised crop name |
| `calculated_yield_kg_per_ha` | Float | Yield recalculated from formula |
| `yield_discrepancy` | Boolean | True if calculated and recorded yield differ by > 10 kg/ha |

### record_status categories

| Status | Meaning |
|--------|---------|
| `complete_positive` | Area > 0, production > 0, yield available – ideal record |
| `zero_activity` | Area = 0, production = 0 – likely fallow or non-cultivated |
| `missing_prod_yield_with_area` | Area > 0 but production and yield are NaN |
| `missing_prod_yield_zero_area` | Area = 0 and production/yield are NaN |
| `zero_prod_with_positive_area` | Area > 0 but production = 0 (possible genuine zero harvest) |
| `small_area_rounding` | Area > 0 but very small (< 0.1 thousand ha) – possible rounding |

---

## 8. Final Output

**File:** `data/processed/cleaned_agriculture_dataset.csv`

**What it contains:**
- All original columns (unchanged)
- Three clean text columns (state_clean, district_clean, crop_clean)
- Five quality/derived columns
- All records (none deleted without justification)

**What other members should use:**
```python
import pandas as pd
df = pd.read_csv("data/processed/cleaned_agriculture_dataset.csv", low_memory=False)
# Use 'state_clean', 'district_clean', 'crop_clean' for analysis
# Use 'record_status' to filter the quality of records you want
```

---

## 9. How to Reproduce the Pipeline

```
# Step 1 – Activate virtual environment (Windows)
.venv\Scripts\activate

# Step 2 – Install dependencies
pip install -r requirements.txt

# Step 3 – Download dataset
python src/data_collection/collect_dataset.py

# Step 4 – Run preprocessing
python src/preprocessing/preprocess_agriculture.py

# Step 5 – Run full pipeline at once
python main.py

# Step 6 – Run tests
pytest tests/ -v
```

All paths in the code use `pathlib.Path` and are relative to the project root,
so the pipeline will work on any computer without changing file paths.
