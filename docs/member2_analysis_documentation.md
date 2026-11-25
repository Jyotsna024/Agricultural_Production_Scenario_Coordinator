# Member 2 Analysis Documentation
## PS 137 – Agricultural Production Scenario Coordinator

**Member:** 2 – Historical Analysis and Trend Identification  
**Module:** `src/analysis/historical_analysis.py`  
**Runner:** `run_analysis.py`  
**Tests:** `tests/test_historical_analysis.py`  
**Reports:** `reports/member2/`

---

## 1. What Member 2's Module Does

Member 2 consumes the cleaned dataset produced by Member 1's pipeline and delivers:

- A comprehensive **historical analysis** of Indian agricultural production (2010–2017).
- **Year-wise**, **state-wise**, and **crop-wise** aggregated summaries.
- **Weighted yield** calculations using the correct formula.
- Interactive filtering functions by state, district, crop, and year.
- A **baseline retrieval function** (`get_historical_baseline`) that Member 3 can call to obtain the actual historical area, production, and yield for any combination.
- Factual **analytical insights** extracted directly from the dataset.
- Saved **charts** and **CSV/text reports** in `reports/member2/`.

> **Scope boundary:** Member 2 produces only historical analysis and descriptive statistics.  
> Scenario simulation is Member 3's responsibility. GenAI agents and dashboards are Member 4's.

---

## 2. Input Dataset

| Property | Value |
|----------|-------|
| File | `data/processed/cleaned_agriculture_dataset.csv` |
| Produced by | Member 1 (`src/preprocessing/preprocess_agriculture.py`) |
| Records | 72,036 |
| Original columns | 11 (including `record_uid`, `state_code`, `district_code`, `source`) |
| Derived columns (Member 1) | 9 (see below) |

### Source columns used by Member 2

| Column | Type | Description |
|--------|------|-------------|
| `state` | Text | Indian state name (original, as downloaded) |
| `district` | Text | District name (original) |
| `year` | Integer | Calendar year |
| `crop` | Text | Crop name (original, often upper-case) |
| `area_1000_ha` | Float | Land area cultivated (thousand hectares) |
| `production_1000_t` | Float | Total production (thousand tonnes) |
| `yield_kg_per_ha` | Float | Recorded yield (kilograms per hectare) |

### Derived columns used (from Member 1 preprocessing)

| Column | Type | Description |
|--------|------|-------------|
| `state_clean` | Text | Title-cased, standardised state name |
| `district_clean` | Text | Title-cased district name |
| `crop_clean` | Text | Title-cased crop name |
| `calculated_yield_kg_per_ha` | Float | Yield recalculated from formula |
| `yield_discrepancy` | Boolean | True if recorded vs calculated yield differ > 10 kg/ha |
| `has_production_data` | Boolean | True if `production_1000_t` is not NaN |
| `is_zero_activity` | Boolean | True if area = 0 AND production = 0 |
| `is_small_area_rounding` | Boolean | True if 0 < area < 0.1 |
| `record_status` | Category | Quality classification of each record |

> Member 2 always uses `state_clean`, `district_clean`, and `crop_clean` for text matching  
> when they are available, falling back to the original columns if not.

---

## 3. Unit Conventions

All units are **inherited from Member 1** and preserved throughout.

| Measure | Unit |
|---------|------|
| Area | Thousand hectares (1,000 ha). A value of 1.0 means 1,000 hectares. |
| Production | Thousand tonnes (1,000 t). A value of 1.0 means 1,000 tonnes. |
| Yield | Kilograms per hectare (kg/ha). Already a ratio; not further scaled. |

---

## 4. Aggregation Methods

### Why weighted yield (not simple average)?

Suppose district A produces 300,000 t from 100,000 ha (3,000 kg/ha) and district B produces 100,000 t from 200,000 ha (500 kg/ha). A simple average gives (3000 + 500) / 2 = 1,750 kg/ha. The correct combined yield is (300,000 + 100,000) × 1,000 / (100,000 + 200,000) × 1,000 = 1,333 kg/ha — very different.

### Weighted yield formula

```
weighted_yield_kg_per_ha = (Σ production_1000_t × 1000) / Σ area_1000_ha
```

Both inputs are sums over the records in the aggregation group. The factor of 1,000 converts production (in thousands of tonnes) to tonnes; dividing by area in thousands of hectares gives kg/ha. The "thousands" cancel in the ratio for area, but the ×1,000 is needed to convert t/ha → kg/ha.

### Production and area totals

Simple sums of `production_1000_t` and `area_1000_ha` after applying the analysis policy filter.

### Year-over-year change

```python
yoy_change     = production_this_year − production_previous_year
yoy_pct_change = (yoy_change / production_previous_year) × 100
```

The first year in the dataset has NaN for both columns. `inf` values (when previous year production is 0) are replaced with NaN.

---

## 5. Analysis Policy

The `AnalysisPolicy` class (in `historical_analysis.py`) controls which records are included:

```python
from src.analysis.historical_analysis import AnalysisPolicy

policy = AnalysisPolicy(
    use_complete_only=False,       # True: only 'complete_positive' records
    exclude_zero_area=True,        # exclude records where area == 0
    exclude_zero_production=False, # include zero-production records
    min_area_threshold=0.0,        # exclude records where area < threshold
)
```

The default policy (`AnalysisPolicy()`) excludes zero-area records but includes records with zero production.

All exclusions are printed to the console and documented in the reports.

---

## 6. Filtering Functions

### `filter_data()`

```python
from src.analysis.historical_analysis import load_cleaned_data, filter_data

df = load_cleaned_data()

# Filter by one criterion
rice_records = filter_data(df, crops=["Rice"])

# Combined filter (AND logic)
subset = filter_data(
    df,
    states=["Karnataka"],
    districts=["Mysore"],
    crops=["Rice"],
    years=[2015],
)
```

- Matching is case-insensitive by default.
- Uses `state_clean`, `district_clean`, `crop_clean` when available.
- Returns an empty DataFrame (not an error) when no records match.
- Prints a warning message when no records match.

---

## 7. Baseline Retrieval Function for Member 3

### `get_historical_baseline()`

**This is Member 3's primary integration point.**

```python
from src.analysis.historical_analysis import load_cleaned_data, get_historical_baseline

df = load_cleaned_data()

result = get_historical_baseline(
    df,
    state="Chhattisgarh",
    district="Durg",
    crop="Rice",
    year=2010,
)
```

#### Return structure

```python
{
    "found"                      : True,          # False if no match or duplicates
    "match_count"                : 1,             # number of matching records
    "state"                      : "Chhattisgarh",
    "district"                   : "Durg",
    "crop"                       : "Rice",
    "year"                       : 2010,
    "area_1000_ha"               : 777.92,        # thousand hectares (None if not found)
    "production_1000_t"          : 1319.18,       # thousand tonnes (None if not found)
    "yield_kg_per_ha"            : 1695.77,       # kg/ha, original recorded value
    "calculated_yield_kg_per_ha" : 1695.74,       # formula-derived (may differ slightly)
    "record_status"              : "complete_positive",
    "has_production_data"        : True,
    "units": {
        "area"      : "Thousand hectares (1000 ha)",
        "production": "Thousand tonnes (1000 t)",
        "yield"     : "Kilograms per hectare (kg/ha)",
    },
    "source_records"             : [{ ... full row dict ... }],
    "warning"                    : None,          # str if something is wrong
}
```

#### Rules

| Scenario | Behaviour |
|----------|-----------|
| Exactly one match | `found=True`, all values populated |
| No match | `found=False`, all measurements `None`, `warning` explains |
| Multiple matches (duplicates) | `found=False`, all source records in `source_records`, `warning` explains |

**Member 3 must check `result["found"]` before using values. Never fabricate a baseline.**

### `get_aggregated_baseline()`

For state-level or national-level aggregation:

```python
from src.analysis.historical_analysis import load_cleaned_data, get_aggregated_baseline

df = load_cleaned_data()

# All-Karnataka Rice production in 2015
agg = get_aggregated_baseline(df, crop="Rice", year=2015, state="Karnataka")

# National-level
nat = get_aggregated_baseline(df, crop="Wheat", year=2014)
```

#### Return structure

```python
{
    "found"                   : True,
    "level"                   : "state",         # "district" | "state" | "national"
    "state"                   : "Karnataka",
    "district"                : None,
    "crop"                    : "Rice",
    "year"                    : 2015,
    "record_count"            : 42,              # number of records aggregated
    "total_area_1000_ha"      : 1234.5,
    "total_production_1000_t" : 4567.8,
    "weighted_yield_kg_per_ha": 3698.2,
    "units"                   : { ... },
    "source_rows_count"       : 42,
    "warning"                 : None,
}
```

> **Important:** `get_aggregated_baseline` sums area and production across matching records.  
> Do **not** combine its output with `get_historical_baseline` output — this would double-count records.

---

## 8. Historical Analysis Functions Reference

| Function | Description |
|----------|-------------|
| `load_cleaned_data(csv_path=None)` | Load and validate the cleaned CSV. |
| `validate_dataset(df)` | Return a summary dict of data quality metrics. |
| `generate_dataset_overview(df)` | Generate a formatted text overview. |
| `analyse_year_wise(df, policy)` | Year-by-year production, area, weighted yield, YoY change. |
| `analyse_state_wise(df, policy)` | State-by-state totals, weighted yield, production share. |
| `analyse_crop_wise(df, policy)` | Crop-by-crop totals, weighted yield, production share. |
| `analyse_crop_trends(df, policy)` | Pivot table: crops × years → production. |
| `identify_crop_changes(df, policy)` | Crops with largest absolute production increase/decrease. |
| `filter_data(df, states, districts, crops, years)` | Multi-criterion filter. |
| `get_historical_baseline(df, state, district, crop, year)` | Single-record baseline for Member 3. |
| `get_aggregated_baseline(df, crop, year, state, district)` | Aggregated baseline for Member 3. |
| `generate_historical_insights(df, year_df, state_df, crop_df)` | Factual text insights. |
| `save_all_outputs(...)` | Save CSVs and text reports to `reports/member2/`. |

---

## 9. Chart Outputs

All charts are saved to `reports/member2/plots/`.

| Filename | Description |
|----------|-------------|
| `yearly_production_trend.png` | Total production by year (line chart) |
| `yearly_area_trend.png` | Total cultivated area by year (line chart) |
| `yearly_weighted_yield_trend.png` | Weighted yield by year (line chart) |
| `yearly_production_yoy_change.png` | Year-over-year % change (bar chart) |
| `state_production_comparison.png` | All states ranked by total production (bar chart) |
| `top_10_states.png` | Top 10 states by production (bar chart) |
| `state_year_heatmap.png` | Top 15 states × years heatmap (requires seaborn) |
| `crop_production_comparison.png` | All crops ranked by total production (bar chart) |
| `crop_area_comparison.png` | All crops ranked by cultivated area (bar chart) |
| `crop_weighted_yield.png` | All crops by weighted yield (bar chart) |
| `crop_production_trends.png` | Top 8 crops' production over years (line chart) |

---

## 10. How Member 3 Imports the Baseline Function

```python
# In Member 3's simulation module:
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # adjust as needed
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.historical_analysis import load_cleaned_data, get_historical_baseline

# Load once at startup, pass df to functions
df = load_cleaned_data()

# Example: get baseline for scenario simulation
baseline = get_historical_baseline(
    df,
    state    = "Punjab",
    district = "Ludhiana",
    crop     = "Wheat",
    year     = 2015,
)

if baseline["found"]:
    area       = baseline["area_1000_ha"]         # thousand hectares
    production = baseline["production_1000_t"]    # thousand tonnes
    yield_val  = baseline["yield_kg_per_ha"]      # kg/ha
    # Use these for scenario adjustment calculations
else:
    print(f"Warning: {baseline['warning']}")
    # Handle missing baseline (do not proceed with fabricated data)
```

---

## 11. How to Run the Analysis

### Prerequisites

Member 1's pipeline must have been run first:

```bash
python main.py          # runs collection + preprocessing + validation
```

### Run Member 2's full analysis

```bash
python run_analysis.py
```

This generates all reports, CSVs, and charts.

### Individual steps

```python
from src.analysis.historical_analysis import (
    load_cleaned_data,
    analyse_year_wise,
    analyse_state_wise,
    analyse_crop_wise,
    get_historical_baseline,
)

df = load_cleaned_data()
year_df  = analyse_year_wise(df)
state_df = analyse_state_wise(df)
crop_df  = analyse_crop_wise(df)
```

### Run tests

```bash
pytest tests/ -v
pytest tests/test_historical_analysis.py -v
pytest tests/test_preprocessing.py -v        # Member 1 tests
```

---

## 12. Known Limitations

1. **Year range is 2010–2017 only.** The dataset does not cover years outside this range.  
   Any scenario extending beyond 2017 must acknowledge this limitation.

2. **14,904 records (20.7%) have missing production values.** These are preserved as NaN  
   and excluded from production totals but not from record counts.

3. **26,178 records have zero area.** These may represent genuine non-cultivation  
   (fallow, seasonal gaps) rather than data errors. They are excluded from yield  
   calculations but preserved in the dataset.

4. **State/district names may have minor inconsistencies** even after text standardisation.  
   Always use case-insensitive matching (the default) when calling filter functions.

5. **Member 2 does not predict future values.** All outputs are descriptive statistics  
   based on historical records. Do not interpret insights as forecasts.

6. **Weighted yield calculations require positive area.** If area is 0 or NaN,  
   yield is reported as NaN. This is correct behaviour.

7. **Chart generation requires matplotlib** (and seaborn for the heatmap).  
   Charts are skipped gracefully if the library is unavailable.

---

## 13. Record Status Categories (from Member 1)

| Status | Meaning |
|--------|---------|
| `complete_positive` | Area > 0, production > 0 — ideal record |
| `zero_activity` | Area = 0, production = 0 — likely fallow |
| `missing_prod_yield_with_area` | Area > 0 but production is NaN |
| `missing_prod_yield_zero_area` | Area = 0, production is NaN |
| `zero_prod_with_positive_area` | Area > 0 but production = 0 |
| `small_area_rounding` | 0 < area < 0.1 (possible rounding artefact) |
| `unknown` | None of the above |

Member 2 recommends that Member 3 use only `complete_positive` records for  
high-confidence scenario simulations, and treat other statuses with caution.
