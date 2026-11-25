# PS 137 – Dataset Documentation

## Dataset Source

| Field | Value |
|-------|-------|
| Name | India Public Agriculture Dataset |
| Provider | Hugging Face |
| Dataset ID | `therajasekhar/india-public-agriculture` |
| URL | https://huggingface.co/datasets/therajasekhar/india-public-agriculture |
| Collected by | Member 1 |
| Collection date | See `data/raw/sources.json` (populated after running collect_dataset.py) |

---

## Dataset Description

This dataset contains historical agricultural production records for India.
Each row represents one crop's area, production, and yield for a specific
state–district–year combination.

The data covers approximately 72,036 records across 20 states, 311 districts,
and 29 crop categories for the years 2010 to 2017.

> **Important:** These are recorded figures, not predictions.
> Any scenario comparisons built on this data are simulated "what-if" situations.

---

## Column Descriptions

| Column | Data Type | Description |
|--------|-----------|-------------|
| `state` | Text | Name of the Indian state |
| `district` | Text | Name of the district within the state |
| `year` | Integer | Calendar year of the record |
| `crop` | Text | Name of the crop or crop category |
| `area_1000_ha` | Float | Land area cultivated (in thousand hectares) |
| `production_1000_t` | Float | Total production (in thousand tonnes) |
| `yield_kg_per_ha` | Float | Yield (in kilograms per hectare) |

---

## Units

| Measure | Unit | Explanation |
|---------|------|-------------|
| Area | Thousand hectares (1,000 ha) | 1 record value of 1.0 means 1,000 hectares |
| Production | Thousand tonnes (1,000 t) | 1 record value of 1.0 means 1,000 tonnes |
| Yield | Kilograms per hectare (kg/ha) | Direct ratio; not scaled |

### Yield consistency formula
```
yield_kg_per_ha = (production_1000_t × 1,000) / area_1000_ha
```
Both production and area are in thousands, so:
- Numerator: production_1000_t × 1,000 = production in tonnes
- Denominator: area_1000_ha (in thousand ha, same scale cancels)
- Result: kg/ha (the ×1,000 converts the t/ha ratio to kg/ha)

---

## Geographic Coverage

| Property | Value |
|----------|-------|
| Country | India |
| Approximate states | 20 |
| Approximate districts | 311 |

> Exact counts are verified after download in `reports/data_quality_audit.txt`.

---

## Year Coverage

| Property | Value |
|----------|-------|
| Start year | 2010 |
| End year | 2017 |
| Number of years | 8 |

> Exact year range is verified after download.

---

## Dataset Limitations

1. **Year range is limited:** Only covers 2010–2017. It cannot reflect recent events such as climate changes after 2017.
2. **Missing values:** Some records have missing production or yield values. These are documented and handled transparently.
3. **Zero entries:** Some records have zero area or zero production. These may represent genuine non-cultivation (seasonal, fallow), not errors.
4. **State name inconsistencies:** The dataset may use older state names (e.g., "Orissa" instead of "Odisha"). This is handled during preprocessing.
5. **Not a forecast dataset:** This is historical recorded data. It should not be used directly as a prediction model.
6. **Single source:** This project intentionally uses only this dataset. No external records are added.
7. **Rounding:** Small area values may cause yield calculations to differ slightly from recorded values due to rounding.
