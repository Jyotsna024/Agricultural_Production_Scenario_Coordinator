# PS 137 – Agricultural Production Scenario Coordinator

## Project Description

This project develops a **GenAI-powered Agricultural Production Scenario Coordinator** for India. The system processes historical crop production records and allows users to test hypothetical assumptions (e.g., "What if rainfall increases by 10%?") and then compares multiple scenarios with verified calculations.

> **Important:** This system produces *scenario comparisons*, not forecasts. It clearly distinguishes between what has been observed historically and what is being simulated.

**SDGs addressed:** SDG 1 (No Poverty), SDG 2 (Zero Hunger), SDG 15 (Life on Land)

---

## Folder Structure

```
PS137_Agricultural_Scenario_Coordinator/
│
├── README.md                        ← This file
├── requirements.txt                 ← Python dependencies
├── .gitignore                       ← Files excluded from Git
├── main.py                          ← Run the full Member 1 pipeline
│
├── data/
│   ├── raw/                         ← Original downloaded data (never modified)
│   │   ├── crop_statistics.parquet  ← Raw Hugging Face dataset
│   │   └── sources.json             ← Collection metadata
│   ├── interim/                     ← Intermediate files (if any)
│   └── processed/
│       └── cleaned_agriculture_dataset.csv  ← Final cleaned dataset
│
├── docs/                            ← Project and dataset documentation
├── reports/                         ← Audit and validation reports
├── notebooks/                       ← Jupyter EDA notebooks
│
├── src/
│   ├── data_collection/             ← Member 1: data download
│   ├── preprocessing/               ← Member 1: data cleaning
│   ├── analysis/                    ← Member 2: trend analysis (placeholder)
│   ├── simulation/                  ← Member 3: scenario simulation (placeholder)
│   ├── agents/                      ← Member 4: AI agents (placeholder)
│   ├── validation/                  ← Member 4: output validation (placeholder)
│   └── visualization/               ← Shared charts (placeholder)
│
├── app/
│   └── app.py                       ← Streamlit web application (Member 4)
│
└── tests/
    └── test_preprocessing.py        ← Automated tests (Member 1)
```

---

## Windows Setup Instructions

### Step 1 – Check Python version
Open **Command Prompt** or **PowerShell** and type:
```
python --version
```
You need Python 3.9 or later. Download from https://python.org if needed.

### Step 2 – Open the project folder
```
cd C:\Users\devik\Desktop\E9\PS137_Agricultural_Scenario_Coordinator
```

### Step 3 – Create a virtual environment (recommended)
```
python -m venv .venv
.venv\Scripts\activate
```
Your prompt will change to `(.venv)` when activated.

### Step 4 – Install dependencies
```
pip install -r requirements.txt
```

---

## Running the Pipeline

### Collect the dataset (download from Hugging Face)
```
python main.py --collect-only
```
Or run the script directly:
```
python src/data_collection/collect_dataset.py
```

### Run preprocessing (clean and prepare the dataset)
```
python main.py --preprocess-only
```
Or run the script directly:
```
python src/preprocessing/preprocess_agriculture.py
```

### Run the full pipeline (collect + preprocess + validate)
```
python main.py
```

### Run a quick validation of the cleaned dataset
```
python main.py --validate-only
```

---

## Running Tests
```
pytest tests/ -v
```

---

## Running the Streamlit App (Member 4)
```
streamlit run app/app.py
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Hugging Face – `therajasekhar/india-public-agriculture` |
| URL | https://huggingface.co/datasets/therajasekhar/india-public-agriculture |
| Format | Parquet (raw), CSV (processed) |
| Expected rows | ~72,036 |
| Geographic scope | India (~20 states, ~311 districts) |
| Year range | 2010 – 2017 |
| Key columns | state, district, year, crop, area_1000_ha, production_1000_t, yield_kg_per_ha |

---

## Output Files

| File | Description |
|------|-------------|
| `data/raw/crop_statistics.parquet` | Original unmodified dataset |
| `data/raw/sources.json` | Collection metadata |
| `data/processed/cleaned_agriculture_dataset.csv` | Final cleaned dataset for all members |
| `reports/data_quality_audit.txt` | Data quality findings |
| `reports/cleaning_summary_report.txt` | What was cleaned and why |
| `reports/final_validation_report.txt` | Validation results |

---

## Team Responsibilities

| Member | Module | Status |
|--------|--------|--------|
| Member 1 | Dataset collection, EDA, preprocessing | ✅ In progress |
| Member 2 | Historical analysis and trend identification | ⏳ Placeholder |
| Member 3 | Scenario simulation and production calculations | ⏳ Placeholder |
| Member 4 | Agentic AI, validation, Streamlit application | ⏳ Placeholder |
