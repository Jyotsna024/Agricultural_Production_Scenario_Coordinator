# PS 137 – Agricultural Production Scenario Coordinator
# Project Overview

## Project Title
Agricultural Production Scenario Coordinator

## Project ID
PS 137

## Domain
Agriculture, Food Security and Rural Development

## Sustainable Development Goals (SDGs)
- **SDG 1 – No Poverty:** Better crop production insights help farmers plan and earn sustainably.
- **SDG 2 – Zero Hunger:** Improved agricultural planning supports food security.
- **SDG 15 – Life on Land:** Responsible land use analysis through data-driven scenarios.

---

## Problem Statement

Develop a GenAI application that processes historical agricultural production records and hypothetical user assumptions. The system should produce scenario comparisons with verified calculations. The prototype must clearly distinguish scenarios (simulated "what-if" situations) from forecasts (predictions about the future).

---

## Objectives

1. Collect and clean a real historical agricultural dataset for India.
2. Explore the data to understand crop, region, and year-wise patterns.
3. Enable users to input hypothetical assumptions (e.g., change in area, yield improvement).
4. Generate and compare multiple production scenarios.
5. Use GenAI to explain the scenario differences in plain language.
6. Present the results in an interactive Streamlit application.

---

## Proposed System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER (Streamlit App)                          │
│           Enters hypothetical assumptions / selects scenarios    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              MEMBER 4 – AI Agent + Application Layer             │
│    Receives user input, calls simulation, explains output        │
└───────────┬─────────────────────────────────┬───────────────────┘
            │                                 │
            ▼                                 ▼
┌─────────────────────────┐       ┌───────────────────────────────┐
│  MEMBER 3 – Scenario    │       │  MEMBER 2 – Historical         │
│  Simulation Engine       │       │  Analysis & Trend Module       │
│  Runs what-if calcs      │       │  Provides baseline context     │
└───────────┬─────────────┘       └───────────────────────────────┘
            │                                 │
            └─────────────┬───────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│            MEMBER 1 – Processed Dataset                          │
│    data/processed/cleaned_agriculture_dataset.csv               │
│    Original source: Hugging Face India Agriculture Dataset       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Team Responsibilities

### Member 1 – Dataset Collection and Preprocessing
- Download the India Public Agriculture dataset from Hugging Face.
- Explore the dataset (EDA): understand shape, columns, missing values, distributions.
- Perform a data quality audit: document all issues found.
- Clean and preprocess the data: standardise text, handle missing values, validate numerics.
- Create derived quality columns to flag record status.
- Produce the final cleaned CSV for all other members.
- Document every step clearly.

### Member 2 – Historical Analysis and Trend Identification
- Use Member 1's cleaned dataset.
- Perform historical trend analysis: year-wise, state-wise, and crop-wise.
- Identify patterns, seasonal effects, and notable changes.
- Provide baseline statistics for the scenario engine.

### Member 3 – Scenario Simulation and Production Calculations
- Build the scenario calculation engine.
- Accept hypothetical inputs (e.g., area change, yield improvement).
- Compute projected production under each scenario.
- Compare scenarios side-by-side with verified formulas.
- Clearly label output as "SCENARIO (not forecast)".

### Member 4 – Agentic AI Integration and Application
- Build the Streamlit user interface.
- Integrate a GenAI agent that explains scenario results in plain English.
- Validate AI outputs against calculated numbers.
- Ensure the application clearly distinguishes scenarios from forecasts.
- Handle edge cases and produce user-friendly error messages.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Data processing | Python, Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Notebooks | Jupyter |
| Web application | Streamlit |
| AI / LLM | Agentic AI (Member 4 to specify) |
| Data format | Parquet (raw), CSV (processed) |
| Testing | pytest |
| Version control | Git |
