# Agricultural Production Scenario Coordinator (PS 137)

A GenAI-powered scenario exploration platform for Indian agriculture that translates natural-language assumptions into deterministic what-if crop production analyses grounded in official historical records (2010–2017).

> ⚠️ **Important:** This system is an **agricultural scenario simulation tool**, **NOT a forecasting or prediction model**. All computed outputs represent arithmetic what-if simulations based on historical baselines and user assumptions.

---

## 🌟 Key Features

- **100% Local LLM with Ollama:** Runs entirely offline using your locally installed Ollama instance (default: `llama3.2`). **Zero Gemini API keys or cloud billing required.**
- **Deterministic Python Calculation Engine:** Pure arithmetic simulations ensuring the LLM never fabricates baseline data or performs error-prone calculations:
  $$\text{New Production (1000 t)} = \frac{\text{New Area (1000 ha)} \times \text{New Yield (kg/ha)}}{1000}$$
- **State-of-the-Art LangGraph Orchestration:**
  1. Understands natural language requests.
  2. Extracts structured parameters (`state`, `district`, `crop`, `year`, `area_change_%`, `yield_change_%`).
  3. Validates required fields and asks intelligent clarification questions if details are missing.
  4. Retrieves verified historical baselines from Member 1 & 2's cleaned dataset.
  5. Runs Member 3's scenario engine and multi-scenario comparator.
  6. Enforces output validation (no NaN/infinite numbers, `is_forecast: false`).
  7. Generates clear, non-technical explanations using Ollama.
- **Clean Single-Page Streamlit Dashboard:** User-friendly natural language prompt interface with clickable example prompts, metric cards, Plotly charts, and manual calculation overrides.

---

## 🛠️ Prerequisites & Setup

### 1. Verify Local Ollama Installation

Ollama must be installed and running on your laptop:

```bash
# Check Ollama version
ollama --version

# View installed models
ollama list
```

If Ollama is not already running as a background service, start it in a separate terminal:

```bash
ollama serve
```

If you don't have a model installed, pull the recommended lightweight model:

```bash
ollama pull llama3.2
```

### 2. Configure Environment

Copy `.env.example` to `.env` (optional, default values work out-of-the-box):

```bash
cp .env.example .env
```

Default configuration in `.env.example`:
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_TEMPERATURE=0.0
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### Launch Streamlit Dashboard

```bash
streamlit run app/app.py
```

Open your browser to `http://localhost:8501`.

### Try an Example Prompt

Paste into the dashboard prompt box:
- *"What happens if rice cultivation area in Karnataka increases by 10% and yield decreases by 5% in 2017?"*
- *"Suppose maize area in Karnataka increases by 15% in 2015 while yield remains unchanged. Calculate the new production."*
- *"Compare two scenarios for wheat in Madhya Pradesh in 2016: Scenario A with yield drop of 12%, and Scenario B with area growth of 8%."*
- *"Show me the historical production trend of rice in Karnataka from 2010 to 2017."*

---

## 🧪 Running Automated Tests

All tests run deterministically with mocked LLMs, so the full test suite runs without requiring an active Ollama connection:

```bash
pytest tests/ -v
```

Current test coverage: **91/91 tests passing** across data collection, preprocessing, historical analysis, scenario simulation, validator, and LangGraph workflow.
