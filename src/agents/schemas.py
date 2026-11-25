"""
schemas.py
==========
Pydantic v2 models for structured LLM extraction and scenario results.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# 1. ScenarioAssumption
# ---------------------------------------------------------------------------

class ScenarioAssumption(BaseModel):
    """Captures the user-specified assumptions for a single scenario."""

    state: Optional[str] = None
    district: Optional[str] = None
    crop: Optional[str] = None
    year: Optional[int] = None
    area_change_percent: float = 0.0
    yield_change_percent: float = 0.0
    scenario_name: str = "Scenario"
    scenario_description: Optional[str] = None

    @field_validator("area_change_percent", "yield_change_percent", mode="before")
    @classmethod
    def coerce_to_float(cls, v):
        if v is None:
            return 0.0
        return float(v)

    @field_validator("year", mode="before")
    @classmethod
    def coerce_year(cls, v):
        if v is None:
            return None
        return int(v)


# ---------------------------------------------------------------------------
# 2. ExtractionResult
# ---------------------------------------------------------------------------

class ExtractionResult(BaseModel):
    """
    Result of parsing / classifying a user natural-language request.

    intent values:
        "historical_analysis"  – user wants to look up historical data
        "scenario_simulation"  – user wants to run what-if scenarios
        "comparison"           – user wants to compare multiple scenarios
        "clarification_needed" – request is too ambiguous to act on
        "invalid"              – request is off-topic / cannot be processed
    """

    intent: str = Field(
        ...,
        description=(
            "One of: historical_analysis, scenario_simulation, "
            "comparison, clarification_needed, invalid"
        ),
    )
    scenarios: List[ScenarioAssumption] = Field(
        default_factory=list,
        description="Extracted scenario assumptions (may be empty for historical queries).",
    )
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    missing_fields: List[str] = Field(default_factory=list)
    original_request: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    historical_query: Optional[str] = Field(
        default=None,
        description="Reformulated query string for historical data look-up.",
    )

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        allowed = {
            "historical_analysis",
            "scenario_simulation",
            "comparison",
            "clarification_needed",
            "invalid",
        }
        if v not in allowed:
            raise ValueError(f"intent must be one of {allowed}, got: {v!r}")
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v):
        v = float(v)
        return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# 3. ScenarioResult
# ---------------------------------------------------------------------------

class ScenarioResult(BaseModel):
    """
    Structured result for a single scenario simulation,
    suitable for passing between pipeline modules.
    """

    success: bool
    scenario_name: str
    state: Optional[str] = None
    district: Optional[str] = None
    crop: Optional[str] = None
    year: Optional[int] = None

    # --- Production figures (all in units stored in `units`) ---
    baseline_production: Optional[float] = None
    new_production: Optional[float] = None
    production_change: Optional[float] = None
    production_change_percent: Optional[float] = None

    # --- Area figures ---
    baseline_area: Optional[float] = None
    new_area: Optional[float] = None

    # --- Yield figures ---
    baseline_yield: Optional[float] = None
    new_yield: Optional[float] = None

    # --- Input assumptions ---
    area_change_percent: float = 0.0
    yield_change_percent: float = 0.0

    # --- Metadata ---
    is_forecast: bool = False
    error_message: Optional[str] = None
    units: dict = Field(default_factory=dict)

    @field_validator(
        "area_change_percent", "yield_change_percent", mode="before"
    )
    @classmethod
    def coerce_pct(cls, v):
        if v is None:
            return 0.0
        return float(v)
