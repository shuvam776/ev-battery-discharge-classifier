"""
schemas.py
===========
Pydantic models for all API request/response structures.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request schemas ────────────────────────────────────────────────────────────

class BatteryData(BaseModel):
    """
    Numerical battery telemetry for the electrical branch.
    All fields are optional — missing values are imputed by the preprocessor.
    At least one field must be provided.
    """
    # Voltage features
    voltage_mean: Optional[float] = Field(None, description="Mean voltage (V)")
    voltage_std: Optional[float] = Field(None, description="Voltage standard deviation")
    voltage_min: Optional[float] = Field(None, description="Minimum voltage (V)")
    voltage_max: Optional[float] = Field(None, description="Maximum voltage (V)")
    voltage_range: Optional[float] = Field(None, description="Voltage range (max - min)")

    # Current features
    current_mean: Optional[float] = Field(None, description="Mean current (A)")
    current_std: Optional[float] = Field(None, description="Current standard deviation")
    current_min: Optional[float] = Field(None, description="Minimum current (A)")
    current_max: Optional[float] = Field(None, description="Maximum current (A)")
    current_range: Optional[float] = Field(None, description="Current range")

    # Temperature features
    temperature_mean: Optional[float] = Field(None, description="Mean temperature (°C)")
    temperature_std: Optional[float] = Field(None, description="Temperature std dev")
    temperature_min: Optional[float] = Field(None, description="Min temperature (°C)")
    temperature_max: Optional[float] = Field(None, description="Max temperature (°C)")
    temperature_range: Optional[float] = Field(None, description="Temperature range")

    # Temporal
    cycle_duration: Optional[float] = Field(None, description="Cycle duration (s)")
    cycle_number: Optional[float] = Field(None, description="Cycle number (age proxy)")

    # Energy
    energy_wh: Optional[float] = Field(None, description="Energy delivered (Wh)")

    # Impedance
    Re: Optional[float] = Field(None, description="Electrolyte resistance (Ohm)")
    Rct: Optional[float] = Field(None, description="Charge transfer resistance (Ohm)")

    class Config:
        json_schema_extra = {
            "example": {
                "voltage_mean": 3.62,
                "voltage_std": 0.08,
                "voltage_min": 3.20,
                "voltage_max": 4.18,
                "voltage_range": 0.98,
                "current_mean": -1.5,
                "current_std": 0.3,
                "temperature_mean": 29.4,
                "temperature_std": 1.2,
                "cycle_duration": 4200.0,
                "cycle_number": 620,
                "energy_wh": 1.72,
                "Re": 0.16,
                "Rct": 0.08,
            }
        }


# ── Response schemas ───────────────────────────────────────────────────────────

class ElectricalResult(BaseModel):
    grade: str = Field(..., description="Predicted battery grade: A, B, C, or SCRAP")
    confidence: float = Field(..., description="Confidence of top prediction (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")


class ThermalResult(BaseModel):
    available: bool
    grade: Optional[str] = Field(None, description="Predicted thermal grade: A, B, C, SCRAP")
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    is_synthetic_model: bool = Field(
        True,
        description="PROTOTYPE FLAG: Indicates model was trained on synthetic patterns. Real EV battery thermal dataset required for production."
    )


class XRayResult(BaseModel):
    available: bool
    grade: Optional[str] = Field(None, description="Predicted structural grade: A, B, C, SCRAP")
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None


class FinalDecision(BaseModel):
    grade: str = Field(..., description="Economic grade from electrical model")
    decision: str = Field(
        ...,
        description="PASS | PASS_WITH_CAUTION | HOLD_FOR_MANUAL_INSPECTION"
    )
    recommended_application: str
    disclaimer: str = "PRELIMINARY AI ASSESSMENT. Final certification requires PunarShakti six-stage physical testing workflow."


class AnalyzeResponse(BaseModel):
    electrical: ElectricalResult
    thermal: ThermalResult
    xray: XRayResult
    final: FinalDecision
    model_version: str


class HealthResponse(BaseModel):
    status: str
    electrical_model_loaded: bool
    thermal_model_loaded: bool
    xray_model_loaded: bool
    version: str


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class FeaturesResponse(BaseModel):
    features: List[FeatureImportanceItem]
    model_version: str


class ThermalAnalysisResponse(BaseModel):
    thermal: ThermalResult
    model_version: str


class XRayAnalysisResponse(BaseModel):
    xray: XRayResult
    model_version: str


class ElectricalAnalysisResponse(BaseModel):
    electrical: ElectricalResult
    model_version: str


class CSVAnalyzeResponse(BaseModel):
    results: List[AnalyzeResponse]
    total_processed: int
    summary: Dict[str, int]
    model_version: str
