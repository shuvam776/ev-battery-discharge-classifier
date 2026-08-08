"""
decision_engine.py
=====================
The PunarShakti conservative fusion decision engine.

DESIGN PHILOSOPHY:
──────────────────
The system makes a CRITICAL distinction between:

  ECONOMIC HEALTH  →  Electrical branch (grade A/B/C/SCRAP)
  SAFETY RISK      →  Thermal branch (LOW/MEDIUM/HIGH)
  STRUCTURAL RISK  →  X-ray branch (NORMAL/SUSPICIOUS)

These are combined using a RULE-BASED SAFETY GATE.

The safety gate is CONSERVATIVE:
  - HIGH thermal risk → HOLD, regardless of electrical grade
  - SUSPICIOUS x-ray → HOLD, regardless of electrical grade
  - A "good" electrical result CANNOT override a safety flag

The final grade always comes from the electrical model.
The final decision reflects the safety assessment.

This design is intentional for regulatory credibility and
investor/judge confidence. Battery safety must not be
compromised by optimistic economic grades.
"""

import logging
from typing import Optional

from .config import settings
from .schemas import (
    AnalyzeResponse,
    ElectricalResult,
    FinalDecision,
    ThermalResult,
    XRayResult,
)

logger = logging.getLogger(__name__)

# ── Decision logic ────────────────────────────────────────────────────────────

GRADE_RECOMMENDATIONS = settings.GRADE_RECOMMENDATIONS

DISCLAIMER = (
    "PRELIMINARY AI ASSESSMENT. "
    "Final certification requires the PunarShakti six-stage physical testing workflow. "
    "This system does not certify batteries."
)


def apply_safety_gate(
    electrical_grade: str,
    thermal_grade: Optional[str],
    xray_grade: Optional[str],
) -> str:
    """
    Apply conservative safety decision rules based on Grade A, B, C, SCRAP outputs.
    
    Rules:
    1. Any SCRAP prediction from thermal or xray -> HOLD_FOR_MANUAL_INSPECTION
    2. Any Grade C prediction from thermal or xray with Grade A electrical -> PASS_WITH_CAUTION
    3. All clear -> PASS
    """
    if thermal_grade == "SCRAP" or xray_grade == "SCRAP":
        logger.info(f"Safety gate: HOLD (thermal={thermal_grade}, xray={xray_grade})")
        return "HOLD_FOR_MANUAL_INSPECTION"

    if (thermal_grade == "C" or xray_grade == "C") and electrical_grade == "A":
        logger.info(f"Safety gate: PASS_WITH_CAUTION (thermal={thermal_grade}, xray={xray_grade}, electrical=A)")
        return "PASS_WITH_CAUTION"

    return "PASS"


def build_final_response(
    electrical_grade: str,
    electrical_confidence: float,
    electrical_probs: dict,
    thermal_grade: Optional[str] = None,
    thermal_confidence: Optional[float] = None,
    thermal_probs: Optional[dict] = None,
    xray_grade: Optional[str] = None,
    xray_confidence: Optional[float] = None,
    xray_probs: Optional[dict] = None,
    model_version: str = "1.0.0",
) -> AnalyzeResponse:
    """
    Build the complete API response from branch outputs.
    """
    # ── Electrical result ──
    electrical = ElectricalResult(
        grade=electrical_grade,
        confidence=round(electrical_confidence, 4),
        probabilities={k: round(v, 4) for k, v in electrical_probs.items()},
    )

    # ── Thermal result ──
    if thermal_grade is not None:
        thermal = ThermalResult(
            available=True,
            grade=thermal_grade,
            confidence=round(thermal_confidence, 4) if thermal_confidence is not None else None,
            probabilities={k: round(v, 4) for k, v in thermal_probs.items()} if thermal_probs else None,
        )
    else:
        thermal = ThermalResult(available=False, grade=None, confidence=None)

    # ── X-ray result ──
    if xray_grade is not None:
        xray = XRayResult(
            available=True,
            grade=xray_grade,
            confidence=round(xray_confidence, 4) if xray_confidence is not None else None,
            probabilities={k: round(v, 4) for k, v in xray_probs.items()} if xray_probs else None,
        )
    else:
        xray = XRayResult(available=False, grade=None)

    # ── Apply safety gate ──
    decision = apply_safety_gate(electrical_grade, thermal_grade, xray_grade)

    # ── Final decision ──
    final = FinalDecision(
        grade=electrical_grade,
        decision=decision,
        recommended_application=GRADE_RECOMMENDATIONS.get(
            electrical_grade, "Consult PunarShakti engineer"
        ),
        disclaimer=DISCLAIMER,
    )

    return AnalyzeResponse(
        electrical=electrical,
        thermal=thermal,
        xray=xray,
        final=final,
        model_version=model_version,
    )
