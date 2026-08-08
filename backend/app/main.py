"""
main.py
========
PunarShakti AI — FastAPI Backend

Endpoints:
    GET  /health                    → System and model status
    POST /analyze                   → Full multimodal analysis
    POST /analyze/csv               → Batch CSV telemetry analysis
    POST /analyze/thermal           → Thermal branch only
    POST /analyze/xray              → X-ray branch only
    GET  /model/features            → Feature importance (electrical)
    POST /explain/thermal           → Grad-CAM heatmap

CORS:
    Configurable via FRONTEND_URL env var for Next.js compatibility.
"""

import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from .config import settings
from .decision_engine import build_final_response
from .model_manager import model_manager
from .preprocessing import battery_data_to_array, csv_row_to_array, get_feature_names
from .schemas import (
    AnalyzeResponse,
    BatteryData,
    CSVAnalyzeResponse,
    ElectricalAnalysisResponse,
    ElectricalResult,
    FeaturesResponse,
    FeatureImportanceItem,
    HealthResponse,
    ThermalAnalysisResponse,
    ThermalResult,
    XRayAnalysisResponse,
    XRayResult,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("punarshakti")


# ── Lifespan (model loading) ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    logger.info("PunarShakti AI backend starting...")
    model_manager.load_all()
    logger.info("Ready to serve requests.")
    yield
    logger.info("PunarShakti AI backend shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PunarShakti AI — Battery Assessment API",
    description=(
        "Preliminary AI-powered EV battery grading system for PunarShakti Energy. "
        "Combines electrical telemetry (XGBoost), thermal imaging (ResNet-18), "
        "and optional X-ray analysis to provide a preliminary safety and economic "
        "grade assessment. NOT a replacement for physical certification."
    ),
    version=settings.MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
# Parse comma-separated origins
_allowed_origins = [
    o.strip() for o in settings.FRONTEND_URL.split(",") if o.strip()
]
# Always allow localhost in development
if "http://localhost:3000" not in _allowed_origins:
    _allowed_origins.append("http://localhost:3000")
if "http://localhost:8000" not in _allowed_origins:
    _allowed_origins.append("http://localhost:8000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper: parse battery_data JSON from multipart ────────────────────────────

def _parse_battery_data(battery_data_json: str) -> BatteryData:
    """Parse stringified JSON battery data from multipart form."""
    try:
        data = json.loads(battery_data_json)
        return BatteryData(**data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid battery_data JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid battery_data: {e}")


def _run_electrical(battery_data: BatteryData) -> tuple:
    """Run electrical inference. Raises 503 if model not loaded."""
    if not model_manager.electrical.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Electrical model not available. Run: python scripts/train_electrical.py",
        )
    features = battery_data_to_array(battery_data)
    return model_manager.electrical.predict(features)


def _run_thermal(image_bytes: bytes) -> tuple:
    """Run thermal inference. Returns (None, None, None) if model not loaded."""
    if not model_manager.thermal.is_loaded:
        return None, None, None
    return model_manager.thermal.predict(image_bytes)


def _run_xray(image_bytes: bytes) -> tuple:
    """Run X-ray inference. Returns (None, None, None) if model not loaded."""
    if not model_manager.xray.is_loaded:
        return None, None, None
    return model_manager.xray.predict(image_bytes)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    tags=["System"],
)
async def health() -> HealthResponse:
    """
    Returns the API status and which models are currently loaded.
    
    Use this to verify the backend is running and which branches
    are available before calling /analyze.
    """
    return HealthResponse(
        status="ok",
        electrical_model_loaded=model_manager.electrical.is_loaded,
        thermal_model_loaded=model_manager.thermal.is_loaded,
        xray_model_loaded=model_manager.xray.is_loaded,
        version=settings.MODEL_VERSION,
    )


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Full multimodal battery analysis",
    tags=["Analysis"],
)
async def analyze(
    battery_data: str = Form(
        ...,
        description="JSON string of battery telemetry features",
        example='{"voltage_mean": 3.62, "cycle_number": 620}',
    ),
    thermal_image: Optional[UploadFile] = File(
        None,
        description="Optional thermal image (PNG/JPG)",
    ),
    xray_image: Optional[UploadFile] = File(
        None,
        description="Optional X-ray image (PNG/JPG)",
    ),
) -> AnalyzeResponse:
    """
    Main multimodal endpoint. Accepts battery telemetry + optional images.
    
    **Modes:**
    - Numerical only: `battery_data` only
    - Numerical + thermal: `battery_data` + `thermal_image`
    - Numerical + thermal + X-ray: all three

    **Safety gate:**
    If thermal risk is HIGH or X-ray risk is SUSPICIOUS, the final
    decision is HOLD_FOR_MANUAL_INSPECTION regardless of electrical grade.

    **Important:** This is a PRELIMINARY AI ASSESSMENT, not a certification.
    """
    # ── Electrical (required) ──
    bd = _parse_battery_data(battery_data)
    elec_grade, elec_conf, elec_probs = _run_electrical(bd)

    # ── Thermal (optional) ──
    thermal_grade = thermal_conf = thermal_probs = None
    if thermal_image is not None:
        image_bytes = await thermal_image.read()
        if len(image_bytes) > 0:
            try:
                thermal_grade, thermal_conf, thermal_probs = _run_thermal(image_bytes)
            except Exception as e:
                logger.warning(f"Thermal inference failed: {e}")

    # ── X-ray (optional) ──
    xray_grade = xray_conf = xray_probs = None
    if xray_image is not None:
        xray_bytes = await xray_image.read()
        if len(xray_bytes) > 0:
            try:
                xray_grade, xray_conf, xray_probs = _run_xray(xray_bytes)
            except Exception as e:
                logger.warning(f"X-ray inference failed: {e}")

    return build_final_response(
        electrical_grade=elec_grade,
        electrical_confidence=elec_conf,
        electrical_probs=elec_probs,
        thermal_grade=thermal_grade,
        thermal_confidence=thermal_conf,
        thermal_probs=thermal_probs,
        xray_grade=xray_grade,
        xray_confidence=xray_conf,
        xray_probs=xray_probs,
        model_version=settings.MODEL_VERSION,
    )


@app.post(
    "/analyze/csv",
    response_model=CSVAnalyzeResponse,
    summary="Batch analysis from CSV telemetry file",
    tags=["Analysis"],
)
async def analyze_csv(
    file: UploadFile = File(..., description="CSV file with battery telemetry rows"),
) -> CSVAnalyzeResponse:
    """
    Process a CSV file where each row is one battery cycle/measurement.
    
    Expected columns: same as the battery_data JSON fields.
    Missing columns are imputed. Each row is graded independently.
    Thermal/X-ray branches are not available via CSV upload.
    """
    if not model_manager.electrical.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Electrical model not available.",
        )

    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")

    results = []
    grade_counts = {"A": 0, "B": 0, "C": 0, "SCRAP": 0}

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        features = csv_row_to_array(row_dict)
        try:
            elec_grade, elec_conf, elec_probs = model_manager.electrical.predict(features)
        except Exception as e:
            logger.warning(f"Row inference error: {e}")
            continue

        response = build_final_response(
            electrical_grade=elec_grade,
            electrical_confidence=elec_conf,
            electrical_probs=elec_probs,
            model_version=settings.MODEL_VERSION,
        )
        results.append(response)
        grade_counts[elec_grade] = grade_counts.get(elec_grade, 0) + 1

    return CSVAnalyzeResponse(
        results=results,
        total_processed=len(results),
        summary=grade_counts,
        model_version=settings.MODEL_VERSION,
    )


@app.post(
    "/analyze/electrical",
    response_model=ElectricalAnalysisResponse,
    summary="Electrical telemetry branch only",
    tags=["Analysis"],
)
async def analyze_electrical(
    battery_data: BatteryData,
) -> ElectricalAnalysisResponse:
    """
    Independent Electrical Telemetry Analyzer endpoint.
    Accepts telemetry JSON directly and returns grade (A, B, C, SCRAP) with probabilities.
    """
    if not model_manager.electrical.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Electrical model not available.",
        )

    grade, confidence, probs = _run_electrical(battery_data)

    return ElectricalAnalysisResponse(
        electrical=ElectricalResult(
            grade=grade,
            confidence=round(confidence, 4),
            probabilities={k: round(v, 4) for k, v in probs.items()},
        ),
        model_version=settings.MODEL_VERSION,
    )


@app.post(
    "/analyze/thermal",
    response_model=ThermalAnalysisResponse,
    summary="Thermal branch only",
    tags=["Analysis"],
)
async def analyze_thermal(
    image: UploadFile = File(..., description="Thermal image (PNG/JPG)"),
) -> ThermalAnalysisResponse:
    """
    Test the thermal branch in isolation.
    Returns LOW/MEDIUM/HIGH risk with confidence scores.
    """
    if not model_manager.thermal.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Thermal model not available. Run: python scripts/train_thermal.py",
        )

    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=422, detail="Empty image file.")

    grade, confidence, probs = model_manager.thermal.predict(image_bytes)

    return ThermalAnalysisResponse(
        thermal=ThermalResult(
            available=True,
            grade=grade,
            confidence=round(confidence, 4),
            probabilities={k: round(v, 4) for k, v in probs.items()},
        ),
        model_version=settings.MODEL_VERSION,
    )


@app.post(
    "/analyze/xray",
    response_model=XRayAnalysisResponse,
    summary="X-ray branch only",
    tags=["Analysis"],
)
async def analyze_xray(
    image: UploadFile = File(..., description="X-ray image (PNG/JPG)"),
) -> XRayAnalysisResponse:
    """
    Test the X-ray branch in isolation.
    Returns Grade A, B, C, SCRAP with confidence.
    """
    if not model_manager.xray.is_loaded:
        return XRayAnalysisResponse(
            xray=XRayResult(
                available=False,
                grade=None,
                confidence=None,
            ),
            model_version=settings.MODEL_VERSION,
        )

    image_bytes = await image.read()
    grade, confidence, probs = model_manager.xray.predict(image_bytes)

    return XRayAnalysisResponse(
        xray=XRayResult(
            available=True,
            grade=grade,
            confidence=round(confidence, 4),
            probabilities={k: round(v, 4) for k, v in probs.items()} if probs else None,
        ),
        model_version=settings.MODEL_VERSION,
    )


@app.get(
    "/model/features",
    response_model=FeaturesResponse,
    summary="Electrical model feature importance",
    tags=["Explainability"],
)
async def get_features() -> FeaturesResponse:
    """
    Returns the feature importance scores from the electrical XGBoost model.
    Useful for understanding which battery measurements most influence grading.
    """
    importance_path = Path(settings.ELECTRICAL_FEATURE_IMPORTANCE_PATH)
    if importance_path.exists():
        with open(importance_path) as f:
            importance_data = json.load(f)
        features = [
            FeatureImportanceItem(feature=item["feature"], importance=item["importance"])
            for item in importance_data
        ]
    elif model_manager.electrical.is_loaded:
        # Fallback: extract from model directly
        try:
            model_clf = model_manager.electrical.model.named_steps.get("clf")
            if model_clf and hasattr(model_clf, "feature_importances_"):
                feature_names = get_feature_names()
                importances = model_clf.feature_importances_
                idx = np.argsort(importances)[::-1]
                features = [
                    FeatureImportanceItem(
                        feature=feature_names[i],
                        importance=round(float(importances[i]), 6),
                    )
                    for i in idx
                ]
            else:
                features = []
        except Exception:
            features = []
    else:
        raise HTTPException(
            status_code=503,
            detail="Electrical model not loaded.",
        )

    return FeaturesResponse(features=features, model_version=settings.MODEL_VERSION)


@app.post(
    "/explain/thermal",
    summary="Grad-CAM heatmap for thermal image",
    tags=["Explainability"],
    responses={200: {"content": {"image/png": {}}}},
)
async def explain_thermal(
    image: UploadFile = File(..., description="Thermal image to explain"),
):
    """
    Returns a Grad-CAM heatmap showing which regions of the thermal image
    most influenced the risk prediction.

    Returns PNG image bytes. Useful for hackathon demo and operator trust.
    
    Note: Requires opencv-python-headless to be installed.
    Returns 501 if Grad-CAM is unavailable.
    """
    if not model_manager.thermal.is_loaded:
        raise HTTPException(status_code=503, detail="Thermal model not loaded.")

    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=422, detail="Empty image file.")

    heatmap_bytes = model_manager.thermal.grad_cam(image_bytes)

    if heatmap_bytes is None:
        raise HTTPException(
            status_code=501,
            detail="Grad-CAM not available (requires opencv-python-headless).",
        )

    return Response(content=heatmap_bytes, media_type="image/png")


# ── Root redirect ──────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "PunarShakti AI Battery Assessment API",
        "version": settings.MODEL_VERSION,
        "docs": "/docs",
        "health": "/health",
        "disclaimer": (
            "PRELIMINARY AI ASSESSMENT SYSTEM. "
            "Not a replacement for physical battery certification."
        ),
    }
