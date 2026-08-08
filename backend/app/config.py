"""
config.py
==========
Centralized configuration for the PunarShakti AI backend.
Uses pydantic-settings for environment variable loading.
"""

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # ── CORS ──────────────────────────────────────────────────────────────────
    FRONTEND_URL: str = "http://localhost:3000"

    # ── Model paths ───────────────────────────────────────────────────────────
    ELECTRICAL_MODEL_PATH: str = "models/electrical_model.joblib"
    ELECTRICAL_PREPROCESSOR_PATH: str = "models/electrical_preprocessor.joblib"
    ELECTRICAL_FEATURE_META_PATH: str = "models/electrical_feature_meta.json"
    ELECTRICAL_FEATURE_IMPORTANCE_PATH: str = "models/electrical_feature_importance.json"

    THERMAL_MODEL_PATH: str = "models/thermal_model.pt"
    XRAY_MODEL_PATH: str = "models/xray_model.pt"

    # ── System ────────────────────────────────────────────────────────────────
    MODEL_VERSION: str = "1.0.0"

    # ── Grade config ─────────────────────────────────────────────────────────
    GRADE_LABELS: List[str] = ["A", "B", "C", "SCRAP"]
    THERMAL_RISK_LABELS: List[str] = ["LOW", "MEDIUM", "HIGH"]
    XRAY_RISK_LABELS: List[str] = ["NORMAL", "SUSPICIOUS"]

    GRADE_RECOMMENDATIONS: dict = {
        "A": "Solar / Telecom backup storage",
        "B": "Home & commercial backup storage",
        "C": "Low-cycle stationary storage / street lighting",
        "SCRAP": "Material recovery & recycling",
    }

    # ── Thermal image ─────────────────────────────────────────────────────────
    THERMAL_IMAGE_SIZE: int = 128

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
