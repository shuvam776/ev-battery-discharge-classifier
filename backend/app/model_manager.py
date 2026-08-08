"""
model_manager.py
==================
Singleton model manager that loads all models at startup
and provides a single access point for inference.

Uses lifespan context manager (FastAPI best practice) so models
are loaded once and reused across all requests.
"""

import logging

from .electrical_model import ElectricalModel
from .thermal_model import ThermalModel
from .xray_model import XRayModel

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages all model instances for the PunarShakti AI backend."""

    def __init__(self):
        self.electrical = ElectricalModel()
        self.thermal = ThermalModel()
        self.xray = XRayModel()

    def load_all(self) -> None:
        """Load all available models. Missing models are skipped gracefully."""
        logger.info("Loading PunarShakti AI models...")

        el_ok = self.electrical.load()
        th_ok = self.thermal.load()
        xr_ok = self.xray.load()

        logger.info(
            f"Model status — Electrical: {'✓' if el_ok else '✗'} | "
            f"Thermal: {'✓' if th_ok else '✗'} | "
            f"X-ray: {'✓' if xr_ok else '✗ (V1 stub)'}"
        )

        if not el_ok:
            logger.warning(
                "Electrical model not loaded. "
                "Run: python scripts/train_electrical.py"
            )

    @property
    def status(self) -> dict:
        return {
            "electrical_model_loaded": self.electrical.is_loaded,
            "thermal_model_loaded": self.thermal.is_loaded,
            "xray_model_loaded": self.xray.is_loaded,
        }


# ── Singleton instance ─────────────────────────────────────────────────────────
model_manager = ModelManager()
