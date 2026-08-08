"""
electrical_model.py
=====================
Electrical branch inference wrapper.

Loads the trained XGBoost pipeline from disk and exposes
a predict() method that returns grade probabilities.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np

from .config import settings

logger = logging.getLogger(__name__)

GRADE_LABELS = ["A", "B", "C", "SCRAP"]


class ElectricalModel:
    """Wraps the trained XGBoost electrical model pipeline."""

    def __init__(self):
        self.model = None
        self.is_loaded = False

    def load(self) -> bool:
        """Load model from disk. Returns True if successful."""
        model_path = Path(settings.ELECTRICAL_MODEL_PATH)
        if not model_path.exists():
            logger.warning(f"Electrical model not found at {model_path}")
            return False

        try:
            self.model = joblib.load(model_path)
            self.is_loaded = True
            logger.info(f"Electrical model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load electrical model: {e}")
            return False

    def predict(self, features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Run inference on a feature array.
        
        Args:
            features: shape (1, n_features) numpy array
            
        Returns:
            (grade_str, confidence, probabilities_dict)
        """
        if not self.is_loaded:
            raise RuntimeError("Electrical model not loaded.")

        # Get class probabilities
        proba = self.model.predict_proba(features)[0]  # shape: (4,)

        # Top class
        top_idx = int(np.argmax(proba))
        top_grade = GRADE_LABELS[top_idx]
        confidence = float(proba[top_idx])

        probabilities = {label: float(p) for label, p in zip(GRADE_LABELS, proba)}

        return top_grade, confidence, probabilities
