"""
preprocessing.py
=================
Feature extraction utilities for the electrical branch.

Converts raw BatteryData into the feature vector expected by the
trained XGBoost model, using the exact same feature order as training.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import settings

# Load feature names from training metadata
_feature_meta: Optional[Dict] = None


def _load_feature_meta() -> Dict:
    global _feature_meta
    if _feature_meta is None:
        meta_path = Path(settings.ELECTRICAL_FEATURE_META_PATH)
        if meta_path.exists():
            with open(meta_path) as f:
                _feature_meta = json.load(f)
        else:
            # Fallback: hardcoded feature order matching prepare script
            _feature_meta = {
                "feature_names": [
                    "voltage_mean", "voltage_std", "voltage_min",
                    "voltage_max", "voltage_range",
                    "current_mean", "current_std", "current_min",
                    "current_max", "current_range",
                    "temperature_mean", "temperature_std", "temperature_min",
                    "temperature_max", "temperature_range",
                    "cycle_duration", "energy_wh", "cycle_number",
                    "Re", "Rct",
                ],
                "grade_labels": ["A", "B", "C", "SCRAP"],
            }
    return _feature_meta


def get_feature_names() -> List[str]:
    """Return the ordered feature names used during training."""
    return _load_feature_meta()["feature_names"]


def battery_data_to_array(battery_data) -> np.ndarray:
    """
    Convert a BatteryData object to a numpy feature vector.
    
    Missing values are set to NaN and handled by the pipeline's imputer.
    The feature order MUST match what was used during training.
    """
    meta = _load_feature_meta()
    feature_names = meta["feature_names"]

    # Map field names to values
    data_dict = battery_data.model_dump()

    row = []
    for feat in feature_names:
        val = data_dict.get(feat, None)
        row.append(float(val) if val is not None else float("nan"))

    return np.array(row, dtype=np.float32).reshape(1, -1)


def csv_row_to_array(row_dict: Dict) -> np.ndarray:
    """
    Convert a CSV row (dict) to a numpy feature vector.
    Handles missing or non-numeric values gracefully.
    """
    meta = _load_feature_meta()
    feature_names = meta["feature_names"]

    row = []
    for feat in feature_names:
        val = row_dict.get(feat, None)
        try:
            row.append(float(val) if val is not None and str(val).strip() != "" else float("nan"))
        except (ValueError, TypeError):
            row.append(float("nan"))

    return np.array(row, dtype=np.float32).reshape(1, -1)
