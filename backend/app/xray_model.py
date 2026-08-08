"""
xray_model.py
===============
X-ray branch inference wrapper.

STATUS: ARCHITECTURE READY — MODEL NOT TRAINED IN V1

The model interface is fully implemented and integrated into the
decision engine and API. The branch activates automatically
when xray_model.pt is present in models/.

When not available:
- /health returns xray_model_loaded: false
- /analyze accepts but ignores xray_image input
- /analyze/xray returns available: false with explanation

To enable: train the model using scripts/train_xray.py
"""

import io
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from .config import settings

logger = logging.getLogger(__name__)

RISK_LABELS = ["A", "B", "C", "SCRAP"]

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _build_xray_architecture(n_classes: int = 4) -> nn.Module:
    """ResNet-18 head for structural grade classification."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 64),
        nn.ReLU(),
        nn.Linear(64, n_classes),
    )
    return model


class XRayModel:
    """Wraps the X-ray structural grade model."""

    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        self.class_to_idx: Dict[str, int] = {"A": 0, "B": 1, "C": 2, "SCRAP": 3}
        self.idx_to_class: Dict[int, str] = {0: "A", 1: "B", 2: "C", 3: "SCRAP"}
        self.n_classes = 4

    def load(self) -> bool:
        """Load model checkpoint from disk. Returns True if successful."""
        model_path = Path(settings.XRAY_MODEL_PATH)
        if not model_path.exists():
            logger.info("X-ray model not found (expected — V1 architecture stub).")
            return False

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.n_classes = checkpoint.get("n_classes", 2)
            self.class_to_idx = checkpoint.get("class_to_idx", self.class_to_idx)
            self.idx_to_class = checkpoint.get("idx_to_class", self.idx_to_class)

            self.model = _build_xray_architecture(self.n_classes)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info(f"X-ray model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load X-ray model: {e}")
            return False

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return VAL_TRANSFORM(img).unsqueeze(0).to(self.device)

    def predict(self, image_bytes: bytes) -> Tuple[str, float, Dict[str, float]]:
        """
        Run structural risk inference on image bytes.
        
        Returns:
            (risk_label, confidence, probabilities_dict)
        """
        if not self.is_loaded:
            raise RuntimeError("X-ray model not loaded.")

        tensor = self.preprocess_image(image_bytes)
        with torch.no_grad():
            logits = self.model(tensor)
            proba = F.softmax(logits, dim=1)[0].cpu().numpy()

        import numpy as np
        top_idx = int(np.argmax(proba))
        risk_label = self.idx_to_class.get(top_idx, "UNKNOWN")
        confidence = float(proba[top_idx])
        probabilities = {
            self.idx_to_class.get(i, str(i)): float(p)
            for i, p in enumerate(proba)
        }

        return risk_label, confidence, probabilities
