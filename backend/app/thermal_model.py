"""
thermal_model.py
==================
Thermal branch inference wrapper.

Loads the trained ResNet-18 PyTorch model and exposes
a predict() method that returns thermal risk probabilities.

Also implements optional Grad-CAM for explainability.
"""

import io
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from .config import settings

logger = logging.getLogger(__name__)

RISK_LABELS = ["LOW", "MEDIUM", "HIGH"]

# ImageNet normalization (same as training)
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((settings.THERMAL_IMAGE_SIZE, settings.THERMAL_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _build_model_architecture(n_classes: int = 4) -> nn.Module:
    """Reconstruct the ResNet-18 architecture used during training."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, n_classes),
    )
    return model


class ThermalModel:
    """Wraps the trained ResNet-18 thermal grade model."""

    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        self.class_to_idx: Dict[str, int] = {"A": 0, "B": 1, "C": 2, "SCRAP": 3}
        self.idx_to_class: Dict[int, str] = {0: "A", 1: "B", 2: "C", 3: "SCRAP"}
        self.n_classes = 4

    def load(self) -> bool:
        """Load model checkpoint from disk. Returns True if successful."""
        model_path = Path(settings.THERMAL_MODEL_PATH)
        if not model_path.exists():
            logger.warning(f"Thermal model not found at {model_path}")
            return False

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.n_classes = checkpoint.get("n_classes", 3)
            self.class_to_idx = checkpoint.get("class_to_idx", self.class_to_idx)
            self.idx_to_class = checkpoint.get("idx_to_class", self.idx_to_class)

            self.model = _build_model_architecture(self.n_classes)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info(f"Thermal model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load thermal model: {e}")
            return False

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Convert raw image bytes to a model-ready tensor."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = VAL_TRANSFORM(img).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, image_bytes: bytes) -> Tuple[str, float, Dict[str, float]]:
        """
        Run thermal risk inference on image bytes.
        
        Returns:
            (risk_label, confidence, probabilities_dict)
        """
        if not self.is_loaded:
            raise RuntimeError("Thermal model not loaded.")

        tensor = self.preprocess_image(image_bytes)

        with torch.no_grad():
            logits = self.model(tensor)
            proba = F.softmax(logits, dim=1)[0].cpu().numpy()

        top_idx = int(np.argmax(proba))
        risk_label = self.idx_to_class.get(top_idx, "UNKNOWN")
        confidence = float(proba[top_idx])
        probabilities = {
            self.idx_to_class.get(i, str(i)): float(p)
            for i, p in enumerate(proba)
        }

        return risk_label, confidence, probabilities

    def grad_cam(self, image_bytes: bytes) -> Optional[bytes]:
        """
        Generate a Grad-CAM heatmap for the predicted class.
        
        Returns PNG bytes of the heatmap overlay, or None if unavailable.
        
        Grad-CAM shows which spatial regions of the thermal image
        most influenced the risk prediction — useful for demo/explainability.
        """
        if not self.is_loaded:
            return None

        try:
            import cv2

            tensor = self.preprocess_image(image_bytes)

            # Hook into the last convolutional layer (layer4[-1])
            gradients = []
            activations = []

            def save_gradient(grad):
                gradients.append(grad)

            def forward_hook(module, input, output):
                activations.append(output)
                output.register_hook(save_gradient)

            target_layer = self.model.layer4[-1]
            hook = target_layer.register_forward_hook(forward_hook)

            self.model.eval()
            logits = self.model(tensor)
            pred_class = logits.argmax(dim=1).item()

            self.model.zero_grad()
            logits[0, pred_class].backward()

            hook.remove()

            if not gradients or not activations:
                return None

            grad = gradients[0].squeeze(0).cpu().numpy()   # (C, H, W)
            act = activations[0].squeeze(0).cpu().numpy()  # (C, H, W)

            weights = grad.mean(axis=(1, 2))               # (C,)
            cam = np.sum(weights[:, None, None] * act, axis=0)
            cam = np.maximum(cam, 0)
            if cam.max() > 0:
                cam = cam / cam.max()

            # Resize to image dimensions
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_np = np.array(img.resize((settings.THERMAL_IMAGE_SIZE, settings.THERMAL_IMAGE_SIZE)))
            cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))

            # Apply colormap and overlay
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

            # Return as PNG bytes
            result_img = Image.fromarray(overlay)
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            return buf.getvalue()

        except Exception as e:
            logger.warning(f"Grad-CAM failed: {e}")
            return None
