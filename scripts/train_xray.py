"""
train_xray.py
===============
X-ray branch training script for PunarShakti AI.

STATUS: ARCHITECTURE READY - NOT TRAINED (V1)

REASON:
No freely available, reliably labeled battery X-ray dataset exists
for second-life EV battery structural defect detection. The available
industrial X-ray datasets (e.g., MVTec, DAGM) are general-purpose and
not specifically labeled for battery structural defects.

WHAT THIS SCRIPT DOES:
1. Checks for a dataset in data/raw/xray/
2. If found with correct structure, trains a ResNet-18 classifier
3. If not found, prints clear instructions and exits gracefully

TO ENABLE X-RAY BRANCH:
  Place a dataset in data/raw/xray/ with this structure:
    data/raw/xray/
        train/
            NORMAL/     - images of structurally normal batteries
            SUSPICIOUS/ - images showing structural defects/anomalies
        val/
            NORMAL/
            SUSPICIOUS/
        test/
            NORMAL/
            SUSPICIOUS/

  Then run: python scripts/train_xray.py

ARCHITECTURE (pluggable, same as thermal):
    X-ray image (224-224 RGB)
         -
    ImageNet pretrained ResNet-18
         -
    Custom head: 512 - 2 (NORMAL / SUSPICIOUS)
         -
    Structural risk: NORMAL / SUSPICIOUS

FUTURE DATASETS TO INVESTIGATE:
- PunarShakti proprietary battery X-ray dataset (primary)
- Battery electrode CT scan datasets from academic papers
- Industrial defect datasets if applicable labels can be curated
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# -- Paths ---------------------------------------------------------------------
XRAY_DIR = Path("data/raw/xray")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# -- Config --------------------------------------------------------------------
CLASSES = ["A", "B", "C", "SCRAP"]
N_CLASSES = 4
CLASS_TO_IDX = {"A": 0, "B": 1, "C": 2, "SCRAP": 3}
IDX_TO_CLASS = {0: "A", 1: "B", 2: "C", 3: "SCRAP"}

IMAGE_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STAGE1_EPOCHS = 8
STAGE1_LR = 1e-3
STAGE2_EPOCHS = 10
STAGE2_LR = 1e-4


# -- Dataset check -------------------------------------------------------------

def check_dataset() -> bool:
    """Return True if a valid dataset structure exists."""
    required = [
        XRAY_DIR / "train" / "A",
        XRAY_DIR / "train" / "B",
        XRAY_DIR / "train" / "C",
        XRAY_DIR / "train" / "SCRAP",
        XRAY_DIR / "val" / "A",
        XRAY_DIR / "val" / "B",
        XRAY_DIR / "val" / "C",
        XRAY_DIR / "val" / "SCRAP",
    ]
    for path in required:
        if not path.exists():
            return False
        images = list(path.glob("*.png")) + list(path.glob("*.jpg")) + list(path.glob("*.jpeg"))
        if len(images) == 0:
            return False
    return True


# -- Dataset -------------------------------------------------------------------

class XRayDataset(Dataset):
    def __init__(self, root_dir: Path, split: str, transform=None):
        self.transform = transform
        self.samples = []
        self.labels = []

        split_dir = root_dir / split
        for cls_name in CLASSES:
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.glob("*")):
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    self.samples.append(str(img_path))
                    self.labels.append(CLASS_TO_IDX[cls_name])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# -- Model ---------------------------------------------------------------------

def build_xray_model() -> nn.Module:
    """ResNet-18 with binary head for structural risk classification."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 64),
        nn.ReLU(),
        nn.Linear(64, N_CLASSES),
    )
    return model


# -- Training (reuse pattern from thermal) ------------------------------------

def train_xray() -> None:
    """Full training pipeline for X-ray branch."""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = XRayDataset(XRAY_DIR, "train", transform)
    val_ds = XRayDataset(XRAY_DIR, "val", val_transform)
    print(f"X-ray dataset: {len(train_ds)} train, {len(val_ds)} val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_xray_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    t0 = time.time()

    # Stage 1: head only
    optimizer = optim.Adam(model.fc.parameters(), lr=STAGE1_LR)
    for epoch in range(1, STAGE1_EPOCHS + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        print(f"  Stage1 Epoch {epoch}/{STAGE1_EPOCHS}")

    # Stage 2: unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=STAGE2_LR
    )
    for epoch in range(1, STAGE2_EPOCHS + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        print(f"  Stage2 Epoch {epoch}/{STAGE2_EPOCHS}")

    training_time = time.time() - t0

    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "n_classes": N_CLASSES,
        "image_size": IMAGE_SIZE,
        "training_time": training_time,
    }, MODELS_DIR / "xray_model.pt")
    print(f"\n- X-ray model saved: {MODELS_DIR / 'xray_model.pt'}")
    print(f"  Training time: {training_time:.1f}s")


# -- Main ----------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("PunarShakti AI - X-ray Branch Training")
    print("=" * 60)

    if not check_dataset():
        print("\n-  X-ray dataset NOT FOUND or incomplete.")
        print("\nStatus: X-ray branch is ARCHITECTURE-READY but not trained.")
        print("\nTo enable the X-ray branch:")
        print("  1. Obtain a labeled battery X-ray dataset")
        print("  2. Organize it as follows:")
        print("       data/raw/xray/")
        print("           train/NORMAL/    - structurally normal images")
        print("           train/SUSPICIOUS/- defect/anomaly images")
        print("           val/NORMAL/")
        print("           val/SUSPICIOUS/")
        print("  3. Re-run: python scripts/train_xray.py")
        print("\nThe X-ray model interface is fully implemented in:")
        print("  backend/app/xray_model.py")
        print("  backend/app/decision_engine.py")
        print("  backend/app/main.py")
        print("\nThe /health endpoint will show: xray_model_loaded: false")
        print("The /analyze endpoint accepts (but ignores) xray images until trained.")

        # Save a status file
        status = {
            "available": False,
            "reason": "No labeled battery X-ray dataset available for V1",
            "architecture": "ResNet-18 (pluggable)",
            "classes": CLASSES,
            "enable_instructions": "See scripts/train_xray.py for dataset format",
        }
        with open(MODELS_DIR / "xray_status.json", "w") as f:
            json.dump(status, f, indent=2)

        sys.exit(0)

    print("\nDataset found! Training X-ray branch...")
    train_xray()


if __name__ == "__main__":
    main()
