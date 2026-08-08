"""
train_thermal.py
=================
Trains the PunarShakti thermal branch using transfer learning.

Architecture:
    Thermal Image (128-128 RGB)
         -
    ImageNet pretrained ResNet-18 (frozen backbone)
         -
    Replace final FC layer: 512 - 3 (LOW/MEDIUM/HIGH)
         -
    Stage 1: Train head only (5 epochs)
         -
    Stage 2: Unfreeze layer4, fine-tune (10 epochs, low LR)
         -
    LOW / MEDIUM / HIGH risk prediction

Notes:
- Uses CPU-friendly training (small images, ResNet-18)
- Early stopping on validation loss
- Class weighting if imbalanced
- Saves best checkpoint
"""

import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")

# -- Paths ---------------------------------------------------------------------
THERMAL_DIR = Path("data/raw/thermal")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# -- Config --------------------------------------------------------------------
CLASSES = ["A", "B", "C", "SCRAP"]
N_CLASSES = 4
CLASS_TO_IDX = {"A": 0, "B": 1, "C": 2, "SCRAP": 3}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

BATCH_SIZE = 32
IMAGE_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
# Windows: num_workers > 0 requires __main__ guard; we use 0 for compatibility
NUM_WORKERS = 0

# Stage 1: head only
STAGE1_EPOCHS = 8
STAGE1_LR = 1e-3

# Stage 2: fine-tune last block
STAGE2_EPOCHS = 12
STAGE2_LR = 1e-4

PATIENCE = 5  # Early stopping patience


# -- Transforms ----------------------------------------------------------------

# Training: mild augmentations appropriate for thermal images
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),  # Physically valid small rotation
    transforms.ColorJitter(brightness=0.15, contrast=0.15),  # Sensor variation only
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Val/test: deterministic
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# -- Dataset -------------------------------------------------------------------

class ThermalDataset(Dataset):
    def __init__(self, root_dir: Path, split: str, transform=None):
        self.transform = transform
        self.samples = []
        self.labels = []

        split_dir = root_dir / split
        for cls_name in CLASSES:
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.glob("*.png")):
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

def build_thermal_model(n_classes: int = N_CLASSES) -> nn.Module:
    """
    ImageNet-pretrained ResNet-18 with custom head for thermal risk classification.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze all layers initially (Stage 1: train head only)
    for param in model.parameters():
        param.requires_grad = False

    # Replace final classification layer
    in_features = model.fc.in_features  # 512 for ResNet-18
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, n_classes),
    )

    return model


def unfreeze_layer4(model: nn.Module) -> None:
    """Stage 2: Unfreeze layer4 for fine-tuning."""
    for param in model.layer4.parameters():
        param.requires_grad = True
    # Also unfreeze BN in earlier layers for stability
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = True


# -- Training utilities --------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def compute_class_weights(dataset: ThermalDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    counts = np.bincount(dataset.labels, minlength=N_CLASSES).astype(float)
    if counts.min() == 0:
        return None
    weights = counts.sum() / (N_CLASSES * counts)
    return torch.FloatTensor(weights).to(DEVICE)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += len(labels)
    return total_loss / total, total_correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total += len(labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, total_correct / total, np.array(all_preds), np.array(all_labels)


def run_training_stage(model, train_loader, val_loader, criterion,
                        optimizer, scheduler, n_epochs: int, stage_name: str):
    """Run one training stage with early stopping."""
    early_stopper = EarlyStopping(patience=PATIENCE)
    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n{stage_name}")
    for epoch in range(1, n_epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc, _, _ = eval_epoch(model, val_loader, criterion)
        scheduler.step(v_loss)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        print(f"  Epoch {epoch:2d}/{n_epochs} | "
              f"train_loss={t_loss:.4f} acc={t_acc:.3f} | "
              f"val_loss={v_loss:.4f} acc={v_acc:.3f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if early_stopper(v_loss):
            print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return history


# -- Main ----------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("PunarShakti AI - Thermal Branch Training")
    print("=" * 60)

    # Load metadata
    with open(PROCESSED_DIR / "thermal_metadata.json") as f:
        thermal_meta = json.load(f)
    print(f"\nDataset: {thermal_meta['dataset_type']}")
    print(f"Note: {thermal_meta['note'][:80]}...")

    # -- Datasets --
    train_ds = ThermalDataset(THERMAL_DIR, "train", train_transform)
    val_ds = ThermalDataset(THERMAL_DIR, "val", val_transform)
    test_ds = ThermalDataset(THERMAL_DIR, "test", val_transform)

    print(f"\nDataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    if len(train_ds) == 0:
        print("ERROR: No training images found. Run prepare_thermal_data.py first.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # -- Model --
    model = build_thermal_model()
    model = model.to(DEVICE)

    # Class weights
    class_weights = compute_class_weights(train_ds)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # -- STAGE 1: Train head only --
    t0 = time.time()
    optimizer_s1 = optim.Adam(model.fc.parameters(), lr=STAGE1_LR, weight_decay=1e-4)
    scheduler_s1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s1, patience=2, factor=0.5)

    hist_s1 = run_training_stage(
        model, train_loader, val_loader, criterion,
        optimizer_s1, scheduler_s1, STAGE1_EPOCHS, "STAGE 1: Training head only"
    )

    # -- STAGE 2: Fine-tune layer4 --
    unfreeze_layer4(model)
    optimizer_s2 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE2_LR, weight_decay=1e-4
    )
    scheduler_s2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s2, patience=3, factor=0.5)

    hist_s2 = run_training_stage(
        model, train_loader, val_loader, criterion,
        optimizer_s2, scheduler_s2, STAGE2_EPOCHS, "STAGE 2: Fine-tuning layer4"
    )

    total_time = time.time() - t0
    print(f"\nTotal training time: {total_time:.1f}s")

    # -- Final evaluation --
    print("\n-- Final Test Evaluation --")
    _, test_acc, y_pred, y_true = eval_epoch(model, test_loader, criterion)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Macro F1:      {macro_f1:.4f}")
    print(f"Weighted F1:   {weighted_f1:.4f}")
    print("\nClassification Report:")
    print(report)

    # -- Save model --
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "n_classes": N_CLASSES,
        "image_size": IMAGE_SIZE,
        "accuracy": float(test_acc),
        "macro_f1": float(macro_f1),
        "dataset_type": "synthetic",
    }, MODELS_DIR / "thermal_model.pt")
    print(f"\n- Saved: {MODELS_DIR / 'thermal_model.pt'}")

    # -- Reports --
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("ResNet-18 - Thermal Risk Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "thermal_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  Saved: {REPORTS_DIR / 'thermal_confusion_matrix.png'}")

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    s1_len = len(hist_s1["train_loss"])
    s2_len = len(hist_s2["train_loss"])
    all_train_loss = hist_s1["train_loss"] + hist_s2["train_loss"]
    all_val_loss = hist_s1["val_loss"] + hist_s2["val_loss"]
    all_train_acc = hist_s1["train_acc"] + hist_s2["train_acc"]
    all_val_acc = hist_s1["val_acc"] + hist_s2["val_acc"]

    epochs = list(range(1, len(all_train_loss) + 1))
    axes[0].plot(epochs, all_train_loss, label="Train Loss")
    axes[0].plot(epochs, all_val_loss, label="Val Loss")
    axes[0].axvline(x=s1_len + 0.5, color="red", linestyle="--", alpha=0.5, label="Stage 2 start")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[1].plot(epochs, all_train_acc, label="Train Acc")
    axes[1].plot(epochs, all_val_acc, label="Val Acc")
    axes[1].axvline(x=s1_len + 0.5, color="red", linestyle="--", alpha=0.5, label="Stage 2 start")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    plt.suptitle("Thermal Branch - Training Curves (ResNet-18)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "thermal_training_curves.png", dpi=150)
    plt.close()

    # Text report
    with open(REPORTS_DIR / "thermal_classification_report.txt", "w") as f:
        f.write("PunarShakti AI - Thermal Branch Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: ResNet-18 (ImageNet pretrained, transfer learning)\n")
        f.write(f"Dataset: {thermal_meta['dataset_type']}\n")
        f.write(f"Training time: {total_time:.1f}s\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Macro F1:      {macro_f1:.4f}\n")
        f.write(f"Weighted F1:   {weighted_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\n\nIMPORTANT NOTE:\n")
        f.write(f"This model was trained on SYNTHETIC thermal images.\n")
        f.write(f"It simulates LOW/MEDIUM/HIGH risk patterns based on thermal\n")
        f.write(f"physics principles. Performance on real battery thermal images\n")
        f.write(f"will differ. Replace with PunarShakti proprietary data for\n")
        f.write(f"production deployment.\n")

    print(f"  Saved: {REPORTS_DIR / 'thermal_classification_report.txt'}")
    print("\n- Thermal branch training complete.")


if __name__ == "__main__":
    main()
