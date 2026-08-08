"""
prepare_xray_data.py
====================
Generates synthetic battery X-ray images for Grade A, B, C, SCRAP classification.
"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

RAW_XRAY_DIR = Path("data/raw/xray")
PROCESSED_DIR = Path("data/processed")
IMAGE_SIZE = (128, 128)
SAMPLES_PER_CLASS = 200
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

CLASSES = ["A", "B", "C", "SCRAP"]


def generate_base_xray() -> np.ndarray:
    h, w = IMAGE_SIZE
    img = np.full((h, w), 180.0, dtype=np.float32)
    img[:, :12] -= 80
    img[:, -12:] -= 80
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    jelly_roll = 25 * np.sin(r * 0.45)
    img += jelly_roll
    img += np.random.normal(0, 4.0, (h, w))
    return img


def generate_grade_a_xray() -> np.ndarray:
    """Grade A: Pristine internal structure."""
    return np.clip(generate_base_xray(), 0, 255).astype(np.float32)


def generate_grade_b_xray() -> np.ndarray:
    """Grade B: Slight internal layer waviness."""
    img = generate_base_xray()
    h, w = IMAGE_SIZE
    y, x = np.ogrid[:h, :w]
    img += 12 * np.sin(x * 0.1) * np.exp(-((y - 64) ** 2) / (2 * 30 ** 2))
    return np.clip(img, 0, 255).astype(np.float32)


def generate_grade_c_xray() -> np.ndarray:
    """Grade C: Noticeable delamination gap / layer shift."""
    img = generate_base_xray()
    y0 = random.randint(40, 80)
    img[y0:y0 + 5, 20:108] += 45.0
    return np.clip(img, 0, 255).astype(np.float32)


def generate_scrap_xray() -> np.ndarray:
    """SCRAP: Severe mechanical deformation / internal metallic debris fragment."""
    img = generate_base_xray()
    h, w = IMAGE_SIZE
    cx, cy = w / 2, h / 2
    y, x = np.ogrid[:h, :w]
    debris = 110.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 7 ** 2))
    img -= debris
    img[30:40, 20:108] += 70.0
    return np.clip(img, 0, 255).astype(np.float32)


GENERATORS = {
    "A": generate_grade_a_xray,
    "B": generate_grade_b_xray,
    "C": generate_grade_c_xray,
    "SCRAP": generate_scrap_xray,
}


def main():
    print("Preparing X-ray dataset for Grade A, B, C, SCRAP classification...")
    splits = {"train": 0.7, "val": 0.15, "test": 0.15}
    for split in splits:
        for cls in CLASSES:
            (RAW_XRAY_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    total = 0
    for cls, generator in GENERATORS.items():
        n_train = int(SAMPLES_PER_CLASS * splits["train"])
        n_val = int(SAMPLES_PER_CLASS * splits["val"])
        n_test = SAMPLES_PER_CLASS - n_train - n_val

        for split, count in [("train", n_train), ("val", n_val), ("test", n_test)]:
            for i in range(count):
                arr = generator()
                img = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
                img = img.resize(IMAGE_SIZE)
                img.save(RAW_XRAY_DIR / split / cls / f"{cls}_{i:04d}.png")
                total += 1

    print(f"Generated {total} X-ray images for Grades A, B, C, SCRAP.")


if __name__ == "__main__":
    main()
