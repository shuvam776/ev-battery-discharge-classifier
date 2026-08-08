"""
prepare_thermal_data.py
=========================
Prepares thermal images labeled directly as battery grades: A, B, C, SCRAP.
"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

RAW_THERMAL_DIR = Path("data/raw/thermal")
PROCESSED_DIR = Path("data/processed")
RAW_THERMAL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)
SAMPLES_PER_CLASS = 200
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

CLASSES = ["A", "B", "C", "SCRAP"]
CLASS_TO_IDX = {"A": 0, "B": 1, "C": 2, "SCRAP": 3}


def gaussian_blob(image: np.ndarray, center_x: float, center_y: float,
                   sigma: float, intensity: float) -> np.ndarray:
    h, w = image.shape
    y, x = np.ogrid[:h, :w]
    blob = intensity * np.exp(-((x - center_x * w) ** 2 + (y - center_y * h) ** 2) / (2 * sigma ** 2))
    return np.clip(image + blob, 0, 255)


def generate_grade_a_image() -> np.ndarray:
    """Grade A: Uniform cool temperature profile."""
    h, w = IMAGE_SIZE
    base_temp = np.random.uniform(25, 35)
    noise_level = np.random.uniform(1, 2)
    y, x = np.ogrid[:h, :w]
    radial = np.exp(-((x - w / 2) ** 2 + (y - h / 2) ** 2) / (2 * (w * 0.4) ** 2))
    image = base_temp + 5 * radial + np.random.normal(0, noise_level, (h, w))
    image = np.clip((image - 20) / 30 * 200 + 30, 0, 255)
    return image.astype(np.float32)


def generate_grade_b_image() -> np.ndarray:
    """Grade B: Mild temperature gradient."""
    h, w = IMAGE_SIZE
    base_temp = np.random.uniform(35, 45)
    image = np.full((h, w), base_temp, dtype=np.float32) + np.random.normal(0, 2, (h, w))
    image = gaussian_blob(image, np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7), h * 0.15, np.random.uniform(10, 20))
    image = np.clip((image - 20) / 40 * 200 + 30, 0, 255)
    return image.astype(np.float32)


def generate_grade_c_image() -> np.ndarray:
    """Grade C: Moderate localized hot zones."""
    h, w = IMAGE_SIZE
    base_temp = np.random.uniform(45, 55)
    image = np.full((h, w), base_temp, dtype=np.float32) + np.random.normal(0, 2.5, (h, w))
    for _ in range(2):
        image = gaussian_blob(image, np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8), h * 0.10, np.random.uniform(25, 45))
    image = np.clip((image - 20) / 55 * 200 + 30, 0, 255)
    return image.astype(np.float32)


def generate_scrap_image() -> np.ndarray:
    """SCRAP: Extreme localized hotspot / thermal failure signature."""
    h, w = IMAGE_SIZE
    base_temp = np.random.uniform(60, 80)
    image = np.full((h, w), base_temp, dtype=np.float32) + np.random.normal(0, 3, (h, w))
    image = gaussian_blob(image, np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7), h * 0.06, np.random.uniform(80, 140))
    image = np.clip((image - 20) / 80 * 220 + 30, 0, 255)
    return image.astype(np.float32)


GENERATORS = {
    "A": generate_grade_a_image,
    "B": generate_grade_b_image,
    "C": generate_grade_c_image,
    "SCRAP": generate_scrap_image,
}


def main() -> None:
    print("Preparing thermal dataset for Grade A, B, C, SCRAP classification...")
    splits = {"train": 0.7, "val": 0.15, "test": 0.15}
    for split in splits:
        for cls in CLASSES:
            (RAW_THERMAL_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    total_generated = 0
    for class_name, generator in GENERATORS.items():
        n_total = SAMPLES_PER_CLASS
        n_train = int(n_total * splits["train"])
        n_val = int(n_total * splits["val"])
        n_test = n_total - n_train - n_val
        split_counts = {"train": n_train, "val": n_val, "test": n_test}

        for split_name, n_samples in split_counts.items():
            for i in range(n_samples):
                img_array = generator()
                img_uint8 = img_array.astype(np.uint8)
                img_pil = Image.fromarray(img_uint8, mode="L").convert("RGB")
                img_pil = img_pil.resize(IMAGE_SIZE, Image.BILINEAR)
                img_pil.save(RAW_THERMAL_DIR / split_name / class_name / f"{class_name}_{i:04d}.png")
                total_generated += 1

    print(f"Generated {total_generated} thermal images for Grades A, B, C, SCRAP.")


if __name__ == "__main__":
    main()
