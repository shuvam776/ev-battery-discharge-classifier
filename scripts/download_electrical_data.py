"""
download_electrical_data.py
============================
Downloads the NASA Prognostics Center of Excellence (PCoE)
Li-ion Battery Aging Dataset.

Dataset: Battery Data Set (ID: 5)
URL: https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip

Contains 18650 Li-ion cells (B0005–B0056) cycled to failure.
Each cell has charge, discharge, and impedance measurements.

NOTE: This dataset uses 18650 cylindrical cells under controlled
laboratory conditions. It serves as a proxy for EV battery telemetry
in this prototype. A production system would use PunarShakti's own
synchronized EV pack dataset.
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────
DATA_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
RAW_DIR = Path("data/raw/electrical")
ZIP_PATH = RAW_DIR / "nasa_battery_dataset.zip"

# ── Helpers ───────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Stream-download a file with a tqdm progress bar."""
    print(f"Downloading: {url}")
    print(f"Destination: {dest}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=dest.name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            size = f.write(chunk)
            bar.update(size)

    print(f"Download complete: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract a ZIP archive, and recursively extract any inner ZIPs."""
    print(f"Extracting {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        print(f"  Archive contains {len(members)} files/dirs.")
        for member in tqdm(members, desc="Extracting outer"):
            zf.extract(member, extract_dir)
    print("Outer extraction complete.")

    # NASA dataset is a ZIP of ZIPs — extract inner ZIPs too
    inner_zips = list(extract_dir.rglob("*.zip"))
    inner_zips = [z for z in inner_zips if z != zip_path]
    if inner_zips:
        print(f"Found {len(inner_zips)} inner ZIP archives — extracting...")
        for inner_zip in tqdm(inner_zips, desc="Extracting inner ZIPs"):
            try:
                with zipfile.ZipFile(inner_zip, "r") as inner_zf:
                    inner_zf.extractall(inner_zip.parent)
                print(f"  Extracted: {inner_zip.name}")
            except Exception as e:
                print(f"  WARNING: Could not extract {inner_zip.name}: {e}")
    print("All extraction complete.")


def check_existing() -> bool:
    """Return True if .mat files already present."""
    mat_files = list(RAW_DIR.rglob("*.mat"))
    if mat_files:
        print(f"Found {len(mat_files)} existing .mat files. Skipping download.")
        for f in mat_files[:5]:
            print(f"  {f}")
        return True
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if check_existing():
        sys.exit(0)

    # Download
    if not ZIP_PATH.exists():
        try:
            download_file(DATA_URL, ZIP_PATH)
        except Exception as e:
            print(f"\nERROR: Download failed: {e}")
            print("Manual download instructions:")
            print(f"  1. Go to: {DATA_URL}")
            print(f"  2. Save to: {ZIP_PATH.resolve()}")
            print("  3. Re-run this script.")
            sys.exit(1)
    else:
        print(f"ZIP already present: {ZIP_PATH}")

    # Extract
    extract_zip(ZIP_PATH, RAW_DIR)

    # Verify
    mat_files = list(RAW_DIR.rglob("*.mat"))
    if not mat_files:
        print("ERROR: No .mat files found after extraction. Check archive contents.")
        sys.exit(1)

    print(f"\nDataset ready. Found {len(mat_files)} .mat files:")
    for f in sorted(mat_files):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
