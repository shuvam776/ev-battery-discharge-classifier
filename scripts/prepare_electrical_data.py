"""
prepare_electrical_data.py
============================
Loads NASA Li-ion .mat files, inspects their actual schema,
extracts per-cycle features, computes capacity retention,
generates A/B/C/SCRAP labels, and saves a clean feature matrix.

DESIGN PRINCIPLES:
- Only uses features actually present in the data (no fabrication).
- capacity_retention used only for label generation - NEVER as a feature.
- Battery-level train/test split via GroupShuffleSplit (no leakage).
- All feature names are derived from actual measurement columns found.
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

# -- Paths ---------------------------------------------------------------------
RAW_DIR = Path("data/raw/electrical")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# -- Grade thresholds ----------------------------------------------------------
GRADE_THRESHOLDS = {
    "A": (80.0, float("inf")),
    "B": (65.0, 80.0),
    "C": (50.0, 65.0),
    "SCRAP": (float("-inf"), 50.0),
}
GRADE_LABEL = {"A": 0, "B": 1, "C": 2, "SCRAP": 3}
LABEL_GRADE = {v: k for k, v in GRADE_LABEL.items()}


# -- NASA .mat parsing ---------------------------------------------------------

def load_mat_file(path: Path) -> dict:
    """Load a .mat file and return as dict. Handles older MATLAB formats."""
    try:
        return sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"  WARNING: Could not load {path.name}: {e}")
        return {}


def inspect_mat_structure(mat_data: dict, battery_name: str) -> None:
    """Print the nested structure of a .mat file for schema discovery."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {battery_name}")
    print(f"{'='*60}")

    def show_keys(obj, depth=0, max_depth=4):
        indent = "  " * depth
        if depth > max_depth:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.startswith("__"):
                    continue
                print(f"{indent}[dict] {k}: {type(v).__name__}", end="")
                if hasattr(v, "shape"):
                    print(f" shape={v.shape}", end="")
                print()
                show_keys(v, depth + 1, max_depth)
        elif hasattr(obj, "_fieldnames"):
            for field in obj._fieldnames:
                val = getattr(obj, field)
                print(f"{indent}[struct] {field}: {type(val).__name__}", end="")
                if hasattr(val, "shape"):
                    print(f" shape={val.shape}", end="")
                elif isinstance(val, (list, np.ndarray)):
                    print(f" len={len(val)}", end="")
                print()
                show_keys(val, depth + 1, max_depth)
        elif isinstance(obj, np.ndarray) and obj.dtype == object:
            if len(obj) > 0:
                show_keys(obj.flat[0], depth + 1, max_depth)

    show_keys(mat_data)


def extract_nasa_cycles(mat_data: dict, battery_id: str) -> List[Dict]:
    """
    Extract per-cycle features from the NASA battery .mat structure.
    
    NASA structure (typical):
        B0005 - struct with field 'cycle'
        cycle - array of structs, each with:
            .type: 'charge' | 'discharge' | 'impedance'
            .ambient_temperature: float
            .time: datetime struct
            .data: struct with measurement arrays
                For charge/discharge:
                    .Voltage_measured
                    .Current_measured
                    .Temperature_measured
                    .Current_load
                    .Voltage_load
                    .Time
                For impedance:
                    .Sense_current
                    .Battery_current
                    .Current_ratio
                    .Battery_impedance
                    .Rectified_impedance
                    .Re (resistance)
                    .Rct (charge transfer resistance)
    """
    records = []

    # Find the battery struct
    battery_struct = None
    for key, val in mat_data.items():
        if key.startswith("__"):
            continue
        battery_struct = val
        break

    if battery_struct is None:
        return records

    # Get cycles array
    try:
        cycles = battery_struct.cycle
        if not hasattr(cycles, "__len__"):
            cycles = [cycles]
    except AttributeError:
        return records

    # --- Track initial discharge capacity ---
    initial_capacity = None
    cycle_idx = 0

    # --- Per-cycle impedance accumulator (last available) ---
    last_Re = np.nan
    last_Rct = np.nan

    for cycle in cycles:
        try:
            cycle_type = str(cycle.type).strip().lower()
        except Exception:
            continue

        # --- Impedance cycles: store last Re, Rct ---
        if cycle_type == "impedance":
            try:
                data = cycle.data
                if hasattr(data, "Re"):
                    re_val = np.atleast_1d(data.Re)
                    if len(re_val) > 0 and not np.all(np.isnan(re_val)):
                        last_Re = float(np.nanmean(re_val))
                if hasattr(data, "Rct"):
                    rct_val = np.atleast_1d(data.Rct)
                    if len(rct_val) > 0 and not np.all(np.isnan(rct_val)):
                        last_Rct = float(np.nanmean(rct_val))
            except Exception:
                pass
            continue

        # --- Only process discharge cycles for capacity/features ---
        if cycle_type != "discharge":
            continue

        try:
            data = cycle.data
        except AttributeError:
            continue

        try:
            voltage = np.atleast_1d(data.Voltage_measured).astype(float)
            current = np.atleast_1d(data.Current_measured).astype(float)
            temperature = np.atleast_1d(data.Temperature_measured).astype(float)
            time_arr = np.atleast_1d(data.Time).astype(float)
        except AttributeError:
            continue

        if len(voltage) < 5:
            continue

        # -- Capacity calculation (Coulomb counting) --
        # capacity (Ah) = -|I|dt using trapezoidal rule
        dt = np.diff(time_arr)
        i_mid = 0.5 * (np.abs(current[:-1]) + np.abs(current[1:]))
        capacity_ah = float(np.sum(i_mid * dt)) / 3600.0  # seconds - hours

        if capacity_ah <= 0:
            continue

        # -- Initial capacity reference --
        if initial_capacity is None or initial_capacity <= 0:
            initial_capacity = capacity_ah

        # -- Capacity retention --
        capacity_retention = (capacity_ah / initial_capacity) * 100.0

        # -- Grade label --
        if capacity_retention >= 80.0:
            grade_str = "A"
        elif capacity_retention >= 65.0:
            grade_str = "B"
        elif capacity_retention >= 50.0:
            grade_str = "C"
        else:
            grade_str = "SCRAP"

        # -- Cycle duration --
        cycle_duration = float(time_arr[-1] - time_arr[0]) if len(time_arr) > 1 else np.nan

        # -- Energy (Wh) --
        try:
            energy_wh = float(np.trapezoid(np.abs(voltage * current), time_arr)) / 3600.0
        except AttributeError:
            # NumPy < 2.0 fallback
            energy_wh = float(np.trapz(np.abs(voltage * current), time_arr)) / 3600.0

        record = {
            "battery_id": battery_id,
            "cycle_number": cycle_idx,
            # Voltage features
            "voltage_mean": float(np.mean(voltage)),
            "voltage_std": float(np.std(voltage)),
            "voltage_min": float(np.min(voltage)),
            "voltage_max": float(np.max(voltage)),
            "voltage_range": float(np.max(voltage) - np.min(voltage)),
            # Current features
            "current_mean": float(np.mean(current)),
            "current_std": float(np.std(current)),
            "current_min": float(np.min(current)),
            "current_max": float(np.max(current)),
            "current_range": float(np.max(current) - np.min(current)),
            # Temperature features
            "temperature_mean": float(np.mean(temperature)),
            "temperature_std": float(np.std(temperature)),
            "temperature_min": float(np.min(temperature)),
            "temperature_max": float(np.max(temperature)),
            "temperature_range": float(np.max(temperature) - np.min(temperature)),
            # Temporal
            "cycle_duration": cycle_duration,
            # Energy
            "energy_wh": energy_wh,
            # Capacity (Ah) - used for retention, NOT as feature
            "capacity_ah": capacity_ah,
            # Impedance (from last impedance cycle)
            "Re": last_Re,
            "Rct": last_Rct,
            # Target derivation intermediaries
            "capacity_retention": capacity_retention,
            "grade": grade_str,
            "grade_label": GRADE_LABEL[grade_str],
        }
        records.append(record)
        cycle_idx += 1

    return records


# -- Feature list (excludes leakage columns) -----------------------------------
FEATURE_COLUMNS = [
    "voltage_mean", "voltage_std", "voltage_min", "voltage_max", "voltage_range",
    "current_mean", "current_std", "current_min", "current_max", "current_range",
    "temperature_mean", "temperature_std", "temperature_min", "temperature_max", "temperature_range",
    "cycle_duration",
    "energy_wh",
    "cycle_number",
    "Re",
    "Rct",
]

TARGET_COLUMN = "grade_label"
GROUP_COLUMN = "battery_id"


# -- Main ----------------------------------------------------------------------

def main() -> None:
    mat_files = sorted(RAW_DIR.rglob("*.mat"))
    if not mat_files:
        print("ERROR: No .mat files found. Run download_electrical_data.py first.")
        sys.exit(1)

    print(f"Found {len(mat_files)} .mat files.")

    # -- Inspect first file for schema discovery --
    first_mat = load_mat_file(mat_files[0])
    inspect_mat_structure(first_mat, mat_files[0].stem)

    # -- Extract all cycles from all batteries --
    all_records = []
    for mat_path in mat_files:
        battery_id = mat_path.stem  # e.g., "B0005"
        mat_data = load_mat_file(mat_path)
        if not mat_data:
            continue
        records = extract_nasa_cycles(mat_data, battery_id)
        print(f"  {battery_id}: {len(records)} discharge cycles extracted.")
        all_records.extend(records)

    if not all_records:
        print("ERROR: No records extracted. Check .mat file format.")
        sys.exit(1)

    df = pd.DataFrame(all_records)
    print(f"\nTotal records: {len(df)}")
    print(f"Batteries: {df['battery_id'].nunique()}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nClass distribution:\n{df['grade'].value_counts()}")
    print(f"\nCapacity retention stats:\n{df['capacity_retention'].describe()}")

    # -- Validate no leakage --
    assert "capacity_retention" not in FEATURE_COLUMNS, "LEAKAGE: capacity_retention in features!"
    assert "capacity_ah" not in FEATURE_COLUMNS, "LEAKAGE: capacity_ah in features!"
    assert "grade" not in FEATURE_COLUMNS, "LEAKAGE: grade in features!"

    # -- Drop rows missing features --
    available_features = [f for f in FEATURE_COLUMNS if f in df.columns]
    print(f"\nUsing features: {available_features}")

    df_clean = df.dropna(subset=available_features + [TARGET_COLUMN]).copy()
    print(f"After dropping NaN rows: {len(df_clean)} records.")

    # -- Battery-level GroupShuffleSplit --
    groups = df_clean[GROUP_COLUMN].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df_clean, groups=groups))

    train_df = df_clean.iloc[train_idx].copy()
    test_df = df_clean.iloc[test_idx].copy()

    # Verify no battery overlap
    train_batteries = set(train_df[GROUP_COLUMN].unique())
    test_batteries = set(test_df[GROUP_COLUMN].unique())
    overlap = train_batteries & test_batteries
    assert len(overlap) == 0, f"DATA LEAKAGE: Batteries in both splits: {overlap}"

    print(f"\nTrain: {len(train_df)} records from {len(train_batteries)} batteries")
    print(f"Test:  {len(test_df)} records from {len(test_batteries)} batteries")
    print(f"Train batteries: {sorted(train_batteries)}")
    print(f"Test batteries:  {sorted(test_batteries)}")

    # -- Save --
    # Save full processed dataset
    df_clean.to_csv(PROCESSED_DIR / "electrical_full.csv", index=False)

    # Save feature matrix only (no leakage columns)
    feature_cols_present = available_features + [TARGET_COLUMN, GROUP_COLUMN, "grade"]
    train_df[feature_cols_present].to_csv(PROCESSED_DIR / "electrical_train.csv", index=False)
    test_df[feature_cols_present].to_csv(PROCESSED_DIR / "electrical_test.csv", index=False)

    # Save feature list for reproducibility
    metadata = {
        "feature_columns": available_features,
        "target_column": TARGET_COLUMN,
        "group_column": GROUP_COLUMN,
        "grade_labels": GRADE_LABEL,
        "label_grades": LABEL_GRADE,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_batteries_train": int(len(train_batteries)),
        "n_batteries_test": int(len(test_batteries)),
        "class_distribution_train": train_df["grade"].value_counts().to_dict(),
        "class_distribution_test": test_df["grade"].value_counts().to_dict(),
    }
    with open(PROCESSED_DIR / "electrical_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n- Saved:")
    print(f"  {PROCESSED_DIR / 'electrical_train.csv'}")
    print(f"  {PROCESSED_DIR / 'electrical_test.csv'}")
    print(f"  {PROCESSED_DIR / 'electrical_full.csv'}")
    print(f"  {PROCESSED_DIR / 'electrical_metadata.json'}")


if __name__ == "__main__":
    main()
