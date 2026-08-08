"""
train_electrical.py
=====================
Trains the PunarShakti electrical branch models:
  1. XGBoost XGBClassifier (primary)
  2. RandomForestClassifier (baseline)

Evaluates and compares both on the held-out test set.
Saves the winning model and preprocessor to models/.
Generates classification reports and figures to reports/.

ANTI-LEAKAGE GUARANTEES:
- capacity_retention is never a feature (used only for label creation)
- Train/test split is battery-level (GroupShuffleSplit in prepare step)
- Test batteries were never seen during training
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# -- Paths ---------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# -- Grade mapping -------------------------------------------------------------
GRADE_NAMES = {0: "A", 1: "B", 2: "C", 3: "SCRAP"}
GRADE_LABELS = ["A", "B", "C", "SCRAP"]


# -- Load data -----------------------------------------------------------------

def load_data():
    meta_path = PROCESSED_DIR / "electrical_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError("Run prepare_electrical_data.py first.")

    with open(meta_path) as f:
        meta = json.load(f)

    feature_cols = meta["feature_columns"]
    target_col = meta["target_column"]

    train_df = pd.read_csv(PROCESSED_DIR / "electrical_train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "electrical_test.csv")

    # Ensure only the available features are used
    available = [c for c in feature_cols if c in train_df.columns]
    print(f"Features used ({len(available)}): {available}")

    X_train = train_df[available].values
    y_train = train_df[target_col].values
    X_test = test_df[available].values
    y_test = test_df[target_col].values

    return X_train, y_train, X_test, y_test, available, meta


# -- Evaluation helper ---------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, target_names=GRADE_LABELS, zero_division=0)

    print(f"\n{'-'*60}")
    print(f"Model: {model_name}")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nClassification Report:")
    print(report)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
        "y_pred": y_pred,
    }


# -- Plot helpers --------------------------------------------------------------

def plot_confusion_matrix(y_test, y_pred, title: str, save_path: Path) -> None:
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=GRADE_LABELS,
        yticklabels=GRADE_LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_importance(model, feature_names: list, save_path: Path) -> None:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        # Pipeline
        for step_name, step in model.named_steps.items():
            if hasattr(step, "feature_importances_"):
                importances = step.feature_importances_
                break
        else:
            return
    else:
        return

    idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_imp = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_names)))
    bars = ax.barh(sorted_names[::-1], sorted_imp[::-1], color=colors[::-1])
    ax.set_xlabel("Feature Importance (Gain)", fontsize=12)
    ax.set_title("XGBoost Feature Importance - Electrical Branch", fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="gray", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

    # Return sorted importance for API exposure
    return [
        {"feature": n, "importance": float(i)}
        for n, i in zip(sorted_names, sorted_imp)
    ]


def plot_class_distribution(y_train, y_test, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, y, title in zip(axes, [y_train, y_test], ["Training Set", "Test Set"]):
        counts = [np.sum(y == i) for i in range(4)]
        bars = ax.bar(GRADE_LABELS, counts, color=["#2ecc71", "#3498db", "#f39c12", "#e74c3c"])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Count")
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(c), ha="center", va="bottom", fontsize=10)
    plt.suptitle("Battery Grade Class Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# -- Models --------------------------------------------------------------------

def build_xgboost_pipeline() -> Pipeline:
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", xgb),
    ])
    return pipeline


def build_rf_pipeline() -> Pipeline:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", rf),
    ])
    return pipeline


# -- Main ----------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("PunarShakti AI - Electrical Branch Training")
    print("=" * 60)

    X_train, y_train, X_test, y_test, feature_names, meta = load_data()

    print(f"\nDataset summary:")
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]} samples")

    # -- Plot class distribution --
    plot_class_distribution(
        y_train, y_test,
        REPORTS_DIR / "electrical_class_distribution.png"
    )

    results = []

    # -- Train XGBoost --
    print("\n[1/2] Training XGBoost...")
    xgb_pipe = build_xgboost_pipeline()
    t0 = time.time()
    xgb_pipe.fit(X_train, y_train)
    xgb_time = time.time() - t0
    print(f"  XGBoost training time: {xgb_time:.1f}s")

    xgb_result = evaluate_model(xgb_pipe, X_test, y_test, "XGBoost")
    xgb_result["training_time"] = xgb_time
    results.append(xgb_result)

    # -- Train Random Forest --
    print("\n[2/2] Training Random Forest baseline...")
    rf_pipe = build_rf_pipeline()
    t0 = time.time()
    rf_pipe.fit(X_train, y_train)
    rf_time = time.time() - t0
    print(f"  RF training time: {rf_time:.1f}s")

    rf_result = evaluate_model(rf_pipe, X_test, y_test, "RandomForest")
    rf_result["training_time"] = rf_time
    results.append(rf_result)

    # -- Model comparison --
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (by Macro F1)")
    print("=" * 60)
    comparison = sorted(results, key=lambda r: r["macro_f1"], reverse=True)
    for r in comparison:
        print(f"  {r['model_name']:20s} | Acc: {r['accuracy']:.4f} | Macro F1: {r['macro_f1']:.4f} | "
              f"Weighted F1: {r['weighted_f1']:.4f} | Time: {r['training_time']:.1f}s")

    winner = comparison[0]
    print(f"\n- Winner: {winner['model_name']} (Macro F1: {winner['macro_f1']:.4f})")

    # Select the winning pipeline
    if winner["model_name"] == "XGBoost":
        winning_pipe = xgb_pipe
        model_clf = xgb_pipe.named_steps["clf"]
    else:
        winning_pipe = rf_pipe
        model_clf = rf_pipe.named_steps["clf"]

    # -- Save models --
    print("\nSaving models...")
    joblib.dump(winning_pipe, MODELS_DIR / "electrical_model.joblib")
    print(f"  Saved: {MODELS_DIR / 'electrical_model.joblib'}")

    # Save preprocessor separately (imputer + scaler)
    preprocessor = Pipeline([
        ("imputer", winning_pipe.named_steps["imputer"]),
        ("scaler", winning_pipe.named_steps["scaler"]),
    ])
    joblib.dump(preprocessor, MODELS_DIR / "electrical_preprocessor.joblib")
    print(f"  Saved: {MODELS_DIR / 'electrical_preprocessor.joblib'}")

    # Save feature names for the API
    feature_meta = {
        "feature_names": feature_names,
        "grade_labels": ["A", "B", "C", "SCRAP"],
        "winning_model": winner["model_name"],
        "macro_f1": winner["macro_f1"],
        "accuracy": winner["accuracy"],
    }
    with open(MODELS_DIR / "electrical_feature_meta.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    # -- Reports --
    print("\nGenerating reports...")

    # XGBoost confusion matrix
    plot_confusion_matrix(
        y_test, xgb_result["y_pred"],
        "XGBoost - Confusion Matrix (Electrical Branch)",
        REPORTS_DIR / "electrical_confusion_matrix.png",
    )

    # Feature importance (XGBoost only)
    importance_list = plot_feature_importance(
        model_clf, feature_names,
        REPORTS_DIR / "electrical_feature_importance.png",
    )

    if importance_list:
        with open(MODELS_DIR / "electrical_feature_importance.json", "w") as f:
            json.dump(importance_list, f, indent=2)

    # Classification report text
    report_path = REPORTS_DIR / "electrical_classification_report.txt"
    with open(report_path, "w") as f:
        f.write("PunarShakti AI - Electrical Branch Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        for r in comparison:
            f.write(f"Model: {r['model_name']}\n")
            f.write(f"Training Time: {r['training_time']:.1f}s\n")
            f.write(f"Accuracy:    {r['accuracy']:.4f}\n")
            f.write(f"Macro F1:    {r['macro_f1']:.4f}\n")
            f.write(f"Weighted F1: {r['weighted_f1']:.4f}\n\n")
            f.write(r["report"])
            f.write("\n" + "-" * 60 + "\n\n")
        f.write(f"\nWinner: {winner['model_name']}\n")
        f.write(f"Features used: {feature_names}\n")
        f.write(f"\nNOTE: capacity_retention was used only to generate labels\n")
        f.write(f"and was EXCLUDED from all input features to prevent leakage.\n")
    print(f"  Saved: {report_path}")

    print("\n- Electrical branch training complete.")


if __name__ == "__main__":
    main()
