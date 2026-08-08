# PunarShakti AI — Multimodal EV Battery Grading System

<div align="center">

```
██████╗ ██╗   ██╗███╗   ██╗ █████╗ ██████╗ ███████╗██╗  ██╗ █████╗ ██╗  ██╗████████╗██╗
██╔══██╗██║   ██║████╗  ██║██╔══██╗██╔══██╗██╔════╝██║  ██║██╔══██╗██║ ██╔╝╚══██╔══╝██║
██████╔╝██║   ██║██╔██╗ ██║███████║██████╔╝███████╗███████║███████║█████╔╝    ██║   ██║
██╔═══╝ ██║   ██║██║╚██╗██║██╔══██║██╔══██╗╚════██║██╔══██║██╔══██║██╔═██╗   ██║   ██║
██║     ╚██████╔╝██║ ╚████║██║  ██║██║  ██║███████║██║  ██║██║  ██║██║  ██╗  ██║   ██║
╚═╝      ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═╝   ╚═╝
```

**Preliminary AI Assessment Engine for Second-Life EV Battery Certification**

*Electrical Degradation · Thermal Safety · Conservative Decision Engine*

</div>

---

> ⚠️ **IMPORTANT**: This is a **PRELIMINARY AI ASSESSMENT SYSTEM**. It does not replace
> the PunarShakti six-stage physical battery certification workflow. Final certification
> always requires physical testing by a qualified engineer.

---

## Table of Contents

1. [Business Problem](#1-business-problem)
2. [ML Architecture](#2-ml-architecture)
3. [Dataset Sources](#3-dataset-sources)
4. [Dataset Limitations](#4-dataset-limitations)
5. [Electrical Model](#5-electrical-model)
6. [Thermal Model](#6-thermal-model)
7. [X-ray Model](#7-x-ray-model)
8. [Fusion Strategy](#8-fusion-strategy)
9. [Safety Decision Engine](#9-safety-decision-engine)
10. [Training Instructions](#10-training-instructions)
11. [Evaluation](#11-evaluation)
12. [API Documentation](#12-api-documentation)
13. [Next.js Integration](#13-nextjs-integration)
14. [Deployment](#14-deployment)
15. [Limitations](#15-limitations)
16. [Production Roadmap](#16-production-roadmap)

---

## 1. Business Problem

PunarShakti Energy builds trusted second-life EV battery certification and resale
infrastructure. Before a used EV battery can be resold or repurposed, it must be
evaluated for:

- **Economic health** — how much capacity remains, and for which use case
- **Safety** — is the battery thermally stable and structurally intact?

The PunarShakti six-stage physical certification workflow covers:
1. Visual & Mechanical Inspection
2. BMS & Usage History
3. Insulation & Leakage
4. Capacity Test
5. Impedance Spectroscopy
6. Thermal Ramp & Imaging

This ML system provides a **preliminary grading and safety assessment** before
physical certification — speeding up the triage process and flagging high-risk
batteries for priority inspection.

**Battery Grade System:**

| Grade | Capacity Retained | Recommended Application |
|-------|---|---|
| **A** | ≥ 80% | Solar / Telecom backup storage |
| **B** | 65–79% | Home & commercial backup storage |
| **C** | 50–64% | Low-cycle stationary / street lighting |
| **SCRAP** | < 50% | Material recovery & recycling |

---

## 2. ML Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PUNARSHAKTI AI SYSTEM                        │
├──────────────────┬──────────────────┬───────────────────────────┤
│  BRANCH 1        │  BRANCH 2        │  BRANCH 3                 │
│  ELECTRICAL      │  THERMAL         │  X-RAY (V2)               │
│                  │                  │                           │
│  Battery         │  Thermal         │  X-ray                    │
│  Telemetry       │  Image           │  Image                    │
│      ↓           │      ↓           │      ↓                    │
│  Feature         │  ResNet-18       │  ResNet-18                │
│  Engineering     │  (ImageNet       │  (ImageNet                │
│      ↓           │   pretrained)    │   pretrained)             │
│  XGBoost         │      ↓           │      ↓                    │
│      ↓           │  Thermal Risk    │  Structural Risk          │
│  A/B/C/SCRAP     │  LOW/MED/HIGH    │  NORMAL/SUSPICIOUS        │
│  probabilities   │  probabilities   │  probabilities            │
└────────┬─────────┴────────┬─────────┴──────────┬────────────────┘
         │                  │                     │
         └──────────────────┼─────────────────────┘
                            ↓
              ┌─────────────────────────┐
              │  LATE FUSION            │
              │  SAFETY GATE            │
              │                         │
              │  HIGH thermal → HOLD    │
              │  SUSPICIOUS xray → HOLD │
              │  else → PASS            │
              └─────────────┬───────────┘
                            ↓
              ┌─────────────────────────┐
              │  FINAL RESPONSE         │
              │  Grade: B               │
              │  Decision: PASS         │
              │  Application: Home...   │
              └─────────────────────────┘
```

---

## 3. Dataset Sources

### Electrical Branch
- **Dataset**: NASA Prognostics Center of Excellence (PCoE) Li-ion Battery Aging Dataset
- **URL**: https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
- **Batteries**: B0005–B0056 (18650 cylindrical Li-ion cells)
- **Measurements per cycle**: Voltage, Current, Temperature, Time, Impedance (Re, Rct)
- **Labeling**: Capacity retention computed via Coulomb counting (∫|I|dt)
- **License**: NASA open data

### Thermal Branch
- **Dataset**: Synthetic thermal images (V1)
- **Generation method**: Physics-based pattern simulation (uniform/localized/hotspot)
- **Classes**: LOW / MEDIUM / HIGH risk
- **Future**: Replace with PunarShakti proprietary synchronized thermal imaging dataset

### X-ray Branch
- **Status**: Architecture implemented, model not trained in V1
- **Reason**: No freely available, reliably labeled EV battery X-ray dataset exists
- **Future**: PunarShakti proprietary battery X-ray dataset

---

## 4. Dataset Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| NASA dataset uses 18650 cells, not EV packs | Feature distributions may differ | Documented; production uses real EV data |
| Thermal dataset is synthetic | Model may not generalize to real images | Clearly flagged in all API responses |
| No X-ray training data | Branch unavailable | Architecture ready; pluggable |
| Datasets are from different experiments | Not a synchronized multimodal dataset | Late fusion architecture mitigates coupling |

**The multimodal system is an engineering prototype.** Production performance requires
PunarShakti's own synchronized dataset.

---

## 5. Electrical Model

### Feature Engineering
Features are extracted per discharge cycle from `.mat` files:

| Feature | Description |
|---|---|
| `voltage_mean/std/min/max/range` | Voltage distribution across discharge |
| `current_mean/std/min/max/range` | Current distribution |
| `temperature_mean/std/min/max/range` | Temperature across cycle |
| `cycle_duration` | Total cycle time (seconds) |
| `cycle_number` | Aging proxy (monotonically increasing) |
| `energy_wh` | Energy delivered (Wh, via numerical integration) |
| `Re` | Electrolyte resistance (from impedance cycles) |
| `Rct` | Charge transfer resistance |

**Anti-leakage guarantee:**
- `capacity_retention` is computed but NEVER used as a feature
- `capacity_ah` is NEVER used as a feature
- Only used to generate the A/B/C/SCRAP label

### Train/Test Split
```python
GroupShuffleSplit(test_size=0.2, random_state=42)
groups = battery_id  # e.g. "B0005", "B0007"
```
The same physical battery NEVER appears in both train and test sets.

### Model
- **Primary**: `XGBClassifier(objective="multi:softprob", num_class=4)`
- **Baseline**: `RandomForestClassifier(class_weight="balanced")`
- **Winner selected by**: Macro F1 score

---

## 6. Thermal Model

### Architecture
```
Input: 128×128 RGB image
    ↓
ImageNet pretrained ResNet-18 (frozen backbone)
    ↓
Stage 1: Train classification head (FC layers)
    ↓
Stage 2: Unfreeze layer4, fine-tune with lr=1e-4
    ↓
Output: LOW / MEDIUM / HIGH risk probabilities
```

### Augmentation (thermal-appropriate only)
- Random horizontal flip
- Random vertical flip
- Random rotation ±15°
- Mild brightness/contrast (sensor calibration variation)
- Gaussian noise (sensor noise)
- **NOT used**: aggressive color jitter, heavy crops, distortions

### Explainability
Grad-CAM available via `POST /explain/thermal` — shows which image
regions influenced the risk prediction.

---

## 7. X-ray Model

**Status: ARCHITECTURE READY — NOT TRAINED IN V1**

The full inference pipeline is implemented. To enable:

```
data/raw/xray/
    train/
        NORMAL/       ← structurally normal battery images
        SUSPICIOUS/   ← defect/anomaly images
    val/
        NORMAL/
        SUSPICIOUS/
```

Then: `python scripts/train_xray.py`

The API automatically detects and activates the branch when `models/xray_model.pt` exists.

---

## 8. Fusion Strategy

**Late Fusion** — each branch runs independently, outputs combined at decision time.

This approach was chosen over early/intermediate fusion because:
- Branches use different datasets (not synchronized)
- Interpretable: each branch's output is independently auditable
- Robust: one branch failure doesn't corrupt others
- Hackathon-appropriate: fast, debuggable, explainable

---

## 9. Safety Decision Engine

```python
def apply_safety_gate(electrical_grade, thermal_risk, xray_risk):
    # Rule 1: HIGH thermal risk = absolute hold
    if thermal_risk == "HIGH":
        return "HOLD_FOR_MANUAL_INSPECTION"
    
    # Rule 2: Suspicious X-ray structural risk = hold
    if xray_risk == "SUSPICIOUS":
        return "HOLD_FOR_MANUAL_INSPECTION"
    
    # Rule 3: Medium thermal with top electrical grade
    if thermal_risk == "MEDIUM" and electrical_grade == "A":
        return "PASS_WITH_CAUTION"
    
    # Rule 4: All clear
    return "PASS"
```

**Key design principle**: The **electrical grade is preserved** even when the
decision is HOLD. A battery graded A economically but with HIGH thermal risk
returns `grade: A, decision: HOLD_FOR_MANUAL_INSPECTION` — the economic
assessment is preserved while the safety flag is raised.

---

## 10. Training Instructions

### Prerequisites
```bash
pip install -r requirements.txt
# For PyTorch (CPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 1: Download electrical data
```bash
python scripts/download_electrical_data.py
```

### Step 2: Prepare electrical data
```bash
python scripts/prepare_electrical_data.py
```
This inspects the actual `.mat` schema, extracts features, computes labels,
and creates a grouped battery-level train/test split.

### Step 3: Train electrical model
```bash
python scripts/train_electrical.py
```
Trains XGBoost + RF baseline, selects winner by Macro F1, saves to `models/`.

### Step 4: Prepare thermal data
```bash
python scripts/prepare_thermal_data.py
```
Generates synthetic thermal images (300 per class).

### Step 5: Train thermal model
```bash
python scripts/train_thermal.py
```
ResNet-18 transfer learning, ~20 epochs total.

### Step 6: X-ray (optional)
```bash
python scripts/train_xray.py
# Will skip gracefully if no dataset is available
```

### Step 7: Start the API
```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 11. Evaluation

Reports are saved to `reports/` after training:

| File | Content |
|---|---|
| `electrical_classification_report.txt` | Accuracy, F1, per-class metrics |
| `electrical_confusion_matrix.png` | Confusion matrix heatmap |
| `electrical_feature_importance.png` | XGBoost feature importance bar chart |
| `electrical_class_distribution.png` | Train/test class balance |
| `thermal_classification_report.txt` | Thermal branch metrics |
| `thermal_confusion_matrix.png` | Thermal confusion matrix |
| `thermal_training_curves.png` | Loss/accuracy curves |

---

## 12. API Documentation

Interactive docs: http://localhost:8000/docs

### GET /health
```json
{
  "status": "ok",
  "electrical_model_loaded": true,
  "thermal_model_loaded": true,
  "xray_model_loaded": false,
  "version": "1.0.0"
}
```

### POST /analyze
Multipart form:
- `battery_data` (required): JSON string of telemetry
- `thermal_image` (optional): PNG/JPG thermal image
- `xray_image` (optional): PNG/JPG X-ray image

```json
{
  "electrical": {
    "grade": "B",
    "confidence": 0.94,
    "probabilities": {"A": 0.03, "B": 0.94, "C": 0.02, "SCRAP": 0.01}
  },
  "thermal": {
    "available": true,
    "risk": "LOW",
    "confidence": 0.91
  },
  "xray": {"available": false, "risk": null},
  "final": {
    "grade": "B",
    "decision": "PASS",
    "recommended_application": "Home & commercial backup storage",
    "disclaimer": "PRELIMINARY AI ASSESSMENT..."
  },
  "model_version": "1.0.0"
}
```

### POST /analyze/csv
Upload a CSV file. Returns per-row grades + summary counts.

### GET /model/features
Returns XGBoost feature importance scores.

### POST /explain/thermal
Returns Grad-CAM heatmap PNG for the uploaded thermal image.

---

## 13. Next.js Integration

```typescript
// Example: Send battery data + thermal image
const formData = new FormData();
formData.append('battery_data', JSON.stringify({
  voltage_mean: 3.62,
  cycle_number: 620,
  temperature_mean: 29.4,
}));
if (thermalFile) {
  formData.append('thermal_image', thermalFile);
}

const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/analyze`, {
  method: 'POST',
  body: formData,
});
const result = await response.json();
// result.final.grade → "B"
// result.final.decision → "PASS"
// result.thermal.risk → "LOW"
```

**Environment variable:**
```env
NEXT_PUBLIC_API_URL=https://your-punarshakti-api.render.com
```

---

## 14. Deployment

### Local
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t punarshakti-ai .
docker run -p 8000:8000 -v ./models:/app/models punarshakti-ai
```

### Docker Compose
```bash
docker-compose up --build
```

### Render / Railway
1. Push to GitHub
2. Connect to Render/Railway
3. Set environment variables:
   - `FRONTEND_URL=https://your-nextjs-app.vercel.app`
   - `MODEL_VERSION=1.0.0`
4. Deploy — Docker is auto-detected

### Environment Variables
```env
HOST=0.0.0.0
PORT=8000
FRONTEND_URL=https://your-nextjs-app.vercel.app
ELECTRICAL_MODEL_PATH=models/electrical_model.joblib
THERMAL_MODEL_PATH=models/thermal_model.pt
MODEL_VERSION=1.0.0
LOG_LEVEL=INFO
```

---

## 15. Limitations

| Limitation | Detail |
|---|---|
| Synthetic thermal data | ResNet-18 trained on simulated images, not real battery thermals |
| NASA dataset scope | 18650 cells, not EV packs; different operating conditions |
| No X-ray training data | Branch is architecture-only in V1 |
| Single-cycle inference | Model predicts from one cycle's features, not time-series |
| No uncertainty quantification | Confidence = softmax probability, not calibrated |
| Class imbalance | SCRAP class may be underrepresented in NASA dataset |

---

## 16. Production Roadmap

### PunarShakti Data Flywheel

```
Every battery tested by PunarShakti
            ↓
    BMS data + electrical measurements
    + capacity + impedance
    + thermal images
    + physical inspection
    + final certified grade
            ↓
    PunarShakti proprietary dataset
            ↓
    Continuous model improvement
```

### V2 Architecture (Future)

```
BMS time series → Temporal Transformer / TCN
Thermal image   → EfficientNet / ConvNeXt
X-ray           → Vision Transformer
Impedance       → MLP
                    ↓
            Multimodal Fusion Transformer
                    ↓
    Health + Safety + Remaining Useful Life (RUL)
                    ↓
        Final Certification Recommendation
```

### V2 Milestones
- [ ] Collect 500+ real battery thermal images with LOW/MEDIUM/HIGH labels
- [ ] Source or create battery X-ray dataset (partner with battery recycler)
- [ ] Replace synthetic thermal with real data; retrain ResNet-18
- [ ] Implement BMS time-series branch (TCN/Transformer)
- [ ] Add Remaining Useful Life (RUL) regression
- [ ] Calibrate confidence scores (Platt scaling / temperature scaling)
- [ ] Implement attention-based multimodal fusion
- [ ] Add online learning as new certified batteries accumulate

---

*Built for PunarShakti Energy — Trusted Second-Life Battery Infrastructure*
#   e v - b a t t e r y - d i s c h a r g e - c l a s s i f i e r  
 