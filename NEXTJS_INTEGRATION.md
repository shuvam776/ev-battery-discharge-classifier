# PunarShakti AI API — Next.js Integration Guide

This guide explains how to connect your Next.js application to the deployed **PunarShakti AI API** backend.

---

## 📍 API Base URL

- **Production (Render)**: `https://<your-render-app-name>.onrender.com`
- **Local Development**: `http://localhost:8000`

Add this to your Next.js `.env.local`:

```env
NEXT_PUBLIC_PUNARSHAKTI_API_URL=https://<your-render-app-name>.onrender.com
```

---

## 🛠 Available API Endpoints Summary

All routes predict directly into battery health grades: **`GRADE A`**, **`GRADE B`**, **`GRADE C`**, or **`SCRAP`**.

| Endpoint | Method | Input Type | Description |
|---|---|---|---|
| `/analyze/electrical` | `POST` | `JSON` | Independent Electrical Telemetry Analyzer |
| `/analyze/thermal` | `POST` | `multipart/form-data` | Independent Thermal Scan Analyzer |
| `/analyze/xray` | `POST` | `multipart/form-data` | Independent Structural X-Ray Scan Analyzer |
| `/analyze` | `POST` | `multipart/form-data` | Combined Multimodal Safety Fusion Analyzer |
| `/health` | `GET` | None | API Status & Model Loading Verification |

---

## 1️⃣ Independent Electrical Telemetry Route (`POST /analyze/electrical`)

Use this endpoint when you want to grade a battery based purely on electrical measurements.

### Next.js API Call Example

```typescript
// app/api/grade-electrical/route.ts OR client component
const API_URL = process.env.NEXT_PUBLIC_PUNARSHAKTI_API_URL || 'http://localhost:8000';

export async function analyzeElectricalTelemetry(telemetryData: {
  voltage_mean: number;
  voltage_std?: number;
  voltage_min?: number;
  voltage_max?: number;
  current_mean?: number;
  temperature_mean?: number;
  cycle_number?: number;
  energy_wh?: number;
  Re?: number;
  Rct?: number;
}) {
  const response = await fetch(`${API_URL}/analyze/electrical`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(telemetryData),
  });

  if (!response.ok) {
    throw new Error(`API Error: ${response.statusText}`);
  }

  const data = await response.json();
  return data;
}
```

### JSON Response Payload
```json
{
  "electrical": {
    "grade": "C",
    "confidence": 0.3377,
    "probabilities": {
      "A": 0.3019,
      "B": 0.1982,
      "C": 0.3377,
      "SCRAP": 0.1622
    }
  },
  "model_version": "1.0.0"
}
```

---

## 2️⃣ Independent Thermal Scan Route (`POST /analyze/thermal`)

Use this endpoint when uploading a thermal camera image (`PNG`/`JPG`).

### Next.js API Call Example

```typescript
export async function analyzeThermalScan(file: File) {
  const API_URL = process.env.NEXT_PUBLIC_PUNARSHAKTI_API_URL || 'http://localhost:8000';
  
  const formData = new FormData();
  formData.append('image', file);

  const response = await fetch(`${API_URL}/analyze/thermal`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`API Error: ${response.statusText}`);
  }

  const data = await response.json();
  return data;
}
```

### JSON Response Payload
```json
{
  "thermal": {
    "available": true,
    "grade": "C",
    "confidence": 0.8343,
    "probabilities": {
      "A": 0.0739,
      "B": 0.0649,
      "C": 0.8343,
      "SCRAP": 0.0269
    },
    "is_synthetic_model": true
  },
  "model_version": "1.0.0"
}
```

---

## 3️⃣ Independent Structural X-Ray Route (`POST /analyze/xray`)

Use this endpoint when uploading an internal structural X-ray scan.

### Next.js API Call Example

```typescript
export async function analyzeXrayScan(file: File) {
  const API_URL = process.env.NEXT_PUBLIC_PUNARSHAKTI_API_URL || 'http://localhost:8000';

  const formData = new FormData();
  formData.append('image', file);

  const response = await fetch(`${API_URL}/analyze/xray`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`API Error: ${response.statusText}`);
  }

  const data = await response.json();
  return data;
}
```

### JSON Response Payload
```json
{
  "xray": {
    "available": true,
    "grade": "C",
    "confidence": 0.9395,
    "probabilities": {
      "A": 0.0002,
      "B": 0.0586,
      "C": 0.9395,
      "SCRAP": 0.0016
    }
  },
  "model_version": "1.0.0"
}
```

---

## 4️⃣ Multimodal Combined Fusion Route (`POST /analyze`)

Combines telemetry JSON, optional thermal image, and optional X-ray scan to compute the final safety gate decision (`PASS`, `PASS_WITH_CAUTION`, `HOLD_FOR_MANUAL_INSPECTION`).

### Next.js API Call Example

```typescript
export async function analyzeMultimodal({
  telemetry,
  thermalFile,
  xrayFile,
}: {
  telemetry: Record<string, number>;
  thermalFile?: File;
  xrayFile?: File;
}) {
  const API_URL = process.env.NEXT_PUBLIC_PUNARSHAKTI_API_URL || 'http://localhost:8000';

  const formData = new FormData();
  formData.append('battery_data', JSON.stringify(telemetry));

  if (thermalFile) formData.append('thermal_image', thermalFile);
  if (xrayFile) formData.append('xray_image', xrayFile);

  const response = await fetch(`${API_URL}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`API Error: ${response.statusText}`);
  }

  const data = await response.json();
  return data;
}
```

### JSON Response Payload
```json
{
  "electrical": {
    "grade": "A",
    "confidence": 0.3052,
    "probabilities": { "A": 0.3052, "B": 0.2023, "C": 0.2691, "SCRAP": 0.2234 }
  },
  "thermal": {
    "available": true,
    "grade": "C",
    "confidence": 0.9526,
    "probabilities": { "A": 0.0065, "B": 0.0319, "C": 0.9526, "SCRAP": 0.0091 }
  },
  "xray": {
    "available": false,
    "grade": null,
    "confidence": null,
    "probabilities": null
  },
  "final": {
    "grade": "A",
    "decision": "PASS_WITH_CAUTION",
    "recommended_application": "Solar / Telecom backup storage",
    "disclaimer": "PRELIMINARY AI ASSESSMENT. Final certification requires PunarShakti six-stage physical testing workflow."
  },
  "model_version": "1.0.0"
}
```

---

## 🔍 React/Next.js Custom Hook Pattern

```typescript
import { useState } from 'react';

export function useBatteryAnalyzer() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const API_URL = process.env.NEXT_PUBLIC_PUNARSHAKTI_API_URL || 'http://localhost:8000';

  const analyzeElectrical = async (telemetry: Record<string, number>) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/analyze/electrical`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(telemetry),
      });
      const data = await res.json();
      return data.electrical;
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return { analyzeElectrical, loading, error };
}
```
