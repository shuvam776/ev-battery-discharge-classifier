"""
test_api.py
=============
Quick smoke-test script for the PunarShakti AI FastAPI backend.

Tests all major endpoints:
  GET  /health
  POST /analyze  (numerical only)
  POST /analyze  (numerical + thermal image)
  GET  /model/features
  POST /analyze/thermal
  POST /analyze/xray  (should return available: false)

Run with: python test_api.py
Requires the server to be running:
  uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
"""

import io
import json
import sys
import time

import numpy as np
import requests
from PIL import Image

BASE_URL = "http://localhost:8000"


def print_result(label: str, response, expected_status: int = 200):
    """Pretty-print an API test result."""
    status_ok = response.status_code == expected_status
    icon = "PASS" if status_ok else "FAIL"
    print(f"\n[{icon}] {label}")
    print(f"  Status: {response.status_code}")
    try:
        data = response.json()
        print(f"  Response: {json.dumps(data, indent=2)[:500]}")
    except Exception:
        print(f"  Response (raw): {response.text[:200]}")
    return status_ok


def make_thermal_image_bytes() -> bytes:
    """Generate a synthetic thermal-like test image."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    # Simulate a hotspot in the center
    y, x = np.ogrid[:128, :128]
    hotspot = np.exp(-((x - 64)**2 + (y - 64)**2) / (2 * 20**2))
    arr[:, :, 0] = (255 * hotspot).astype(np.uint8)  # Red channel = heat
    arr[:, :, 1] = (100 * hotspot * 0.5).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    print("=" * 60)
    print("PunarShakti AI -- API Smoke Tests")
    print("=" * 60)
    print(f"Target: {BASE_URL}")

    # Wait a moment for server
    time.sleep(1)

    results = []

    # ── Test 1: Health ──────────────────────────────────────────────────────
    r = requests.get(f"{BASE_URL}/health")
    ok = print_result("GET /health", r)
    results.append(ok)
    if ok:
        health = r.json()
        print(f"  Models loaded:")
        print(f"    Electrical: {health.get('electrical_model_loaded')}")
        print(f"    Thermal:    {health.get('thermal_model_loaded')}")
        print(f"    X-ray:      {health.get('xray_model_loaded')}")

    # ── Test 2: Root ────────────────────────────────────────────────────────
    r = requests.get(f"{BASE_URL}/")
    ok = print_result("GET / (root)", r)
    results.append(ok)

    # ── Test 3: Analyze (numerical only) ───────────────────────────────────
    battery_data = json.dumps({
        "voltage_mean": 3.62,
        "voltage_std": 0.08,
        "voltage_min": 3.20,
        "voltage_max": 4.18,
        "voltage_range": 0.98,
        "current_mean": -1.5,
        "current_std": 0.3,
        "temperature_mean": 29.4,
        "temperature_std": 1.2,
        "cycle_duration": 4200.0,
        "cycle_number": 100,
        "energy_wh": 1.72,
        "Re": 0.045,
        "Rct": 0.069,
    })
    r = requests.post(
        f"{BASE_URL}/analyze",
        data={"battery_data": battery_data},
    )
    ok = print_result("POST /analyze (numerical only)", r)
    results.append(ok)
    if ok:
        result = r.json()
        print(f"\n  === PunarShakti Battery Assessment ===")
        print(f"  Electrical Grade: {result['electrical']['grade']} ({result['electrical']['confidence']:.0%} confidence)")
        print(f"  Thermal:          {result['thermal']['available']}")
        print(f"  X-ray:            {result['xray']['available']}")
        print(f"  Final Decision:   {result['final']['decision']}")
        print(f"  Application:      {result['final']['recommended_application']}")
        print(f"  Disclaimer:       {result['final']['disclaimer'][:60]}...")

    # ── Test 4: Analyze with thermal image ─────────────────────────────────
    thermal_bytes = make_thermal_image_bytes()
    r = requests.post(
        f"{BASE_URL}/analyze",
        data={"battery_data": battery_data},
        files={"thermal_image": ("thermal.png", thermal_bytes, "image/png")},
    )
    ok = print_result("POST /analyze (numerical + thermal)", r)
    results.append(ok)
    if ok:
        result = r.json()
        print(f"\n  === PunarShakti Battery Assessment (Multimodal) ===")
        print(f"  Electrical Grade: {result['electrical']['grade']} ({result['electrical']['confidence']:.0%})")
        print(f"  Thermal Risk:     {result['thermal'].get('risk')} ({result['thermal'].get('confidence', 0):.0%})")
        print(f"  Final Decision:   {result['final']['decision']}")

    # Test high-risk scenario
    high_risk_data = json.dumps({
        "cycle_number": 800,
        "temperature_mean": 45.0,
        "voltage_mean": 3.1,
        "current_mean": -2.0,
    })
    r = requests.post(
        f"{BASE_URL}/analyze",
        data={"battery_data": high_risk_data},
        files={"thermal_image": ("thermal.png", thermal_bytes, "image/png")},
    )
    ok = print_result("POST /analyze (high-degradation scenario)", r)
    results.append(ok)

    # ── Test 5: Thermal branch only ─────────────────────────────────────────
    r = requests.post(
        f"{BASE_URL}/analyze/thermal",
        files={"image": ("thermal.png", thermal_bytes, "image/png")},
    )
    ok = print_result("POST /analyze/thermal (standalone)", r)
    results.append(ok)

    # ── Test 6: X-ray (should return available: false) ──────────────────────
    r = requests.post(
        f"{BASE_URL}/analyze/xray",
        files={"image": ("xray.png", thermal_bytes, "image/png")},
    )
    ok = print_result("POST /analyze/xray (stub - expects available:false)", r)
    results.append(ok)
    if ok:
        xray = r.json()
        print(f"  X-ray available: {xray['xray']['available']} (expected: False)")

    # ── Test 7: Feature importance ──────────────────────────────────────────
    r = requests.get(f"{BASE_URL}/model/features")
    ok = print_result("GET /model/features", r)
    results.append(ok)
    if ok:
        features = r.json()
        print(f"  Top 5 features:")
        for feat in features["features"][:5]:
            print(f"    {feat['feature']:25s}: {feat['importance']:.4f}")

    # ── Summary ─────────────────────────────────────────────────────────────
    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{'=' * 60}")
    print(f"Tests: {n_pass}/{n_total} PASSED")
    print(f"{'=' * 60}")

    if n_pass == n_total:
        print("All tests passed! PunarShakti AI backend is functional.")
    else:
        print("Some tests failed. Check server logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
