"""
Automated Test Suite for Dual-Domain Inspection (Glass Bottle & Plastic Bottle).
Verifies the 7 mandatory scenarios from Requirement 22:
  Test 1: Normal plastic-bottle test image -> Expected: GOOD
  Test 2: Defective plastic-bottle test image -> Expected: DEFECTIVE
  Test 3: Slightly drifted external plastic bottle -> Expected: LOW_CONFIDENCE / HUMAN REVIEW
  Test 4: Strongly different external image -> Expected: OUTSIDE_DOMAIN
  Test 5: Non-bottle image (Random Noise / Abstract) -> Expected: OUTSIDE_DOMAIN
  Test 6: Glass bottle uploaded in Plastic Bottle mode -> Expected: OUTSIDE VALIDATED PLASTIC DOMAIN
  Test 7: Plastic bottle uploaded in Glass Bottle mode -> Expected: OUTSIDE VALIDATED GLASS DOMAIN
"""

import os
import requests
import numpy as np
import cv2
from PIL import Image

API = "http://localhost:8000"

def run_tests():
    print("=================================================================")
    print("DUAL-DOMAIN INSPECTION & CROSS-DOMAIN SAFETY TEST SUITE")
    print("=================================================================")

    # Healthcheck
    h = requests.get(f"{API}/health").json()
    print("1. API Healthcheck:")
    print("   Glass Bottle Coreset:", h["domains"]["glass_bottle"]["coreset_size"], "| Threshold:", h["domains"]["glass_bottle"]["threshold"])
    print("   Plastic Bottle Coreset:", h["domains"]["plastic_bottle"]["coreset_size"], "| Threshold:", h["domains"]["plastic_bottle"]["threshold"])
    print("-----------------------------------------------------------------")

    # TEST 1: Normal plastic-bottle test image
    p_good = os.path.join("plastic_bottle_dataset", "test", "good", "000.png")
    with open(p_good, "rb") as f:
        r1 = requests.post(f"{API}/plastic/predict", files={"file": ("good_plastic.png", f, "image/png")}).json()
    print("TEST 1: Normal Plastic Bottle:")
    print(f"   Verdict: {r1['verdict']} | Grade: {r1['grade']}")
    print(f"   Compatibility: {r1['input_compatibility_status']} ({r1['input_compatibility_score']}%)")
    assert "GOOD" in r1["verdict"] and r1["input_compatibility_status"] == "COMPATIBLE", "Test 1 failed!"
    print("   -> PASS")
    print("-----------------------------------------------------------------")

    # TEST 2: Defective plastic-bottle test image
    p_dent = os.path.join("plastic_bottle_dataset", "test", "dent", "000.png")
    with open(p_dent, "rb") as f:
        r2 = requests.post(f"{API}/plastic/predict", files={"file": ("dent_plastic.png", f, "image/png")}).json()
    print("TEST 2: Defective Plastic Bottle (Dent):")
    print(f"   Verdict: {r2['verdict']} | Grade: {r2['grade']}")
    print(f"   Zone: {r2['defect_zone']} | Score: {r2['anomaly_score']} (Thresh: {r2['threshold']})")
    assert r2["verdict"] == "DEFECTIVE" and r2["input_compatibility_status"] == "COMPATIBLE", "Test 2 failed!"
    print("   -> PASS")
    print("-----------------------------------------------------------------")

    # TEST 3a: Slightly different but compatible external plastic-bottle image (Mult 0.72)
    img_pil = Image.open(p_good).convert('RGB')
    compat_np = np.clip(np.array(img_pil).astype(np.float32) * 0.72, 0, 255).astype(np.uint8)
    _, compat_buf = cv2.imencode('.png', cv2.cvtColor(compat_np, cv2.COLOR_RGB2BGR))
    r3a = requests.post(f"{API}/plastic/predict", files={"file": ("compat_external.png", compat_buf.tobytes(), "image/png")}).json()
    print("TEST 3a: Compatible External Plastic Bottle (Mild Lighting Shift):")
    print(f"   Verdict: {r3a['verdict']} | Grade: {r3a['grade']}")
    print(f"   Compatibility: {r3a['input_compatibility_status']} ({r3a['input_compatibility_score']}%)")
    assert r3a["input_compatibility_status"] == "COMPATIBLE" and "GOOD" in r3a["verdict"], "Test 3a failed!"
    print("   -> PASS (Expected: GOOD / COMPATIBLE)")

    # TEST 3b: Moderate external shift (Mult 0.48 -> LOW_CONFIDENCE)
    drifted_np = np.clip(np.array(img_pil).astype(np.float32) * 0.48, 0, 255).astype(np.uint8)
    _, drifted_buf = cv2.imencode('.png', cv2.cvtColor(drifted_np, cv2.COLOR_RGB2BGR))
    r3b = requests.post(f"{API}/plastic/predict", files={"file": ("drifted_plastic.png", drifted_buf.tobytes(), "image/png")}).json()
    print("TEST 3b: Drifted Plastic Bottle (Simulated External Illumination Shift):")
    print(f"   Verdict: {r3b['verdict']} | Grade: {r3b['grade']}")
    print(f"   Compatibility: {r3b['input_compatibility_status']} ({r3b['input_compatibility_score']}%)")
    print(f"   Human Review Required: {r3b['human_review_required']}")
    assert r3b["input_compatibility_status"] == "LOW_CONFIDENCE" and r3b["human_review_required"] is True, "Test 3b failed!"
    print("   -> PASS (Expected: LOW_CONFIDENCE / HUMAN REVIEW)")
    print("-----------------------------------------------------------------")

    # TEST 4: Strongly different external image (Google Side-view Bottle / Desk photo)
    desk_img = np.full((256, 256, 3), 90, dtype=np.uint8)
    cv2.putText(desk_img, "Web Desk Photo", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
    _, desk_buf = cv2.imencode('.png', desk_img)
    r4 = requests.post(f"{API}/plastic/predict", files={"file": ("google_external.png", desk_buf.tobytes(), "image/png")}).json()
    print("TEST 4: Strongly Different External Image:")
    print(f"   Verdict: {r4['verdict']} | Grade: {r4['grade']}")
    print(f"   Compatibility: {r4['input_compatibility_status']} ({r4['input_compatibility_score']}%)")
    print(f"   Inspection Mode: {r4['inspection_mode']}")
    assert r4["input_compatibility_status"] == "OUTSIDE_DOMAIN", "Test 4 failed!"
    print("   -> PASS")
    print("-----------------------------------------------------------------")

    # TEST 5: Non-bottle image
    noise_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    _, noise_buf = cv2.imencode('.png', noise_img)
    r5 = requests.post(f"{API}/plastic/predict", files={"file": ("noise.png", noise_buf.tobytes(), "image/png")}).json()
    print("TEST 5: Non-Bottle Image (Random Noise Proxy):")
    print(f"   Verdict: {r5['verdict']} | Grade: {r5['grade']}")
    print(f"   Compatibility: {r5['input_compatibility_status']} ({r5['input_compatibility_score']}%)")
    assert r5["input_compatibility_status"] == "OUTSIDE_DOMAIN", "Test 5 failed!"
    print("   -> PASS")
    print("-----------------------------------------------------------------")

    # TEST 6: Glass bottle uploaded in Plastic Bottle mode
    g_img = os.path.join("mvtec_bottle_dataset", "bottle", "test", "good", "000.png")
    with open(g_img, "rb") as f:
        r6 = requests.post(f"{API}/plastic/predict", files={"file": ("glass_topview_in_plastic.png", f, "image/png")}).json()
    print("TEST 6: Glass Bottle Uploaded in Plastic Bottle Mode (Cross-Domain Protection):")
    print(f"   Verdict: {r6['verdict']}")
    print(f"   Compatibility: {r6['input_compatibility_status']} ({r6['input_compatibility_score']}%)")
    print(f"   Warning: {r6['domain_warning']}")
    assert r6["input_compatibility_status"] == "OUTSIDE_DOMAIN", "Test 6 failed: Glass bottle was not rejected in Plastic mode!"
    print("   -> PASS (Successfully detected as OUTSIDE_DOMAIN in Plastic Mode!)")
    print("-----------------------------------------------------------------")

    # TEST 7: Plastic bottle uploaded in Glass Bottle mode
    with open(p_good, "rb") as f:
        r7 = requests.post(f"{API}/predict", files={"file": ("plastic_frontview_in_glass.png", f, "image/png")}).json()
    print("TEST 7: Plastic Bottle Uploaded in Glass Bottle Mode (Cross-Domain Protection):")
    print(f"   Verdict: {r7['verdict']}")
    print(f"   Compatibility: {r7['input_compatibility_status']} ({r7['input_compatibility_score']}%)")
    print(f"   Warning: {r7['domain_warning']}")
    assert r7["input_compatibility_status"] == "OUTSIDE_DOMAIN", "Test 7 failed: Plastic bottle was not rejected in Glass mode!"
    print("   -> PASS (Successfully detected as OUTSIDE_DOMAIN in Glass Mode!)")
    print("=================================================================")
    print("ALL 7 MANDATORY TEST SCENARIOS PASSED WITH 100% SUCCESS!")
    print("=================================================================")

if __name__ == "__main__":
    run_tests()
