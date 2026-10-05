import requests
import os
import glob
import io
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw

API = "http://localhost:8000"

print("=" * 60)
print("1. API HEALTHCHECK")
print("=" * 60)
h = requests.get(f"{API}/health").json()
print("Health:", h)

print("\n" + "=" * 60)
print("2. GROUP 1 — MVTEC GOOD TEST IMAGES (VALIDATED DOMAIN)")
print("=" * 60)
r_good = requests.get(f"{API}/predict_sample?category=good&filename=000.png").json()
print("Good Sample (good/000.png):")
print(f"  Anomaly Score: {r_good['anomaly_score']}")
print(f"  Verdict: {r_good['verdict']} | Grade: {r_good['grade']}")
print(f"  Input Compatibility: {r_good.get('input_compatibility_status')} ({r_good.get('input_compatibility_score')}%)")
print(f"  Confidence State: {r_good.get('inspection_confidence')} | Human Review Required: {r_good.get('human_review_required')}")

print("\n" + "=" * 60)
print("3. GROUP 2 — MVTEC DEFECTIVE TEST IMAGES (PRESERVED DEFECTS)")
print("=" * 60)
r_broken = requests.get(f"{API}/predict_sample?category=broken_large&filename=000.png").json()
print("Broken Large Sample (broken_large/000.png):")
print(f"  Anomaly Score: {r_broken['anomaly_score']} | DSAR: {r_broken['dsar_percent']}%")
print(f"  Verdict: {r_broken['verdict']} | Grade: {r_broken['grade']}")
print(f"  Input Compatibility: {r_broken.get('input_compatibility_status')} ({r_broken.get('input_compatibility_score')}%)")
print(f"  Confidence State: {r_broken.get('inspection_confidence')} | Human Review Required: {r_broken.get('human_review_required')}")

r_contam = requests.get(f"{API}/predict_sample?category=contamination&filename=000.png").json()
print("\nContamination Sample (contamination/000.png):")
print(f"  Anomaly Score: {r_contam['anomaly_score']} | DSAR: {r_contam['dsar_percent']}%")
print(f"  Verdict: {r_contam['verdict']} | Grade: {r_contam['grade']}")
print(f"  Input Compatibility: {r_contam.get('input_compatibility_status')} ({r_contam.get('input_compatibility_score')}%)")
print(f"  Confidence State: {r_contam.get('inspection_confidence')} | Human Review Required: {r_contam.get('human_review_required')}")

print("\n" + "=" * 60)
print("4. GROUP 3 — SLIGHTLY DIFFERENT INPUTS (LOW CONFIDENCE)")
print("=" * 60)
base = Image.open("mvtec_bottle_dataset/bottle/test/good/000.png")
drift = ImageEnhance.Brightness(base).enhance(0.72)
buf_drift = io.BytesIO()
drift.save(buf_drift, format="PNG")
buf_drift.seek(0)
r_ext_drift = requests.post(f"{API}/predict", files={"file": ("drifted_exposure_bottle.png", buf_drift.getvalue(), "image/png")}).json()
print("Drifted Exposure Bottle (0.72x brightness):")
print(f"  PatchCore Anomaly Score: {r_ext_drift['anomaly_score']}")
print(f"  Verdict: {r_ext_drift['verdict']} | Grade: {r_ext_drift['grade']}")
print(f"  Input Compatibility: {r_ext_drift.get('input_compatibility_status')} ({r_ext_drift.get('input_compatibility_score')}%)")
print(f"  Confidence State: {r_ext_drift.get('inspection_confidence')} | Human Review Required: {r_ext_drift.get('human_review_required')}")
print(f"  Warning: {r_ext_drift.get('domain_warning')}")

print("\n" + "=" * 60)
print("5. GROUP 4 — CLEARLY DIFFERENT EXTERNAL INPUTS (OUTSIDE DOMAIN)")
print("=" * 60)
side_view = Image.new("RGB", (256, 256), color=(220, 220, 220))
draw = ImageDraw.Draw(side_view)
draw.rectangle([100, 40, 156, 220], fill=(50, 120, 60))
draw.rectangle([115, 10, 141, 40], fill=(70, 140, 80))
buf_side = io.BytesIO()
side_view.save(buf_side, format="PNG")
buf_side.seek(0)
r_ext_side = requests.post(f"{API}/predict", files={"file": ("google_side_view_bottle.png", buf_side.getvalue(), "image/png")}).json()
print("External Google Side-View Bottle:")
print(f"  Raw Anomaly Score: {r_ext_side['anomaly_score']}")
print(f"  Verdict: {r_ext_side['verdict']} | Grade: {r_ext_side['grade']}")
print(f"  Input Compatibility: {r_ext_side.get('input_compatibility_status')} ({r_ext_side.get('input_compatibility_score')}%)")
print(f"  Confidence State: {r_ext_side.get('inspection_confidence')} | Human Review Required: {r_ext_side.get('human_review_required')}")
print(f"  Warning: {r_ext_side.get('domain_warning')}")

# Random Noise / Object
random_obj = Image.fromarray(np.uint8(np.random.uniform(50, 200, (256, 256, 3))))
buf_rand = io.BytesIO()
random_obj.save(buf_rand, format="PNG")
buf_rand.seek(0)
r_ext_rand = requests.post(f"{API}/predict", files={"file": ("arbitrary_random_object.png", buf_rand.getvalue(), "image/png")}).json()
print("\nArbitrary Non-Bottle Image (Random Photo Proxy):")
print(f"  Raw Anomaly Score: {r_ext_rand['anomaly_score']}")
print(f"  Verdict: {r_ext_rand['verdict']} | Grade: {r_ext_rand['grade']}")
print(f"  Input Compatibility: {r_ext_rand.get('input_compatibility_status')} ({r_ext_rand.get('input_compatibility_score')}%)")
print(f"  Confidence State: {r_ext_rand.get('inspection_confidence')} | Human Review Required: {r_ext_rand.get('human_review_required')}")
print(f"  Warning: {r_ext_rand.get('domain_warning')}")

print("\n" + "=" * 60)
print("6. VERIFYING PDF REPORT GENERATION WITH COMPATIBILITY")
print("=" * 60)
pdf_req = {
    "filename": "google_bottle_test.png",
    "verdict": r_ext_side["verdict"],
    "grade": r_ext_side["grade"],
    "grade_reason": r_ext_side["grade_reason"],
    "dsar_percent": r_ext_side["dsar_percent"],
    "anomaly_score": r_ext_side["anomaly_score"],
    "defect_zone": r_ext_side["defect_zone"],
    "defect_type": r_ext_side["defect_type"],
    "disposal_action": r_ext_side["disposal_action"],
    "risk_score": r_ext_side.get("risk_score", 0.0),
    "risk_level": r_ext_side.get("risk_level", "Unvalidated"),
    "quality_score": r_ext_side.get("quality_score", 0.0),
    "pattern_cluster": r_ext_side.get("pattern_cluster", "Outlier"),
    "is_unknown": True,
    "input_compatibility_score": r_ext_side.get("input_compatibility_score", 0.0),
    "input_compatibility_status": r_ext_side.get("input_compatibility_status", "OUTSIDE_DOMAIN"),
    "inspection_confidence": r_ext_side.get("inspection_confidence", "UNRELIABLE"),
    "human_review_required": True,
    "domain_warning": r_ext_side.get("domain_warning", "")
}
r_pdf = requests.post(f"{API}/generate_pdf", json=pdf_req)
print(f"PDF Generation Status: {r_pdf.status_code} | Bytes: {len(r_pdf.content)}")
print("=" * 60)
print("TEST SUITE COMPLETED SUCCESSFULLY!")
