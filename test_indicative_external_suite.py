"""
Comprehensive Validation Test Suite for Indicative AI Assessment & External Image Handling
Tests 7 distinct scenarios across Glass Bottle and Plastic Bottle domains:
  1. Good dataset plastic bottle (Official Dataset Mode)
  2. Defective dataset plastic bottle (Official Dataset Mode)
  3. Good external plastic bottle (External Mode)
  4. Defective-looking external plastic bottle (External Mode)
  5. Borderline external plastic bottle (External Mode)
  6. Good external glass bottle (External Mode)
  7. Clearly incompatible image (Outside Domain Mode)
"""

import os
import glob
import numpy as np
import cv2
from PIL import Image
import torch

from defect_model import BottleDefectDetector
from plastic_defect_model import PlasticBottleDetector
import quality_intelligence as qi
import plastic_quality_intelligence as plastic_qi

def run_tests():
    print("=" * 80)
    print(" EVALUATION SUITE: INDICATIVE AI ASSESSMENT & EXTERNAL IMAGE HANDLING")
    print("=" * 80)
    
    # 1. Load detectors
    glass_detector = BottleDefectDetector(device='cpu', backbone='resnet18', image_size=256)
    glass_detector.load_model("bottle_defect_model.pkl")
    
    plastic_detector = PlasticBottleDetector(device='cpu', image_size=256)
    plastic_detector.load_model(os.path.join("plastic_bottle_model", "plastic_defect_model.pkl"))
    
    results = {}
    
    # -------------------------------------------------------------------------
    # CASE 1: Good Dataset Plastic Bottle (Official Dataset)
    # -------------------------------------------------------------------------
    good_plastic_files = sorted(glob.glob("plastic_bottle_dataset/test/good/*.png"))
    test_img_1 = good_plastic_files[0] if good_plastic_files else "plastic_bottle_dataset/train/good/good_000.png"
    p1 = plastic_detector.predict(test_img_1, filename=os.path.basename(test_img_1))
    prof1 = plastic_qi.generate_plastic_quality_profile(
        anomaly_score=p1["anomaly_score"],
        dsar_percent=p1["dsar_percent"],
        defect_zone=p1["defect_zone"],
        defect_type=p1["defect_type"],
        threshold=p1["threshold"],
        input_compatibility_score=p1["input_compatibility_score"],
        input_compatibility_status=p1["input_compatibility_status"]
    )
    results["Case 1: Good Dataset Plastic Bottle"] = {
        "Domain": "Plastic Bottle",
        "Mode": "Official Dataset Benchmark",
        "File": os.path.basename(test_img_1),
        "Verdict": p1["verdict"],
        "Grade": p1["grade"],
        "Anomaly Score": p1["anomaly_score"],
        "Threshold": p1["threshold"],
        "Input Compatibility": f"{p1['input_compatibility_score']:.1f}%",
        "Compatibility Status": p1["input_compatibility_status"],
        "Overall Quality Score": f"{prof1['overall_quality_score']}/100",
        "Risk Level": prof1["risk_level"]
    }
    
    # -------------------------------------------------------------------------
    # CASE 2: Defective Dataset Plastic Bottle (Official Dataset)
    # -------------------------------------------------------------------------
    dent_files = sorted(glob.glob("plastic_bottle_dataset/test/dent/*.png"))
    test_img_2 = dent_files[0]
    p2 = plastic_detector.predict(test_img_2, filename=os.path.basename(test_img_2))
    prof2 = plastic_qi.generate_plastic_quality_profile(
        anomaly_score=p2["anomaly_score"],
        dsar_percent=p2["dsar_percent"],
        defect_zone=p2["defect_zone"],
        defect_type=p2["defect_type"],
        threshold=p2["threshold"],
        input_compatibility_score=p2["input_compatibility_score"],
        input_compatibility_status=p2["input_compatibility_status"]
    )
    results["Case 2: Defective Dataset Plastic Bottle"] = {
        "Domain": "Plastic Bottle",
        "Mode": "Official Dataset Benchmark",
        "File": os.path.basename(test_img_2),
        "Verdict": p2["verdict"],
        "Grade": p2["grade"],
        "Anomaly Score": p2["anomaly_score"],
        "Threshold": p2["threshold"],
        "Input Compatibility": f"{p2['input_compatibility_score']:.1f}%",
        "Compatibility Status": p2["input_compatibility_status"],
        "Overall Quality Score": f"{prof2['overall_quality_score']}/100",
        "Risk Level": prof2["risk_level"]
    }

    # -------------------------------------------------------------------------
    # CASE 3: Good External Plastic Bottle (Simulated Google/External Upload)
    # -------------------------------------------------------------------------
    # Create an external unseen plastic bottle with slight lighting/hue shift
    base_good_pil = Image.open(good_plastic_files[1] if len(good_plastic_files) > 1 else test_img_1).convert('RGB')
    np_good_ext = np.array(base_good_pil)
    # Add subtle external lighting variation (soft warm tone)
    np_good_ext[:, :, 0] = np.clip(np_good_ext[:, :, 0].astype(np.float32) * 1.03, 0, 255).astype(np.uint8)
    p3 = plastic_detector.predict(np_good_ext, filename="external_google_plastic_good.jpg")
    ind3 = plastic_qi.compute_plastic_indicative_assessment(
        anomaly_score=p3["anomaly_score"],
        threshold=p3["threshold"],
        input_compatibility_score=p3["input_compatibility_score"],
        input_compatibility_status=p3["input_compatibility_status"],
        risk_score=plastic_qi.calculate_plastic_risk_score(p3["anomaly_score"], p3["defect_zone"], p3["dsar_percent"], p3["threshold"])[0],
        dsar_percent=p3["dsar_percent"],
        defect_zone=p3["defect_zone"]
    )
    results["Case 3: Good External Plastic Bottle"] = {
        "Domain": "Plastic Bottle",
        "Mode": "External / Unseen Input",
        "File": "external_google_plastic_good.jpg",
        "Input Compatibility": f"{ind3['input_compatibility_score']:.1f}%",
        "Anomaly Score": ind3["anomaly_score"],
        "Threshold": p3["threshold"],
        "Indicative Quality Score": ind3["indicative_quality_display"],
        "Indicative Assessment": ind3["indicative_assessment"],
        "Risk Level": ind3["risk_level"],
        "Human Review Recommendation": ind3["human_review_recommendation"]
    }

    # -------------------------------------------------------------------------
    # CASE 4: Defective-Looking External Plastic Bottle
    # -------------------------------------------------------------------------
    deform_files = sorted(glob.glob("plastic_bottle_dataset/test/deformation/*.png"))
    test_img_4 = deform_files[0] if deform_files else test_img_2
    base_def_pil = Image.open(test_img_4).convert('RGB')
    np_def_ext = np.array(base_def_pil)
    # Add external brightness variation
    np_def_ext = np.clip(np_def_ext.astype(np.float32) * 0.96, 0, 255).astype(np.uint8)
    p4 = plastic_detector.predict(np_def_ext, filename="external_google_plastic_crushed.jpg")
    ind4 = plastic_qi.compute_plastic_indicative_assessment(
        anomaly_score=p4["anomaly_score"],
        threshold=p4["threshold"],
        input_compatibility_score=p4["input_compatibility_score"],
        input_compatibility_status=p4["input_compatibility_status"],
        risk_score=plastic_qi.calculate_plastic_risk_score(p4["anomaly_score"], p4["defect_zone"], p4["dsar_percent"], p4["threshold"])[0],
        dsar_percent=p4["dsar_percent"],
        defect_zone=p4["defect_zone"]
    )
    results["Case 4: Defective-Looking External Plastic Bottle"] = {
        "Domain": "Plastic Bottle",
        "Mode": "External / Unseen Input",
        "File": "external_google_plastic_crushed.jpg",
        "Input Compatibility": f"{ind4['input_compatibility_score']:.1f}%",
        "Anomaly Score": ind4["anomaly_score"],
        "Threshold": p4["threshold"],
        "Indicative Quality Score": ind4["indicative_quality_display"],
        "Indicative Assessment": ind4["indicative_assessment"],
        "Risk Level": ind4["risk_level"],
        "Human Review Recommendation": ind4["human_review_recommendation"]
    }

    # -------------------------------------------------------------------------
    # CASE 5: Borderline External Plastic Bottle
    # -------------------------------------------------------------------------
    # Create an image with minor subtle surface scuff that hovers near threshold
    np_borderline = np.array(base_good_pil).copy()
    # Add slight scuff in label zone
    cv2.line(np_borderline, (115, 110), (135, 120), (160, 160, 160), 2)
    p5 = plastic_detector.predict(np_borderline, filename="external_google_plastic_borderline.jpg")
    ind5 = plastic_qi.compute_plastic_indicative_assessment(
        anomaly_score=p5["anomaly_score"],
        threshold=p5["threshold"],
        input_compatibility_score=p5["input_compatibility_score"],
        input_compatibility_status=p5["input_compatibility_status"],
        risk_score=plastic_qi.calculate_plastic_risk_score(p5["anomaly_score"], p5["defect_zone"], p5["dsar_percent"], p5["threshold"])[0],
        dsar_percent=p5["dsar_percent"],
        defect_zone=p5["defect_zone"]
    )
    results["Case 5: Borderline External Plastic Bottle"] = {
        "Domain": "Plastic Bottle",
        "Mode": "External / Unseen Input",
        "File": "external_google_plastic_borderline.jpg",
        "Input Compatibility": f"{ind5['input_compatibility_score']:.1f}%",
        "Anomaly Score": ind5["anomaly_score"],
        "Threshold": p5["threshold"],
        "Indicative Quality Score": ind5["indicative_quality_display"],
        "Indicative Assessment": ind5["indicative_assessment"],
        "Risk Level": ind5["risk_level"],
        "Human Review Recommendation": ind5["human_review_recommendation"]
    }

    # -------------------------------------------------------------------------
    # CASE 6: Good External Glass Bottle
    # -------------------------------------------------------------------------
    glass_good_files = sorted(glob.glob("mvtec_bottle_dataset/bottle/test/good/*.png"))
    test_img_6 = glass_good_files[0] if glass_good_files else "mvtec_bottle_dataset/bottle/train/good/000.png"
    np_glass_good = np.array(Image.open(test_img_6).convert('RGB'))
    # Simulate slight illumination shift for external web upload
    np_glass_good = np.clip(np_glass_good.astype(np.float32) * 1.02, 0, 255).astype(np.uint8)
    p6 = glass_detector.predict(np_glass_good, filename="external_google_glass_topview.jpg")
    r6, r6_lvl = qi.compute_risk_profile(p6["anomaly_score"], p6["dsar_percent"], p6["defect_zone"], p6["grade"], p6["threshold"])
    ind6 = qi.compute_indicative_assessment(
        anomaly_score=p6["anomaly_score"],
        threshold=p6["threshold"],
        input_compatibility_score=p6["input_compatibility_score"],
        input_compatibility_status=p6["input_compatibility_status"],
        risk_score=r6,
        dsar_percent=p6["dsar_percent"],
        defect_zone=p6["defect_zone"]
    )
    results["Case 6: Good External Glass Bottle"] = {
        "Domain": "Glass Bottle",
        "Mode": "External / Unseen Input",
        "File": "external_google_glass_topview.jpg",
        "Input Compatibility": f"{ind6['input_compatibility_score']:.1f}%",
        "Anomaly Score": ind6["anomaly_score"],
        "Threshold": p6["threshold"],
        "Indicative Quality Score": ind6["indicative_quality_display"],
        "Indicative Assessment": ind6["indicative_assessment"],
        "Risk Level": ind6["risk_level"],
        "Human Review Recommendation": ind6["human_review_recommendation"]
    }

    # -------------------------------------------------------------------------
    # CASE 7: Clearly Incompatible Image (Non-Bottle / Natural Scene / Random Texture)
    # -------------------------------------------------------------------------
    # Generate a natural landscape-like / random incompatible RGB image (e.g., grass/textured pattern)
    np.random.seed(42)
    incompatible_img = np.random.randint(40, 180, (256, 256, 3), dtype=np.uint8)
    # Add a horizontal gradient / non-bottle content
    for y in range(256):
        incompatible_img[y, :, 0] = np.clip(incompatible_img[y, :, 0] + y // 3, 0, 255)
        incompatible_img[y, :, 1] = np.clip(incompatible_img[y, :, 1] + (255 - y) // 3, 0, 255)
    
    p7 = plastic_detector.predict(incompatible_img, filename="incompatible_random_image.jpg")
    ind7 = plastic_qi.compute_plastic_indicative_assessment(
        anomaly_score=p7["anomaly_score"],
        threshold=p7["threshold"],
        input_compatibility_score=p7["input_compatibility_score"],
        input_compatibility_status=p7["input_compatibility_status"],
        risk_score=100.0,
        dsar_percent=p7["dsar_percent"],
        defect_zone=p7["defect_zone"]
    )
    results["Case 7: Clearly Incompatible Image"] = {
        "Domain": "Plastic / General",
        "Mode": "Outside Domain / Invalid",
        "File": "incompatible_random_image.jpg",
        "Input Compatibility": f"{ind7['input_compatibility_score']:.1f}%",
        "Anomaly Score": ind7["anomaly_score"],
        "Indicative Quality Score": ind7["indicative_quality_display"],
        "Indicative Assessment": ind7["indicative_assessment"],
        "Risk Level": ind7["risk_level"],
        "Human Review Recommendation": ind7["human_review_recommendation"],
        "Assessment Banner": ind7.get("assessment_banner", "")
    }

    # Print Formatted Results Table
    for case_name, data in results.items():
        print(f"\n--- {case_name} ---")
        for k, v in data.items():
            print(f"  {k:30s}: {v}")

if __name__ == "__main__":
    run_tests()
