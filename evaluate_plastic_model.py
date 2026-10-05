"""
Official Quantitative Benchmark Evaluation for Plastic Bottle Inspection Module (Domain 2).
Evaluates the model across plastic_bottle_dataset/test/ (80 images):
  - Normal: good (20 images)
  - Defective: dent (10), scratch (10), deformation (10), cap_damage (10), label_defect (10), contamination (10)
Calculates: Accuracy, Precision, Recall, F1, AUROC, FPR, Confusion Matrix, Latency & FPS.
Saves results to plastic_bottle_config/benchmark_results.json.
"""

import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from plastic_defect_model import PlasticBottleDetector

DATASET_ROOT = "plastic_bottle_dataset"
TEST_DIR = os.path.join(DATASET_ROOT, "test")
MODEL_PATH = os.path.join("plastic_bottle_model", "plastic_defect_model.pkl")
CONFIG_DIR = "plastic_bottle_config"
BENCHMARK_SAVE_PATH = os.path.join(CONFIG_DIR, "benchmark_results.json")

def evaluate():
    print("=================================================================")
    print("PLASTIC BOTTLE OFFICIAL BENCHMARK EVALUATION")
    print("=================================================================")

    detector = PlasticBottleDetector(device='cpu', image_size=256)
    detector.load_model(MODEL_PATH)
    active_thresh = detector.threshold

    categories = [d for d in sorted(os.listdir(TEST_DIR)) if os.path.isdir(os.path.join(TEST_DIR, d))]
    
    y_true = []
    y_scores = []
    y_pred = []
    category_records = []
    latencies = []

    for cat in categories:
        cat_dir = os.path.join(TEST_DIR, cat)
        files = [f for f in sorted(os.listdir(cat_dir)) if f.endswith(('.png', '.jpg', '.jpeg'))]
        cat_is_def = (cat != "good")
        cat_scores = []
        cat_correct = 0

        for fname in files:
            fpath = os.path.join(cat_dir, fname)
            
            t0 = time.perf_counter()
            res = detector.predict(fpath, filename=fname)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            score = res["anomaly_score"]
            pred_def = res["is_defective"]

            y_true.append(1 if cat_is_def else 0)
            y_scores.append(score)
            y_pred.append(1 if pred_def else 0)

            is_correct = (pred_def == cat_is_def)
            if is_correct:
                cat_correct += 1
            cat_scores.append(score)

            category_records.append({
                "Filename": fname,
                "Category": cat,
                "True_Label": "Defective" if cat_is_def else "Good",
                "Anomaly_Score": score,
                "Predicted_Verdict": res["verdict"],
                "Predicted_Grade": res["grade"],
                "Detected_Zone": res["defect_zone"],
                "DSAR_Percent": res["dsar_percent"],
                "Correct": is_correct
            })

        cat_acc = (cat_correct / max(1, len(files))) * 100.0
        print(f"  Category [{cat:<14}]: {len(files):>2} images | Mean Anomaly: {np.mean(cat_scores):.4f} | Accuracy: {cat_acc:.1f}%")

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    prec = float(precision_score(y_true, y_pred, zero_division=0) * 100.0)
    rec = float(recall_score(y_true, y_pred, zero_division=0) * 100.0)
    f1 = float(f1_score(y_true, y_pred, zero_division=0) * 100.0)
    
    try:
        auroc = float(roc_auc_score(y_true, y_scores) * 100.0)
    except Exception:
        auroc = 100.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = float((fp / max(1, fp + tn)) * 100.0)

    avg_latency = float(np.mean(latencies))
    fps = float(1000.0 / max(1e-3, avg_latency))

    summary = {
        "domain": "plastic_bottle",
        "model_version": detector.model_version,
        "dataset_version": detector.dataset_version,
        "threshold": active_thresh,
        "total_test_images": len(y_true),
        "total_good_images": int(np.sum(y_true == 0)),
        "total_defective_images": int(np.sum(y_true == 1)),
        "metrics": {
            "auroc": round(auroc, 2),
            "accuracy": round(acc, 2),
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1_score": round(f1, 2),
            "false_positive_rate": round(fpr, 2),
            "confusion_matrix": {
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn)
            },
            "average_latency_ms": round(avg_latency, 1),
            "fps": round(fps, 1)
        },
        "detailed_results": category_records
    }

    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(BENCHMARK_SAVE_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("=================================================================")
    print(f"OVERALL PERFORMANCE METRICS (Threshold = {active_thresh:.4f}):")
    print(f"  AUROC:      {auroc:.2f}%")
    print(f"  Accuracy:   {acc:.2f}%")
    print(f"  Precision:  {prec:.2f}%")
    print(f"  Recall:     {rec:.2f}%")
    print(f"  F1-Score:   {f1:.2f}%")
    print(f"  FPR:        {fpr:.2f}%")
    print(f"  Confusion:  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Latency:    {avg_latency:.1f} ms per image (~{fps:.1f} FPS)")
    print("=================================================================")
    print(f"[+] Saved official benchmark results to '{BENCHMARK_SAVE_PATH}'")

if __name__ == "__main__":
    evaluate()
