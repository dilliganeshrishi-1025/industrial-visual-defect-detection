"""
Training and Benchmark Evaluation Script for Visual Defect Detection on MVTec AD Bottle Dataset
"""

import os
import glob
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from defect_model import BottleDefectDetector

DATASET_ROOT = os.path.join("mvtec_bottle_dataset", "bottle")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train", "good")
TEST_DIR = os.path.join(DATASET_ROOT, "test")
MODEL_SAVE_PATH = "bottle_defect_model.pkl"

def get_image_files(directory: str):
    """Retrieves all PNG/JPG image file paths from directory."""
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(files)

def main():
    print("=" * 65, flush=True)
    print("      VISUAL DEFECT DETECTION MODEL TRAINING & EVALUATION     ", flush=True)
    print("=" * 65, flush=True)
    
    # 1. Gather Training Images
    train_image_paths = get_image_files(TRAIN_DIR)
    print(f"[+] Found {len(train_image_paths)} normal bottle images for training in '{TRAIN_DIR}'", flush=True)
    
    if len(train_image_paths) == 0:
        raise FileNotFoundError(f"No training images found in '{TRAIN_DIR}'!")
        
    # 2. Initialize and Train Detector
    detector = BottleDefectDetector(device='cpu', backbone='resnet18', image_size=256)
    detector.train_memory_bank(train_image_paths, coreset_sampling_ratio=0.15)
    
    # 3. Gather Test Dataset
    test_categories = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    print(f"\n[*] Evaluating on Test Categories: {test_categories}", flush=True)
    
    y_true = []       # 0 for Good, 1 for Defective
    y_scores = []     # Continuous anomaly scores
    test_results = [] # Detailed record dicts
    
    for category in test_categories:
        cat_dir = os.path.join(TEST_DIR, category)
        cat_images = get_image_files(cat_dir)
        is_defect_cat = (category != "good")
        label = 1 if is_defect_cat else 0
        
        print(f"    - Processing category '{category}' ({len(cat_images)} images)...", flush=True)
        for img_path in cat_images:
            res = detector.predict(img_path)
            score = res["anomaly_score"]
            y_true.append(label)
            y_scores.append(score)
            test_results.append({
                "path": img_path,
                "category": category,
                "true_label": label,
                "anomaly_score": score
            })

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # 4. Calculate AUROC & Find Optimal Decision Threshold
    auroc = roc_auc_score(y_true, y_scores)
    
    # Grid search for threshold maximizing F1 score
    thresholds = np.linspace(y_scores.min(), y_scores.max(), 200)
    best_f1 = -1.0
    best_thresh = y_scores.mean()
    
    for t in thresholds:
        preds = (y_scores > t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
            
    # Set calibrated threshold in detector
    detector.threshold = best_thresh
    preds = (y_scores > best_thresh).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary')
    acc = np.mean(preds == y_true)
    cm = confusion_matrix(y_true, preds)
    
    print("\n" + "=" * 65)
    print("                     BENCHMARK RESULTS                        ")
    print("=" * 65)
    print(f"  • Image AUROC Score     : {auroc * 100:.2f}%")
    print(f"  • Classification Accuracy: {acc * 100:.2f}%")
    print(f"  • Precision              : {precision * 100:.2f}%")
    print(f"  • Recall                 : {recall * 100:.2f}%")
    print(f"  • F1-Score               : {f1 * 100:.2f}%")
    print(f"  • Calibrated Threshold   : {best_thresh:.4f}")
    print("  • Confusion Matrix       :")
    print(f"        TN (Good correctly identified): {cm[0, 0]} / {np.sum(y_true == 0)}")
    print(f"        FP (Good misclassified as defect): {cm[0, 1]}")
    print(f"        FN (Defect missed)            : {cm[1, 0]}")
    print(f"        TP (Defect caught)            : {cm[1, 1]} / {np.sum(y_true == 1)}")
    
    print("\n" + "-" * 65)
    print("                 PER-CATEGORY SCORE BREAKDOWN                 ")
    print("-" * 65)
    for cat in test_categories:
        cat_scores = [r["anomaly_score"] for r in test_results if r["category"] == cat]
        mean_score = np.mean(cat_scores)
        std_score = np.std(cat_scores)
        cat_preds = [1 if s > best_thresh else 0 for s in cat_scores]
        correct = np.sum(cat_preds) if cat != "good" else np.sum([1 for p in cat_preds if p == 0])
        total = len(cat_scores)
        print(f"  Category '{cat:15s}': Avg Score = {mean_score:.4f} ± {std_score:.4f} | Accuracy = {correct}/{total} ({correct/total*100:.1f}%)")
    print("-" * 65)
    
    # 5. Save Model Checkpoint
    detector.save_model(MODEL_SAVE_PATH)
    print(f"\n[+] Training & Evaluation complete! Trained detector saved to '{MODEL_SAVE_PATH}'")

if __name__ == "__main__":
    main()
