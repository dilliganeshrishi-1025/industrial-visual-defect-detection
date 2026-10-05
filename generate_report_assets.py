"""
Automated Figure Generator for Visual Defect Detection Project Report & Presentation
Generates:
1. ROC Curve (roc_curve.png)
2. Confusion Matrix (confusion_matrix.png)
3. Anomaly Score Distribution Plot (score_distribution.png)
4. Sample Defect Heatmaps Comparison Grid (sample_heatmaps.png)
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from defect_model import BottleDefectDetector

DATASET_ROOT = os.path.join("mvtec_bottle_dataset", "bottle")
TEST_DIR = os.path.join(DATASET_ROOT, "test")
MODEL_PATH = "bottle_defect_model.pkl"
OUTPUT_DIR = "report_figures"

def get_image_files(directory: str):
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(files)

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Trained model file '{MODEL_PATH}' not found. Run train_and_evaluate.py first.")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[*] Loading model from '{MODEL_PATH}'...", flush=True)
    
    detector = BottleDefectDetector(device='cpu', backbone='resnet18', image_size=256)
    detector.load_model(MODEL_PATH)
    
    test_categories = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    
    y_true = []
    y_scores = []
    category_data = {cat: [] for cat in test_categories}
    sample_images = {}
    
    for category in test_categories:
        cat_dir = os.path.join(TEST_DIR, category)
        cat_images = get_image_files(cat_dir)
        is_defect = (category != "good")
        label = 1 if is_defect else 0
        
        # Keep first 2 images for sample visual grid
        sample_images[category] = cat_images[:2]
        
        for path in cat_images:
            res = detector.predict(path)
            score = res["anomaly_score"]
            y_true.append(label)
            y_scores.append(score)
            category_data[category].append(score)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Set aesthetics style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # --- 1. ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auroc = roc_auc_score(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot(fpr, tpr, color='#1f77b4', lw=3, label=f'PatchCore ResNet18 (AUROC = {auroc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Chance (AUROC = 0.50)')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#1f77b4')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve\nVisual Defect Detection - MVTec Bottle', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
    ax.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[+] Saved ROC Curve -> {roc_path}", flush=True)

    # --- 2. Confusion Matrix ---
    preds = (y_scores > detector.threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Good (Normal)', 'Defective'],
                yticklabels=['Good (Normal)', 'Defective'],
                annot_kws={'size': 16, 'weight': 'bold'}, ax=ax)
    ax.set_title(f'Confusion Matrix (Threshold = {detector.threshold:.4f})\nAccuracy = {np.mean(preds == y_true)*100:.1f}%', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('Actual True Label', fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Saved Confusion Matrix -> {cm_path}", flush=True)

    # --- 3. Anomaly Score Distribution ---
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    colors = {'good': '#2ca02c', 'broken_large': '#d62728', 'broken_small': '#ff7f0e', 'contamination': '#9467bd'}
    
    for cat, scores in category_data.items():
        sns.kdeplot(scores, label=f'{cat} (n={len(scores)})', color=colors.get(cat, 'blue'), lw=2.5, ax=ax, fill=True, alpha=0.2)
        
    ax.axvline(detector.threshold, color='red', linestyle='--', lw=2, label=f'Decision Threshold ({detector.threshold:.4f})')
    ax.set_title('Anomaly Score Density Distribution Across Defect Classes', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('PatchCore Anomaly Distance Score', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    dist_path = os.path.join(OUTPUT_DIR, "score_distribution.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"[+] Saved Score Distribution -> {dist_path}", flush=True)

    # --- 4. Sample Defect Heatmaps Grid ---
    fig, axes = plt.subplots(len(test_categories), 3, figsize=(10, 3 * len(test_categories)), dpi=200)
    
    for idx, cat in enumerate(test_categories):
        img_path = sample_images[cat][0]
        res = detector.predict(img_path)
        
        orig = res["original_image"]
        heatmap = res["heatmap_overlay"]
        score = res["anomaly_score"]
        verdict = res["verdict"]
        
        # Col 0: Original
        axes[idx, 0].imshow(orig)
        axes[idx, 0].set_title(f"Class: {cat}\nOriginal Image", fontsize=10)
        axes[idx, 0].axis('off')
        
        # Col 1: Heatmap
        axes[idx, 1].imshow(heatmap)
        axes[idx, 1].set_title(f"Defect Localization Heatmap\nScore: {score:.4f}", fontsize=10)
        axes[idx, 1].axis('off')
        
        # Col 2: Verdict Banner
        color = 'green' if verdict == "GOOD" else 'red'
        axes[idx, 2].imshow(orig)
        axes[idx, 2].text(128, 128, f"{verdict}", color='white', fontsize=16, fontweight='bold',
                          ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.85))
        axes[idx, 2].set_title(f"Model Verdict\nThreshold: {detector.threshold:.4f}", fontsize=10)
        axes[idx, 2].axis('off')
        
    plt.suptitle("MVTec Bottle Defect Detection & Pixel-Level Heatmap Inspection", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    grid_path = os.path.join(OUTPUT_DIR, "sample_heatmaps.png")
    plt.savefig(grid_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Heatmaps Grid -> {grid_path}", flush=True)

    print(f"\n[+] All report figures successfully generated in folder '{OUTPUT_DIR}/'", flush=True)

if __name__ == "__main__":
    main()
