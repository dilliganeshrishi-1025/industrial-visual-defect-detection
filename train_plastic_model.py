"""
Training & Calibration Script for Full-Body Plastic Bottle Module
Trains the Plastic Bottle normal memory bank strictly on train/good images,
calibrates the decision threshold using the validation split,
and saves persistent model weights and metadata.
"""

import os
import json
from plastic_defect_model import PlasticBottleDetector

DATASET_ROOT = "plastic_bottle_dataset"
TRAIN_GOOD_DIR = os.path.join(DATASET_ROOT, "train", "good")
VAL_DIR = os.path.join(DATASET_ROOT, "validation")

MODEL_DIR = "plastic_bottle_model"
MODEL_PATH = os.path.join(MODEL_DIR, "plastic_defect_model.pkl")

CONFIG_DIR = "plastic_bottle_config"
META_CONFIG_PATH = os.path.join(DATASET_ROOT, "metadata", "threshold_config.json")
CONFIG_PATH = os.path.join(CONFIG_DIR, "threshold_config.json")

def train_and_calibrate():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

    print("=================================================================")
    print("PLASTIC BOTTLE MODEL TRAINING & THRESHOLD CALIBRATION")
    print("=================================================================")

    detector = PlasticBottleDetector(device='cpu', image_size=256)

    # 1. Train Memory Bank (strictly from normal train/good samples)
    detector.train_memory_bank(train_good_dir=TRAIN_GOOD_DIR, coreset_size=5000)

    # 2. Calibrate Threshold on Validation Set
    calib_meta = detector.calibrate_threshold(val_dir=VAL_DIR)

    # 3. Save Model Checkpoint
    detector.save_model(MODEL_PATH)

    # 4. Save Calibration Configuration to Metadata and Config Dirs
    with open(META_CONFIG_PATH, "w") as f:
        json.dump(calib_meta, f, indent=2)

    with open(CONFIG_PATH, "w") as f:
        json.dump(calib_meta, f, indent=2)

    print(f"[+] Saved calibration configuration to '{CONFIG_PATH}' and '{META_CONFIG_PATH}'")
    print("=================================================================")
    print("TRAINING & CALIBRATION COMPLETE!")
    print("=================================================================")

if __name__ == "__main__":
    train_and_calibrate()
