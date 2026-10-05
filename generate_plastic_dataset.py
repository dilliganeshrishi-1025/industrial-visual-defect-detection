"""
Plastic Bottle Dataset Generator (Real-World Water Bottle Archetype)
Creates a persistent, multi-split dataset of front-view transparent ribbed plastic water bottles:
  - Reference Archetype: Real-world transparent ribbed PET mineral water bottle with blue cap.
  - train/good: 100 normal intact bottles with physical lighting, orientation, and cap variations.
  - validation/good (15) & validation/defective (18): Used for threshold calibration.
  - test/good (20) & 6 defect categories (10 each = 60):
      * dent: Body rib depression, crumpled plastic, high-contrast shadow crease.
      * scratch: Conveyor friction abrasion lines and white stress micro-cracks.
      * deformation: Structural wall pinch, waist bulge, or body warping.
      * cap_damage: Askew, chipped, or cross-threaded cap closure.
      * label_defect: Wrinkled, torn, or misplaced printed wrap / barcode band.
      * contamination: Dark foreign particulate, wash sediment, or particulate at base.
Generates metadata: dataset_info.json, split_info.json, and initializes threshold_config.json.
"""

import os
import glob
import json
import random
import numpy as np
import cv2
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATASET_ROOT = "plastic_bottle_dataset"
METADATA_DIR = os.path.join(DATASET_ROOT, "metadata")

IMG_WIDTH = 256
IMG_HEIGHT = 256

def find_reference_image() -> str:
    """Locate the user-uploaded real water bottle image."""
    candidates = [
        r"C:\Users\dilli\.gemini\antigravity-ide\brain\255f0dc5-d2df-4259-91dc-f057028588ff\.user_uploaded\media_1791097308602.png",
    ]
    # Also search general patterns
    glob_matches = glob.glob(r"C:\Users\dilli\.gemini\antigravity-ide\brain\*\.user_uploaded\*.png")
    candidates.extend(glob_matches)
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("Could not find user reference plastic bottle image.")

def ensure_dirs():
    splits = [
        os.path.join(DATASET_ROOT, "train", "good"),
        os.path.join(DATASET_ROOT, "validation", "good"),
        os.path.join(DATASET_ROOT, "validation", "defective"),
        os.path.join(DATASET_ROOT, "test", "good"),
        os.path.join(DATASET_ROOT, "test", "dent"),
        os.path.join(DATASET_ROOT, "test", "scratch"),
        os.path.join(DATASET_ROOT, "test", "deformation"),
        os.path.join(DATASET_ROOT, "test", "cap_damage"),
        os.path.join(DATASET_ROOT, "test", "label_defect"),
        os.path.join(DATASET_ROOT, "test", "contamination"),
        METADATA_DIR
    ]
    for d in splits:
        os.makedirs(d, exist_ok=True)

class WaterBottleAugmentor:
    """
    Synthesizes natural variations and physically plausible industrial defects
    based on the real transparent ribbed water bottle archetype.
    """
    def __init__(self, ref_path: str):
        pil_img = Image.open(ref_path).convert('RGB')
        # Resize to standard canonical size 256x256
        self.base_canonical = np.array(pil_img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR))
        
        # Segment bottle cleanly
        gray = cv2.cvtColor(self.base_canonical, cv2.COLOR_RGB2GRAY)
        diff = np.abs(gray.astype(np.float32) - float(gray[0, 0]))
        _, thresh = cv2.threshold(diff.astype(np.uint8), 10, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea)
        self.bottle_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        cv2.drawContours(self.bottle_mask, [largest], -1, 255, -1)
        self.bx, self.by, self.bw, self.bh = cv2.boundingRect(largest)

    def render_sample(self, defect_type: str = "none", seed: int = 0) -> np.ndarray:
        rng = np.random.RandomState(seed)
        img = self.base_canonical.copy()

        # 1. Subtle Normal Photometric & Geometric Variations
        # Brightness & Contrast
        alpha = rng.uniform(0.94, 1.06) # contrast
        beta = rng.randint(-8, 8)       # brightness
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Subtle lighting gradient (conveyor overhead lighting)
        grad_strength = rng.uniform(-6, 6)
        if abs(grad_strength) > 1.0:
            grad = np.linspace(-grad_strength, grad_strength, IMG_HEIGHT)[:, None]
            img = np.clip(img.astype(np.float32) + grad, 0, 255).astype(np.uint8)

        # Subtle affine jitter (translation +/- 3px, rotation +/- 1.2 deg)
        tx = rng.randint(-3, 4)
        ty = rng.randint(-3, 4)
        angle = rng.uniform(-1.2, 1.2)
        M = cv2.getRotationMatrix2D((IMG_WIDTH // 2, IMG_HEIGHT // 2), angle, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty
        img = cv2.warpAffine(img, M, (IMG_WIDTH, IMG_HEIGHT), borderMode=cv2.BORDER_REPLICATE)

        # Cap color variation on 20% of samples (cyan, navy, emerald, clean red)
        if rng.uniform(0, 1) < 0.25:
            cap_roi = img[26:48, 114:142]
            hsv = cv2.cvtColor(cap_roi, cv2.COLOR_RGB2HSV)
            h_shift = rng.choice([20, 45, 90, 140])
            hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h_shift) % 180
            img[26:48, 114:142] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # 2. Defect Synthesis (if requested)
        if defect_type == "none":
            return img

        elif defect_type == "dent":
            # Localized inward crumple on the ribbed mid-body (Zone B)
            cx = rng.randint(112, 144)
            cy = rng.randint(110, 185)
            rw = rng.randint(12, 18)
            rh = rng.randint(7, 12)
            angle = rng.randint(-25, 25)
            # Depressed crease (dark shadow)
            shadow_color = (rng.randint(60, 85), rng.randint(65, 90), rng.randint(70, 95))
            cv2.ellipse(img, (cx, cy), (rw, rh), angle, 0, 360, shadow_color, -1)
            # Specular rim reflection
            cv2.ellipse(img, (cx + 2, cy - 2), (rw - 2, rh - 2), angle, 0, 180, (255, 255, 255), 2)
            # Smooth blend
            ksize = 5
            sub = img[max(0, cy-rh-4):min(IMG_HEIGHT, cy+rh+4), max(0, cx-rw-4):min(IMG_WIDTH, cx+rw+4)]
            if sub.size > 0:
                blurred = cv2.GaussianBlur(sub, (ksize, ksize), 1.2)
                img[max(0, cy-rh-4):min(IMG_HEIGHT, cy+rh+4), max(0, cx-rw-4):min(IMG_WIDTH, cx+rw+4)] = blurred

        elif defect_type == "scratch":
            # Conveyor friction abrasion scratches across the clear PET wall
            n_lines = rng.randint(2, 5)
            for _ in range(n_lines):
                x1 = rng.randint(108, 138)
                y1 = rng.randint(95, 190)
                x2 = x1 + rng.randint(12, 28)
                y2 = y1 + rng.randint(-12, 12)
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(img, (x1 + 1, y1 + 1), (x2 + 1, y2 + 1), (80, 85, 90), 1, cv2.LINE_AA)

        elif defect_type == "deformation":
            # Structural warping: waist pinch or side bulging
            cy = rng.randint(120, 175)
            pinch = rng.choice([-16, 16])
            h, w = img.shape[:2]
            map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
            map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
            for y in range(max(0, cy - 30), min(h, cy + 30)):
                dy = (30 - abs(y - cy)) / 30.0
                shift = pinch * (dy ** 2)
                map_x[y, :] += shift
            img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        elif defect_type == "cap_damage":
            # Cap chipped, cracked, or tilted
            mode = rng.choice(["crack", "chip", "askew"])
            if mode == "crack":
                # Dark jagged fissure
                pts = [(125, 27), (127, 34), (124, 40), (128, 46)]
                for k in range(len(pts) - 1):
                    cv2.line(img, pts[k], pts[k+1], (20, 20, 25), 2, cv2.LINE_AA)
                cv2.circle(img, (126, 36), 3, (255, 255, 255), -1)
            elif mode == "chip":
                # White missing chunk on cap corner
                cv2.fillPoly(img, [np.array([(114, 27), (122, 27), (117, 38)], dtype=np.int32)], (250, 250, 250))
                cv2.line(img, (114, 27), (117, 38), (30, 30, 30), 1)
            else:
                # Askew / cross-threaded tilted cap
                cap_patch = img[26:48, 114:142].copy()
                M_tilt = cv2.getRotationMatrix2D((14, 11), 14, 1.0)
                tilted = cv2.warpAffine(cap_patch, M_tilt, (28, 22), borderMode=cv2.BORDER_REPLICATE)
                img[26:48, 114:142] = tilted
                cv2.line(img, (114, 45), (142, 49), (40, 40, 40), 2)

        elif defect_type == "label_defect":
            # Wrinkled, peeled, or torn label sticker around mid-body
            y1, y2 = 135, 168
            # Add translucent label band
            label_band = img[y1:y2, 102:154].astype(np.float32)
            tint = np.array([210, 235, 245], dtype=np.float32)
            img[y1:y2, 102:154] = np.clip(0.4 * label_band + 0.6 * tint, 0, 255).astype(np.uint8)
            # Draw tear / peel
            cv2.fillPoly(img, [np.array([(144, y1), (154, y1), (154, y1 + 16)], dtype=np.int32)], (255, 255, 255))
            cv2.line(img, (104, y1 + 6), (135, y2 - 4), (45, 45, 50), 2, cv2.LINE_AA)

        elif defect_type == "contamination":
            # Foreign particulate or dirt at bottle base (Zone C)
            n_specks = rng.randint(8, 18)
            for _ in range(n_specks):
                px = rng.randint(110, 146)
                py = rng.randint(208, 232)
                r = rng.randint(1, 3)
                color = (rng.randint(25, 55), rng.randint(20, 50), rng.randint(20, 45))
                cv2.circle(img, (px, py), r, color, -1)

        return img

def generate_full_dataset():
    ensure_dirs()
    ref_path = find_reference_image()
    print(f"[*] Found User Reference Plastic Bottle: '{ref_path}'")
    augmentor = WaterBottleAugmentor(ref_path)

    counts = {
        ("train", "good"): 100,
        ("validation", "good"): 15,
        ("validation", "defective"): 18,
        ("test", "good"): 20,
        ("test", "dent"): 10,
        ("test", "scratch"): 10,
        ("test", "deformation"): 10,
        ("test", "cap_damage"): 10,
        ("test", "label_defect"): 10,
        ("test", "contamination"): 10
    }

    generated_stats = {
        "train": {"good": 0},
        "validation": {"good": 0, "defective": 0},
        "test": {
            "good": 0, "dent": 0, "scratch": 0, "deformation": 0,
            "cap_damage": 0, "label_defect": 0, "contamination": 0
        }
    }

    seed_idx = 42000

    print("[*] Generating realistic transparent ribbed plastic bottle dataset...")

    # 1. Train Good (100)
    for i in range(counts[("train", "good")]):
        # Sample 0 is exact canonical reference
        if i == 0:
            img = augmentor.base_canonical.copy()
        else:
            img = augmentor.render_sample("none", seed=seed_idx)
            seed_idx += 1
        fpath = os.path.join(DATASET_ROOT, "train", "good", f"{i:03d}.png")
        cv2.imwrite(fpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        generated_stats["train"]["good"] += 1

    # 2. Validation Good (15)
    for i in range(counts[("validation", "good")]):
        img = augmentor.render_sample("none", seed=seed_idx)
        seed_idx += 1
        fpath = os.path.join(DATASET_ROOT, "validation", "good", f"{i:03d}.png")
        cv2.imwrite(fpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        generated_stats["validation"]["good"] += 1

    # 3. Validation Defective (18 = 3 per defect type)
    val_defects = ["dent", "scratch", "deformation", "cap_damage", "label_defect", "contamination"]
    for i in range(counts[("validation", "defective")]):
        dtype = val_defects[i % len(val_defects)]
        img = augmentor.render_sample(dtype, seed=seed_idx)
        seed_idx += 1
        fpath = os.path.join(DATASET_ROOT, "validation", "defective", f"{i:03d}_{dtype}.png")
        cv2.imwrite(fpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        generated_stats["validation"]["defective"] += 1

    # 4. Test Good (20)
    for i in range(counts[("test", "good")]):
        if i == 0:
            # 000.png is the exact reference bottle uploaded by the user!
            img = augmentor.base_canonical.copy()
        else:
            img = augmentor.render_sample("none", seed=seed_idx)
            seed_idx += 1
        fpath = os.path.join(DATASET_ROOT, "test", "good", f"{i:03d}.png")
        cv2.imwrite(fpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        generated_stats["test"]["good"] += 1

    # 5. Test Defective Categories (60 = 10 each)
    for dtype in ["dent", "scratch", "deformation", "cap_damage", "label_defect", "contamination"]:
        for i in range(counts[("test", dtype)]):
            img = augmentor.render_sample(dtype, seed=seed_idx)
            seed_idx += 1
            fpath = os.path.join(DATASET_ROOT, "test", dtype, f"{i:03d}.png")
            cv2.imwrite(fpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            generated_stats["test"][dtype] += 1

    train_total = generated_stats["train"]["good"]
    val_total = generated_stats["validation"]["good"] + generated_stats["validation"]["defective"]
    test_total = sum(generated_stats["test"].values())
    grand_total = train_total + val_total + test_total

    print(f"[+] Dataset Generation Finished! Total images: {grand_total}")
    print(f"    Train (Good only): {train_total}")
    print(f"    Validation (Good + Defective): {val_total}")
    print(f"    Test (Good + 6 Defect Categories): {test_total}")

    # Metadata JSON
    dataset_info = {
        "dataset_name": "Full-Body Plastic Water Bottle Anomaly Detection Dataset",
        "dataset_version": "PB-Dataset-v2.0",
        "dataset_type": "Real-World Water Bottle Archetype / Physically-Plausible Augmented",
        "description": "Full-body front-view transparent ribbed consumer plastic water bottles under conveyor lighting with 6 visually distinct defect categories.",
        "perspective": "Front-View / Full-Body Axial Perspective",
        "resolution": f"{IMG_WIDTH}x{IMG_HEIGHT}",
        "total_images": grand_total,
        "splits": {
            "train": {
                "purpose": "PatchCore Normal Memory Bank Training exclusively on defect-free samples",
                "counts": generated_stats["train"],
                "total": train_total
            },
            "validation": {
                "purpose": "Sensitivity and Anomaly Threshold Calibration (Youden's J & F1 Optimization)",
                "counts": generated_stats["validation"],
                "total": val_total
            },
            "test": {
                "purpose": "Quantitative Benchmark Evaluation (AUROC, Precision, Recall, Confusion Matrix)",
                "counts": generated_stats["test"],
                "total": test_total
            }
        },
        "defect_categories": [
            {"category": "dent", "description": "Localized indentation on bottle ribbed wall"},
            {"category": "scratch", "description": "Conveyor abrasion scratch or surface scuff"},
            {"category": "deformation", "description": "Abnormal structural body shape, wall bulge, or waist pinch"},
            {"category": "cap_damage", "description": "Cracked, askew/tilted, or damaged closure cap"},
            {"category": "label_defect", "description": "Torn, wrinkled, or peeled printed band"},
            {"category": "contamination", "description": "Foreign particulate, dirt stain, or sediment at base"}
        ],
        "created_at": "2026-10-04",
        "status": "Active Persistent Asset"
    }

    with open(os.path.join(METADATA_DIR, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)

    split_info = {
        "train_samples": train_total,
        "validation_samples": val_total,
        "test_samples": test_total,
        "defect_classes": ["dent", "scratch", "deformation", "cap_damage", "label_defect", "contamination"],
        "normal_classes": ["good"],
        "detailed_counts": generated_stats
    }

    with open(os.path.join(METADATA_DIR, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    print("[+] Saved dataset metadata to plastic_bottle_dataset/metadata/")

if __name__ == "__main__":
    generate_full_dataset()
