"""
Full-Body Plastic Bottle Visual Defect Detection Engine
Domain: Front-View / Full-Body Axial Container Inspection
Architecture:
  - Bottle ROI Extractor & Background Suppressor
  - 3-Zone Anatomical Segmentation (Cap/Neck, Label/Mid-Body, Lower-Body/Base)
  - Pretrained ResNet-18 Feature Extractor (Layers 2 & 3)
  - Dedicated Plastic Bottle Coreset Memory Bank (PB-v1.0)
  - Input Compatibility & Uncertainty-Aware Domain Gating
"""

import os
import json
import pickle
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

MODEL_VERSION = "PB-v1.0"
DATASET_VERSION = "PB-Dataset-v1.0"
THRESHOLD_VERSION = "PB-Threshold-v1.0"

class PlasticFeatureExtractor(nn.Module):
    """
    Extracts multi-scale patch features from Layer 2 (medium-level structural contours)
    and Layer 3 (high-level surface & material semantics) of ResNet-18.
    """
    def __init__(self, pretrained: bool = True):
        super(PlasticFeatureExtractor, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.stem(x)
            x = self.layer1(x)
            f2 = self.layer2(x)   # (B, 128, H/8, W/8)
            f3 = self.layer3(f2)  # (B, 256, H/16, W/16)
            
            f3_up = F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
            features = torch.cat([f2, f3_up], dim=1) # (B, 384, H/8, W/8)
            features = F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)
        return features

class PlasticBottlePreprocessor:
    """
    Extracts bottle region of interest (ROI) and segments the bottle into
    three anatomical zones:
      - Zone A: Cap / Neck (Top 25%)
      - Zone B: Label / Mid Body (Middle 50%: 25% - 75%)
      - Zone C: Lower Body / Base (Bottom 25%: 75% - 100%)
    """
    @staticmethod
    def extract_bottle_roi(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Segments bottle body from studio background.
        Returns:
          - masked_img: RGB image with background suppressed
          - bottle_mask: Binary mask (255 for bottle, 0 for background)
          - roi_meta: Bounding box (x, y, w, h), aspect ratio, zone boundaries
        """
        h, w = img_rgb.shape[:2]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Invert from high-key studio background (bottle is darker/textured than pure white bg)
        bg_val = np.median([gray[0:10, 0:10], gray[0:10, -10:], gray[-10:, 0:10], gray[-10:, -10:]])
        diff = cv2.absdiff(gray, int(bg_val))
        
        _, thresh = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        has_bottle = False
        if contours:
            largest_c = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(largest_c)
            # Refine if reasonable size
            if bw * bh > (h * w * 0.08) and bh > (h * 0.40):
                bottle_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(bottle_mask, [largest_c], -1, 255, -1)
                aspect_ratio = bh / float(max(1, bw))
                has_bottle = True
            else:
                bottle_mask = np.zeros((h, w), dtype=np.uint8)
                bx, by, bw, bh = int(w*0.35), int(h*0.1), int(w*0.3), int(h*0.8)
                bottle_mask[by:by+bh, bx:bx+bw] = 255
                aspect_ratio = 1.0
        else:
            bottle_mask = np.zeros((h, w), dtype=np.uint8)
            bx, by, bw, bh = int(w*0.35), int(h*0.1), int(w*0.3), int(h*0.8)
            bottle_mask[by:by+bh, bx:bx+bw] = 255
            aspect_ratio = 1.0

        # Suppress background: blend background towards clean uniform neutral
        masked_img = img_rgb.copy()
        bg_clean = np.full_like(img_rgb, int(bg_val))
        mask_3c = np.stack([bottle_mask]*3, axis=-1) / 255.0
        masked_img = (masked_img * mask_3c + bg_clean * (1.0 - mask_3c)).astype(np.uint8)

        # 3-Zone vertical splits within bounding box
        zone_a_bottom = by + int(bh * 0.25)
        zone_b_bottom = by + int(bh * 0.75)
        
        roi_meta = {
            "bbox": (int(bx), int(by), int(bw), int(bh)),
            "aspect_ratio": float(aspect_ratio),
            "has_bottle": has_bottle,
            "zone_a_bounds": (by, zone_a_bottom),
            "zone_b_bounds": (zone_a_bottom, zone_b_bottom),
            "zone_c_bounds": (zone_b_bottom, by + bh),
            "bottle_area_pixels": int(np.sum(bottle_mask > 0))
        }
        
        return masked_img, bottle_mask, roi_meta

class PlasticBottleDetector:
    """
    Dedicated PatchCore visual defect detection engine for Full-Body Plastic Bottles.
    Maintains separate weights, memory bank, calibration thresholds, and quality intelligence.
    """
    def __init__(self, device: str = 'cpu', image_size: int = 256):
        self.device = torch.device(device if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.image_size = image_size
        self.feature_extractor = PlasticFeatureExtractor(pretrained=True).to(self.device)
        self.memory_bank: Optional[np.ndarray] = None
        self.threshold: float = 0.4500  # Default initial prior, calibrated via validation
        self.model_version = MODEL_VERSION
        self.dataset_version = DATASET_VERSION
        self.threshold_version = THRESHOLD_VERSION
        
        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, img_input) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, Dict]:
        """Loads and pre-processes image with bottle ROI extraction."""
        if isinstance(img_input, str):
            pil_img = Image.open(img_input).convert('RGB')
        elif isinstance(img_input, Image.Image):
            pil_img = img_input.convert('RGB')
        else:
            pil_img = Image.fromarray(img_input).convert('RGB')
            
        pil_resized = pil_img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        img_np = np.array(pil_resized)
        
        masked_np, bottle_mask, roi_meta = PlasticBottlePreprocessor.extract_bottle_roi(img_np)
        tensor = self.transform(Image.fromarray(masked_np)).unsqueeze(0).to(self.device)
        return tensor, img_np, bottle_mask, roi_meta

    def extract_patch_features(self, tensor: torch.Tensor) -> np.ndarray:
        """Extracts L2-normalized spatial patch feature vectors from ResNet-18."""
        with torch.no_grad():
            feat_map = self.feature_extractor(tensor) # (1, 384, H_map, W_map)
            b, c, h_m, w_m = feat_map.shape
            feat_reshaped = feat_map.squeeze(0).permute(1, 2, 0).reshape(-1, c) # (N_patches, 384)
            feat_norm = F.normalize(feat_reshaped, p=2, dim=1)
        return feat_norm.cpu().numpy()

    def train_memory_bank(self, train_good_dir: str, coreset_size: int = 5000):
        """
        Populates the Plastic Bottle Normal Feature Memory Bank strictly from defect-free images.
        Subsamples using greedy coreset selection to maintain ultra-fast inference.
        """
        print(f"[*] Training Plastic Bottle Memory Bank from '{train_good_dir}'...")
        image_files = [os.path.join(train_good_dir, f) for f in os.listdir(train_good_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        all_patches = []

        for idx, fpath in enumerate(image_files):
            tensor, _, _, _ = self.preprocess_image(fpath)
            patches = self.extract_patch_features(tensor)
            # Subsample 15% randomly per training image to keep initial pool manageable
            n_sample = max(10, int(len(patches) * 0.15))
            chosen_indices = np.random.choice(len(patches), n_sample, replace=False)
            all_patches.append(patches[chosen_indices])

        raw_memory = np.vstack(all_patches).astype(np.float32)
        print(f"[*] Accumulated {len(raw_memory)} normal patch vectors. Running minimax coreset subsampling...")

        # Greedy k-center minimax coreset selection
        target_size = min(coreset_size, len(raw_memory))
        coreset_indices = [np.random.randint(0, len(raw_memory))]
        
        # Initialize min distance array with distance to first center
        first_center = raw_memory[coreset_indices[0]]
        sims = np.dot(raw_memory, first_center)
        min_distances = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * sims))

        for step in range(1, target_size):
            new_idx = int(np.argmax(min_distances))
            coreset_indices.append(new_idx)
            new_center = raw_memory[new_idx]
            new_sims = np.dot(raw_memory, new_center)
            new_dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * new_sims))
            min_distances = np.minimum(min_distances, new_dists)

        self.memory_bank = raw_memory[coreset_indices]
        print(f"[+] Plastic Bottle Memory Bank trained successfully with {len(self.memory_bank)} coreset vectors.")

    def calibrate_threshold(self, val_dir: str) -> Dict:
        """
        Calibrates anomaly decision threshold using the validation split.
        Applies Youden's J statistic to maximize balanced sensitivity and specificity.
        """
        print(f"[*] Calibrating Plastic Bottle threshold on '{val_dir}'...")
        good_dir = os.path.join(val_dir, "good")
        defect_dir = os.path.join(val_dir, "defective")

        good_files = [os.path.join(good_dir, f) for f in os.listdir(good_dir) if f.endswith(('.png', '.jpg'))] if os.path.exists(good_dir) else []
        defect_files = [os.path.join(defect_dir, f) for f in os.listdir(defect_dir) if f.endswith(('.png', '.jpg'))] if os.path.exists(defect_dir) else []

        scores_good = []
        scores_defect = []

        for f in good_files:
            res = self.predict(f)
            scores_good.append(res["anomaly_score"])

        for f in defect_files:
            res = self.predict(f)
            scores_defect.append(res["anomaly_score"])

        all_scores = scores_good + scores_defect
        all_labels = [0] * len(scores_good) + [1] * len(scores_defect)

        best_thresh = 0.45
        best_j = -1.0
        best_f1 = 0.0

        for candidate in np.linspace(min(all_scores), max(all_scores), 100):
            preds = [1 if s >= candidate else 0 for s in all_scores]
            tp = sum(1 for p, y in zip(preds, all_labels) if p == 1 and y == 1)
            tn = sum(1 for p, y in zip(preds, all_labels) if p == 0 and y == 0)
            fp = sum(1 for p, y in zip(preds, all_labels) if p == 1 and y == 0)
            fn = sum(1 for p, y in zip(preds, all_labels) if p == 0 and y == 1)

            tpr = tp / float(max(1, tp + fn))
            fpr = fp / float(max(1, fp + tn))
            j_stat = tpr - fpr
            precision = tp / float(max(1, tp + fp))
            f1 = (2 * precision * tpr) / float(max(1e-6, precision + tpr))

            if j_stat > best_j or (abs(j_stat - best_j) < 1e-4 and f1 > best_f1):
                best_j = j_stat
                best_f1 = f1
                best_thresh = float(candidate)

        self.threshold = round(best_thresh, 4)
        
        calib_meta = {
            "inspection_domain": "plastic_bottle",
            "model_version": self.model_version,
            "threshold_version": self.threshold_version,
            "threshold": self.threshold,
            "calibration_method": "Youden J Statistic & Maximum F1 Optimization",
            "validation_samples": {
                "good": len(scores_good),
                "defective": len(scores_defect),
                "total": len(all_scores)
            },
            "score_distribution": {
                "good_mean": float(np.mean(scores_good)) if scores_good else 0.0,
                "good_max": float(np.max(scores_good)) if scores_good else 0.0,
                "defect_mean": float(np.mean(scores_defect)) if scores_defect else 0.0,
                "defect_min": float(np.min(scores_defect)) if scores_defect else 0.0
            },
            "calibrated_at": "2026-10-04"
        }
        
        print(f"[+] Calibrated Plastic Bottle Threshold: {self.threshold:.4f} (Good Max: {max(scores_good):.4f}, Defect Min: {min(scores_defect):.4f})")
        return calib_meta

    def compute_input_compatibility(
        self,
        patch_distances: np.ndarray,
        roi_meta: Dict,
        img_np: np.ndarray,
        bottle_mask: Optional[np.ndarray] = None
    ) -> Tuple[float, str, str, bool, str]:
        """
        Evaluates Input Compatibility for the Plastic Bottle Domain.
        Signals:
          1. Nearest-Neighbor Patch Distances (q25, d_mean) inside Bottle ROI against Plastic Memory Bank
          2. Aspect ratio: Front-view bottle must be vertical (aspect ratio >= 1.6)
          3. Studio background luminance
        Returns:
          - score: 0 to 100
          - status: COMPATIBLE / LOW_CONFIDENCE / OUTSIDE_DOMAIN
          - confidence: VALIDATED DOMAIN / UNCERTAIN / UNRELIABLE
          - human_review_required: True/False
          - warning: str
        """
        # 1. Feature Distance Signal (Inside Bottle ROI)
        if bottle_mask is not None and np.sum(bottle_mask > 0) > 100:
            grid_dim = int(np.sqrt(len(patch_distances)))
            dist_grid = patch_distances.reshape(grid_dim, grid_dim)
            mask_down = cv2.resize(bottle_mask, (grid_dim, grid_dim), interpolation=cv2.INTER_NEAREST)
            bottle_dists = dist_grid[mask_down > 0]
            if len(bottle_dists) > 10:
                q25 = float(np.percentile(bottle_dists, 25))
                d_mean = float(np.mean(bottle_dists))
            else:
                q25 = float(np.percentile(patch_distances, 25))
                d_mean = float(np.mean(patch_distances))
        else:
            q25 = float(np.percentile(patch_distances, 25))
            d_mean = float(np.mean(patch_distances))

        dist_factor = float(np.clip(1.0 - (q25 - 0.16) / 0.18, 0.0, 1.0))
        
        # 2. Geometric Aspect Ratio Signal
        aspect_ratio = roi_meta.get("aspect_ratio", 1.0)
        has_bottle = roi_meta.get("has_bottle", True)
        
        # Geometry component: aspect ratio penalty & bottle presence
        if not has_bottle or aspect_ratio < 1.35:
            # Likely top-view or non-bottle image!
            geo_factor = 0.0
            is_vertical = False
        elif aspect_ratio < 1.6:
            geo_factor = 0.50
            is_vertical = False
        else:
            geo_factor = 1.0
            is_vertical = True

        # 3. Corner Studio Background Signal
        c1 = np.mean(img_np[0:12, 0:12])
        c2 = np.mean(img_np[0:12, -12:])
        c3 = np.mean(img_np[-12:, 0:12])
        c4 = np.mean(img_np[-12:, -12:])
        corner_lum = float(np.mean([c1, c2, c3, c4]))

        # Background component: studio background >= 180
        bg_factor = float(np.clip((corner_lum - 80) / 160.0, 0.0, 1.0))

        # Weighted compatibility score (0 - 100)
        if dist_factor == 0.0:
            # Container features completely unvalidated / mismatch with training domain
            comp_score = 0.0
        else:
            comp_score = round(float(0.60 * (dist_factor * 100) + 0.25 * (geo_factor * 100) + 0.15 * (bg_factor * 100)), 1)
        comp_score = float(np.clip(comp_score, 0.0, 100.0))

        if comp_score >= 60.0 and is_vertical:
            status = "COMPATIBLE"
            confidence = "VALIDATED DOMAIN"
            human_rev = False
            warning = ""
        elif comp_score >= 30.0:
            status = "LOW_CONFIDENCE"
            confidence = "UNCERTAIN"
            human_rev = True
            warning = "Input differs somewhat from the validated plastic-bottle inspection distribution. Human review recommended."
        else:
            status = "OUTSIDE_DOMAIN"
            confidence = "UNRELIABLE"
            human_rev = True
            if not is_vertical:
                warning = f"Geometry mismatch (Aspect Ratio: {aspect_ratio:.2f}). Expected vertical full-body plastic bottle. Automatic defect classification disabled."
            else:
                warning = "This image is outside the validated plastic-bottle inspection distribution. Automatic defect decision disabled."

        return comp_score, status, confidence, human_rev, warning

    def predict(self, img_input, custom_threshold: Optional[float] = None, filename: str = "plastic_bottle.png") -> Dict:
        """
        Runs complete Full-Body Plastic Bottle inference:
          - Input Compatibility / OOD gating
          - Bottle ROI extraction
          - ResNet-18 patch feature distance computation
          - Spatial anomaly heatmap & 3-zone localization
          - DSAR calculation & Quality Triage
        """
        if self.memory_bank is None:
            raise RuntimeError("Plastic Bottle model has no trained memory bank. Train or load checkpoint first.")

        active_thresh = custom_threshold if custom_threshold is not None else self.threshold
        tensor, original_rgb, bottle_mask, roi_meta = self.preprocess_image(img_input)
        
        # 1. Extract Patch Features
        patches = self.extract_patch_features(tensor) # (N_patches, 384)
        
        # 2. Nearest-Neighbor Memory Bank Distance Matrix
        sim_matrix = np.dot(patches, self.memory_bank.T)
        max_sims = np.max(sim_matrix, axis=1)
        patch_dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * max_sims))

        # 3. Input Compatibility & Domain Check
        comp_score, comp_status, conf_state, human_rev, domain_warn = self.compute_input_compatibility(
            patch_dists, roi_meta, original_rgb, bottle_mask
        )

        # 4. Spatial Anomaly Heatmap Construction
        grid_dim = int(np.sqrt(len(patches)))
        dist_grid = patch_dists.reshape(grid_dim, grid_dim)
        
        raw_heatmap = cv2.resize(dist_grid, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        raw_heatmap = cv2.GaussianBlur(raw_heatmap, (15, 15), sigmaX=3.5)

        # Suppress background in heatmap using bottle ROI mask
        mask_float = (bottle_mask > 0).astype(np.float32)
        masked_heatmap = raw_heatmap * mask_float

        # Anomaly Score (top 1% highest patch distance inside bottle ROI)
        bottle_patch_scores = masked_heatmap[bottle_mask > 0]
        if len(bottle_patch_scores) > 0:
            top_k_count = max(1, int(len(bottle_patch_scores) * 0.01))
            anomaly_score = float(np.mean(np.partition(bottle_patch_scores, -top_k_count)[-top_k_count:]))
        else:
            anomaly_score = float(np.max(raw_heatmap))

        # Defect Segmentation Mask (Pixels >= threshold on bottle)
        defect_binary = ((masked_heatmap >= active_thresh) & (bottle_mask > 0)).astype(np.uint8) * 255
        defect_pixel_count = int(np.sum(defect_binary > 0))
        bottle_area = max(1, roi_meta.get("bottle_area_pixels", 1))
        dsar_percent = round(float((defect_pixel_count / bottle_area) * 100.0), 2)

        # 5. Zone-Aware Anatomical Diagnosis (Cap/Neck, Label/Mid-Body, Base)
        za_top, za_bot = roi_meta["zone_a_bounds"]
        zb_top, zb_bot = roi_meta["zone_b_bounds"]
        zc_top, zc_bot = roi_meta["zone_c_bounds"]

        defect_za = int(np.sum(defect_binary[za_top:za_bot, :] > 0))
        defect_zb = int(np.sum(defect_binary[zb_top:zb_bot, :] > 0))
        defect_zc = int(np.sum(defect_binary[zc_top:zc_bot, :] > 0))

        if defect_pixel_count == 0:
            detected_zone = "None"
        elif defect_za >= defect_zb and defect_za >= defect_zc:
            detected_zone = "Zone A: Cap / Neck"
        elif defect_zb >= defect_za and defect_zb >= defect_zc:
            detected_zone = "Zone B: Label / Mid Body"
        else:
            detected_zone = "Zone C: Lower Body / Base"

        # 6. Quality Triage & Uncertainty-Aware Action
        is_defective = (anomaly_score >= active_thresh) or (dsar_percent >= 0.5)

        if comp_status == "OUTSIDE_DOMAIN":
            verdict = "UNVALIDATED EXTERNAL INPUT"
            grade = "Unvalidated Input — Outside Domain"
            action = "HOLD CONTAINER (Manual Inspection Required)"
            grade_reason = domain_warn
            insp_mode = "HUMAN REVIEW REQUIRED (AUTOMATIC DECISION DISABLED)"
            defect_type = "Unseen External Image"
        elif comp_status == "LOW_CONFIDENCE":
            if is_defective:
                verdict = "DEFECTIVE (UNCERTAIN)"
                grade = "Grade C — Defective (Uncertain)" if "Cap" in detected_zone else "Grade B — Review Required"
                action = "Manual Line Inspection Recommended"
                defect_type = "Potential Surface Defect"
            else:
                verdict = "GOOD (UNCERTAIN)"
                grade = "Grade A — Good (Uncertain)"
                action = "Human Audit Sampling"
                defect_type = "Intact Container"
            grade_reason = f"Low confidence input ({comp_score:.1f}%). {domain_warn}"
            insp_mode = "HUMAN REVIEW RECOMMENDED"
        else: # COMPATIBLE
            insp_mode = "AUTOMATIC"
            if not is_defective:
                verdict = "GOOD"
                grade = "Grade A — Intact (Pass)"
                action = "Approved for Bottling / Packaging"
                defect_type = "Good (Normal Plastic Bottle)"
                grade_reason = f"Bottle within structural tolerances. Max anomaly score {anomaly_score:.4f} < threshold {active_thresh:.4f}."
            else:
                verdict = "DEFECTIVE"
                if "Cap" in detected_zone:
                    grade = "Grade C — Critical Cap Defect"
                    action = "Immediate Reject / Scrap"
                    defect_type = "Cap / Closure Anomaly"
                    grade_reason = f"Closure integrity compromised in Zone A (DSAR: {dsar_percent}%). Cap seal failure risk."
                elif dsar_percent >= 4.0 or anomaly_score >= (active_thresh * 1.35):
                    grade = "Grade C — Structural Deformation / Severe Defect"
                    action = "Immediate Scrap Bin"
                    defect_type = "Structural Deformation / Severe Dent"
                    grade_reason = f"Severe anomaly extent (DSAR: {dsar_percent}%). Structural failure risk."
                elif "Label" in detected_zone:
                    grade = "Grade B — Label / Cosmetic Defect"
                    action = "Rework / Re-labeling Station"
                    defect_type = "Label / Surface Defect"
                    grade_reason = f"Localized label / cosmetic anomaly in Zone B (DSAR: {dsar_percent}%)."
                else:
                    grade = "Grade B — Base Contamination / Minor Scuff"
                    action = "Secondary Wash / Manual Audit"
                    defect_type = "Base Scuff / Particulate"
                    grade_reason = f"Surface anomaly detected in Zone C (DSAR: {dsar_percent}%)."

        # 7. Render Color Heatmap Overlay
        norm_map = np.clip((masked_heatmap - active_thresh * 0.5) / (active_thresh * 1.5 + 1e-6), 0.0, 1.0)
        heat_uint8 = (norm_map * 255).astype(np.uint8)
        color_map = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
        
        # Blend overlay onto original
        alpha = 0.55 * mask_float[:, :, None]
        heatmap_overlay = np.clip((1.0 - alpha) * original_rgb + alpha * color_map, 0, 255).astype(np.uint8)
        
        # Draw zone demarcation lines on overlay for interpretability
        cv2.line(heatmap_overlay, (0, za_bot), (self.image_size, za_bot), (255, 255, 0), 1, cv2.LINE_AA)
        cv2.line(heatmap_overlay, (0, zb_bot), (self.image_size, zb_bot), (255, 255, 0), 1, cv2.LINE_AA)

        return {
            "inspection_domain": "plastic_bottle",
            "filename": filename,
            "anomaly_score": round(anomaly_score, 4),
            "threshold": round(active_thresh, 4),
            "is_defective": bool(is_defective),
            "verdict": verdict,
            "grade": grade,
            "grade_reason": grade_reason,
            "dsar_percent": dsar_percent,
            "defect_zone": detected_zone,
            "defect_type": defect_type,
            "disposal_action": action,
            "input_compatibility_score": comp_score,
            "input_compatibility_status": comp_status,
            "inspection_confidence": conf_state,
            "inspection_mode": insp_mode,
            "human_review_required": human_rev,
            "domain_warning": domain_warn,
            "heatmap_overlay": heatmap_overlay,
            "original_image": original_rgb,
            "bottle_mask": bottle_mask,
            "roi_meta": roi_meta
        }

    def save_model(self, save_path: str):
        """Saves model state and memory bank checkpoint."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data = {
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "threshold_version": self.threshold_version,
            "memory_bank": self.memory_bank,
            "threshold": self.threshold,
            "image_size": self.image_size
        }
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"[+] Saved Plastic Bottle Model Checkpoint to '{save_path}'")

    def load_model(self, load_path: str):
        """Loads model state and memory bank checkpoint."""
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        self.memory_bank = data["memory_bank"]
        self.threshold = float(data["threshold"])
        self.image_size = int(data.get("image_size", 256))
        self.model_version = data.get("model_version", MODEL_VERSION)
        self.dataset_version = data.get("dataset_version", DATASET_VERSION)
        self.threshold_version = data.get("threshold_version", THRESHOLD_VERSION)
        print(f"[+] Loaded Plastic Bottle Model from '{load_path}' (Memory bank: {len(self.memory_bank)} vectors, Threshold: {self.threshold:.4f})")
