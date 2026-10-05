"""
Quality Intelligence & Adaptive Risk Management for Full-Body Plastic Bottles (Domain 2).
Provides:
  - 3-Zone Anatomical Risk Engineering (Cap/Neck, Label/Mid-Body, Lower-Body/Base)
  - Full-Body Plastic Bottle Quality Profile (Structural, Cap, Body, Label, Surface)
  - Dedicated Batch Quality Intelligence (BQS 0–100)
  - Manufacturing Pareto Defect Analysis (80/20 Rule)
  - SQLite Historical Audit Trail Persistence (`plastic_inspections`)
"""

import os
import sqlite3
import datetime
import uuid
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple

DB_PATH = "quality_intelligence.db"

# -----------------------------------------------------------------------------
# 1. SQLite Database Schema for Plastic Bottle Domain
# -----------------------------------------------------------------------------
def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_plastic_db():
    """Initializes the dedicated plastic_inspections and plastic_human_reviews SQLite tables."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plastic_inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                bottle_id TEXT,
                filename TEXT NOT NULL,
                category TEXT,
                anomaly_score REAL NOT NULL,
                dsar_percent REAL NOT NULL,
                defect_zone TEXT NOT NULL,
                defect_type TEXT NOT NULL,
                grade TEXT NOT NULL,
                grade_reason TEXT,
                disposal_action TEXT,
                threshold REAL NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                quality_score REAL NOT NULL,
                structural_integrity REAL NOT NULL,
                cap_integrity REAL NOT NULL,
                body_integrity REAL NOT NULL,
                label_condition REAL NOT NULL,
                surface_condition REAL NOT NULL,
                input_compatibility_score REAL NOT NULL DEFAULT 100.0,
                input_compatibility_status TEXT NOT NULL DEFAULT 'COMPATIBLE',
                inspection_confidence TEXT NOT NULL DEFAULT 'VALIDATED DOMAIN',
                human_review_required INTEGER NOT NULL DEFAULT 0,
                domain_warning TEXT,
                batch_id TEXT,
                reviewed INTEGER NOT NULL DEFAULT 0
            )
        """)
        
        # Migrations for existing databases
        try:
            cursor.execute("ALTER TABLE plastic_inspections ADD COLUMN reviewed INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass

        # Dedicated Human Review Table for Plastic Domain
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plastic_human_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id TEXT NOT NULL,
                review_timestamp TEXT NOT NULL,
                reviewer_name TEXT NOT NULL,
                action_type TEXT NOT NULL,
                defect_name TEXT,
                description TEXT,
                validated_zone TEXT,
                validated_severity TEXT,
                recommended_action TEXT,
                FOREIGN KEY (inspection_id) REFERENCES plastic_inspections (inspection_id)
            )
        """)
        conn.commit()

# Ensure table initialized on import
init_plastic_db()

# -----------------------------------------------------------------------------
# 2. Zone-Aware Adaptive Risk Engine (Upgrade 1)
# -----------------------------------------------------------------------------
ZONE_WEIGHTS = {
    "Zone A: Cap / Neck": 1.40,       # Critical closure seal breach, leakage, carbonation loss
    "Zone B: Label / Mid Body": 0.90,  # Aesthetic branding or label alignment
    "Zone C: Lower Body / Base": 1.20, # Structural container stability, tipping, burst pressure
    "None": 0.10
}

def calculate_plastic_risk_score(
    anomaly_score: float,
    defect_zone: str,
    dsar_percent: float,
    threshold: float = 0.3377
) -> Tuple[float, str]:
    """
    Computes an engineering Risk Score (0 - 100) and discrete Risk Level.
    Combines anomaly feature distance, anatomical zone importance, and spatial spread (DSAR %).
    """
    zone_multiplier = ZONE_WEIGHTS.get(defect_zone, 1.0)
    
    if anomaly_score < threshold and dsar_percent < 0.2:
        # Intact bottle baseline risk
        base_risk = (anomaly_score / max(1e-4, threshold)) * 15.0
        risk_score = min(20.0, base_risk)
    else:
        # Defective bottle risk scaling
        severity_ratio = (anomaly_score - threshold) / max(1e-4, threshold)
        severity_component = np.clip(severity_ratio * 40.0, 0.0, 50.0)
        dsar_component = np.clip(dsar_percent * 6.0, 0.0, 40.0)
        
        raw_risk = (15.0 + severity_component + dsar_component) * zone_multiplier
        risk_score = float(np.clip(raw_risk, 0.0, 100.0))

    risk_score = round(risk_score, 1)

    if risk_score <= 25.0:
        risk_level = "Low"
    elif risk_score <= 55.0:
        risk_level = "Moderate"
    elif risk_score <= 80.0:
        risk_level = "High"
    else:
        risk_level = "Critical"

    return risk_score, risk_level

# -----------------------------------------------------------------------------
# 2.5 Indicative AI Assessment Engine (For External / Unseen Plastic Images)
# -----------------------------------------------------------------------------
def compute_plastic_indicative_assessment(
    anomaly_score: float,
    threshold: float,
    input_compatibility_score: float,
    input_compatibility_status: str,
    risk_score: float,
    dsar_percent: float = 0.0,
    defect_zone: str = "None"
) -> Dict[str, Any]:
    """
    Computes a grounded, non-fabricated Indicative AI Assessment (0-100) for external plastic bottles.
    
    Formula for Indicative Quality Score:
      - Q_anomaly (0 - 45 pts):
          If anomaly_score <= threshold:
              Q_anomaly = 45.0 * (1.0 - 0.5 * (anomaly_score / max(1e-4, threshold)))
          Else:
              norm_excess = (anomaly_score - threshold) / max(1e-4, threshold)
              Q_anomaly = max(0.0, 22.5 * (1.0 - min(1.0, norm_excess)))
      - Q_compat (0 - 30 pts):
          Q_compat = 30.0 * (min(100.0, max(0.0, input_compatibility_score)) / 100.0)
      - Q_risk (0 - 25 pts):
          Q_risk = 25.0 * (1.0 - min(100.0, max(0.0, risk_score)) / 100.0)
          
      Indicative Quality Score = round(Q_anomaly + Q_compat + Q_risk, 1)
    """
    comp_score = float(np.clip(input_compatibility_score, 0.0, 100.0))
    is_ood = (comp_score < 25.0) or (input_compatibility_status == "OUTSIDE_DOMAIN")
    
    if is_ood:
        return {
            "is_external": True,
            "input_compatibility_score": comp_score,
            "anomaly_score": round(float(anomaly_score), 4),
            "indicative_quality_score": None,
            "indicative_quality_display": "N/A",
            "indicative_assessment": "NOT RELIABLE",
            "risk_level": "Unvalidated (OOD)",
            "human_review_recommendation": "REQUIRED",
            "is_reliable": False,
            "assessment_banner": "AUTOMATIC QUALITY ASSESSMENT NOT RELIABLE",
            "note": "This image is outside the validated plastic bottle inspection distribution. Automatic defect decision disabled."
        }

    # Anomaly Component (0 - 45 pts)
    norm_score = anomaly_score / max(1e-4, threshold)
    if norm_score <= 1.0:
        q_anomaly = 45.0 * (1.0 - 0.5 * norm_score)
    else:
        norm_excess = norm_score - 1.0
        q_anomaly = max(0.0, 22.5 * (1.0 - min(1.0, norm_excess)))

    # Compatibility Component (0 - 30 pts)
    q_compat = 30.0 * (comp_score / 100.0)

    # Risk Component (0 - 25 pts)
    q_risk = 25.0 * (1.0 - min(100.0, max(0.0, risk_score)) / 100.0)

    indicative_quality = round(float(np.clip(q_anomaly + q_compat + q_risk, 0.0, 100.0)), 1)

    # Broad Assessment Outcome
    if (anomaly_score <= threshold) and (comp_score >= 50.0) and (risk_score <= 25.0) and (dsar_percent < 0.5):
        assessment = "LIKELY GOOD"
        ind_risk = "LOW"
        human_rec = "RECOMMENDED"
    elif (anomaly_score > (threshold * 1.08)) or (anomaly_score > threshold and dsar_percent >= 0.5) or (risk_score > 55.0):
        assessment = "LIKELY DEFECTIVE"
        ind_risk = "HIGH" if risk_score < 80.0 else "CRITICAL"
        human_rec = "REQUIRED"
    else:
        assessment = "BORDERLINE"
        ind_risk = "MODERATE"
        human_rec = "REQUIRED"

    return {
        "is_external": True,
        "input_compatibility_score": comp_score,
        "anomaly_score": round(float(anomaly_score), 4),
        "indicative_quality_score": indicative_quality,
        "indicative_quality_display": f"{int(round(indicative_quality))}%",
        "indicative_assessment": assessment,
        "risk_level": ind_risk,
        "human_review_recommendation": human_rec,
        "is_reliable": True,
        "assessment_banner": f"INDICATIVE AI ASSESSMENT — {assessment}",
        "note": "This is an indicative assessment for an unseen external image and is not part of the official benchmark evaluation."
    }

# -----------------------------------------------------------------------------
# 3. Plastic Bottle Quality Profile Generator (Upgrade 2)
# -----------------------------------------------------------------------------
def generate_plastic_quality_profile(
    anomaly_score: float,
    dsar_percent: float,
    defect_zone: str,
    defect_type: str,
    threshold: float = 0.3377,
    input_compatibility_score: float = 100.0,
    input_compatibility_status: str = "COMPATIBLE"
) -> Dict[str, Any]:
    """
    Synthesizes localized telemetry into standardized quality dimensions:
      - Overall Quality Score (0 - 100)
      - Structural Integrity (0 - 100%)
      - Cap/Neck Integrity (0 - 100%)
      - Body Integrity (0 - 100%)
      - Label Condition (0 - 100%)
      - Surface Condition (0 - 100%)
      - Anomaly Severity (0 - 100%)
    """
    risk_score, risk_level = calculate_plastic_risk_score(anomaly_score, defect_zone, dsar_percent, threshold)
    overall_quality = round(float(np.clip(100.0 - risk_score, 0.0, 100.0)), 1)
    
    indicative_data = compute_plastic_indicative_assessment(
        anomaly_score=anomaly_score,
        threshold=threshold,
        input_compatibility_score=input_compatibility_score,
        input_compatibility_status=input_compatibility_status,
        risk_score=risk_score,
        dsar_percent=dsar_percent,
        defect_zone=defect_zone
    )
    
    # Defaults for intact bottle
    structural_integrity = 99.0
    cap_integrity = 99.0
    body_integrity = 99.0
    label_condition = 99.0
    surface_condition = 99.0

    if anomaly_score >= threshold or dsar_percent > 0.2:
        penalty = min(90.0, dsar_percent * 8.0 + (anomaly_score - threshold) * 80.0)
        
        if "Cap" in defect_zone or "cap_damage" in defect_type.lower():
            cap_integrity = max(10.0, 99.0 - penalty * 1.3)
            structural_integrity = max(40.0, 99.0 - penalty * 0.7)
        elif "Label" in defect_zone or "label_defect" in defect_type.lower():
            label_condition = max(15.0, 99.0 - penalty * 1.4)
            surface_condition = max(40.0, 99.0 - penalty * 0.6)
        elif "deformation" in defect_type.lower() or "dent" in defect_type.lower():
            body_integrity = max(15.0, 99.0 - penalty * 1.2)
            structural_integrity = max(20.0, 99.0 - penalty * 1.3)
        elif "scratch" in defect_type.lower() or "contamination" in defect_type.lower():
            surface_condition = max(15.0, 99.0 - penalty * 1.3)
            body_integrity = max(50.0, 99.0 - penalty * 0.5)
        else:
            # Base zone
            structural_integrity = max(30.0, 99.0 - penalty)
            body_integrity = max(40.0, 99.0 - penalty * 0.8)

    anomaly_severity = round(float(np.clip((anomaly_score / max(1e-4, threshold * 2.0)) * 100.0, 0.0, 100.0)), 1)

    return {
        "overall_quality_score": overall_quality,
        "structural_integrity": round(float(structural_integrity), 1),
        "cap_integrity": round(float(cap_integrity), 1),
        "body_integrity": round(float(body_integrity), 1),
        "label_condition": round(float(label_condition), 1),
        "surface_condition": round(float(surface_condition), 1),
        "anomaly_severity": anomaly_severity,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "indicative_assessment": indicative_data
    }

# -----------------------------------------------------------------------------
# 4. Persistence / Database Logging & Human Review
# -----------------------------------------------------------------------------
def submit_plastic_human_review(
    inspection_id: str,
    reviewer_name: str,
    action_type: str,
    defect_name: str = "",
    description: str = "",
    validated_zone: str = "",
    validated_severity: str = "",
    recommended_action: str = ""
) -> bool:
    """Submits human review feedback for a plastic bottle inspection."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO plastic_human_reviews (
                inspection_id, review_timestamp, reviewer_name, action_type,
                defect_name, description, validated_zone, validated_severity,
                recommended_action
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            inspection_id, timestamp, reviewer_name, action_type,
            defect_name, description, validated_zone, validated_severity,
            recommended_action
        ))
        cursor.execute("UPDATE plastic_inspections SET reviewed = 1 WHERE inspection_id = ?", (inspection_id,))
        conn.commit()
    return True

def log_plastic_inspection_result(
    filename: str,
    category: str,
    anomaly_score: float,
    dsar_percent: float,
    defect_zone: str,
    defect_type: str,
    grade: str,
    grade_reason: str,
    disposal_action: str,
    threshold: float,
    input_compatibility_score: float = 100.0,
    input_compatibility_status: str = "COMPATIBLE",
    inspection_confidence: str = "VALIDATED DOMAIN",
    human_review_required: bool = False,
    domain_warning: str = "",
    batch_id: Optional[str] = None
) -> Dict[str, Any]:
    """Logs a plastic bottle inspection into SQLite plastic_inspections table."""
    inspection_id = f"PL-{uuid.uuid4().hex[:8].upper()}"
    bottle_id = f"PB-{uuid.uuid4().hex[:6].upper()}"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    profile = generate_plastic_quality_profile(
        anomaly_score=anomaly_score,
        dsar_percent=dsar_percent,
        defect_zone=defect_zone,
        defect_type=defect_type,
        threshold=threshold,
        input_compatibility_score=input_compatibility_score,
        input_compatibility_status=input_compatibility_status
    )

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO plastic_inspections (
                inspection_id, timestamp, bottle_id, filename, category,
                anomaly_score, dsar_percent, defect_zone, defect_type,
                grade, grade_reason, disposal_action, threshold,
                risk_score, risk_level, quality_score,
                structural_integrity, cap_integrity, body_integrity,
                label_condition, surface_condition,
                input_compatibility_score, input_compatibility_status,
                inspection_confidence, human_review_required, domain_warning, batch_id, reviewed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            inspection_id, timestamp, bottle_id, filename, category,
            anomaly_score, dsar_percent, defect_zone, defect_type,
            grade, grade_reason, disposal_action, threshold,
            profile["risk_score"], profile["risk_level"], profile["overall_quality_score"],
            profile["structural_integrity"], profile["cap_integrity"], profile["body_integrity"],
            profile["label_condition"], profile["surface_condition"],
            input_compatibility_score, input_compatibility_status,
            inspection_confidence, 1 if human_review_required else 0, domain_warning, batch_id
        ))
        conn.commit()

    profile["inspection_id"] = inspection_id
    profile["bottle_id"] = bottle_id
    profile["timestamp"] = timestamp
    return profile

def get_plastic_inspection_history(
    limit: int = 100,
    grade_filter: Optional[str] = None,
    risk_filter: Optional[str] = None,
    compatibility_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieves filtered historical records from plastic_inspections table."""
    query = "SELECT * FROM plastic_inspections WHERE 1=1"
    params = []

    if grade_filter and grade_filter != "All":
        query += " AND grade LIKE ?"
        params.append(f"%{grade_filter}%")

    if risk_filter and risk_filter != "All":
        query += " AND risk_level = ?"
        params.append(risk_filter)

    if compatibility_filter and compatibility_filter != "All":
        query += " AND input_compatibility_status = ?"
        params.append(compatibility_filter)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

# -----------------------------------------------------------------------------
# 5. Batch Quality Intelligence & Score (BQS) (Upgrade 3)
# -----------------------------------------------------------------------------
def compute_plastic_batch_intelligence(batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes lot-level quality metrics and Batch Quality Score (BQS 0–100) for plastic bottles.
    BQS = 100 - (Defect Rate % * 0.40) - (Critical Defect Rate % * 0.40) - (Avg Risk Score * 0.20)
    """
    total = len(batch_results)
    if total == 0:
        return {"batch_quality_score": 100.0, "total_inspected": 0, "defect_rate": 0.0, "critical_defect_rate": 0.0, "avg_risk_score": 0.0}

    defective_count = sum(1 for r in batch_results if r.get("Verdict") == "DEFECTIVE" or r.get("is_defective", False))
    critical_count = sum(1 for r in batch_results if "Critical" in str(r.get("Grade", "")) or r.get("Risk_Level") == "Critical")
    
    defect_rate = round((defective_count / total) * 100.0, 1)
    critical_defect_rate = round((critical_count / total) * 100.0, 1)
    
    risk_scores = [float(r.get("Risk_Score", r.get("risk_score", 0.0))) for r in batch_results]
    avg_risk = round(float(np.mean(risk_scores)), 1) if risk_scores else 0.0

    bqs = 100.0 - (defect_rate * 0.40) - (critical_defect_rate * 0.40) - (avg_risk * 0.20)
    bqs = round(float(np.clip(bqs, 0.0, 100.0)), 1)

    return {
        "batch_quality_score": bqs,
        "total_inspected": total,
        "passed_count": total - defective_count,
        "defective_count": defective_count,
        "critical_count": critical_count,
        "defect_rate": defect_rate,
        "critical_defect_rate": critical_defect_rate,
        "avg_risk_score": avg_risk
    }

# -----------------------------------------------------------------------------
# 6. Manufacturing Pareto Defect Analysis (Upgrade 4)
# -----------------------------------------------------------------------------
def compute_plastic_pareto_analysis(defect_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes Pareto distribution (80/20 Rule) for plastic bottle defect categories:
    dent, scratch, deformation, cap_damage, label_defect, contamination.
    """
    if not defect_records:
        return {"categories": [], "primary_issue": "None", "vital_few": []}

    categories = [
        str(r.get("Category", r.get("category", r.get("defect_type", "Unknown")))).replace("Defective (", "").replace(")", "").strip()
        for r in defect_records
    ]
    df_cat = pd.Series(categories).value_counts().reset_index()
    df_cat.columns = ["category", "count"]
    
    total_defects = int(df_cat["count"].sum())
    df_cat["percentage"] = (df_cat["count"] / max(1, total_defects) * 100.0).round(1)
    df_cat["cumulative_pct"] = df_cat["percentage"].cumsum().round(1)
    
    primary_issue = df_cat.iloc[0]["category"] if not df_cat.empty else "None"
    vital_few = df_cat[df_cat["cumulative_pct"] <= 85.0]["category"].tolist()
    if not vital_few and not df_cat.empty:
        vital_few = [primary_issue]

    return {
        "total_defects": total_defects,
        "primary_issue": primary_issue,
        "vital_few": vital_few,
        "categories": df_cat.to_dict(orient="records")
    }
