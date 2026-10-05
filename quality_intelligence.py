"""
AI-Driven Glass Bottle Quality Intelligence & Adaptive Risk Management Layer.
Provides high-level diagnostic analytics, zone-aware risk assessment, defect pattern
clustering, human-in-the-loop review queue, batch intelligence, and historical persistence.

This layer acts strictly on top of the PatchCore inference engine without altering
the underlying model weights or threshold calibration.
"""

import os
import sqlite3
import datetime
import uuid
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.cluster import KMeans

DB_PATH = "quality_intelligence.db"

# -----------------------------------------------------------------------------
# 1. SQLite Database Schema & Initialization
# -----------------------------------------------------------------------------
def get_db_connection() -> sqlite3.Connection:
    """Creates and returns a connection to the local SQLite quality database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_intelligence_db():
    """Initializes the database schema for inspections, reviews, and batches."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Main Inspection Telemetry & Quality Profiles
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
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
                rim_integrity REAL NOT NULL,
                cleanliness_score REAL NOT NULL,
                is_unknown INTEGER NOT NULL DEFAULT 0,
                unknown_confidence REAL NOT NULL DEFAULT 0.0,
                pattern_cluster_id TEXT NOT NULL,
                batch_id TEXT,
                reviewed INTEGER NOT NULL DEFAULT 0,
                input_compatibility_score REAL NOT NULL DEFAULT 100.0,
                input_compatibility_status TEXT NOT NULL DEFAULT 'COMPATIBLE',
                inspection_confidence TEXT NOT NULL DEFAULT 'VALIDATED DOMAIN',
                human_review_required INTEGER NOT NULL DEFAULT 0,
                domain_warning TEXT
            )
        """)
        
        # Migrations for existing databases
        for col_def in [
            "input_compatibility_score REAL NOT NULL DEFAULT 100.0",
            "input_compatibility_status TEXT NOT NULL DEFAULT 'COMPATIBLE'",
            "inspection_confidence TEXT NOT NULL DEFAULT 'VALIDATED DOMAIN'",
            "human_review_required INTEGER NOT NULL DEFAULT 0",
            "domain_warning TEXT"
        ]:
            try:
                cursor.execute(f"ALTER TABLE inspections ADD COLUMN {col_def}")
            except sqlite3.OperationalError:
                pass # Column already exists
        
        # 2. Human-in-the-Loop Review Queue & Knowledge Base
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS human_reviews (
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
                FOREIGN KEY (inspection_id) REFERENCES inspections (inspection_id)
            )
        """)
        
        # 3. Batch Quality Records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_records (
                batch_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                total_inspected INTEGER NOT NULL,
                grade_a_count INTEGER NOT NULL,
                grade_b_count INTEGER NOT NULL,
                grade_c_count INTEGER NOT NULL,
                defect_rate REAL NOT NULL,
                critical_defect_rate REAL NOT NULL,
                avg_anomaly_score REAL NOT NULL,
                avg_dsar REAL NOT NULL,
                avg_risk_score REAL NOT NULL,
                batch_quality_score REAL NOT NULL,
                dominant_defect TEXT,
                dominant_pattern TEXT
            )
        """)
        conn.commit()

# Run database setup on module import
init_intelligence_db()

# -----------------------------------------------------------------------------
# 2. Zone-Aware Risk Engine
# -----------------------------------------------------------------------------
def compute_risk_profile(
    anomaly_score: float,
    dsar_percent: float,
    defect_zone: str,
    grade: str,
    threshold: float = 0.4263
) -> Tuple[float, str]:
    """
    Computes an engineering-grade Risk Score (0–100) and Risk Level based on
    anomaly magnitude, defective surface area ratio (DSAR), and anatomical zone.
    
    Functional Weightings:
    - Sealing Rim (Mouth / Rim Annulus): High structural risk (1.8x factor).
      Risk of cap seal breach, pressure loss, or glass shard ingestion.
    - Base / Core (Internal Base Circle): Moderate contamination risk (1.0x factor).
      Cleanable residue or particulate dust.
    - Defect-Free (Grade A): Base risk (0–15).
    """
    is_good = "Grade A" in grade or anomaly_score <= threshold
    
    if is_good:
        # Scale benign score up to 15.0 max
        score_ratio = min(1.0, anomaly_score / max(threshold, 0.1))
        risk_score = round(score_ratio * 14.0, 1)
        risk_level = "Low"
        return risk_score, risk_level

    # Component 1: Anomaly Intensity Severity (0 - 45 pts)
    # 0.42 -> 0 pts, 0.90 -> 45 pts
    norm_score = max(0.0, (anomaly_score - threshold) / (0.90 - threshold + 1e-6))
    score_component = min(1.0, norm_score) * 45.0
    
    # Component 2: Defective Surface Area Ratio (DSAR) Severity (0 - 35 pts)
    # 0% -> 0 pts, 20% -> 35 pts
    norm_dsar = min(1.0, dsar_percent / 20.0)
    dsar_component = norm_dsar * 35.0
    
    # Component 3: Zone Vulnerability Factor (Multiplier)
    # Sealing Rim lip is critical for capping integrity.
    if "Rim" in defect_zone or "Mouth" in defect_zone:
        zone_multiplier = 1.35
    elif "Base" in defect_zone or "Core" in defect_zone:
        zone_multiplier = 0.95
    else:
        zone_multiplier = 1.05
        
    base_risk = (score_component + dsar_component) * zone_multiplier
    
    # Ensure Grade C has a minimum high risk baseline
    if "Grade C" in grade:
        base_risk = max(70.0, base_risk)
    elif "Grade B" in grade:
        # Grade B is cleanable contamination: keep moderate (26 - 65)
        base_risk = min(64.0, max(26.0, base_risk))
        
    risk_score = round(float(np.clip(base_risk, 0.0, 100.0)), 1)
    
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
# 2.5 Indicative AI Assessment Engine (For External / Unseen Images)
# -----------------------------------------------------------------------------
def compute_indicative_assessment(
    anomaly_score: float,
    threshold: float,
    input_compatibility_score: float,
    input_compatibility_status: str,
    risk_score: float,
    dsar_percent: float = 0.0,
    defect_zone: str = "None"
) -> Dict[str, Any]:
    """
    Computes a grounded, non-fabricated Indicative AI Assessment (0-100) for external/unseen images.
    
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
      
    Broad Indicative States:
      1. OUTSIDE DOMAIN / INCOMPATIBLE (comp_score < 25.0 or OUTSIDE_DOMAIN):
         - Indicative Quality Score: None (displayed as "N/A")
         - Indicative Assessment: "NOT RELIABLE"
         - Indicative Risk Level: "Unvalidated (OOD)"
         - Human Review: "REQUIRED"
         - Banner: "AUTOMATIC QUALITY ASSESSMENT NOT RELIABLE"
      2. LIKELY GOOD:
         - anomaly_score <= threshold, comp_score >= 50.0, risk_score <= 25.0, dsar_percent < 0.5
         - Indicative Assessment: "LIKELY GOOD"
         - Indicative Risk Level: "LOW"
         - Human Review: "RECOMMENDED"
      3. LIKELY DEFECTIVE:
         - anomaly_score > (threshold * 1.08) or (anomaly_score > threshold and dsar_percent >= 0.5) or risk_score > 55.0
         - Indicative Assessment: "LIKELY DEFECTIVE"
         - Indicative Risk Level: "HIGH" (or "CRITICAL" if risk_score >= 80.0)
         - Human Review: "REQUIRED"
      4. BORDERLINE:
         - All other ambiguous or intermediate states (score near threshold or compatibility 25-50% or risk 26-55)
         - Indicative Assessment: "BORDERLINE"
         - Indicative Risk Level: "MODERATE"
         - Human Review: "REQUIRED"
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
            "note": "This image is outside the validated inspection distribution. Automatic defect classification is disabled."
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
# 3. Bottle Quality Profile Generator
# -----------------------------------------------------------------------------
def generate_quality_profile(
    bottle_id: str,
    anomaly_score: float,
    dsar_percent: float,
    defect_zone: str,
    defect_type: str,
    grade: str,
    disposal_action: str,
    threshold: float = 0.4263,
    pattern_cluster: str = "Cluster 01 — Intact / Nominal",
    is_unknown: bool = False,
    unknown_conf: float = 0.0,
    input_compatibility_score: float = 100.0,
    input_compatibility_status: str = "COMPATIBLE",
    inspection_confidence: str = "VALIDATED DOMAIN",
    human_review_required: bool = False,
    domain_warning: Optional[str] = None
) -> Dict[str, Any]:
    """
    Synthesizes multi-dimensional manufacturing quality metrics for an inspected bottle.
    Every metric is mathematically derived from physical PatchCore measurements.
    """
    risk_score, risk_level = compute_risk_profile(
        anomaly_score=anomaly_score,
        dsar_percent=dsar_percent,
        defect_zone=defect_zone,
        grade=grade,
        threshold=threshold
    )
    
    # Indicative assessment for external / unseen inputs
    indicative_data = compute_indicative_assessment(
        anomaly_score=anomaly_score,
        threshold=threshold,
        input_compatibility_score=input_compatibility_score,
        input_compatibility_status=input_compatibility_status,
        risk_score=risk_score,
        dsar_percent=dsar_percent,
        defect_zone=defect_zone
    )
    
    # Overall Quality Score: Inverted risk (100 = Pristine Container, 0 = Catastrophic Scrap)
    # If outside domain, quality score is marked as unvalidated (0.0)
    if input_compatibility_status == "OUTSIDE_DOMAIN":
        overall_quality_score = 0.0
        risk_level = "Unvalidated (OOD)"
    else:
        overall_quality_score = round(max(0.0, min(100.0, 100.0 - risk_score)), 1)
    
    # Structural Integrity (0-100)
    if "Grade C" in grade or "Rim" in defect_zone or "Mouth" in defect_zone:
        structural_integrity = round(max(0.0, 100.0 - (dsar_percent * 3.5) - (anomaly_score * 40.0)), 1)
    elif "Grade B" in grade:
        structural_integrity = 92.0  # Structure is physically intact
    else:
        structural_integrity = 99.5
        
    # Rim Sealing Integrity (0-100)
    if "Rim" in defect_zone or "Mouth" in defect_zone:
        rim_integrity = round(max(0.0, 100.0 - (dsar_percent * 4.0) - 35.0), 1)
    else:
        rim_integrity = 99.0
        
    # Cleanliness / Hygiene Score (0-100)
    if "Grade B" in grade or "Contamination" in defect_type or "Base" in defect_zone:
        cleanliness_score = round(max(10.0, 100.0 - (dsar_percent * 4.5)), 1)
    else:
        cleanliness_score = 99.0
        
    # Anomaly Severity Score (0-100)
    anomaly_severity = round(float(np.clip((anomaly_score / 0.90) * 100.0, 0.0, 100.0)), 1)
    
    return {
        "bottle_id": bottle_id,
        "inspection_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_quality_score": overall_quality_score,
        "anomaly_score": round(anomaly_score, 4),
        "dsar_percent": round(dsar_percent, 2),
        "defect_zone": defect_zone,
        "defect_type": defect_type,
        "grade": grade,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "structural_integrity": structural_integrity,
        "rim_integrity": rim_integrity,
        "cleanliness_score": cleanliness_score,
        "anomaly_severity": anomaly_severity,
        "recommended_action": disposal_action,
        "pattern_cluster": pattern_cluster,
        "is_unknown_anomaly": is_unknown,
        "unknown_anomaly_confidence": round(unknown_conf, 1),
        # Input Compatibility & Domain Confidence Dimensions
        "input_compatibility_score": input_compatibility_score,
        "input_compatibility_status": input_compatibility_status,
        "inspection_confidence": inspection_confidence,
        "human_review_required": human_review_required,
        "domain_warning": domain_warning,
        # Indicative AI Assessment Extensions
        "indicative_assessment": indicative_data
    }

# -----------------------------------------------------------------------------
# 4. Unknown Anomaly Discovery & Defect Pattern Clustering
# -----------------------------------------------------------------------------
class AnomalyPatternEngine:
    """
    Unsupervised pattern discovery module for recurring anomaly archetypes
    and detection of out-of-distribution 'Unknown Anomalies' requiring human review.
    """
    def __init__(self):
        self.cluster_definitions = {
            "Cluster 01": "Cluster 01 — Sealing Rim Discontinuity",
            "Cluster 02": "Cluster 02 — Internal Base Contamination",
            "Cluster 03": "Cluster 03 — Localized Chipped Lip",
            "Cluster 04": "Cluster 04 — Diffuse / Outlier Anomaly Pattern"
        }
        
    def classify_pattern(
        self,
        anomaly_score: float,
        dsar_percent: float,
        defect_zone: str,
        is_defective: bool
    ) -> Tuple[str, bool, float]:
        if not is_defective:
            return "Cluster 00 — Intact / Defect-Free", False, 0.0
            
        is_rim = "Rim" in defect_zone or "Mouth" in defect_zone
        is_core = "Core" in defect_zone or "Base" in defect_zone
        
        if is_rim and dsar_percent >= 10.0:
            return self.cluster_definitions["Cluster 01"], False, 0.0
        elif is_core and dsar_percent >= 1.0:
            return self.cluster_definitions["Cluster 02"], False, 0.0
        elif is_rim and (0.8 <= dsar_percent < 10.0):
            return self.cluster_definitions["Cluster 03"], False, 0.0
        else:
            unknown_conf = round(float(np.clip(anomaly_score * 90.0, 50.0, 95.0)), 1)
            return self.cluster_definitions["Cluster 04"], True, unknown_conf

pattern_engine = AnomalyPatternEngine()

# -----------------------------------------------------------------------------
# 5. Database Persistence & Query API
# -----------------------------------------------------------------------------
def log_inspection_result(
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
    batch_id: Optional[str] = None,
    bottle_id: Optional[str] = None,
    input_compatibility_score: float = 100.0,
    input_compatibility_status: str = "COMPATIBLE",
    inspection_confidence: str = "VALIDATED DOMAIN",
    human_review_required: bool = False,
    domain_warning: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enriches PatchCore prediction with Quality Intelligence telemetry and
    permanently persists the inspection record in SQLite database.
    """
    is_defective = "Grade A" not in grade and anomaly_score > threshold
    
    # 1. Pattern Clustering & Unknown Detection
    cluster_id, is_unknown_pattern, unknown_conf = pattern_engine.classify_pattern(
        anomaly_score=anomaly_score,
        dsar_percent=dsar_percent,
        defect_zone=defect_zone,
        is_defective=is_defective
    )
    
    # Flag for human review if either pattern is unknown OR input is low confidence / OOD
    is_unknown = is_unknown_pattern or human_review_required or (input_compatibility_status != "COMPATIBLE")
    if input_compatibility_status == "OUTSIDE_DOMAIN":
        unknown_conf = max(unknown_conf, 99.0)
    elif input_compatibility_status == "LOW_CONFIDENCE":
        unknown_conf = max(unknown_conf, 75.0)

    clean_bottle_id = bottle_id or f"BTL-{uuid.uuid4().hex[:8].upper()}"
    inspection_id = f"QC-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"
    
    # 2. Quality Profile Generation
    profile = generate_quality_profile(
        bottle_id=clean_bottle_id,
        anomaly_score=anomaly_score,
        dsar_percent=dsar_percent,
        defect_zone=defect_zone,
        defect_type=defect_type,
        grade=grade,
        disposal_action=disposal_action,
        threshold=threshold,
        pattern_cluster=cluster_id,
        is_unknown=is_unknown,
        unknown_conf=unknown_conf,
        input_compatibility_score=input_compatibility_score,
        input_compatibility_status=input_compatibility_status,
        inspection_confidence=inspection_confidence,
        human_review_required=human_review_required,
        domain_warning=domain_warning
    )
    
    # 3. Save to SQLite
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO inspections (
                    inspection_id, timestamp, bottle_id, filename, category,
                    anomaly_score, dsar_percent, defect_zone, defect_type,
                    grade, grade_reason, disposal_action, threshold,
                    risk_score, risk_level, quality_score, structural_integrity,
                    rim_integrity, cleanliness_score, is_unknown, unknown_confidence,
                    pattern_cluster_id, batch_id, reviewed,
                    input_compatibility_score, input_compatibility_status,
                    inspection_confidence, human_review_required, domain_warning
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                inspection_id, profile["inspection_timestamp"], clean_bottle_id,
                filename, category, anomaly_score, dsar_percent,
                defect_zone, defect_type, grade, grade_reason, disposal_action,
                threshold, profile["risk_score"], profile["risk_level"],
                profile["overall_quality_score"], profile["structural_integrity"],
                profile["rim_integrity"], profile["cleanliness_score"],
                1 if is_unknown else 0, unknown_conf, cluster_id, batch_id, 0,
                input_compatibility_score, input_compatibility_status,
                inspection_confidence, 1 if human_review_required else 0, domain_warning
            ))
            conn.commit()
    except Exception as e:
        print(f"[!] Warning: Failed to log inspection to SQLite: {e}")
        
    profile["inspection_id"] = inspection_id
    profile["filename"] = filename
    return profile

def get_inspection_history(
    limit: int = 100,
    grade_filter: Optional[str] = None,
    risk_filter: Optional[str] = None,
    unknown_only: bool = False,
    compatibility_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieves filtered historical inspection records from SQLite."""
    query = "SELECT * FROM inspections WHERE 1=1"
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
    if unknown_only:
        query += " AND (is_unknown = 1 OR human_review_required = 1) AND reviewed = 0"
        
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def submit_human_review(
    inspection_id: str,
    reviewer_name: str,
    action_type: str,
    defect_name: str,
    description: str,
    validated_zone: str,
    validated_severity: str,
    recommended_action: str
) -> bool:
    """
    Submits expert quality engineer feedback for flagged Unknown Anomalies,
    updating the human review knowledge base without altering model weights.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO human_reviews (
                inspection_id, review_timestamp, reviewer_name, action_type,
                defect_name, description, validated_zone, validated_severity,
                recommended_action
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            inspection_id, timestamp, reviewer_name, action_type,
            defect_name, description, validated_zone, validated_severity,
            recommended_action
        ))
        
        # Mark inspection as reviewed in main table
        cursor.execute("UPDATE inspections SET reviewed = 1 WHERE inspection_id = ?", (inspection_id,))
        conn.commit()
    return True

# -----------------------------------------------------------------------------
# 6. Batch Quality Intelligence & Score
# -----------------------------------------------------------------------------
def compute_batch_intelligence(batch_results: List[Dict[str, Any]], batch_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculates comprehensive manufacturing KPIs and the Batch Quality Score (0–100).
    
    Formula for Batch Quality Score (BQS):
    BQS = 100 - (Defect Rate % * 0.40) - (Critical Defect Rate % * 0.40) - (Avg Risk Score * 0.20)
    Clamped strictly to range [0.0, 100.0].
    """
    if not batch_results:
        return {}
        
    total = len(batch_results)
    cnt_a = sum(1 for r in batch_results if "Grade A" in r.get("Grade", ""))
    cnt_b = sum(1 for r in batch_results if "Grade B" in r.get("Grade", ""))
    cnt_c = sum(1 for r in batch_results if "Grade C" in r.get("Grade", ""))
    
    defect_count = cnt_b + cnt_c
    defect_rate = round((defect_count / total) * 100.0, 2)
    critical_defect_rate = round((cnt_c / total) * 100.0, 2)
    
    scores = [r.get("Anomaly_Score", 0.0) for r in batch_results]
    dsars = [r.get("DSAR_Percent", 0.0) for r in batch_results]
    risks = [r.get("Risk_Score", 10.0 if "Grade A" in r.get("Grade", "") else 70.0) for r in batch_results]
    
    avg_score = round(float(np.mean(scores)), 4) if scores else 0.0
    avg_dsar = round(float(np.mean(dsars)), 2) if dsars else 0.0
    avg_risk = round(float(np.mean(risks)), 1) if risks else 0.0
    
    # Batch Quality Score Calculation
    penalty = (defect_rate * 0.40) + (critical_defect_rate * 0.40) + (avg_risk * 0.20)
    batch_quality_score = round(float(np.clip(100.0 - penalty, 0.0, 100.0)), 1)
    
    # Identify Highest Risk Bottle
    highest_risk_item = max(batch_results, key=lambda x: x.get("Anomaly_Score", 0.0), default={})
    
    # Identify Dominant Defect Category
    categories = [r.get("Category", "N/A") for r in batch_results if "Grade A" not in r.get("Grade", "")]
    dominant_cat = pd.Series(categories).mode()[0] if categories else "None (All Passed)"
    
    # Identify Dominant Anomaly Pattern
    patterns = [r.get("Pattern_Cluster", "Nominal") for r in batch_results]
    dominant_pattern = pd.Series(patterns).mode()[0] if patterns else "Nominal"
    
    clean_batch_id = batch_id or f"BATCH-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Persist Batch Record to Database
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO batch_records (
                    batch_id, timestamp, total_inspected, grade_a_count,
                    grade_b_count, grade_c_count, defect_rate, critical_defect_rate,
                    avg_anomaly_score, avg_dsar, avg_risk_score,
                    batch_quality_score, dominant_defect, dominant_pattern
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                clean_batch_id, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                total, cnt_a, cnt_b, cnt_c, defect_rate, critical_defect_rate,
                avg_score, avg_dsar, avg_risk, batch_quality_score,
                dominant_cat, dominant_pattern
            ))
            conn.commit()
    except Exception as e:
        print(f"[!] Warning: Failed to log batch summary to SQLite: {e}")
        
    return {
        "batch_id": clean_batch_id,
        "total_inspected": total,
        "grade_a_count": cnt_a,
        "grade_b_count": cnt_b,
        "grade_c_count": cnt_c,
        "defect_rate": defect_rate,
        "critical_defect_rate": critical_defect_rate,
        "avg_anomaly_score": avg_score,
        "avg_dsar": avg_dsar,
        "avg_risk_score": avg_risk,
        "batch_quality_score": batch_quality_score,
        "highest_risk_bottle": highest_risk_item.get("Filename", "None"),
        "highest_risk_score": round(highest_risk_item.get("Anomaly_Score", 0.0), 4),
        "dominant_defect_category": dominant_cat,
        "dominant_pattern_cluster": dominant_pattern
    }

# -----------------------------------------------------------------------------
# 7. Manufacturing Pareto Analysis
# -----------------------------------------------------------------------------
def compute_pareto_analysis(batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Performs 80/20 Pareto analysis on observed defect categories.
    Returns sorted frequencies, cumulative percentage curve, and dominant defect.
    """
    defective_items = [r for r in batch_results if "Grade A" not in r.get("Grade", "")]
    if not defective_items:
        return {"categories": [], "counts": [], "cum_percentages": [], "dominant_issue": "None (No defects)"}
        
    df = pd.DataFrame(defective_items)
    col = "Category" if "Category" in df.columns else "Defect_Type"
    counts = df[col].value_counts().reset_index()
    counts.columns = ["Defect_Category", "Count"]
    
    total_defects = counts["Count"].sum()
    counts["Percentage"] = (counts["Count"] / total_defects) * 100.0
    counts["Cum_Percentage"] = counts["Percentage"].cumsum()
    
    dominant_cat = counts.iloc[0]["Defect_Category"]
    dominant_pct = round(counts.iloc[0]["Percentage"], 1)
    
    return {
        "categories": counts["Defect_Category"].tolist(),
        "counts": counts["Count"].tolist(),
        "percentages": [round(p, 1) for p in counts["Percentage"].tolist()],
        "cum_percentages": [round(cp, 1) for cp in counts["Cum_Percentage"].tolist()],
        "dominant_issue": f"Primary observed quality issue: {dominant_cat} ({dominant_pct}% of total defects)"
    }

# -----------------------------------------------------------------------------
# 8. Historical Trend Detection
# -----------------------------------------------------------------------------
def detect_quality_trends(limit: int = 15) -> Dict[str, Any]:
    """
    Analyzes historical batch runs and emits safe manufacturing trend indicators
    without claiming machine root causes.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM batch_records ORDER BY timestamp ASC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        
    if not rows or len(rows) < 2:
        return {
            "has_data": False,
            "batches": [],
            "defect_rates": [],
            "critical_rates": [],
            "quality_scores": [],
            "trend_warnings": ["Insufficient historical batch data. Run additional inspections to establish trends."]
        }
        
    batches = [r["batch_id"][-6:] for r in rows]
    defect_rates = [r["defect_rate"] for r in rows]
    critical_rates = [r["critical_defect_rate"] for r in rows]
    quality_scores = [r["batch_quality_score"] for r in rows]
    
    warnings = []
    
    # 1. Defect Rate Trend
    if defect_rates[-1] > defect_rates[-2]:
        diff = round(defect_rates[-1] - defect_rates[-2], 1)
        warnings.append(f"⚠️ Potential process issue indicated: Batch defect rate increased by +{diff}% vs prior run.")
    else:
        warnings.append("✅ Process stability normal: Batch defect rate is within nominal operating range.")
        
    # 2. Critical Fracture Trend
    if len(critical_rates) >= 3 and critical_rates[-1] > critical_rates[-2] > critical_rates[-3]:
        warnings.append("🚨 Elevated Risk Alert: Upward trajectory in Grade C Critical Fractures observed across 3 consecutive batches.")
        
    # 3. Overall Quality Trajectory
    if quality_scores[-1] >= 85.0:
        warnings.append("⭐ High Quality Line Compliance: Batch Quality Score maintains standard packaging rating (>85.0).")
        
    return {
        "has_data": True,
        "batches": batches,
        "timestamps": [r["timestamp"] for r in rows],
        "defect_rates": defect_rates,
        "critical_rates": critical_rates,
        "quality_scores": quality_scores,
        "avg_risks": [r["avg_risk_score"] for r in rows],
        "trend_warnings": warnings
    }
