"""
FastAPI Backend API Server for Visual Defect Detection System (MVTec Bottle Dataset)
Provides RESTful Endpoints for inference, heatmaps, dataset exploration, and model benchmarking.
"""

from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import io
import json
import glob
import base64
import numpy as np
import pandas as pd
import uuid
from PIL import Image
import cv2
from defect_model import BottleDefectDetector
from pdf_generator import generate_qa_pdf_report
import quality_intelligence as qi

from plastic_defect_model import PlasticBottleDetector
from plastic_pdf_generator import generate_plastic_qa_pdf_report
import plastic_quality_intelligence as plastic_qi

app = FastAPI(
    title="Deep Learning-Based Anomaly Detection and Quality Triage Web Portal for Glass Bottle Inspection",
    description="Decoupled Backend REST API for PatchCore Anomaly Detection on MVTec Glass Bottle Dataset",
    version="1.0.0"
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "bottle_defect_model.pkl"
DATASET_ROOT = os.path.join("mvtec_bottle_dataset", "bottle")
TEST_DIR = os.path.join(DATASET_ROOT, "test")

PLASTIC_MODEL_PATH = os.path.join("plastic_bottle_model", "plastic_defect_model.pkl")
PLASTIC_DATASET_ROOT = "plastic_bottle_dataset"
PLASTIC_TEST_DIR = os.path.join(PLASTIC_DATASET_ROOT, "test")

# Global Detector Instances (Separated Memory Banks & Weights)
detector = BottleDefectDetector(device='cpu', backbone='resnet18', image_size=256)
plastic_detector = PlasticBottleDetector(device='cpu', image_size=256)

@app.on_event("startup")
def startup_event():
    """Load model checkpoints for both inspection domains on server launch."""
    if os.path.exists(MODEL_PATH):
        detector.load_model(MODEL_PATH)
        print(f"[+] Loaded Glass Bottle model checkpoint from '{MODEL_PATH}'")
    else:
        print(f"[!] Warning: Glass bottle checkpoint '{MODEL_PATH}' not found.")
        
    if os.path.exists(PLASTIC_MODEL_PATH):
        plastic_detector.load_model(PLASTIC_MODEL_PATH)
        print(f"[+] Loaded Plastic Bottle model checkpoint from '{PLASTIC_MODEL_PATH}'")
    else:
        print(f"[!] Warning: Plastic bottle checkpoint '{PLASTIC_MODEL_PATH}' not found.")

def np_to_base64(img_np: np.ndarray) -> str:
    """Converts RGB/BGR numpy image array to base64 PNG string."""
    pil_img = Image.fromarray(img_np)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

@app.get("/")
def read_root():
    """Serves the main web dashboard if index.html exists."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Visual Defect Detection API is Online. Visit /docs for API documentation."}

@app.get("/style.css")
def read_style():
    """Serves CSS stylesheet."""
    if os.path.exists("style.css"):
        return FileResponse("style.css", media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found")

@app.get("/dashboard.js")
def read_dashboard_js():
    """Serves JavaScript logic."""
    if os.path.exists("dashboard.js"):
        return FileResponse("dashboard.js", media_type="application/javascript")
    raise HTTPException(status_code=404, detail="dashboard.js not found")


@app.get("/health")
def health_check():
    """Healthcheck endpoint returning server & dual domain model status."""
    return {
        "status": "online",
        "model_loaded": detector.memory_bank is not None,
        "coreset_size": detector.memory_bank.shape[0] if detector.memory_bank is not None else 0,
        "threshold": detector.threshold if detector.memory_bank is not None else 0.4263,
        "domains": {
            "glass_bottle": {
                "name": "Glass Bottle Inspection (Top-View)",
                "loaded": detector.memory_bank is not None,
                "coreset_size": detector.memory_bank.shape[0] if detector.memory_bank is not None else 0,
                "threshold": round(detector.threshold, 4) if detector.memory_bank is not None else 0.4263,
                "dataset": "MVTec AD Bottle"
            },
            "plastic_bottle": {
                "name": "Full-Body Plastic Bottle Inspection (Front-View)",
                "loaded": plastic_detector.memory_bank is not None,
                "coreset_size": len(plastic_detector.memory_bank) if plastic_detector.memory_bank is not None else 0,
                "threshold": round(plastic_detector.threshold, 4) if plastic_detector.memory_bank is not None else 0.3309,
                "model_version": plastic_detector.model_version,
                "dataset": "PB-Dataset-v2.0"
            }
        }
    }

@app.post("/plastic/reload")
def reload_plastic_model():
    """Reloads the plastic bottle model from disk."""
    if os.path.exists(PLASTIC_MODEL_PATH):
        plastic_detector.load_model(PLASTIC_MODEL_PATH)
        return {
            "status": "reloaded",
            "threshold": plastic_detector.threshold,
            "coreset_size": len(plastic_detector.memory_bank) if plastic_detector.memory_bank is not None else 0
        }
    raise HTTPException(status_code=404, detail="Plastic bottle model file not found")

def base64_to_pil(b64_str: str) -> Image.Image:
    """Decodes base64 string to PIL Image."""
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert('RGB')

class PDFReportRequest(BaseModel):
    filename: str = "bottle_inspection.png"
    verdict: str = "GOOD"
    grade: str = "Grade A — Good"
    grade_reason: str = ""
    dsar_percent: float = 0.0
    anomaly_score: float = 0.0
    defect_zone: str = "None"
    defect_type: str = "N/A"
    disposal_action: str = "Approved for Packaging"
    original_base64: str = ""
    heatmap_base64: str = ""
    polar_base64: str = ""
    risk_score: float = 0.0
    risk_level: str = "Low"
    quality_score: float = 100.0
    pattern_cluster: str = "Cluster 00 — Intact / Defect-Free"
    is_unknown: bool = False
    input_compatibility_score: float = 100.0
    input_compatibility_status: str = "COMPATIBLE"
    inspection_confidence: str = "VALIDATED DOMAIN"
    human_review_required: bool = False
    domain_warning: str = ""

class HumanReviewRequest(BaseModel):
    inspection_id: str
    reviewer_name: str = "Quality Engineer"
    action_type: str = "CONFIRMED_GOOD"  # "CONFIRMED_GOOD", "CONFIRMED_DEFECTIVE", "SEND_FOR_REINSPECTION"
    defect_name: str = ""
    description: str = ""
    validated_zone: str = ""
    validated_severity: str = ""
    recommended_action: str = ""

@app.post("/submit_review")
def submit_glass_review_endpoint(req: HumanReviewRequest):
    """Saves human review feedback for a Glass Bottle inspection."""
    success = qi.submit_human_review(
        inspection_id=req.inspection_id,
        reviewer_name=req.reviewer_name,
        action_type=req.action_type,
        defect_name=req.defect_name,
        description=req.description,
        validated_zone=req.validated_zone,
        validated_severity=req.validated_severity,
        recommended_action=req.recommended_action
    )
    return {"status": "success" if success else "failed", "inspection_id": req.inspection_id}

@app.post("/plastic/submit_review")
def submit_plastic_review_endpoint(req: HumanReviewRequest):
    """Saves human review feedback for a Plastic Bottle inspection."""
    success = plastic_qi.submit_plastic_human_review(
        inspection_id=req.inspection_id,
        reviewer_name=req.reviewer_name,
        action_type=req.action_type,
        defect_name=req.defect_name,
        description=req.description,
        validated_zone=req.validated_zone,
        validated_severity=req.validated_severity,
        recommended_action=req.recommended_action
    )
    return {"status": "success" if success else "failed", "inspection_id": req.inspection_id}

@app.post("/generate_pdf")
def generate_pdf_endpoint(data: PDFReportRequest):
    """Generates and returns downloadable QA inspection PDF certificate."""
    try:
        orig_pil = base64_to_pil(data.original_base64) if data.original_base64 else Image.new('RGB', (256, 256), color='gray')
        heat_pil = base64_to_pil(data.heatmap_base64) if data.heatmap_base64 else Image.new('RGB', (256, 256), color='gray')
        polar_pil = base64_to_pil(data.polar_base64) if data.polar_base64 else Image.new('RGB', (256, 256), color='gray')

        pdf_bytes = generate_qa_pdf_report(
            filename=data.filename,
            verdict=data.verdict,
            grade=data.grade,
            dsar_percent=data.dsar_percent,
            anomaly_score=data.anomaly_score,
            defect_zone=data.defect_zone,
            defect_type=data.defect_type,
            disposal_action=data.disposal_action,
            orig_img_pil=orig_pil,
            heatmap_img_pil=heat_pil,
            polar_img_pil=polar_pil,
            grade_reason=data.grade_reason,
            risk_score=data.risk_score,
            risk_level=data.risk_level,
            quality_score=data.quality_score,
            pattern_cluster=data.pattern_cluster,
            is_unknown=data.is_unknown,
            input_compatibility_score=data.input_compatibility_score,
            input_compatibility_status=data.input_compatibility_status,
            inspection_confidence=data.inspection_confidence,
            human_review_required=data.human_review_required,
            domain_warning=data.domain_warning
        )
        safe_fname = os.path.splitext(os.path.basename(data.filename))[0]
        
        # Save local copy in QA_Inspection_Reports folder
        reports_dir = "QA_Inspection_Reports"
        os.makedirs(reports_dir, exist_ok=True)
        local_report_path = os.path.join(reports_dir, f"QA_Certificate_{safe_fname}.pdf")
        with open(local_report_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"[+] Saved QA Certificate to '{local_report_path}'", flush=True)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=QA_Certificate_{safe_fname}.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Generation failed: {str(e)}")


@app.post("/predict")
async def predict_image_file(file: UploadFile = File(...), threshold: Optional[float] = Query(default=None)):
    """
    Receives an uploaded bottle image file and returns defect score, verdict, heatmap, polar unwrap ribbon, DSAR %, and triage actions.
    Supports dynamic operational threshold overrides and logs Quality Intelligence metrics.
    """
    if detector.memory_bank is None:
        raise HTTPException(status_code=500, detail="Model is not trained or loaded.")
        
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        res = detector.predict(pil_img, custom_threshold=threshold, filename=file.filename)
        active_thresh = float(res.get("threshold", detector.threshold))
        
        # Quality Intelligence & Risk Assessment Layer
        qi_profile = qi.log_inspection_result(
            filename=file.filename,
            category=res.get("defect_type", "Standard Inspection"),
            anomaly_score=float(res["anomaly_score"]),
            dsar_percent=float(res["dsar_percent"]),
            defect_zone=res["defect_zone"],
            defect_type=res["defect_type"],
            grade=res["grade"],
            grade_reason=res.get("grade_reason", ""),
            disposal_action=res["disposal_action"],
            threshold=active_thresh,
            input_compatibility_score=res.get("input_compatibility_score", 100.0),
            input_compatibility_status=res.get("input_compatibility_status", "COMPATIBLE"),
            inspection_confidence=res.get("inspection_confidence", "VALIDATED DOMAIN"),
            human_review_required=res.get("human_review_required", False),
            domain_warning=res.get("domain_warning")
        )
        
        heatmap_b64 = np_to_base64(res["heatmap_overlay"])
        polar_b64 = np_to_base64(res["polar_ribbon"])
        orig_b64 = np_to_base64(res["original_image"])
        
        return {
            "filename": file.filename,
            "anomaly_score": round(res["anomaly_score"], 4),
            "is_defective": res["is_defective"],
            "verdict": res["verdict"],
            "grade": res["grade"],
            "grade_reason": res.get("grade_reason", ""),
            "dsar_percent": round(res["dsar_percent"], 2),
            "defect_zone": res["defect_zone"],
            "defect_type": res["defect_type"],
            "disposal_action": res["disposal_action"],
            "threshold": round(active_thresh, 4),
            "heatmap_base64": heatmap_b64,
            "polar_base64": polar_b64,
            "original_base64": orig_b64,
            # Input Compatibility & Uncertainty Extensions
            "input_compatibility_score": res.get("input_compatibility_score", 100.0),
            "input_compatibility_status": res.get("input_compatibility_status", "COMPATIBLE"),
            "inspection_confidence": res.get("inspection_confidence", "VALIDATED DOMAIN"),
            "inspection_mode": res.get("inspection_mode", "AUTOMATIC"),
            "human_review_required": res.get("human_review_required", False),
            "domain_warning": res.get("domain_warning"),
            # Quality Intelligence Extensions
            "risk_score": qi_profile["risk_score"],
            "risk_level": qi_profile["risk_level"],
            "quality_score": qi_profile["overall_quality_score"],
            "pattern_cluster": qi_profile["pattern_cluster"],
            "is_unknown_anomaly": qi_profile["is_unknown_anomaly"],
            "unknown_anomaly_confidence": qi_profile["unknown_anomaly_confidence"],
            "inspection_id": qi_profile["inspection_id"],
            "bottle_id": qi_profile["bottle_id"],
            "quality_profile": qi_profile,
            "is_external": True,
            "indicative_assessment": qi_profile.get("indicative_assessment", {})
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

@app.get("/predict_sample")
def predict_sample_image(category: str = Query(...), filename: str = Query(...), threshold: Optional[float] = Query(default=None)):
    """
    Runs prediction on a dataset sample file path with optional dynamic sensitivity threshold override.
    Enriches with Quality Intelligence layer and logs to SQLite database.
    """
    if detector.memory_bank is None:
        raise HTTPException(status_code=500, detail="Model is not trained or loaded.")
        
    img_path = os.path.join(TEST_DIR, category, filename)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail=f"Sample file '{img_path}' not found.")
        
    res = detector.predict(img_path, custom_threshold=threshold, filename=f"{category}_{filename}")
    active_thresh = float(res.get("threshold", detector.threshold))
    
    # Quality Intelligence & Risk Assessment Layer
    qi_profile = qi.log_inspection_result(
        filename=filename,
        category=category,
        anomaly_score=float(res["anomaly_score"]),
        dsar_percent=float(res["dsar_percent"]),
        defect_zone=res["defect_zone"],
        defect_type=res["defect_type"],
        grade=res["grade"],
        grade_reason=res.get("grade_reason", ""),
        disposal_action=res["disposal_action"],
        threshold=active_thresh,
        input_compatibility_score=res.get("input_compatibility_score", 100.0),
        input_compatibility_status=res.get("input_compatibility_status", "COMPATIBLE"),
        inspection_confidence=res.get("inspection_confidence", "VALIDATED DOMAIN"),
        human_review_required=res.get("human_review_required", False),
        domain_warning=res.get("domain_warning")
    )
    
    heatmap_b64 = np_to_base64(res["heatmap_overlay"])
    polar_b64 = np_to_base64(res["polar_ribbon"])
    orig_b64 = np_to_base64(res["original_image"])
    
    return {
        "category": category,
        "filename": filename,
        "anomaly_score": round(res["anomaly_score"], 4),
        "is_defective": res["is_defective"],
        "verdict": res["verdict"],
        "grade": res["grade"],
        "grade_reason": res.get("grade_reason", ""),
        "dsar_percent": round(res["dsar_percent"], 2),
        "defect_zone": res["defect_zone"],
        "defect_type": res["defect_type"],
        "disposal_action": res["disposal_action"],
        "threshold": round(active_thresh, 4),
        "heatmap_base64": heatmap_b64,
        "polar_base64": polar_b64,
        "original_base64": orig_b64,
        # Input Compatibility & Uncertainty Extensions
        "input_compatibility_score": res.get("input_compatibility_score", 100.0),
        "input_compatibility_status": res.get("input_compatibility_status", "COMPATIBLE"),
        "inspection_confidence": res.get("inspection_confidence", "VALIDATED DOMAIN"),
        "inspection_mode": res.get("inspection_mode", "AUTOMATIC"),
        "human_review_required": res.get("human_review_required", False),
        "domain_warning": res.get("domain_warning"),
        # Quality Intelligence Extensions
        "risk_score": qi_profile["risk_score"],
        "risk_level": qi_profile["risk_level"],
        "quality_score": qi_profile["overall_quality_score"],
        "pattern_cluster": qi_profile["pattern_cluster"],
        "is_unknown_anomaly": qi_profile["is_unknown_anomaly"],
        "unknown_anomaly_confidence": qi_profile["unknown_anomaly_confidence"],
        "inspection_id": qi_profile["inspection_id"],
        "bottle_id": qi_profile["bottle_id"],
        "quality_profile": qi_profile,
        "is_external": False,
        "indicative_assessment": qi_profile.get("indicative_assessment", {})
    }

@app.get("/categories")
def list_categories():
    """Lists available test categories and files."""
    if not os.path.exists(TEST_DIR):
        return {"categories": {}}
        
    categories = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    result = {}
    for cat in categories:
        cat_dir = os.path.join(TEST_DIR, cat)
        files = [os.path.basename(f) for f in glob.glob(os.path.join(cat_dir, "*")) if f.endswith(('.png', '.jpg'))]
        result[cat] = sorted(files)
        
    return {"categories": result}

@app.post("/batch_upload_predict")
async def batch_upload_predict(files: List[UploadFile] = File(...)):
    """
    Receives multiple uploaded bottle image files, runs defect inspection on each image,
    classifies defect type, saves results to SQLite & 'batch_inspection_results.csv',
    and computes Batch Quality Score (BQS), Pareto analysis, and risk intelligence.
    """
    if detector.memory_bank is None:
        raise HTTPException(status_code=500, detail="Model is not trained or loaded.")
        
    results = []
    batch_id = f"BATCH-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
    
    for file in files:
        try:
            contents = await file.read()
            pil_img = Image.open(io.BytesIO(contents)).convert('RGB')
            res = detector.predict(pil_img, filename=file.filename)
            
            score = round(res["anomaly_score"], 4)
            verdict = res["verdict"]
            is_def = res["is_defective"]
            grade = res["grade"]
            grade_reason = res.get("grade_reason", "")
            dsar = round(res["dsar_percent"], 2)
            zone = res["defect_zone"]
            defect_type = res["defect_type"]
            action = res["disposal_action"]
                
            # Infer or deduce category from filename or AI diagnostics
            fname_lower = file.filename.lower()
            if "broken_large" in fname_lower:
                category = "Broken Large"
            elif "broken_small" in fname_lower:
                category = "Broken Small"
            elif "contamination" in fname_lower:
                category = "Contamination"
            elif "good" in fname_lower or "train" in fname_lower:
                category = "Good (Normal)"
            elif is_def:
                if "Grade B" in grade:
                    category = "Contamination"
                elif "Mouth" in zone or "Rim" in zone:
                    category = "Broken Large" if dsar > 10.0 else "Broken Small"
                else:
                    category = "Contamination"
            else:
                category = "Good (Normal)"
                
            # Quality Intelligence Layer
            qi_profile = qi.log_inspection_result(
                filename=file.filename,
                category=category,
                anomaly_score=score,
                dsar_percent=dsar,
                defect_zone=zone,
                defect_type=defect_type,
                grade=grade,
                grade_reason=grade_reason,
                disposal_action=action,
                threshold=float(detector.threshold),
                batch_id=batch_id,
                input_compatibility_score=res.get("input_compatibility_score", 100.0),
                input_compatibility_status=res.get("input_compatibility_status", "COMPATIBLE"),
                inspection_confidence=res.get("inspection_confidence", "VALIDATED DOMAIN"),
                human_review_required=res.get("human_review_required", False),
                domain_warning=res.get("domain_warning")
            )
            
            heatmap_b64 = np_to_base64(res["heatmap_overlay"])
            polar_b64 = np_to_base64(res["polar_ribbon"])
            orig_b64 = np_to_base64(res["original_image"])
            
            results.append({
                "Filename": file.filename,
                "Category": category,
                "Anomaly_Score": score,
                "Verdict": verdict,
                "Grade": grade,
                "Grade_Reason": grade_reason,
                "DSAR_Percent": dsar,
                "Defect_Zone": zone,
                "Is_Defective": is_def,
                "Defect_Type": defect_type,
                "Disposal_Action": action,
                "Threshold": round(detector.threshold, 4),
                "Heatmap_Base64": heatmap_b64,
                "Polar_Base64": polar_b64,
                "Original_Base64": orig_b64,
                "Risk_Score": qi_profile["risk_score"],
                "Risk_Level": qi_profile["risk_level"],
                "Quality_Score": qi_profile["overall_quality_score"],
                "Pattern_Cluster": qi_profile["pattern_cluster"],
                "Is_Unknown_Anomaly": qi_profile["is_unknown_anomaly"],
                "Unknown_Confidence": qi_profile["unknown_anomaly_confidence"],
                "Inspection_ID": qi_profile["inspection_id"],
                "Bottle_ID": qi_profile["bottle_id"],
                "Input_Compatibility_Score": res.get("input_compatibility_score", 100.0),
                "Input_Compatibility_Status": res.get("input_compatibility_status", "COMPATIBLE"),
                "Inspection_Confidence": res.get("inspection_confidence", "VALIDATED DOMAIN"),
                "Human_Review_Required": res.get("human_review_required", False),
                "Domain_Warning": res.get("domain_warning")
            })
        except Exception as e:
            results.append({
                "Filename": file.filename,
                "Category": "Error",
                "Anomaly_Score": 0.0,
                "Verdict": "ERROR",
                "Grade": "ERROR",
                "Grade_Reason": f"Error: {str(e)}",
                "DSAR_Percent": 0.0,
                "Defect_Zone": "Unknown",
                "Is_Defective": True,
                "Defect_Type": f"Processing Error: {str(e)}",
                "Disposal_Action": "ERROR",
                "Threshold": round(detector.threshold, 4),
                "Heatmap_Base64": "",
                "Polar_Base64": "",
                "Original_Base64": "",
                "Risk_Score": 100.0,
                "Risk_Level": "Critical",
                "Quality_Score": 0.0,
                "Pattern_Cluster": "Error",
                "Is_Unknown_Anomaly": True,
                "Unknown_Confidence": 100.0,
                "Inspection_ID": f"ERR-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
                "Bottle_ID": "BTL-ERR",
                "Input_Compatibility_Score": 0.0,
                "Input_Compatibility_Status": "OUTSIDE_DOMAIN",
                "Inspection_Confidence": "UNRELIABLE",
                "Human_Review_Required": True,
                "Domain_Warning": "Processing Error"
            })
            
    # Compute Batch Quality Intelligence and Pareto Analysis
    batch_intelligence = qi.compute_batch_intelligence(results, batch_id=batch_id)
    pareto_analysis = qi.compute_pareto_analysis(results)
    
    # Export results to batch_inspection_results.csv file
    csv_records = [
        {
            "Filename": r["Filename"],
            "Category": r.get("Category", "N/A"),
            "Anomaly_Score": r["Anomaly_Score"],
            "Verdict": r["Verdict"],
            "Grade": r["Grade"],
            "Grade_Reason": r.get("Grade_Reason", ""),
            "DSAR_Percent": r["DSAR_Percent"],
            "Defect_Zone": r["Defect_Zone"],
            "Defect_Type": r["Defect_Type"],
            "Disposal_Action": r["Disposal_Action"],
            "Risk_Score": r.get("Risk_Score", 0.0),
            "Risk_Level": r.get("Risk_Level", "Low"),
            "Quality_Score": r.get("Quality_Score", 100.0),
            "Pattern_Cluster": r.get("Pattern_Cluster", ""),
            "Is_Unknown": r.get("Is_Unknown_Anomaly", False),
            "Input_Compatibility": r.get("Input_Compatibility_Status", "COMPATIBLE"),
            "Compatibility_Score": r.get("Input_Compatibility_Score", 100.0),
            "Confidence_State": r.get("Inspection_Confidence", "VALIDATED DOMAIN"),
            "Human_Review_Required": r.get("Human_Review_Required", False),
            "Threshold": r["Threshold"],
            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        } for r in results
    ]
    df_csv = pd.DataFrame(csv_records)
    csv_file_path = "batch_inspection_results.csv"
    df_csv.to_csv(csv_file_path, index=False)
    
    return {
        "batch_id": batch_id,
        "total_processed": len(results),
        "total_passed": sum(1 for r in results if r["Verdict"] == "GOOD"),
        "total_failed": sum(1 for r in results if r["Verdict"] == "DEFECTIVE"),
        "csv_saved_path": csv_file_path,
        "batch_intelligence": batch_intelligence,
        "pareto_analysis": pareto_analysis,
        "results": results
    }

@app.get("/benchmark")
def run_benchmark():
    """Runs batch benchmark evaluation across the entire test set and exports CSV."""
    if detector.memory_bank is None or not os.path.exists(TEST_DIR):
        raise HTTPException(status_code=400, detail="Model or dataset unavailable.")
        
    test_cats = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    results = []
    
    for cat in test_cats:
        cat_dir = os.path.join(TEST_DIR, cat)
        files = glob.glob(os.path.join(cat_dir, "*"))
        for fpath in files:
            if not fpath.endswith(('.png', '.jpg', '.jpeg')):
                continue
            res = detector.predict(fpath)
            score = round(res["anomaly_score"], 4)
            is_def = res["is_defective"]
            
            results.append({
                "Filename": os.path.basename(fpath),
                "Category": cat,
                "True_Label": "Good" if cat == "good" else "Defective",
                "Anomaly_Score": score,
                "Predicted_Verdict": res["verdict"],
                "Defect_Type": f"Good (Normal)" if cat == "good" else f"Defective ({cat})",
                "Correct": (res["verdict"] == "GOOD" and cat == "good") or (res["verdict"] == "DEFECTIVE" and cat != "good")
            })
            
    df = pd.DataFrame(results)
    accuracy = float(df["Correct"].mean() * 100)
    
    # Export benchmark to CSV file
    csv_file_path = "batch_inspection_results.csv"
    df.to_csv(csv_file_path, index=False)
    
    summary = df.groupby("Category").agg(
        Total=("Filename", "count"),
        Avg_Score=("Anomaly_Score", "mean"),
        Accuracy=("Correct", lambda x: float(x.mean() * 100))
    ).reset_index().to_dict(orient="records")
    
    return {
        "total_test_images": len(df),
        "overall_accuracy": round(accuracy, 2),
        "csv_saved_path": csv_file_path,
        "category_summary": summary,
        "detailed_results": results
    }

@app.get("/download_csv")
def download_csv():
    """Returns batch inspection CSV report as a downloadable attachment."""
    csv_path = "batch_inspection_results.csv"
    if os.path.exists(csv_path):
        return FileResponse(csv_path, media_type="text/csv", filename="batch_inspection_results.csv")
    raise HTTPException(status_code=404, detail="No batch inspection results CSV found yet.")

@app.post("/update_threshold")
def update_threshold(threshold: float = Query(..., ge=0.1, le=2.0)):
    """Updates active decision threshold in model and saves checkpoint."""
    detector.threshold = threshold
    detector.save_model(MODEL_PATH)
    return {"status": "updated", "new_threshold": threshold}

# =====================================================================
# QUALITY INTELLIGENCE & ADAPTIVE RISK MANAGEMENT ENDPOINTS
# =====================================================================

class HumanReviewRequest(BaseModel):
    inspection_id: str
    reviewer_name: str = "Quality Engineer"
    action_type: str = "Confirm Known Defect"
    defect_name: Optional[str] = ""
    description: Optional[str] = ""
    validated_zone: Optional[str] = ""
    validated_severity: Optional[str] = ""
    recommended_action: Optional[str] = ""

@app.get("/intelligence/history")
def get_history_endpoint(
    limit: int = Query(default=100, ge=1, le=500),
    grade: Optional[str] = Query(default=None),
    risk: Optional[str] = Query(default=None),
    compatibility: Optional[str] = Query(default=None),
    unknown_only: bool = Query(default=False)
):
    """Retrieves filtered historical inspection records from the SQLite database."""
    records = qi.get_inspection_history(
        limit=limit,
        grade_filter=grade,
        risk_filter=risk,
        unknown_only=unknown_only,
        compatibility_filter=compatibility
    )
    return {"total": len(records), "records": records}

@app.get("/intelligence/unknown_queue")
def get_unknown_queue_endpoint():
    """Retrieves all pending Unknown Anomalies requiring human quality engineer review."""
    queue = qi.get_inspection_history(limit=100, unknown_only=True)
    return {"queue_length": len(queue), "pending_reviews": queue}

@app.post("/intelligence/submit_review")
def submit_review_endpoint(data: HumanReviewRequest):
    """Submits human quality engineer feedback for flagged Unknown Anomalies."""
    success = qi.submit_human_review(
        inspection_id=data.inspection_id,
        reviewer_name=data.reviewer_name,
        action_type=data.action_type,
        defect_name=data.defect_name or "",
        description=data.description or "",
        validated_zone=data.validated_zone or "",
        validated_severity=data.validated_severity or "",
        recommended_action=data.recommended_action or ""
    )
    return {"status": "success", "inspection_id": data.inspection_id, "action_type": data.action_type}

@app.get("/intelligence/trends")
def get_trends_endpoint():
    """Returns historical quality trend analysis and safe process warnings."""
    trends = qi.detect_quality_trends()
    return trends

# =====================================================================
# DATASET EXPLORER ENDPOINTS (MVTEC AD GLASS BOTTLE)
# =====================================================================

@app.get("/api/datasets/summary")
def get_datasets_summary():
    """Returns comprehensive metadata, categories, and sample file lists for MVTec AD Glass Bottle dataset."""
    mvtec_cats = {}
    if os.path.exists(TEST_DIR):
        for cat in sorted(os.listdir(TEST_DIR)):
            cdir = os.path.join(TEST_DIR, cat)
            if os.path.isdir(cdir):
                mvtec_cats[cat] = sorted([os.path.basename(f) for f in glob.glob(os.path.join(cdir, "*")) if f.endswith(('.png', '.jpg'))])

    return {
        "mvtec": {
            "title": "MVTec AD Glass Bottle Benchmark",
            "type": "Unsupervised PatchCore Anomaly Localization",
            "material": "Glass Container Mouth Rim & Base (Top-down)",
            "categories": mvtec_cats,
            "total_images": sum(len(v) for v in mvtec_cats.values()) + 209
        }
    }

@app.get("/api/dataset/image")
def get_dataset_image(dataset: str = Query("mvtec"), category: str = Query(""), filename: str = Query(...)):
    """Serves image file from MVTec Glass Bottle dataset for gallery browsing."""
    safe_filename = os.path.basename(filename)
    target = os.path.join(TEST_DIR, category, safe_filename)
    if not os.path.exists(target):
        target = os.path.join(DATASET_ROOT, "train", category, safe_filename)
        
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail="Image not found.")
    
    media_type = "image/png" if target.lower().endswith(".png") else "image/jpeg"
    return FileResponse(target, media_type=media_type)

# =====================================================================
# DOMAIN 2: FULL-BODY PLASTIC BOTTLE ENDPOINTS
# =====================================================================

class PlasticPDFReportRequest(BaseModel):
    filename: str = "plastic_bottle.png"
    verdict: str = "GOOD"
    grade: str = "Grade A — Intact (Pass)"
    grade_reason: str = ""
    dsar_percent: float = 0.0
    anomaly_score: float = 0.0
    defect_zone: str = "None"
    defect_type: str = "Good (Normal Plastic Bottle)"
    disposal_action: str = "Approved for Bottling / Packaging"
    original_base64: str = ""
    heatmap_base64: str = ""
    threshold: float = 0.3377
    risk_score: float = 0.0
    risk_level: str = "Low"
    quality_score: float = 100.0
    structural_integrity: float = 99.0
    cap_integrity: float = 99.0
    body_integrity: float = 99.0
    label_condition: float = 99.0
    input_compatibility_score: float = 100.0
    input_compatibility_status: str = "COMPATIBLE"
    inspection_confidence: str = "VALIDATED DOMAIN"
    human_review_required: bool = False
    domain_warning: str = ""

@app.post("/plastic/predict")
async def plastic_predict_image(file: UploadFile = File(...), threshold: Optional[float] = Query(default=None)):
    """Inference endpoint for Full-Body Plastic Bottles."""
    if plastic_detector.memory_bank is None:
        raise HTTPException(status_code=500, detail="Plastic Bottle model is not trained or loaded.")
        
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert('RGB')
        res = plastic_detector.predict(pil_img, custom_threshold=threshold, filename=file.filename)
        active_thresh = float(res.get("threshold", plastic_detector.threshold))

        # Log into dedicated plastic_inspections table
        qi_profile = plastic_qi.log_plastic_inspection_result(
            filename=file.filename,
            category="custom_upload",
            anomaly_score=float(res["anomaly_score"]),
            dsar_percent=float(res["dsar_percent"]),
            defect_zone=res["defect_zone"],
            defect_type=res["defect_type"],
            grade=res["grade"],
            grade_reason=res.get("grade_reason", ""),
            disposal_action=res["disposal_action"],
            threshold=active_thresh,
            input_compatibility_score=res.get("input_compatibility_score", 100.0),
            input_compatibility_status=res.get("input_compatibility_status", "COMPATIBLE"),
            inspection_confidence=res.get("inspection_confidence", "VALIDATED DOMAIN"),
            human_review_required=res.get("human_review_required", False),
            domain_warning=res.get("domain_warning")
        )

        heatmap_b64 = np_to_base64(res["heatmap_overlay"])
        orig_b64 = np_to_base64(res["original_image"])
        mask_b64 = np_to_base64(np.stack([res["bottle_mask"]]*3, axis=-1))

        return {
            "inspection_domain": "plastic_bottle",
            "filename": file.filename,
            "anomaly_score": round(res["anomaly_score"], 4),
            "threshold": round(active_thresh, 4),
            "is_defective": res["is_defective"],
            "verdict": res["verdict"],
            "grade": res["grade"],
            "grade_reason": res.get("grade_reason", ""),
            "dsar_percent": round(res["dsar_percent"], 2),
            "defect_zone": res["defect_zone"],
            "defect_type": res["defect_type"],
            "disposal_action": res["disposal_action"],
            "input_compatibility_score": res.get("input_compatibility_score", 100.0),
            "input_compatibility_status": res.get("input_compatibility_status", "COMPATIBLE"),
            "inspection_confidence": res.get("inspection_confidence", "VALIDATED DOMAIN"),
            "inspection_mode": res.get("inspection_mode", "AUTOMATIC"),
            "human_review_required": res.get("human_review_required", False),
            "domain_warning": res.get("domain_warning"),
            "heatmap_base64": heatmap_b64,
            "original_base64": orig_b64,
            "mask_base64": mask_b64,
            "risk_score": qi_profile["risk_score"],
            "risk_level": qi_profile["risk_level"],
            "quality_score": qi_profile["overall_quality_score"],
            "inspection_id": qi_profile["inspection_id"],
            "bottle_id": qi_profile["bottle_id"],
            "quality_profile": qi_profile,
            "is_external": True,
            "indicative_assessment": qi_profile.get("indicative_assessment", {})
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Plastic bottle processing failed: {str(e)}")

@app.get("/plastic/predict_sample")
def plastic_predict_sample(category: str = Query(...), filename: str = Query(...), threshold: Optional[float] = Query(default=None)):
    """Runs prediction on a sample file from plastic_bottle_dataset/test."""
    if plastic_detector.memory_bank is None:
        raise HTTPException(status_code=500, detail="Plastic Bottle model is not trained or loaded.")
        
    img_path = os.path.join(PLASTIC_TEST_DIR, category, filename)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail=f"Sample file '{img_path}' not found.")
        
    res = plastic_detector.predict(img_path, custom_threshold=threshold, filename=f"{category}_{filename}")
    active_thresh = float(res.get("threshold", plastic_detector.threshold))

    qi_profile = plastic_qi.log_plastic_inspection_result(
        filename=filename,
        category=category,
        anomaly_score=float(res["anomaly_score"]),
        dsar_percent=float(res["dsar_percent"]),
        defect_zone=res["defect_zone"],
        defect_type=res["defect_type"],
        grade=res["grade"],
        grade_reason=res.get("grade_reason", ""),
        disposal_action=res["disposal_action"],
        threshold=active_thresh,
        input_compatibility_score=res.get("input_compatibility_score", 100.0),
        input_compatibility_status=res.get("input_compatibility_status", "COMPATIBLE"),
        inspection_confidence=res.get("inspection_confidence", "VALIDATED DOMAIN"),
        human_review_required=res.get("human_review_required", False),
        domain_warning=res.get("domain_warning")
    )

    heatmap_b64 = np_to_base64(res["heatmap_overlay"])
    orig_b64 = np_to_base64(res["original_image"])
    mask_b64 = np_to_base64(np.stack([res["bottle_mask"]]*3, axis=-1))

    return {
        "inspection_domain": "plastic_bottle",
        "category": category,
        "filename": filename,
        "anomaly_score": round(res["anomaly_score"], 4),
        "threshold": round(active_thresh, 4),
        "is_defective": res["is_defective"],
        "verdict": res["verdict"],
        "grade": res["grade"],
        "grade_reason": res.get("grade_reason", ""),
        "dsar_percent": round(res["dsar_percent"], 2),
        "defect_zone": res["defect_zone"],
        "defect_type": res["defect_type"],
        "disposal_action": res["disposal_action"],
        "input_compatibility_score": res.get("input_compatibility_score", 100.0),
        "input_compatibility_status": res.get("input_compatibility_status", "COMPATIBLE"),
        "inspection_confidence": res.get("inspection_confidence", "VALIDATED DOMAIN"),
        "inspection_mode": res.get("inspection_mode", "AUTOMATIC"),
        "human_review_required": res.get("human_review_required", False),
        "domain_warning": res.get("domain_warning"),
        "heatmap_base64": heatmap_b64,
        "original_base64": orig_b64,
        "mask_base64": mask_b64,
        "risk_score": qi_profile["risk_score"],
        "risk_level": qi_profile["risk_level"],
        "quality_score": qi_profile["overall_quality_score"],
        "inspection_id": qi_profile["inspection_id"],
        "bottle_id": qi_profile["bottle_id"],
        "quality_profile": qi_profile,
        "is_external": False,
        "indicative_assessment": qi_profile.get("indicative_assessment", {})
    }

@app.get("/plastic/categories")
def get_plastic_categories():
    """Returns categories and sample image file names from plastic_bottle_dataset/test."""
    cats = {}
    if os.path.exists(PLASTIC_TEST_DIR):
        for cat in sorted(os.listdir(PLASTIC_TEST_DIR)):
            cdir = os.path.join(PLASTIC_TEST_DIR, cat)
            if os.path.isdir(cdir):
                cats[cat] = sorted([f for f in os.listdir(cdir) if f.endswith(('.png', '.jpg'))])
    return {"categories": cats}

@app.post("/plastic/batch_upload_predict")
async def plastic_batch_upload(files: List[UploadFile] = File(...)):
    """Runs batch prediction across multiple uploaded plastic bottle images."""
    if plastic_detector.memory_bank is None:
        raise HTTPException(status_code=500, detail="Plastic Bottle model is not trained or loaded.")
        
    try:
        results = []
        batch_id = f"PL-BATCH-{uuid.uuid4().hex[:6].upper()}"

        for f in files:
            contents = await f.read()
            pil_img = Image.open(io.BytesIO(contents)).convert('RGB')
            res = plastic_detector.predict(pil_img, filename=f.filename)
            
            qi_prof = plastic_qi.log_plastic_inspection_result(
                filename=f.filename,
                category="batch_upload",
                anomaly_score=float(res["anomaly_score"]),
                dsar_percent=float(res["dsar_percent"]),
                defect_zone=res["defect_zone"],
                defect_type=res["defect_type"],
                grade=res["grade"],
                grade_reason=res.get("grade_reason", ""),
                disposal_action=res["disposal_action"],
                threshold=float(res["threshold"]),
                input_compatibility_score=res.get("input_compatibility_score", 100.0),
                input_compatibility_status=res.get("input_compatibility_status", "COMPATIBLE"),
                inspection_confidence=res.get("inspection_confidence", "VALIDATED DOMAIN"),
                human_review_required=res.get("human_review_required", False),
                domain_warning=res.get("domain_warning"),
                batch_id=batch_id
            )

            results.append({
                "Filename": f.filename,
                "Category": "Nominal / Good" if res["verdict"] == "GOOD" else res.get("defect_type", res.get("defect_zone", "Defective")),
                "Input_Compatibility_Status": res.get("input_compatibility_status", "COMPATIBLE"),
                "Input_Compatibility_Score": res.get("input_compatibility_score", 100.0),
                "Inspection_Confidence": res.get("inspection_confidence", "VALIDATED DOMAIN"),
                "Anomaly_Score": round(res["anomaly_score"], 4),
                "Verdict": res["verdict"],
                "Grade": res["grade"],
                "Risk_Score": qi_prof["risk_score"],
                "Risk_Level": qi_prof["risk_level"],
                "Quality_Score": qi_prof["overall_quality_score"],
                "DSAR_Percent": round(res["dsar_percent"], 2),
                "Defect_Zone": res["defect_zone"],
                "Disposal_Action": res["disposal_action"],
                "Heatmap_Base64": np_to_base64(res["heatmap_overlay"])
            })

        batch_intel = plastic_qi.compute_plastic_batch_intelligence(results)
        defects_only = [r for r in results if r["Verdict"] != "GOOD"]
        pareto_data = plastic_qi.compute_plastic_pareto_analysis(defects_only)

        # Save to CSV
        csv_path = "plastic_batch_inspection_results.csv"
        pd.DataFrame(results).to_csv(csv_path, index=False)

        return {
            "batch_id": batch_id,
            "total_processed": len(results),
            "batch_intelligence": batch_intel,
            "pareto_analysis": pareto_data,
            "csv_saved_path": csv_path,
            "results": results
        }
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"[!] Error in plastic_batch_upload: {err_msg}")
        raise HTTPException(status_code=500, detail=f"Plastic batch error: {str(e)}")

@app.get("/plastic/benchmark")
def run_plastic_benchmark():
    """Returns official benchmark evaluation across plastic_bottle_dataset/test."""
    bench_file = os.path.join("plastic_bottle_config", "benchmark_results.json")
    if os.path.exists(bench_file):
        with open(bench_file, "r") as f:
            data = json.load(f)
        return data
        
    raise HTTPException(status_code=404, detail="Benchmark results not found. Run evaluate_plastic_model.py first.")

@app.get("/plastic/history")
def get_plastic_history(
    limit: int = Query(default=100, ge=1, le=500),
    grade: Optional[str] = Query(default=None),
    risk: Optional[str] = Query(default=None),
    compatibility: Optional[str] = Query(default=None)
):
    """Retrieves filtered historical records from plastic_inspections table."""
    records = plastic_qi.get_plastic_inspection_history(
        limit=limit,
        grade_filter=grade,
        risk_filter=risk,
        compatibility_filter=compatibility
    )
    return {"total": len(records), "records": records}

@app.post("/plastic/generate_pdf")
def generate_plastic_pdf_endpoint(req: PlasticPDFReportRequest):
    """Generates official ISO PDF certificate for Plastic Bottle inspection."""
    orig_img = base64_to_pil(req.original_base64) if req.original_base64 else Image.new('RGB', (100, 100), 'white')
    heat_img = base64_to_pil(req.heatmap_base64) if req.heatmap_base64 else Image.new('RGB', (100, 100), 'white')

    pdf_bytes = generate_plastic_qa_pdf_report(
        filename=req.filename,
        verdict=req.verdict,
        grade=req.grade,
        dsar_percent=req.dsar_percent,
        anomaly_score=req.anomaly_score,
        defect_zone=req.defect_zone,
        defect_type=req.defect_type,
        disposal_action=req.disposal_action,
        orig_img_pil=orig_img,
        heatmap_img_pil=heat_img,
        threshold=req.threshold,
        grade_reason=req.grade_reason,
        risk_score=req.risk_score,
        risk_level=req.risk_level,
        quality_score=req.quality_score,
        structural_integrity=req.structural_integrity,
        cap_integrity=req.cap_integrity,
        body_integrity=req.body_integrity,
        label_condition=req.label_condition,
        input_compatibility_score=req.input_compatibility_score,
        input_compatibility_status=req.input_compatibility_status,
        inspection_confidence=req.inspection_confidence,
        human_review_required=req.human_review_required,
        domain_warning=req.domain_warning
    )
    return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=QA_Certificate_{req.filename}.pdf"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_api:app", host="0.0.0.0", port=8000, reload=True)
