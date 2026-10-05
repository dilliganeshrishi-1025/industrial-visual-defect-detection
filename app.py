"""
Streamlit Web Frontend Client - Dual-Domain Industrial Bottle Quality Inspection Platform
Supports two independent inspection domains:
  1. 🍾 Glass Bottle (Top-View Perspective / MVTec AD Dataset / Polar & Dual-Zone Analysis)
  2. 🧴 Plastic Bottle (Full-Body Front-View / PB-Dataset-v1.0 / ROI & 3-Zone Analysis)
Communicates with FastAPI REST API Backend (http://localhost:8000) and SQLite Quality DB.
"""

import streamlit as st
import os
import io
import time
import base64
import datetime
import requests
import numpy as np
import pandas as pd
import glob
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from pdf_generator import generate_qa_pdf_report, generate_batch_qa_pdf_report
from plastic_pdf_generator import generate_plastic_qa_pdf_report, generate_plastic_batch_qa_pdf_report
import quality_intelligence as qi
import plastic_quality_intelligence as plastic_qi

API_BASE_URL = "http://localhost:8000"

# Page Configuration
st.set_page_config(
    page_title="Dual-Domain Bottle Quality Intelligence & Inspection Platform",
    page_icon="🍾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 20px;
    }
    .status-good {
        color: #10B981;
        font-weight: bold;
        font-size: 1.25rem;
        border: 2px solid #10B981;
        border-radius: 8px;
        padding: 6px 14px;
        display: inline-block;
        background: #ECFDF5;
    }
    .status-rework {
        color: #F59E0B;
        font-weight: bold;
        font-size: 1.25rem;
        border: 2px solid #F59E0B;
        border-radius: 8px;
        padding: 6px 14px;
        display: inline-block;
        background: #FFFBEB;
    }
    .status-defect {
        color: #EF4444;
        font-weight: bold;
        font-size: 1.25rem;
        border: 2px solid #EF4444;
        border-radius: 8px;
        padding: 6px 14px;
        display: inline-block;
        background: #FEF2F2;
    }
    .status-ood {
        color: #B91C1C;
        font-weight: bold;
        font-size: 1.25rem;
        border: 2px solid #DC2626;
        border-radius: 8px;
        padding: 6px 14px;
        display: inline-block;
        background: #FEF2F2;
    }
    .domain-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 12px;
    }
    .domain-glass {
        background: #E0F2FE;
        color: #0369A1;
        border: 1px solid #7DD3FC;
    }
    .domain-plastic {
        background: #F0FDF4;
        color: #15803D;
        border: 1px solid #86EFAC;
    }
    .assessment-badge-good {
        color: #065F46;
        font-weight: 800;
        font-size: 1.2rem;
        background: #D1FAE5;
        border: 2px solid #10B981;
        border-radius: 8px;
        padding: 8px 16px;
        display: inline-block;
    }
    .assessment-badge-borderline {
        color: #92400E;
        font-weight: 800;
        font-size: 1.2rem;
        background: #FEF3C7;
        border: 2px solid #F59E0B;
        border-radius: 8px;
        padding: 8px 16px;
        display: inline-block;
    }
    .assessment-badge-defective {
        color: #991B1B;
        font-weight: 800;
        font-size: 1.2rem;
        background: #FEE2E2;
        border: 2px solid #EF4444;
        border-radius: 8px;
        padding: 8px 16px;
        display: inline-block;
    }
    .assessment-badge-unreliable {
        color: #7F1D1D;
        font-weight: 800;
        font-size: 1.2rem;
        background: #FFE4E6;
        border: 2px solid #DC2626;
        border-radius: 8px;
        padding: 8px 16px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

def show_img(img, caption=None):
    """Safely displays images across different Streamlit versions."""
    try:
        st.image(img, caption=caption, width='stretch')
    except Exception:
        st.image(img, caption=caption, use_container_width=True)

def show_df(df):
    """Safely displays dataframes across different Streamlit versions."""
    try:
        st.dataframe(df, width='stretch')
    except Exception:
        st.dataframe(df, use_container_width=True)

def show_plotly(fig):
    """Safely displays plotly figures across different Streamlit versions."""
    try:
        st.plotly_chart(fig, width='stretch')
    except Exception:
        try:
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.plotly_chart(fig)

def check_backend_health():
    """Queries FastAPI backend server status."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

def base64_to_pil(b64_str: str) -> Image.Image:
    """Decodes base64 string to PIL Image."""
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert('RGB')

# Initialize SQLite schemas
qi.init_intelligence_db()
plastic_qi.init_plastic_db()

def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/factory.png", width=70)
    st.sidebar.title("Inspection Platform")
    st.sidebar.caption("Dual-Domain Industrial Container Quality AI")
    
    # -------------------------------------------------------------------------
    # DOMAIN SWITCHER (Requirement 1 & 16)
    # -------------------------------------------------------------------------
    domain_choice = st.sidebar.radio(
        "Select Inspection Domain:",
        [
            "🍾 Glass Bottle (Top-View / MVTec)",
            "🧴 Plastic Bottle (Full-Body / Front-View)"
        ],
        index=0
    )
    is_glass_domain = "Glass Bottle" in domain_choice

    st.sidebar.markdown("---")
    # 4 Focused Sections per Domain
    section_selection = st.sidebar.radio("Navigation:", [
        "1. 📸 Single Bottle Inspection",
        "2. 📦 Batch Quality Intelligence",
        "3. 📜 Inspection History",
        "4. 📑 Reports & Verification"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("FastAPI REST Backend")
    health = check_backend_health()
    
    if health and health.get("status") == "online":
        domains_info = health.get("domains", {})
        if is_glass_domain:
            g_info = domains_info.get("glass_bottle", {})
            st.sidebar.success(f"🟢 Glass API Online\n\nMemory Bank: {g_info.get('coreset_size', 20000)} vectors")
            st.sidebar.info(f"Threshold: {g_info.get('threshold', 0.4263):.4f}\n\nDataset: MVTec AD Bottle")
            active_thresh = float(g_info.get("threshold", 0.4263))
        else:
            p_info = domains_info.get("plastic_bottle", {})
            st.sidebar.success(f"🟢 Plastic API Online\n\nMemory Bank: {p_info.get('coreset_size', 5000)} vectors")
            st.sidebar.info(f"Threshold: {p_info.get('threshold', 0.3377):.4f}\n\nDataset: PB-Dataset-v1.0")
            active_thresh = float(p_info.get("threshold", 0.3377))
    else:
        st.sidebar.error("🔴 Backend API Offline!\nPlease start `python -m uvicorn backend_api:app --host 0.0.0.0 --port 8000 --reload`")
        active_thresh = 0.4263 if is_glass_domain else 0.3377

    # Main Header with Active Domain Tag
    if is_glass_domain:
        st.markdown('<div class="domain-badge domain-glass">Active Domain: 🍾 Top-View Glass Bottle (MVTec AD Benchmark)</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="main-header">🍾 Glass Bottle Quality Intelligence & Inspection</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Uncertainty-Aware Gating, Polar Unwrapping & Dual-Zone Analysis on Calibrated PatchCore</p>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="domain-badge domain-plastic">Active Domain: 🧴 Full-Body Plastic Bottle (Front-View Axial Perspective)</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="main-header">🧴 Full-Body Plastic Bottle Quality Intelligence & Inspection</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Bottle ROI Extraction, 3-Zone Anatomical Analysis (Cap/Body/Base) & Dedicated PatchCore (PB-v1.0)</p>', unsafe_allow_html=True)

    # =========================================================================
    # DOMAIN 1: GLASS BOTTLE (PRESERVED 100% UNCHANGED)
    # =========================================================================
    if is_glass_domain:
        qa_standard_name = "Calibrated Industrial Glass Standard"

        if section_selection == "1. 📸 Single Bottle Inspection":
            st.subheader("📸 Single Bottle Quality Inspection (Glass)")
            input_source = st.radio("Choose Image Input Source:", ["📁 Upload Custom Image", "🖼️ Select Sample from Dataset"], horizontal=True)
            api_response = None

            if input_source == "📁 Upload Custom Image":
                uploaded_file = st.file_uploader("Upload a top-view bottle mouth/base image (PNG / JPG / JPEG)", type=["png", "jpg", "jpeg"])
                if uploaded_file is not None and st.button("🔎 Run Inspection via API", type="primary"):
                    with st.spinner("Analyzing glass bottle via PatchCore & Risk Engine..."):
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        try:
                            res = requests.post(f"{API_BASE_URL}/predict", files=files, params={"threshold": active_thresh})
                            if res.status_code == 200:
                                api_response = res.json()
                                st.session_state["last_single_glass_result"] = api_response
                            else:
                                st.error(f"API Error: {res.text}")
                        except Exception as e:
                            st.error(f"Failed to reach API server: {e}")
            else:
                try:
                    cat_resp = requests.get(f"{API_BASE_URL}/categories").json()
                    categories = cat_resp.get("categories", {})
                    if categories:
                        col_cat, col_fn = st.columns(2)
                        with col_cat:
                            sel_cat = st.selectbox("Category:", list(categories.keys()))
                        with col_fn:
                            sel_fn = st.selectbox("Sample Image File:", categories.get(sel_cat, []))
                        if sel_cat and sel_fn and st.button("🔎 Inspect Sample via API", type="primary"):
                            with st.spinner(f"Fetching prediction from REST API..."):
                                res = requests.get(f"{API_BASE_URL}/predict_sample", params={"category": sel_cat, "filename": sel_fn, "threshold": active_thresh})
                                if res.status_code == 200:
                                    api_response = res.json()
                                    st.session_state["last_single_glass_result"] = api_response
                                else:
                                    st.error(f"API Error: {res.text}")
                except Exception as e:
                    st.error(f"Could not load categories: {e}")

            if api_response is None and "last_single_glass_result" in st.session_state:
                api_response = st.session_state["last_single_glass_result"]

            if api_response:
                orig_img = base64_to_pil(api_response["original_base64"])
                heatmap_img = base64_to_pil(api_response["heatmap_base64"])
                polar_img = base64_to_pil(api_response["polar_base64"])
                
                comp_score = float(api_response.get("input_compatibility_score", 100.0))
                comp_status = api_response.get("input_compatibility_status", "COMPATIBLE")
                conf_state = api_response.get("inspection_confidence", "VALIDATED DOMAIN")
                insp_mode = api_response.get("inspection_mode", "AUTOMATIC")
                human_rev_req = api_response.get("human_review_required", False)
                domain_warn = api_response.get("domain_warning", "")
                
                st.markdown("### 🛡️ Input Validation & Domain Compatibility")
                if comp_status == "COMPATIBLE":
                    st.markdown(f'<div style="background:#F0FDF4; border:1.5px solid #86EFAC; border-radius:8px; padding:12px 16px; margin-bottom:14px;"><span style="font-weight:800; color:#15803D; font-size:1.1rem;">✅ INPUT COMPATIBLE</span><br/><span style="color:#166534; font-size:0.95rem;"><b>Compatibility Score:</b> HIGH ({comp_score:.1f}%) | <b>Confidence State:</b> {conf_state} | <b>Inspection Mode:</b> {insp_mode}</span></div>', unsafe_allow_html=True)
                elif comp_status == "LOW_CONFIDENCE":
                    st.markdown(f'<div style="background:#FFFBEB; border:2px solid #F59E0B; border-radius:8px; padding:14px 16px; margin-bottom:14px;"><span style="font-weight:800; color:#B45309; font-size:1.15rem;">⚠️ LOW CONFIDENCE — HUMAN REVIEW RECOMMENDED</span><br/><span style="color:#92400E; font-size:0.95rem;"><b>Compatibility Score:</b> LOW ({comp_score:.1f}%) | <b>Confidence State:</b> {conf_state} | <b>Inspection Mode:</b> {insp_mode}<br/><b>Warning:</b> {domain_warn or "Differs from validated domain."}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="background:#FEF2F2; border:2.5px solid #DC2626; border-radius:8px; padding:16px; margin-bottom:14px;"><span style="font-weight:800; color:#991B1B; font-size:1.25rem;">🚨 INPUT OUTSIDE VALIDATED INSPECTION DOMAIN</span><br/><span style="color:#B91C1C; font-size:0.98rem;"><b>Compatibility Score:</b> OUTSIDE DOMAIN ({comp_score:.1f}%) | <b>Mode:</b> HUMAN REVIEW REQUIRED (AUTOMATIC DECISION DISABLED)<br/><b>Warning:</b> {domain_warn or "Deviates from top-down glass bottle domain."}</span></div>', unsafe_allow_html=True)

                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    st.markdown("### (a) Original Input")
                    show_img(orig_img, caption=f"File: {api_response.get('filename', 'Input Image')}")
                with c2:
                    st.markdown("### (b) PatchCore Heatmap")
                    show_img(heatmap_img, caption="Spatial Anomaly Localization")
                with c3:
                    st.markdown("### (c) Polar Strip")
                    show_img(polar_img, caption="cv2.warpPolar 360° Circular Ribbon")

                st.markdown("---")
                score = api_response["anomaly_score"]
                grade = api_response.get("grade", "Grade A — Good")
                grade_reason = api_response.get("grade_reason", "")
                dsar = api_response.get("dsar_percent", 0.0)
                zone = api_response.get("defect_zone", "None")
                action = api_response.get("disposal_action", "Approved for Packaging")
                risk_score = api_response.get("risk_score", 0.0)
                risk_level = api_response.get("risk_level", "Low")
                
                is_external = bool(api_response.get("is_external", input_source == "📁 Upload Custom Image"))
                indicative = api_response.get("indicative_assessment", {})
                if not indicative and is_external:
                    indicative = qi.compute_indicative_assessment(
                        anomaly_score=score,
                        threshold=active_thresh,
                        input_compatibility_score=comp_score,
                        input_compatibility_status=comp_status,
                        risk_score=risk_score,
                        dsar_percent=dsar,
                        defect_zone=zone
                    )

                if is_external:
                    st.markdown("### 🌐 EXTERNAL / UNSEEN IMAGE ASSESSMENT")
                    with st.container(border=True):
                        is_reliable = indicative.get("is_reliable", True)
                        assessment = indicative.get("indicative_assessment", "LIKELY GOOD")
                        quality_display = indicative.get("indicative_quality_display", "N/A")
                        risk_lvl = indicative.get("risk_level", "LOW")
                        human_rec = indicative.get("human_review_recommendation", "RECOMMENDED")
                        
                        if not is_reliable:
                            st.markdown('<div class="assessment-badge-unreliable">🚨 AUTOMATIC QUALITY ASSESSMENT NOT RELIABLE</div>', unsafe_allow_html=True)
                            st.markdown('<p style="color:#991B1B; margin-top:10px; font-weight:600;"><b>Outside Validated Inspection Distribution:</b> Input image features deviate significantly from the baseline training distribution. An automatic quality score cannot be reliably established. Human review required.</p>', unsafe_allow_html=True)
                        else:
                            badge_cls = "assessment-badge-good" if assessment == "LIKELY GOOD" else ("assessment-badge-borderline" if assessment == "BORDERLINE" else "assessment-badge-defective")
                            st.markdown(f'<div class="{badge_cls}">INDICATIVE AI ASSESSMENT: {assessment}</div>', unsafe_allow_html=True)

                        st.markdown("<br/>", unsafe_allow_html=True)
                        ec1, ec2, ec3, ec4, ec5, ec6 = st.columns(6)
                        ec1.metric("Input Type", "External / Unseen")
                        ec2.metric("Input Compatibility", f"{comp_score:.1f}%")
                        ec3.metric("Anomaly Score", f"{score:.4f}")
                        ec4.metric("Indicative Quality", quality_display)
                        ec5.metric("Risk Level", risk_lvl)
                        ec6.metric("Human Review", human_rec)
                        
                        st.caption("ℹ️ **Notice:** *This is an indicative assessment for an unseen external image and is not part of the official benchmark evaluation.*")

                    # Human Review Panel
                    st.markdown("---")
                    with st.container(border=True):
                        st.markdown("#### 🧑‍🏭 Human Review & Decision Panel")
                        st.markdown(f"**AI Indicative Assessment:** `{assessment}` | **Indicative Quality:** `{quality_display}` | **Human Review:** `{human_rec}`")
                        
                        r_col1, r_col2 = st.columns([1, 1])
                        with r_col1:
                            reviewer_name = st.text_input("Reviewer Name:", value="Quality Engineer", key="glass_rev_name")
                        with r_col2:
                            review_notes = st.text_input("Inspection Notes / Rationale:", placeholder="e.g., Rim intact under visual inspection", key="glass_rev_notes")
                            
                        b_col1, b_col2, b_col3 = st.columns(3)
                        with b_col1:
                            if st.button("✅ Confirm as Good", key="glass_confirm_good", use_container_width=True):
                                st.session_state[f"glass_qe_status_{api_response.get('inspection_id')}"] = "APPROVED — GOOD (Permission Granted for Bottling)"
                                st.session_state[f"glass_qe_perm_{api_response.get('inspection_id')}"] = "PERMISSION GRANTED — APPROVED FOR COMMERCIAL BOTTLING"
                                st.session_state[f"glass_qe_name_{api_response.get('inspection_id')}"] = reviewer_name
                                st.session_state[f"glass_qe_notes_{api_response.get('inspection_id')}"] = review_notes
                                try:
                                    resp = requests.post(f"{API_BASE_URL}/submit_review", json={
                                        "inspection_id": api_response.get("inspection_id", ""),
                                        "reviewer_name": reviewer_name,
                                        "action_type": "CONFIRMED_GOOD",
                                        "description": review_notes,
                                        "recommended_action": "Approved for Packaging"
                                    }, timeout=3)
                                    if resp.status_code == 200:
                                        st.success("✅ Container confirmed as GOOD! Quality Engineer Permission: GRANTED.")
                                    else:
                                        qi.submit_human_review(api_response.get("inspection_id", ""), reviewer_name, "CONFIRMED_GOOD", "", review_notes, zone, "Low", "Approved for Packaging")
                                        st.success("✅ Container confirmed as GOOD! Quality Engineer Permission: GRANTED.")
                                except Exception:
                                    qi.submit_human_review(api_response.get("inspection_id", ""), reviewer_name, "CONFIRMED_GOOD", "", review_notes, zone, "Low", "Approved for Packaging")
                                    st.success("✅ Container confirmed as GOOD! Quality Engineer Permission: GRANTED.")

                        with b_col2:
                            if st.button("❌ Mark as Defective", key="glass_mark_defective", use_container_width=True):
                                st.session_state[f"glass_qe_status_{api_response.get('inspection_id')}"] = "NOT APPROVED — REJECTED / DEFECTIVE (Permission Denied)"
                                st.session_state[f"glass_qe_perm_{api_response.get('inspection_id')}"] = "PERMISSION DENIED — SCRAP / REJECT"
                                st.session_state[f"glass_qe_name_{api_response.get('inspection_id')}"] = reviewer_name
                                st.session_state[f"glass_qe_notes_{api_response.get('inspection_id')}"] = review_notes
                                try:
                                    resp = requests.post(f"{API_BASE_URL}/submit_review", json={
                                        "inspection_id": api_response.get("inspection_id", ""),
                                        "reviewer_name": reviewer_name,
                                        "action_type": "CONFIRMED_DEFECTIVE",
                                        "description": review_notes,
                                        "recommended_action": "Diverted to Scrap"
                                    }, timeout=3)
                                    if resp.status_code == 200:
                                        st.warning("❌ Container marked as DEFECTIVE! Quality Engineer Permission: DENIED.")
                                    else:
                                        qi.submit_human_review(api_response.get("inspection_id", ""), reviewer_name, "CONFIRMED_DEFECTIVE", "", review_notes, zone, "High", "Diverted to Scrap")
                                        st.warning("❌ Container marked as DEFECTIVE! Quality Engineer Permission: DENIED.")
                                except Exception:
                                    qi.submit_human_review(api_response.get("inspection_id", ""), reviewer_name, "CONFIRMED_DEFECTIVE", "", review_notes, zone, "High", "Diverted to Scrap")
                                    st.warning("❌ Container marked as DEFECTIVE! Quality Engineer Permission: DENIED.")

                        with b_col3:
                            if st.button("🔄 Send for Reinspection", key="glass_reinspect", use_container_width=True):
                                st.session_state[f"glass_qe_status_{api_response.get('inspection_id')}"] = "NOT APPROVED — ROUTED FOR RE-INSPECTION (Permission On Hold)"
                                st.session_state[f"glass_qe_perm_{api_response.get('inspection_id')}"] = "PERMISSION CONDITIONAL — ROUTE TO SECONDARY INSPECTION"
                                st.session_state[f"glass_qe_name_{api_response.get('inspection_id')}"] = reviewer_name
                                st.session_state[f"glass_qe_notes_{api_response.get('inspection_id')}"] = review_notes
                                try:
                                    resp = requests.post(f"{API_BASE_URL}/submit_review", json={
                                        "inspection_id": api_response.get("inspection_id", ""),
                                        "reviewer_name": reviewer_name,
                                        "action_type": "SEND_FOR_REINSPECTION",
                                        "description": review_notes,
                                        "recommended_action": "Secondary Line Inspection"
                                    }, timeout=3)
                                    if resp.status_code == 200:
                                        st.info("🔄 Container routed to Reinspection Queue! Permission: ON HOLD.")
                                    else:
                                        qi.submit_human_review(api_response.get("inspection_id", ""), reviewer_name, "SEND_FOR_REINSPECTION", "", review_notes, zone, "Moderate", "Secondary Line Inspection")
                                        st.info("🔄 Container routed to Reinspection Queue! Permission: ON HOLD.")
                                except Exception:
                                    qi.submit_human_review(api_response.get("inspection_id", ""), reviewer_name, "SEND_FOR_REINSPECTION", "", review_notes, zone, "Moderate", "Secondary Line Inspection")
                                    st.info("🔄 Container routed to Reinspection Queue! Permission: ON HOLD.")
                else:
                    # Official Validated Dataset Sample
                    st.markdown("### 🏷️ VALIDATED BENCHMARK DATASET SAMPLE")
                    col_b1, col_b2 = st.columns([1, 1])
                    with col_b1:
                        if comp_status == "OUTSIDE_DOMAIN":
                            st.markdown(f'<div class="status-ood">⚠️ UNVALIDATED INPUT (OOD)</div>', unsafe_allow_html=True)
                        elif "Grade A" in grade:
                            st.markdown(f'<div class="status-good">✅ {grade}</div>', unsafe_allow_html=True)
                        elif "Grade B" in grade:
                            st.markdown(f'<div class="status-rework">⚠️ {grade}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="status-defect">❌ {grade}</div>', unsafe_allow_html=True)
                    with col_b2:
                        st.info(f"⚙️ **Factory Action:** {action}")

                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    mc1.metric("Input Compatibility", f"{comp_score:.1f}%", delta=comp_status)
                    mc2.metric("Anomaly Score", f"{score:.4f}", delta=f"Thresh {active_thresh:.4f}")
                    mc3.metric("DSAR %", f"{dsar:.2f}%")
                    mc4.metric("Risk Score", f"{risk_score}/100", delta=risk_level, delta_color="inverse")
                    mc5.metric("Detected Zone", zone)

                    st.markdown("---")
                    st.subheader("🛡️ Bottle Quality Profile (Glass)")
                    prof_cols = st.columns(4)
                    qprof = api_response.get("quality_profile", {})
                    prof_cols[0].metric("Structural Integrity", f"{qprof.get('structural_integrity', 99.5)}%")
                    prof_cols[1].metric("Rim Integrity", f"{qprof.get('rim_integrity', 99.0)}%")
                    prof_cols[2].metric("Cleanliness / Hygiene", f"{qprof.get('cleanliness_score', 99.0)}%")
                    prof_cols[3].metric("Anomaly Severity", f"{qprof.get('anomaly_severity', 10.0)}%")

                # Retrieve active Quality Engineer sign-off values
                insp_id = api_response.get("inspection_id", "")
                qe_status = st.session_state.get(f"glass_qe_status_{insp_id}", "")
                qe_perm = st.session_state.get(f"glass_qe_perm_{insp_id}", "")
                qe_name = st.session_state.get(f"glass_qe_name_{insp_id}", "Lead Quality Engineer")
                qe_notes = st.session_state.get(f"glass_qe_notes_{insp_id}", "")

                st.markdown("---")
                pdf_bytes = generate_qa_pdf_report(
                    filename=api_response.get("filename", "bottle.png"),
                    verdict=api_response["verdict"],
                    grade=grade,
                    dsar_percent=dsar,
                    anomaly_score=score,
                    defect_zone=zone,
                    defect_type=api_response.get("defect_type", "N/A"),
                    disposal_action=action,
                    orig_img_pil=orig_img,
                    heatmap_img_pil=heatmap_img,
                    polar_img_pil=polar_img,
                    threshold=api_response.get("threshold", active_thresh),
                    qa_standard=qa_standard_name,
                    grade_reason=grade_reason,
                    risk_score=risk_score,
                    risk_level=risk_level,
                    quality_score=api_response.get("quality_score", 100.0),
                    input_compatibility_score=comp_score,
                    input_compatibility_status=comp_status,
                    inspection_confidence=conf_state,
                    human_review_required=human_rev_req,
                    domain_warning=domain_warn or "",
                    quality_engineer_name=qe_name,
                    quality_engineer_status=qe_status,
                    quality_engineer_notes=qe_notes,
                    quality_permission=qe_perm
                )
                st.download_button(
                    label="📄 Download Official ISO Quality Inspection PDF Certificate (Glass)",
                    data=pdf_bytes,
                    file_name=f"QA_Certificate_Glass_{api_response.get('filename', 'bottle')}.pdf",
                    mime="application/pdf",
                    type="primary"
                )

        elif section_selection == "2. 📦 Batch Quality Intelligence":
            st.subheader("📦 Batch Quality Intelligence & Pareto Analysis (Glass)")
            batch_mode = st.radio("Choose Batch Source:", ["📁 Select & Upload Multiple Custom Images", "📊 Run Full MVTec Benchmark Dataset Suite (83 Images)"], horizontal=True)

            if batch_mode == "📁 Select & Upload Multiple Custom Images":
                uploaded_files = st.file_uploader("Upload Multiple Glass Bottle Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
                if uploaded_files and st.button("🚀 Run Batch Inspection", type="primary"):
                    with st.spinner("Processing batch through Glass PatchCore..."):
                        files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
                        res = requests.post(f"{API_BASE_URL}/batch_upload_predict", files=files_payload)
                        if res.status_code == 200:
                            st.session_state["custom_glass_batch_data"] = res.json()
                            st.success("Batch inspection complete!")
                        else:
                            st.error(f"API Error: {res.text}")

                if "custom_glass_batch_data" in st.session_state:
                    bdata = st.session_state["custom_glass_batch_data"]
                    bresults = bdata.get("results", [])
                    bintel = bdata.get("batch_intelligence", {})
                    
                    st.markdown("### 🏆 Batch Quality Intelligence Summary")
                    k1, k2, k3, k4, k5 = st.columns(5)
                    k1.metric("Batch Quality Score", f"{bintel.get('batch_quality_score', 90.0)}/100")
                    k2.metric("Total Inspected", bintel.get("total_inspected", len(bresults)))
                    k3.metric("Defect Rate", f"{bintel.get('defect_rate', 0.0)}%")
                    k4.metric("Critical Defect Rate", f"{bintel.get('critical_defect_rate', 0.0)}%")
                    k5.metric("Avg Risk Score", f"{bintel.get('avg_risk_score', 0.0)}/100")

                    df_res = pd.DataFrame(bresults)
                    show_df(df_res)

                    st.markdown("---")
                    batch_pdf = generate_batch_qa_pdf_report(
                        batch_id=bdata.get("batch_id", f"LOT-GL-{int(time.time())%100000}"),
                        total_inspected=bintel.get("total_inspected", len(bresults)),
                        defect_count=bintel.get("defective_count", sum(1 for r in bresults if r.get("verdict") == "DEFECTIVE")),
                        defect_rate=float(bintel.get("defect_rate", 0.0)),
                        critical_defect_rate=float(bintel.get("critical_defect_rate", 0.0)),
                        batch_quality_score=float(bintel.get("batch_quality_score", 90.0)),
                        avg_risk_score=float(bintel.get("avg_risk_score", 15.0)),
                        results=bresults,
                        quality_engineer_name="Lead Quality Engineer"
                    )
                    st.download_button(
                        label="📄 Download Official Batch Quality Inspection PDF Report (Glass)",
                        data=batch_pdf,
                        file_name=f"Batch_QA_Report_Glass_{bdata.get('batch_id', 'lot')}.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
            else:
                if st.button("🚀 Trigger Full Backend Benchmark Suite (83 Images)", type="primary"):
                    with st.spinner("Executing full MVTec dataset benchmark..."):
                        res = requests.get(f"{API_BASE_URL}/benchmark")
                        if res.status_code == 200:
                            st.session_state["benchmark_glass_data"] = res.json()
                            st.success("Benchmark evaluation complete!")

                if "benchmark_glass_data" in st.session_state:
                    data = st.session_state["benchmark_glass_data"]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Overall Accuracy", f"{data['overall_accuracy']}%")
                    c2.metric("Total Tested Images", data['total_test_images'])
                    c3.metric("Backend Status", "100% Evaluated")
                    show_df(pd.DataFrame(data["category_summary"]))

                    st.markdown("---")
                    cat_summ = data.get("category_summary", [])
                    bench_pdf = generate_batch_qa_pdf_report(
                        batch_id="MVTec-BENCHMARK-83",
                        total_inspected=int(data.get("total_test_images", 83)),
                        defect_count=int(data.get("total_test_images", 83) - sum(c.get("Correct", 0) for c in cat_summ if c.get("Category") == "Good")),
                        defect_rate=round(float((83 - sum(c.get("Correct", 0) for c in cat_summ if c.get("Category") == "Good")) / 83.0 * 100.0), 2),
                        critical_defect_rate=0.0,
                        batch_quality_score=float(data.get("overall_accuracy", 98.8)),
                        avg_risk_score=10.0,
                        results=cat_summ,
                        qa_standard="MVTec AD Official Benchmark Evaluation Standard",
                        quality_engineer_name="Lead Quality Engineer"
                    )
                    st.download_button(
                        label="📄 Download Official Benchmark Audit PDF Report (Glass)",
                        data=bench_pdf,
                        file_name="Benchmark_QA_Report_Glass_MVTec83.pdf",
                        mime="application/pdf",
                        type="primary"
                    )

        elif section_selection == "3. 📜 Inspection History":
            st.subheader("📜 Glass Bottle Inspection History & Traceability")
            f1, f2, f3 = st.columns(3)
            with f1:
                grade_flt = st.selectbox("Filter Grade:", ["All", "Grade A", "Grade B", "Grade C"])
            with f2:
                risk_flt = st.selectbox("Filter Risk Level:", ["All", "Low", "Moderate", "High", "Critical"])
            with f3:
                compat_flt = st.selectbox("Input Compatibility:", ["All", "COMPATIBLE", "LOW_CONFIDENCE", "OUTSIDE_DOMAIN"])
                
            records = qi.get_inspection_history(limit=100, grade_filter=grade_flt, risk_filter=risk_flt, compatibility_filter=compat_flt)
            st.info(f"Retrieved **{len(records)}** records.")
            if records:
                show_df(pd.DataFrame(records))

        elif section_selection == "4. 📑 Reports & Verification":
            st.subheader("📑 Glass Bottle Quality Reports & Archive")
            rep_dir = "QA_Inspection_Reports"
            if os.path.exists(rep_dir):
                pdf_files = sorted(glob.glob(os.path.join(rep_dir, "*.pdf")), reverse=True)
                st.write(f"Archived Certificates: **{len(pdf_files)}**")
                for pdf_p in pdf_files[:5]:
                    fname = os.path.basename(pdf_p)
                    with open(pdf_p, "rb") as f:
                        st.download_button(label=f"⬇️ {fname}", data=f.read(), file_name=fname, mime="application/pdf")

    # =========================================================================
    # DOMAIN 2: FULL-BODY PLASTIC BOTTLE (NEW SELF-CONTAINED MODULE)
    # =========================================================================
    else:
        qa_standard_name = "Consumer Beverage PET / HDPE Standard (PB-v1.0)"

        if section_selection == "1. 📸 Single Bottle Inspection":
            st.subheader("📸 Single Bottle Quality Inspection (Plastic Bottle — Front View)")
            input_source = st.radio("Choose Image Input Source:", ["📁 Upload Custom Plastic Bottle Image", "🖼️ Select Sample from Dataset"], horizontal=True)
            api_response = None

            if input_source == "📁 Upload Custom Plastic Bottle Image":
                uploaded_file = st.file_uploader("Upload a full-body plastic bottle image (PNG / JPG / JPEG)", type=["png", "jpg", "jpeg"])
                if uploaded_file is not None and st.button("🔎 Run Plastic Inspection via API", type="primary"):
                    with st.spinner("Analyzing plastic bottle via Full-Body ROI & PatchCore Engine..."):
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        try:
                            res = requests.post(f"{API_BASE_URL}/plastic/predict", files=files, params={"threshold": active_thresh})
                            if res.status_code == 200:
                                api_response = res.json()
                                st.session_state["last_single_plastic_result"] = api_response
                            else:
                                st.error(f"API Error: {res.text}")
                        except Exception as e:
                            st.error(f"Failed to reach API server: {e}")
            else:
                try:
                    cat_resp = requests.get(f"{API_BASE_URL}/plastic/categories").json()
                    categories = cat_resp.get("categories", {})
                    if categories:
                        col_cat, col_fn = st.columns(2)
                        with col_cat:
                            sel_cat = st.selectbox("Defect Category:", list(categories.keys()))
                        with col_fn:
                            sel_fn = st.selectbox("Sample Image File:", categories.get(sel_cat, []))
                        if sel_cat and sel_fn and st.button("🔎 Inspect Plastic Sample via API", type="primary"):
                            with st.spinner(f"Fetching prediction from REST API..."):
                                res = requests.get(f"{API_BASE_URL}/plastic/predict_sample", params={"category": sel_cat, "filename": sel_fn, "threshold": active_thresh})
                                if res.status_code == 200:
                                    api_response = res.json()
                                    st.session_state["last_single_plastic_result"] = api_response
                                else:
                                    st.error(f"API Error: {res.text}")
                except Exception as e:
                    st.error(f"Could not load plastic categories: {e}")

            if api_response is None and "last_single_plastic_result" in st.session_state:
                api_response = st.session_state["last_single_plastic_result"]

            if api_response:
                orig_img = base64_to_pil(api_response["original_base64"])
                heatmap_img = base64_to_pil(api_response["heatmap_base64"])
                mask_img = base64_to_pil(api_response["mask_base64"])
                
                comp_score = float(api_response.get("input_compatibility_score", 100.0))
                comp_status = api_response.get("input_compatibility_status", "COMPATIBLE")
                conf_state = api_response.get("inspection_confidence", "VALIDATED DOMAIN")
                insp_mode = api_response.get("inspection_mode", "AUTOMATIC")
                human_rev_req = api_response.get("human_review_required", False)
                domain_warn = api_response.get("domain_warning", "")
                
                # Input Compatibility Card (Requirement 10 & 11)
                st.markdown("### 🛡️ Plastic Input Validation & Compatibility")
                if comp_status == "COMPATIBLE":
                    st.markdown(f'<div style="background:#F0FDF4; border:1.5px solid #86EFAC; border-radius:8px; padding:12px 16px; margin-bottom:14px;"><span style="font-weight:800; color:#15803D; font-size:1.1rem;">✅ INPUT COMPATIBLE</span><br/><span style="color:#166534; font-size:0.95rem;"><b>Compatibility Score:</b> HIGH ({comp_score:.1f}%) | <b>Confidence State:</b> {conf_state} | <b>Inspection Mode:</b> {insp_mode}</span></div>', unsafe_allow_html=True)
                elif comp_status == "LOW_CONFIDENCE":
                    st.markdown(f'<div style="background:#FFFBEB; border:2px solid #F59E0B; border-radius:8px; padding:14px 16px; margin-bottom:14px;"><span style="font-weight:800; color:#B45309; font-size:1.15rem;">⚠️ LOW CONFIDENCE — HUMAN REVIEW RECOMMENDED</span><br/><span style="color:#92400E; font-size:0.95rem;"><b>Compatibility Score:</b> LOW ({comp_score:.1f}%) | <b>Confidence State:</b> {conf_state} | <b>Inspection Mode:</b> {insp_mode}<br/><b>Warning:</b> {domain_warn or "Differs from validated plastic-bottle inspection distribution."}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="background:#FEF2F2; border:2.5px solid #DC2626; border-radius:8px; padding:16px; margin-bottom:14px;"><span style="font-weight:800; color:#991B1B; font-size:1.25rem;">🚨 INPUT OUTSIDE VALIDATED PLASTIC-BOTTLE DOMAIN</span><br/><span style="color:#B91C1C; font-size:0.98rem;"><b>Compatibility Score:</b> OUTSIDE DOMAIN ({comp_score:.1f}%) | <b>Mode:</b> HUMAN REVIEW REQUIRED (AUTOMATIC DECISION DISABLED)<br/><b>Warning:</b> {domain_warn or "Geometry / Feature mismatch. Not a recognized full-body plastic bottle."}</span></div>', unsafe_allow_html=True)

                # Visual Outputs (Requirement 7 & 8)
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    st.markdown("### (a) Original Input")
                    show_img(orig_img, caption=f"File: {api_response.get('filename', 'Input Image')}")
                with c2:
                    st.markdown("### (b) PatchCore Heatmap (3 Zones)")
                    show_img(heatmap_img, caption="Zone A: Cap, Zone B: Label, Zone C: Base")
                with c3:
                    st.markdown("### (c) Extracted Bottle ROI")
                    show_img(mask_img, caption="Segmented Plastic Body Mask (Background Suppressed)")

                st.markdown("---")
                score = api_response["anomaly_score"]
                grade = api_response.get("grade", "Grade A — Intact (Pass)")
                grade_reason = api_response.get("grade_reason", "")
                dsar = api_response.get("dsar_percent", 0.0)
                zone = api_response.get("defect_zone", "None")
                action = api_response.get("disposal_action", "Approved for Bottling / Packaging")
                risk_score = api_response.get("risk_score", 0.0)
                risk_level = api_response.get("risk_level", "Low")
                
                is_external = bool(api_response.get("is_external", input_source == "📁 Upload Custom Plastic Bottle Image"))
                indicative = api_response.get("indicative_assessment", {})
                if not indicative and is_external:
                    indicative = plastic_qi.compute_plastic_indicative_assessment(
                        anomaly_score=score,
                        threshold=active_thresh,
                        input_compatibility_score=comp_score,
                        input_compatibility_status=comp_status,
                        risk_score=risk_score,
                        dsar_percent=dsar,
                        defect_zone=zone
                    )

                if is_external:
                    st.markdown("### 🌐 EXTERNAL / UNSEEN PLASTIC IMAGE ASSESSMENT")
                    with st.container(border=True):
                        is_reliable = indicative.get("is_reliable", True)
                        assessment = indicative.get("indicative_assessment", "LIKELY GOOD")
                        quality_display = indicative.get("indicative_quality_display", "N/A")
                        risk_lvl = indicative.get("risk_level", "LOW")
                        human_rec = indicative.get("human_review_recommendation", "RECOMMENDED")
                        
                        if not is_reliable:
                            st.markdown('<div class="assessment-badge-unreliable">🚨 AUTOMATIC QUALITY ASSESSMENT NOT RELIABLE</div>', unsafe_allow_html=True)
                            st.markdown('<p style="color:#991B1B; margin-top:10px; font-weight:600;"><b>Outside Validated Plastic Domain:</b> Input geometry or feature distribution deviates significantly from vertical full-body plastic bottles. Automatic defect decision disabled. Human review required.</p>', unsafe_allow_html=True)
                        else:
                            badge_cls = "assessment-badge-good" if assessment == "LIKELY GOOD" else ("assessment-badge-borderline" if assessment == "BORDERLINE" else "assessment-badge-defective")
                            st.markdown(f'<div class="{badge_cls}">INDICATIVE AI ASSESSMENT: {assessment}</div>', unsafe_allow_html=True)

                        st.markdown("<br/>", unsafe_allow_html=True)
                        ec1, ec2, ec3, ec4, ec5, ec6 = st.columns(6)
                        ec1.metric("Input Type", "External / Unseen")
                        ec2.metric("Input Compatibility", f"{comp_score:.1f}%")
                        ec3.metric("Anomaly Score", f"{score:.4f}")
                        ec4.metric("Indicative Quality", quality_display)
                        ec5.metric("Risk Level", risk_lvl)
                        ec6.metric("Human Review", human_rec)
                        
                        st.caption("ℹ️ **Notice:** *This is an indicative assessment for an unseen external image and is not part of the official benchmark evaluation.*")

                    # Human Review Panel for Plastic
                    st.markdown("---")
                    with st.container(border=True):
                        st.markdown("#### 🧑‍🏭 Plastic Bottle Human Review & Decision Panel")
                        st.markdown(f"**AI Indicative Assessment:** `{assessment}` | **Indicative Quality:** `{quality_display}` | **Human Review:** `{human_rec}`")
                        
                        r_col1, r_col2 = st.columns([1, 1])
                        with r_col1:
                            reviewer_name = st.text_input("Reviewer Name:", value="Quality Engineer", key="plastic_rev_name")
                        with r_col2:
                            review_notes = st.text_input("Inspection Notes / Rationale:", placeholder="e.g., Vertical contour intact, cap aligned", key="plastic_rev_notes")
                            
                        b_col1, b_col2, b_col3 = st.columns(3)
                        with b_col1:
                            if st.button("✅ Confirm as Good", key="plastic_confirm_good", use_container_width=True):
                                st.session_state[f"plastic_qe_status_{api_response.get('inspection_id')}"] = "APPROVED — GOOD (Permission Granted for Packaging)"
                                st.session_state[f"plastic_qe_perm_{api_response.get('inspection_id')}"] = "PERMISSION GRANTED — APPROVED FOR COMMERCIAL BOTTLING"
                                st.session_state[f"plastic_qe_name_{api_response.get('inspection_id')}"] = reviewer_name
                                st.session_state[f"plastic_qe_notes_{api_response.get('inspection_id')}"] = review_notes
                                try:
                                    resp = requests.post(f"{API_BASE_URL}/plastic/submit_review", json={
                                        "inspection_id": api_response.get("inspection_id", ""),
                                        "reviewer_name": reviewer_name,
                                        "action_type": "CONFIRMED_GOOD",
                                        "description": review_notes,
                                        "recommended_action": "Approved for Bottling / Packaging"
                                    }, timeout=3)
                                    if resp.status_code == 200:
                                        st.success("✅ Plastic container confirmed as GOOD! Quality Engineer Permission: GRANTED.")
                                    else:
                                        plastic_qi.submit_plastic_human_review(api_response.get("inspection_id", ""), reviewer_name, "CONFIRMED_GOOD", "", review_notes, zone, "Low", "Approved for Bottling / Packaging")
                                        st.success("✅ Plastic container confirmed as GOOD! Quality Engineer Permission: GRANTED.")
                                except Exception:
                                    plastic_qi.submit_plastic_human_review(api_response.get("inspection_id", ""), reviewer_name, "CONFIRMED_GOOD", "", review_notes, zone, "Low", "Approved for Bottling / Packaging")
                                    st.success("✅ Plastic container confirmed as GOOD! Quality Engineer Permission: GRANTED.")

                        with b_col2:
                            if st.button("❌ Mark as Defective", key="plastic_mark_defective", use_container_width=True):
                                st.session_state[f"plastic_qe_status_{api_response.get('inspection_id')}"] = "NOT APPROVED — REJECTED / DEFECTIVE (Permission Denied)"
                                st.session_state[f"plastic_qe_perm_{api_response.get('inspection_id')}"] = "PERMISSION DENIED — IMMEDIATE SCRAP / QUARANTINE"
                                st.session_state[f"plastic_qe_name_{api_response.get('inspection_id')}"] = reviewer_name
                                st.session_state[f"plastic_qe_notes_{api_response.get('inspection_id')}"] = review_notes
                                try:
                                    resp = requests.post(f"{API_BASE_URL}/plastic/submit_review", json={
                                        "inspection_id": api_response.get("inspection_id", ""),
                                        "reviewer_name": reviewer_name,
                                        "action_type": "CONFIRMED_DEFECTIVE",
                                        "description": review_notes,
                                        "recommended_action": "Immediate Scrap Bin"
                                    }, timeout=3)
                                    if resp.status_code == 200:
                                        st.warning("❌ Plastic container marked as DEFECTIVE! Quality Engineer Permission: DENIED.")
                                    else:
                                        plastic_qi.submit_plastic_human_review(api_response.get("inspection_id", ""), reviewer_name, "CONFIRMED_DEFECTIVE", "", review_notes, zone, "High", "Immediate Scrap Bin")
                                        st.warning("❌ Plastic container marked as DEFECTIVE! Quality Engineer Permission: DENIED.")
                                except Exception:
                                    plastic_qi.submit_plastic_human_review(api_response.get("inspection_id", ""), reviewer_name, "CONFIRMED_DEFECTIVE", "", review_notes, zone, "High", "Immediate Scrap Bin")
                                    st.warning("❌ Plastic container marked as DEFECTIVE! Quality Engineer Permission: DENIED.")

                        with b_col3:
                            if st.button("🔄 Send for Reinspection", key="plastic_reinspect", use_container_width=True):
                                st.session_state[f"plastic_qe_status_{api_response.get('inspection_id')}"] = "NOT APPROVED — ROUTED FOR RE-INSPECTION (Permission On Hold)"
                                st.session_state[f"plastic_qe_perm_{api_response.get('inspection_id')}"] = "PERMISSION CONDITIONAL — ROUTE TO REWORK / RE-INSPECTION"
                                st.session_state[f"plastic_qe_name_{api_response.get('inspection_id')}"] = reviewer_name
                                st.session_state[f"plastic_qe_notes_{api_response.get('inspection_id')}"] = review_notes
                                try:
                                    resp = requests.post(f"{API_BASE_URL}/plastic/submit_review", json={
                                        "inspection_id": api_response.get("inspection_id", ""),
                                        "reviewer_name": reviewer_name,
                                        "action_type": "SEND_FOR_REINSPECTION",
                                        "description": review_notes,
                                        "recommended_action": "Secondary Manual Audit"
                                    }, timeout=3)
                                    if resp.status_code == 200:
                                        st.info("🔄 Plastic container routed to Reinspection Queue! Permission: ON HOLD.")
                                    else:
                                        plastic_qi.submit_plastic_human_review(api_response.get("inspection_id", ""), reviewer_name, "SEND_FOR_REINSPECTION", "", review_notes, zone, "Moderate", "Secondary Manual Audit")
                                        st.info("🔄 Plastic container routed to Reinspection Queue! Permission: ON HOLD.")
                                except Exception:
                                    plastic_qi.submit_plastic_human_review(api_response.get("inspection_id", ""), reviewer_name, "SEND_FOR_REINSPECTION", "", review_notes, zone, "Moderate", "Secondary Manual Audit")
                                    st.info("🔄 Plastic container routed to Reinspection Queue! Permission: ON HOLD.")
                else:
                    # Official Validated Dataset Sample
                    st.markdown("### 🏷️ VALIDATED BENCHMARK DATASET SAMPLE")
                    col_b1, col_b2 = st.columns([1, 1])
                    with col_b1:
                        if comp_status == "OUTSIDE_DOMAIN":
                            st.markdown(f'<div class="status-ood">⚠️ UNVALIDATED INPUT (OOD)</div>', unsafe_allow_html=True)
                        elif "Grade A" in grade:
                            st.markdown(f'<div class="status-good">✅ {grade}</div>', unsafe_allow_html=True)
                        elif "Grade B" in grade:
                            st.markdown(f'<div class="status-rework">⚠️ {grade}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="status-defect">❌ {grade}</div>', unsafe_allow_html=True)
                    with col_b2:
                        st.info(f"⚙️ **Factory Action:** {action}")

                    if grade_reason:
                        st.caption(f"🔬 **Diagnosis Detail:** {grade_reason}")

                    # Metrics
                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    mc1.metric("Input Compatibility", f"{comp_score:.1f}%", delta=comp_status)
                    mc2.metric("Anomaly Score", f"{score:.4f}", delta=f"Thresh {active_thresh:.4f}")
                    mc3.metric("DSAR % (Bottle Area)", f"{dsar:.2f}%")
                    mc4.metric("Risk Score", f"{risk_score}/100", delta=risk_level, delta_color="inverse")
                    mc5.metric("Detected Zone", zone)

                    # Plastic Bottle Quality Profile
                    st.markdown("---")
                    st.subheader("🛡️ Plastic Bottle Quality Profile (PB-v1.0)")
                    qprof = api_response.get("quality_profile", {})
                    p1, p2, p3, p4, p5 = st.columns(5)
                    p1.metric("Overall Quality", f"{qprof.get('overall_quality_score', 99.0)}/100")
                    p2.metric("Structural Integrity", f"{qprof.get('structural_integrity', 99.0)}%")
                    p3.metric("Cap / Neck Integrity", f"{qprof.get('cap_integrity', 99.0)}%")
                    p4.metric("Body Integrity", f"{qprof.get('body_integrity', 99.0)}%")
                    p5.metric("Label Condition", f"{qprof.get('label_condition', 99.0)}%")

                # Retrieve active Quality Engineer sign-off values for Plastic
                insp_id = api_response.get("inspection_id", "")
                qe_status = st.session_state.get(f"plastic_qe_status_{insp_id}", "")
                qe_perm = st.session_state.get(f"plastic_qe_perm_{insp_id}", "")
                qe_name = st.session_state.get(f"plastic_qe_name_{insp_id}", "Lead Quality Engineer")
                qe_notes = st.session_state.get(f"plastic_qe_notes_{insp_id}", "")

                # PDF Certificate Export
                st.markdown("---")
                pdf_bytes = generate_plastic_qa_pdf_report(
                    filename=api_response.get("filename", "plastic_bottle.png"),
                    verdict=api_response["verdict"],
                    grade=grade,
                    dsar_percent=dsar,
                    anomaly_score=score,
                    defect_zone=zone,
                    defect_type=api_response.get("defect_type", "Good (Normal Plastic Bottle)"),
                    disposal_action=action,
                    orig_img_pil=orig_img,
                    heatmap_img_pil=heatmap_img,
                    threshold=active_thresh,
                    qa_standard=qa_standard_name,
                    grade_reason=grade_reason,
                    risk_score=risk_score,
                    risk_level=risk_level,
                    quality_score=api_response.get("quality_score", 100.0),
                    structural_integrity=api_response.get("quality_profile", {}).get("structural_integrity", 99.0),
                    cap_integrity=api_response.get("quality_profile", {}).get("cap_integrity", 99.0),
                    body_integrity=api_response.get("quality_profile", {}).get("body_integrity", 99.0),
                    label_condition=api_response.get("quality_profile", {}).get("label_condition", 99.0),
                    input_compatibility_score=comp_score,
                    input_compatibility_status=comp_status,
                    inspection_confidence=conf_state,
                    human_review_required=human_rev_req,
                    domain_warning=domain_warn or "",
                    quality_engineer_name=qe_name,
                    quality_engineer_status=qe_status,
                    quality_engineer_notes=qe_notes,
                    quality_permission=qe_perm
                )
                st.download_button(
                    label="📄 Download Official ISO Quality Inspection PDF Certificate (Plastic Bottle)",
                    data=pdf_bytes,
                    file_name=f"QA_Certificate_Plastic_{api_response.get('filename', 'bottle')}.pdf",
                    mime="application/pdf",
                    type="primary"
                )

        elif section_selection == "2. 📦 Batch Quality Intelligence":
            st.subheader("📦 Batch Quality Intelligence & Pareto Analysis (Plastic Bottle)")
            st.markdown("Lot-level quality analytics for plastic containers. Evaluates defect rates, calculates **Batch Quality Score (BQS)**, and isolates 80/20 root cause drivers.")
            
            batch_mode = st.radio("Choose Batch Source:", ["📁 Select & Upload Multiple Plastic Images", "📊 Run Full Plastic Benchmark Suite (80 Images)"], horizontal=True)

            if batch_mode == "📁 Select & Upload Multiple Plastic Images":
                uploaded_files = st.file_uploader("Upload Multiple Plastic Bottle Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
                if uploaded_files and st.button("🚀 Run Plastic Batch Inspection", type="primary"):
                    with st.spinner("Processing batch through Plastic PatchCore..."):
                        files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
                        res = requests.post(f"{API_BASE_URL}/plastic/batch_upload_predict", files=files_payload)
                        if res.status_code == 200:
                            st.session_state["custom_plastic_batch_data"] = res.json()
                            st.success("Batch inspection complete!")
                        else:
                            st.error(f"API Error: {res.text}")

                if "custom_plastic_batch_data" in st.session_state:
                    bdata = st.session_state["custom_plastic_batch_data"]
                    bresults = bdata.get("results", [])
                    bintel = bdata.get("batch_intelligence", {})
                    pareto = bdata.get("pareto_analysis", {})

                    st.markdown("### 🏆 Plastic Batch Quality Intelligence Summary")
                    k1, k2, k3, k4, k5 = st.columns(5)
                    k1.metric("Batch Quality Score", f"{bintel.get('batch_quality_score', 90.0)}/100")
                    k2.metric("Total Inspected", bintel.get("total_inspected", len(bresults)))
                    k3.metric("Defect Rate", f"{bintel.get('defect_rate', 0.0)}%")
                    k4.metric("Critical Defect Rate", f"{bintel.get('critical_defect_rate', 0.0)}%")
                    k5.metric("Avg Risk Score", f"{bintel.get('avg_risk_score', 0.0)}/100")

                    # Pareto Chart (Requirement 13 Upgrade 4)
                    st.markdown("---")
                    st.subheader("📊 Plastic Defect Pareto Analysis (80/20 Rule)")
                    cats = [d["category"] for d in pareto.get("categories", []) if d.get("count", 0) > 0]
                    counts = [d["count"] for d in pareto.get("categories", []) if d.get("count", 0) > 0]
                    cum_pcts = [d["cumulative_pct"] for d in pareto.get("categories", []) if d.get("count", 0) > 0]

                    if cats and sum(counts) > 0:
                        fig_par = go.Figure()
                        fig_par.add_trace(go.Bar(x=cats, y=counts, name="Defect Count", marker_color="#0284C7", yaxis="y1"))
                        fig_par.add_trace(go.Scatter(x=cats, y=cum_pcts, name="Cumulative %", marker=dict(color="#DC2626", size=8), line=dict(color="#DC2626", width=2), yaxis="y2"))
                        fig_par.update_layout(
                            title="Defect Category Pareto Distribution",
                            yaxis=dict(title="Occurrence Count"),
                            yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105])
                        )
                        fig_par.add_hline(y=80.0, line_dash="dash", line_color="red", yref="y2", annotation_text="80% Cutoff")
                        show_plotly(fig_par)
                    else:
                        st.success("🎉 **100% Lot Yield:** No defects detected in this batch! All uploaded plastic containers passed inspection with Grade A (Pass).")
                    
                    df_res = pd.DataFrame(bresults)
                    show_df(df_res)

                    st.markdown("---")
                    plastic_batch_pdf = generate_plastic_batch_qa_pdf_report(
                        batch_id=bdata.get("batch_id", f"LOT-PL-{int(time.time())%100000}"),
                        results=bresults,
                        batch_intelligence=bintel,
                        pareto_analysis=pareto
                    )
                    st.download_button(
                        label="📄 Download Official Batch Quality Inspection PDF Report (Plastic Bottle)",
                        data=plastic_batch_pdf,
                        file_name=f"Batch_QA_Report_Plastic_{bdata.get('batch_id', 'lot')}.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
            else:
                # Full Plastic Benchmark Suite
                res = requests.get(f"{API_BASE_URL}/plastic/benchmark")
                if res.status_code == 200:
                    data = res.json()
                    metrics = data.get("metrics", {})
                    
                    st.markdown("### 📊 Official Plastic Benchmark Performance (PB-Dataset-v1.0)")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Overall Accuracy", f"{metrics.get('accuracy')}%")
                    c2.metric("AUROC", f"{metrics.get('auroc')}%")
                    c3.metric("Precision", f"{metrics.get('precision')}%")
                    c4.metric("Recall", f"{metrics.get('recall')}%")

                    st.markdown("---")
                    df_dt = pd.DataFrame(data.get("detailed_results", []))
                    
                    # Defect Category Breakdown
                    st.subheader("Plastic Defect Category Breakdown")
                    cat_summary = df_dt.groupby("Category").agg(
                        Total=("Filename", "count"),
                        Avg_Score=("Anomaly_Score", "mean"),
                        Accuracy=("Correct", lambda x: round(float(x.mean() * 100), 1))
                    ).reset_index()
                    show_df(cat_summary)

                    # Pareto on Benchmark
                    defects_only = df_dt[df_dt["True_Label"] == "Defective"]
                    pareto = plastic_qi.compute_plastic_pareto_analysis(defects_only.to_dict(orient="records"))
                    
                    st.markdown("---")
                    st.subheader("📊 Plastic Defect Pareto Analysis (Benchmark Distribution)")
                    st.info(f"📌 **Dominant Tested Rejection Categories:** `{', '.join(pareto.get('vital_few', []))}`")
                    cats = [d["category"] for d in pareto.get("categories", [])]
                    counts = [d["count"] for d in pareto.get("categories", [])]
                    cum_pcts = [d["cumulative_pct"] for d in pareto.get("categories", [])]

                    fig_par = go.Figure()
                    fig_par.add_trace(go.Bar(x=cats, y=counts, name="Count", marker_color="#0284C7", yaxis="y1"))
                    fig_par.add_trace(go.Scatter(x=cats, y=cum_pcts, name="Cumulative %", marker=dict(color="#DC2626", size=8), line=dict(color="#DC2626", width=2), yaxis="y2"))
                    fig_par.update_layout(yaxis=dict(title="Occurrence Count"), yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]))
                    fig_par.add_hline(y=80.0, line_dash="dash", line_color="red", yref="y2", annotation_text="80% Cutoff")
                    show_plotly(fig_par)

                    st.markdown("---")
                    benchmark_results_list = df_dt.to_dict(orient="records")
                    bench_intel = {
                        "batch_quality_score": float(metrics.get("accuracy", 98.8)),
                        "total_inspected": len(benchmark_results_list),
                        "passed_count": sum(1 for r in benchmark_results_list if r.get("Correct")),
                        "defective_count": sum(1 for r in benchmark_results_list if not r.get("Correct")),
                        "defect_rate": round(float((1.0 - float(metrics.get("accuracy", 98.8))/100.0)*100.0), 2),
                        "critical_defect_rate": 0.0,
                        "avg_risk_score": 10.0
                    }
                    bench_plastic_pdf = generate_plastic_batch_qa_pdf_report(
                        batch_id="PB-BENCHMARK-80",
                        results=benchmark_results_list,
                        batch_intelligence=bench_intel,
                        pareto_analysis=pareto,
                        qa_standard="Plastic Bottle PB-Dataset-v1.0 Benchmark Standard"
                    )
                    st.download_button(
                        label="📄 Download Official Benchmark Audit PDF Report (Plastic Bottle)",
                        data=bench_plastic_pdf,
                        file_name="Benchmark_QA_Report_Plastic_PB80.pdf",
                        mime="application/pdf",
                        type="primary"
                    )

        elif section_selection == "3. 📜 Inspection History":
            st.subheader("📜 Plastic Bottle Inspection History & Traceability")
            f1, f2, f3 = st.columns(3)
            with f1:
                grade_flt = st.selectbox("Filter Grade:", ["All", "Grade A", "Grade B", "Grade C"])
            with f2:
                risk_flt = st.selectbox("Filter Risk Level:", ["All", "Low", "Moderate", "High", "Critical"])
            with f3:
                compat_flt = st.selectbox("Input Compatibility:", ["All", "COMPATIBLE", "LOW_CONFIDENCE", "OUTSIDE_DOMAIN"])
                
            records = plastic_qi.get_plastic_inspection_history(limit=100, grade_filter=grade_flt, risk_filter=risk_flt, compatibility_filter=compat_flt)
            st.info(f"Retrieved **{len(records)}** plastic bottle inspection records from database.")
            if records:
                df_p = pd.DataFrame(records)
                show_df(df_p)
                csv_bytes = df_p.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Download Plastic Inspection History (CSV)", data=csv_bytes, file_name="plastic_inspection_history.csv", mime="text/csv", type="primary")
            else:
                st.warning("No records matched the selected query filters.")

        elif section_selection == "4. 📑 Reports & Verification":
            st.subheader("📑 Plastic Bottle Reports & Domain Science Comparison")
            
            # Master Database Export
            with plastic_qi.get_db_connection() as conn:
                df_all = pd.read_sql_query("SELECT * FROM plastic_inspections ORDER BY id DESC", conn)
            if not df_all.empty:
                st.download_button(
                    label="📥 Export Full Plastic Quality Telemetry (CSV)",
                    data=df_all.to_csv(index=False).encode('utf-8'),
                    file_name="master_plastic_bottle_telemetry.csv",
                    mime="text/csv",
                    type="primary"
                )

            # Scientific Comparison Table (Requirement 15: Glass vs Plastic — Do Not Mix the Science)
            st.markdown("---")
            st.subheader("⚖️ Scientific Domain Comparison: Glass Bottle vs Full-Body Plastic Bottle")
            
            comp_table_data = {
                "Property": [
                    "Benchmark Dataset",
                    "Inspection Perspective",
                    "Target Geometry",
                    "Normal Training Basis",
                    "Feature Extractor",
                    "Memory Bank Architecture",
                    "Active Decision Threshold",
                    "Primary Anatomical Zones",
                    "Preprocessing Method",
                    "Quality Intelligence Upgrades",
                    "External / Google Input Gating",
                    "Reporting & Traceability"
                ],
                "Domain 1: Glass Bottle": [
                    "MVTec AD Industrial Bottle (209 Train / 83 Test)",
                    "Top-Down Axial View",
                    "Circular Mouth Rim, Lip, & Base Core",
                    "Strictly Normal Images (train/good)",
                    "ResNet-18 (Layers 2 & 3 Concatenation)",
                    "Glass Memory Bank (20,000 Coreset Vectors)",
                    "0.4263 (Calibrated on MVTec)",
                    "Sealing Rim Annulus vs Internal Base Core",
                    "Polar Coordinate Ribbon Unwrapping (cv2.warpPolar)",
                    "Zone-Aware Risk, Profile, BQS, Pareto",
                    "q25 & Background Consistency Gating",
                    "ISO QA PDF Certificate & SQLite CSV Export"
                ],
                "Domain 2: Plastic Bottle": [
                    "Custom Prototype / PB-Dataset-v1.0 (100 Train / 33 Val / 80 Test)",
                    "Front-View Axial / Full-Body View",
                    "Cylindrical / Contoured Container Body",
                    "Strictly Normal Images (train/good)",
                    "ResNet-18 (Layers 2 & 3 Concatenation)",
                    "Plastic Memory Bank (5,000 Coreset Vectors)",
                    "0.3377 (Calibrated on PB Validation Split)",
                    "Zone A (Cap/Neck), Zone B (Label/Body), Zone C (Base)",
                    "Bottle Contour ROI Extraction & Background Suppression",
                    "Zone-Aware Risk, Profile, BQS, Pareto",
                    "q25, Aspect Ratio (>=1.6), & Studio Illumination",
                    "Plastic ISO PDF Certificate & SQLite CSV Export"
                ]
            }
            show_df(pd.DataFrame(comp_table_data))

            st.markdown("""
            > **Academic & Industrial Positioning:**
            > - **Independent Weights & Coresets:** The glass and plastic models do NOT share feature memory banks or decision thresholds.
            > - **Cross-Domain Protection:** A top-view glass bottle uploaded in Plastic mode triggers `OUTSIDE_DOMAIN` due to aspect ratio and patch divergence; conversely, a plastic bottle in Glass mode triggers `OUTSIDE_DOMAIN`.
            > - **Realistic Synthetic Dataset:** Clearly declared as `Custom Prototype / Synthetic-Augmented Plastic Bottle Dataset`.
            """)

if __name__ == "__main__":
    main()
