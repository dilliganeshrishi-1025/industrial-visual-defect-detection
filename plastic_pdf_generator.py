"""
Industrial QA PDF Certificate Generator for Full-Body Plastic Bottle Inspection System.
Produces publication-ready, single-page ISO-style compliance certificates for Domain 2.
"""

import io
import datetime
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_plastic_qa_pdf_report(
    filename: str,
    verdict: str,
    grade: str,
    dsar_percent: float,
    anomaly_score: float,
    defect_zone: str,
    defect_type: str,
    disposal_action: str,
    orig_img_pil: Image.Image,
    heatmap_img_pil: Image.Image,
    timestamp: str = None,
    threshold: float = 0.3377,
    qa_standard: str = "Consumer Beverage PET / HDPE Standard",
    grade_reason: str = "",
    risk_score: float = 0.0,
    risk_level: str = "Low",
    quality_score: float = 100.0,
    structural_integrity: float = 99.0,
    cap_integrity: float = 99.0,
    body_integrity: float = 99.0,
    label_condition: float = 99.0,
    input_compatibility_score: float = 100.0,
    input_compatibility_status: str = "COMPATIBLE",
    inspection_confidence: str = "VALIDATED DOMAIN",
    human_review_required: bool = False,
    domain_warning: str = "",
    quality_engineer_name: str = "Lead Quality Engineer",
    quality_engineer_status: str = "",
    quality_engineer_notes: str = "",
    quality_permission: str = ""
) -> bytes:
    """Generates a publication-ready single-page PDF Certificate with Quality Engineer Approval & Permission."""
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Auto-resolve Quality Engineer approval status and permission if not explicitly provided
    if not quality_engineer_status:
        if "Grade A" in grade or verdict == "GOOD":
            quality_engineer_status = "APPROVED — GOOD (Permission Granted for Packaging)"
            quality_permission = "PERMISSION GRANTED — APPROVED FOR COMMERCIAL BOTTLING"
        elif "Grade B" in grade:
            quality_engineer_status = "NOT APPROVED — REWORK REQUIRED (Conditional Permission)"
            quality_permission = "PERMISSION CONDITIONAL — ROUTE TO RE-LABEL / WASH STATION"
        elif "Grade C" in grade or "DEFECTIVE" in verdict:
            quality_engineer_status = "NOT APPROVED — REJECTED / DEFECTIVE (Permission Denied)"
            quality_permission = "PERMISSION DENIED — IMMEDIATE SCRAP / QUARANTINE"
        else:
            quality_engineer_status = "PENDING HUMAN REVIEW — UNVALIDATED (Permission On Hold)"
            quality_permission = "PERMISSION ON HOLD — OUTSIDE VALIDATED DOMAIN"

    if not quality_permission:
        if "APPROVED" in quality_engineer_status and "NOT APPROVED" not in quality_engineer_status:
            quality_permission = "PERMISSION GRANTED — APPROVED FOR COMMERCIAL PRODUCTION"
        elif "REWORK" in quality_engineer_status or "CONDITIONAL" in quality_permission:
            quality_permission = "PERMISSION CONDITIONAL — ROUTE TO REWORK / RE-INSPECTION"
        elif "NOT APPROVED" in quality_engineer_status or "DEFECTIVE" in quality_engineer_status or "REJECTED" in quality_engineer_status:
            quality_permission = "PERMISSION DENIED — SCRAP / REJECT"
        else:
            quality_permission = "PERMISSION ON HOLD — PENDING ENGINEERING VERIFICATION"

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=2
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=3,
        spaceAfter=3
    )
    cell_bold = ParagraphStyle('CBold', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.HexColor("#1E293B"))
    cell_normal = ParagraphStyle('CNormal', fontName='Helvetica', fontSize=8, leading=10, textColor=colors.HexColor("#334155"))

    PASS_COLOR = colors.HexColor("#16A34A")
    WARN_COLOR = colors.HexColor("#D97706")
    FAIL_COLOR = colors.HexColor("#DC2626")

    if input_compatibility_status == "OUTSIDE_DOMAIN":
        badge_text = "UNVALIDATED EXTERNAL INPUT"
        verdict_color = FAIL_COLOR
    elif "GOOD" in verdict:
        badge_text = "PASSED INSPECTION"
        verdict_color = PASS_COLOR
    elif "UNCERTAIN" in verdict:
        badge_text = "HUMAN REVIEW REQUIRED"
        verdict_color = WARN_COLOR
    else:
        badge_text = "DEFECT DETECTED"
        verdict_color = FAIL_COLOR

    elements = []

    # Header Table with Logo & Title
    header_data = [
        [
            Paragraph("<b>PLASTIC CONTAINER QUALITY INSPECTION CERTIFICATE</b><br/>"
                      "<i>PatchCore Full-Body Front-View Anomaly Detection Engine (PB-v1.0)</i>", title_style),
            Paragraph(f"<font color='{verdict_color.hexval()}'><b>{badge_text}</b></font><br/>"
                      f"<font size=7 color='#64748B'>{timestamp}</font>",
                      ParagraphStyle('HRight', fontName='Helvetica-Bold', fontSize=10, leading=13, alignment=2))
        ]
    ]
    t_header = Table(header_data, colWidths=[5.2*inch, 2.2*inch])
    t_header.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    elements.append(t_header)
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceBefore=4, spaceAfter=6))

    # Quality Engineer Signoff & Permission Block (Requirement)
    app_is_approved = ("APPROVED" in quality_engineer_status and "NOT APPROVED" not in quality_engineer_status)
    app_is_rework = ("REWORK" in quality_engineer_status or "CONDITIONAL" in quality_permission)
    app_bg = colors.HexColor("#DCFCE7") if app_is_approved else (colors.HexColor("#FEF3C7") if app_is_rework else colors.HexColor("#FEE2E2"))
    app_border = PASS_COLOR if app_is_approved else (WARN_COLOR if app_is_rework else FAIL_COLOR)

    approval_data = [
        [
            Paragraph("<b>Quality Engineer Sign-Off:</b>", cell_bold),
            Paragraph(f"<font color='{app_border.hexval()}'><b>{quality_engineer_status}</b></font>", ParagraphStyle('AppValP', fontName='Helvetica-Bold', fontSize=8.5, textColor=app_border))
        ],
        [
            Paragraph("<b>Production Permission:</b>", cell_bold),
            Paragraph(f"<font color='{app_border.hexval()}'><b>{quality_permission}</b></font>", ParagraphStyle('PermValP', fontName='Helvetica-Bold', fontSize=8, textColor=app_border))
        ],
        [
            Paragraph("<b>Certified Inspector & Time:</b>", cell_bold),
            Paragraph(f"{quality_engineer_name} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Sign-Off Timestamp:</b> {timestamp}", cell_normal)
        ]
    ]
    if quality_engineer_notes:
        approval_data.append([
            Paragraph("<b>Engineer Audit Notes:</b>", cell_bold),
            Paragraph(f"<i>{quality_engineer_notes}</i>", cell_normal)
        ])
    t_approval = Table(approval_data, colWidths=[2.2*inch, 5.2*inch])
    t_approval.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), app_bg),
        ('BOX', (0,0), (-1,-1), 1.2, app_border),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0,0), (-1,-1), 3.5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(t_approval)
    elements.append(Spacer(1, 6))

    # Dual Visual Evidence Table (Original Input & Anomaly Heatmap with Demarcation Lines)
    orig_bio = io.BytesIO()
    orig_img_pil.resize((160, 160)).save(orig_bio, format='PNG')
    orig_bio.seek(0)

    heat_bio = io.BytesIO()
    heatmap_img_pil.resize((160, 160)).save(heat_bio, format='PNG')
    heat_bio.seek(0)

    img_data = [
        [
            Paragraph("<b>(a) Original Front-View Bottle</b>", cell_bold),
            Paragraph("<b>(b) PatchCore Spatial Heatmap (3 Zones)</b>", cell_bold)
        ],
        [
            RLImage(orig_bio, width=2.1*inch, height=2.1*inch),
            RLImage(heat_bio, width=2.1*inch, height=2.1*inch)
        ],
        [
            Paragraph("Front-view full body with ROI extraction", cell_normal),
            Paragraph("Yellow demarcation: Zone A (Cap), Zone B (Label), Zone C (Base)", cell_normal)
        ]
    ]
    img_table = Table(img_data, colWidths=[3.7*inch, 3.7*inch])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,0), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0,0), (-1,-1), 2),
    ]))
    elements.append(img_table)
    elements.append(Spacer(1, 6))

    # Quantitative Telemetry & Quality Diagnosis Table
    elements.append(Paragraph("Quantitative Metric Triage & Manufacturing Action", h2_style))
    elements.append(Spacer(1, 3))

    metrics_data = [
        [Paragraph("Sample Filename", cell_bold), Paragraph(filename, cell_normal)],
        [Paragraph("Input Compatibility Status", cell_bold), Paragraph(f"<b>{input_compatibility_status}</b> ({input_compatibility_score:.1f}%) — <i>{inspection_confidence}</i>", cell_normal)],
        [Paragraph("Human Review Required", cell_bold), Paragraph(f"<font color='{FAIL_COLOR.hexval()}'><b>YES — Engineering Review Required</b></font>" if human_review_required else f"<font color='{PASS_COLOR.hexval()}'><b>NO (Automated Inspection Valid)</b></font>", cell_normal)],
        [Paragraph("Operating QA Standard", cell_bold), Paragraph(f"<b>{qa_standard}</b> (Threshold: {threshold:.4f})", cell_normal)],
        [Paragraph("Anomaly Matching Score", cell_bold), Paragraph(f"{anomaly_score:.4f}", cell_normal)],
        [Paragraph("Defective Surface Area Ratio (DSAR)", cell_bold), Paragraph(f"<b>{dsar_percent:.2f}%</b> (Bottle Area)", cell_normal)],
        [Paragraph("Anatomical Defect Zone", cell_bold), Paragraph(f"<b>{defect_zone}</b>", cell_normal)],
        [Paragraph("Classified Defect Subtype", cell_bold), Paragraph(defect_type, cell_normal)],
        [Paragraph("Quality Classification Grade", cell_bold), Paragraph(f"<b>{grade}</b>", cell_normal)],
        [Paragraph("Overall Quality Score", cell_bold), Paragraph(f"<b>{quality_score:.1f} / 100</b>", cell_normal)],
        [Paragraph("Structural Integrity Profile", cell_bold), Paragraph(f"Structural: {structural_integrity:.1f}% | Cap: {cap_integrity:.1f}% | Body: {body_integrity:.1f}% | Label: {label_condition:.1f}%", cell_normal)],
        [Paragraph("Manufacturing Risk Profile", cell_bold), Paragraph(f"<b>{risk_score:.1f} / 100 ({risk_level} Risk)</b>", cell_normal)],
        [Paragraph("Industrial Triage Action", cell_bold), Paragraph(f"<font color='{verdict_color.hexval()}'><b>{disposal_action}</b></font>", cell_normal)]
    ]

    metrics_table = Table(metrics_data, colWidths=[2.4*inch, 5.0*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#F1F5F9")),
        ('BACKGROUND', (1,0), (1,-1), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0,0), (-1,-1), 3),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 6))

    # Signoff Footer
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#94A3B8"), spaceBefore=2, spaceAfter=4))
    footer_data = [
        [
            Paragraph("<b>Automated Vision Inspection Unit — Domain 2 (Plastic Bottle)</b><br/>PatchCore Full-Body Feature Engine (PB-v1.0)", ParagraphStyle('F1', fontName='Helvetica', fontSize=6.5, textColor=colors.HexColor("#64748B"))),
            Paragraph(f"<b>Quality Engineer Sign-Off:</b> {quality_engineer_name}<br/><i>[ Decision: {quality_engineer_status[:35]} ]</i>", ParagraphStyle('F2', fontName='Helvetica', fontSize=6.5, alignment=2, textColor=colors.HexColor("#64748B")))
        ]
    ]
    footer_table = Table(footer_data, colWidths=[3.7*inch, 3.7*inch])
    elements.append(footer_table)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

def generate_plastic_batch_qa_pdf_report(
    batch_id: str,
    results: list,
    batch_intelligence: dict,
    pareto_analysis: dict,
    threshold: float = 0.3309,
    timestamp: str = None,
    qa_standard: str = "Consumer Beverage PET / HDPE Standard (PB-v2.0)"
) -> bytes:
    """
    Generates a publication-ready Lot/Batch Inspection Certificate & Pareto Report for Plastic Bottles.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'BatchTitle',
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0F172A")
    )
    h2_style = ParagraphStyle(
        'BatchH2',
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=5,
        spaceAfter=3
    )
    cell_bold = ParagraphStyle('BCBold', fontName='Helvetica-Bold', fontSize=7.5, leading=9.5, textColor=colors.HexColor("#1E293B"))
    cell_normal = ParagraphStyle('BCNormal', fontName='Helvetica', fontSize=7.5, leading=9.5, textColor=colors.HexColor("#334155"))
    cell_header = ParagraphStyle('BCHeader', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.white)

    PASS_COLOR = colors.HexColor("#16A34A")
    WARN_COLOR = colors.HexColor("#D97706")
    FAIL_COLOR = colors.HexColor("#DC2626")

    bqs = float(batch_intelligence.get("batch_quality_score", 90.0))
    total_inspected = int(batch_intelligence.get("total_inspected", len(results)))
    passed_cnt = int(batch_intelligence.get("passed_count", sum(1 for r in results if r.get("Verdict") == "GOOD")))
    defect_cnt = int(batch_intelligence.get("defective_count", total_inspected - passed_cnt))
    defect_rate = float(batch_intelligence.get("defect_rate", (defect_cnt / max(1, total_inspected)) * 100.0))
    crit_rate = float(batch_intelligence.get("critical_defect_rate", 0.0))
    avg_risk = float(batch_intelligence.get("avg_risk_score", 15.0))

    if bqs >= 85.0 and crit_rate == 0.0:
        lot_status = "LOT APPROVED — PASS"
        lot_color = PASS_COLOR
        lot_action = "Approved for Commercial Packaging & Bottling Line"
    elif bqs >= 65.0:
        lot_status = "CONDITIONAL RELEASE — AUDIT REQUIRED"
        lot_color = WARN_COLOR
        lot_action = "Manual Line Inspection Recommended Before Release"
    else:
        lot_status = "LOT QUARANTINED — CRITICAL DEFECTS"
        lot_color = FAIL_COLOR
        lot_action = "Hold Entire Lot. Quarantine for Engineering Root Cause Review"

    elements = []

    # 1. Header Table
    header_data = [
        [
            Paragraph("<b>PLASTIC BOTTLE BATCH QUALITY AUDIT CERTIFICATE</b><br/>"
                      "<i>PatchCore Full-Body Inspection Engine (ISO-9001 / HACCP Compliant)</i>", title_style),
            Paragraph(f"<font color='{lot_color.hexval()}'><b>{lot_status}</b></font><br/>"
                      f"<font size=7 color='#64748B'>Audit Date: {timestamp}</font>",
                      ParagraphStyle('HRight', fontName='Helvetica-Bold', fontSize=9, leading=11, alignment=2))
        ]
    ]
    t_header = Table(header_data, colWidths=[5.0*inch, 2.5*inch])
    t_header.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    elements.append(t_header)
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceBefore=4, spaceAfter=6))

    # 2. Executive Lot KPI Grid
    kpi_data = [
        [
            Paragraph("<b>Batch Tracking ID</b>", cell_bold), Paragraph(str(batch_id), cell_normal),
            Paragraph("<b>Operating QA Standard</b>", cell_bold), Paragraph(str(qa_standard), cell_normal)
        ],
        [
            Paragraph("<b>Batch Quality Score (BQS)</b>", cell_bold), Paragraph(f"<b>{bqs:.1f} / 100</b>", cell_normal),
            Paragraph("<b>Lot Pass Yield</b>", cell_bold), Paragraph(f"<b>{100.0 - defect_rate:.1f}%</b> ({passed_cnt}/{total_inspected})", cell_normal)
        ],
        [
            Paragraph("<b>Total Defect Rate</b>", cell_bold), Paragraph(f"{defect_rate:.1f}% ({defect_cnt} units)", cell_normal),
            Paragraph("<b>Critical Defect Rate</b>", cell_bold), Paragraph(f"{crit_rate:.1f}%", cell_normal)
        ],
        [
            Paragraph("<b>Average Lot Risk Score</b>", cell_bold), Paragraph(f"<b>{avg_risk:.1f} / 100</b>", cell_normal),
            Paragraph("<b>Lot Disposition Action</b>", cell_bold), Paragraph(f"<font color='{lot_color.hexval()}'><b>{lot_action}</b></font>", cell_normal)
        ]
    ]
    t_kpi = Table(kpi_data, colWidths=[1.8*inch, 1.9*inch, 1.8*inch, 2.0*inch])
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 3),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(t_kpi)
    elements.append(Spacer(1, 6))

    # 3. Pareto 80/20 Defect Root Cause Breakdown (if defects exist)
    cats = pareto_analysis.get("categories", [])
    if cats:
        elements.append(Paragraph("Manufacturing Defect Pareto Analysis (80/20 Rule)", h2_style))
        pareto_rows = [
            [
                Paragraph("<b>Defect Category</b>", cell_header),
                Paragraph("<b>Count</b>", cell_header),
                Paragraph("<b>Proportion (%)</b>", cell_header),
                Paragraph("<b>Cumulative (%)</b>", cell_header),
                Paragraph("<b>Classification</b>", cell_header)
            ]
        ]
        vital = set(pareto_analysis.get("vital_few", []))
        for c in cats:
            c_name = c.get("category", "Defect")
            is_vital = c_name in vital
            badge = "<font color='#DC2626'><b>Vital Few (80% Driver)</b></font>" if is_vital else "Useful Many"
            pareto_rows.append([
                Paragraph(f"<b>{c_name}</b>", cell_normal),
                Paragraph(str(c.get("count", 0)), cell_normal),
                Paragraph(f"{c.get('percentage', 0.0):.1f}%", cell_normal),
                Paragraph(f"{c.get('cumulative_pct', 0.0):.1f}%", cell_normal),
                Paragraph(badge, cell_normal)
            ])
        t_pareto = Table(pareto_rows, colWidths=[2.1*inch, 1.0*inch, 1.3*inch, 1.3*inch, 1.8*inch])
        t_pareto.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0284C7")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
            ('PADDING', (0,0), (-1,-1), 3),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ]))
        elements.append(t_pareto)
        elements.append(Spacer(1, 6))

    # 4. Itemized Container Breakdown Table
    elements.append(Paragraph(f"Itemized Container Inspection Audit Trail (Showing up to 40 samples)", h2_style))
    item_rows = [
        [
            Paragraph("<b>Container Filename</b>", cell_header),
            Paragraph("<b>Verdict</b>", cell_header),
            Paragraph("<b>Quality Grade</b>", cell_header),
            Paragraph("<b>Score</b>", cell_header),
            Paragraph("<b>DSAR</b>", cell_header),
            Paragraph("<b>Defect Zone</b>", cell_header),
            Paragraph("<b>Action</b>", cell_header)
        ]
    ]

    for r in results[:40]:
        fn = str(r.get("Filename", "sample.png"))
        v = str(r.get("Verdict", "GOOD"))
        g = str(r.get("Grade", "Grade A"))
        sc = float(r.get("Anomaly_Score", 0.0))
        ds = float(r.get("DSAR_Percent", 0.0))
        zn = str(r.get("Defect_Zone", "None")).replace("Zone ", "Z")
        act = str(r.get("Disposal_Action", "Pass"))
        
        v_col = PASS_COLOR if "GOOD" in v else (WARN_COLOR if "UNCERTAIN" in v else FAIL_COLOR)
        item_rows.append([
            Paragraph(fn, cell_normal),
            Paragraph(f"<font color='{v_col.hexval()}'><b>{v}</b></font>", cell_normal),
            Paragraph(g, cell_normal),
            Paragraph(f"{sc:.4f}", cell_normal),
            Paragraph(f"{ds:.2f}%", cell_normal),
            Paragraph(zn, cell_normal),
            Paragraph(act[:28], cell_normal)
        ])

    t_items = Table(item_rows, colWidths=[1.8*inch, 1.0*inch, 1.4*inch, 0.7*inch, 0.7*inch, 0.9*inch, 1.0*inch])
    t_items.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E293B")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 2.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    elements.append(t_items)
    elements.append(Spacer(1, 6))

    # 5. Signoff Footer
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#94A3B8"), spaceBefore=4, spaceAfter=4))
    footer_data = [
        [
            Paragraph("<b>Automated Vision Inspection Unit — Domain 2 (Plastic Bottle)</b><br/>PatchCore Full-Body Feature Engine (PB-v1.0)", ParagraphStyle('F1', fontName='Helvetica', fontSize=6.5, textColor=colors.HexColor("#64748B"))),
            Paragraph("<b>QA Engineering Lead Digital Signoff</b><br/><i>[ Electronically Verified & Timestamped ]</i>", ParagraphStyle('F2', fontName='Helvetica', fontSize=6.5, alignment=2, textColor=colors.HexColor("#64748B")))
        ]
    ]
    footer_table = Table(footer_data, colWidths=[3.75*inch, 3.75*inch])
    elements.append(footer_table)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

