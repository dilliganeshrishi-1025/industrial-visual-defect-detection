"""
Industrial QA PDF Report Generator for Bottle Defect Inspection System.
Uses ReportLab to generate a publication-ready PDF certificate.
"""

import io
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import datetime

def generate_qa_pdf_report(
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
    polar_img_pil: Image.Image,
    timestamp: str = None,
    threshold: float = 0.4263,
    qa_standard: str = "Commercial Beverage (Balanced)",
    grade_reason: str = "",
    risk_score: float = 0.0,
    risk_level: str = "Low",
    quality_score: float = 100.0,
    pattern_cluster: str = "Cluster 00 — Intact / Defect-Free",
    is_unknown: bool = False,
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
    """
    Generates a single-page PDF Certificate of Quality Inspection with Quality Engineer Approval & Permission.
    Returns PDF content as bytes.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    # Auto-resolve Quality Engineer approval status and permission if not explicitly provided
    if not quality_engineer_status:
        if "Grade A" in grade or verdict == "GOOD":
            quality_engineer_status = "APPROVED — GOOD (Permission Granted for Bottling)"
            quality_permission = "PERMISSION GRANTED — APPROVED FOR COMMERCIAL PRODUCTION"
        elif "Grade B" in grade:
            quality_engineer_status = "NOT APPROVED — REWORK REQUIRED (Conditional Permission)"
            quality_permission = "PERMISSION CONDITIONAL — ROUTE TO INDUSTRIAL WASH / RE-INSPECTION"
        elif "Grade C" in grade or "DEFECTIVE" in verdict:
            quality_engineer_status = "NOT APPROVED — REJECTED / DEFECTIVE (Permission Denied)"
            quality_permission = "PERMISSION DENIED — SCRAP / DESTROY (Cap Sealing / Structural Hazard)"
        else:
            quality_engineer_status = "PENDING HUMAN REVIEW — UNVALIDATED (Permission On Hold)"
            quality_permission = "PERMISSION ON HOLD — OUTSIDE VALIDATED DOMAIN"

    if not quality_permission:
        if "APPROVED" in quality_engineer_status and "NOT APPROVED" not in quality_engineer_status:
            quality_permission = "PERMISSION GRANTED — APPROVED FOR COMMERCIAL PRODUCTION"
        elif "REWORK" in quality_engineer_status or "CONDITIONAL" in quality_engineer_status:
            quality_permission = "PERMISSION CONDITIONAL — ROUTE TO INDUSTRIAL WASH / RE-INSPECTION"
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
    
    # Custom Palette
    PRIMARY = colors.HexColor("#0F172A")    # Dark slate
    ACCENT = colors.HexColor("#2563EB")     # Sapphire blue
    PASS_COLOR = colors.HexColor("#16A34A") # Green
    REWORK_COLOR = colors.HexColor("#D97706") # Amber
    FAIL_COLOR = colors.HexColor("#DC2626") # Red
    BG_LIGHT = colors.HexColor("#F8FAFC")
    
    if "Grade A" in grade or verdict == "GOOD":
        verdict_color = PASS_COLOR
        verdict_bg = colors.HexColor("#DCFCE7")
    elif "Grade B" in grade:
        verdict_color = REWORK_COLOR
        verdict_bg = colors.HexColor("#FEF3C7")
    else:
        verdict_color = FAIL_COLOR
        verdict_bg = colors.HexColor("#FEE2E2")
    
    # Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=PRIMARY,
        alignment=0
    )
    
    h2_style = ParagraphStyle(
        'H2Header',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=PRIMARY,
        spaceBefore=2,
        spaceAfter=3
    )
    
    cell_bold = ParagraphStyle('CB', fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=PRIMARY)
    cell_normal = ParagraphStyle('CN', fontName='Helvetica', fontSize=8.5, leading=11, textColor=PRIMARY)
    
    elements = []
    
    # Header Title Block
    header_data = [
        [
            Paragraph("<b>QUALITY CONTROL & INSPECTION CERTIFICATE</b><br/><font size=8 color='#64748B'>GLASS BOTTLE ANOMALY DETECTION & QUALITY TRIAGE SYSTEM</font>", title_style),
            Paragraph(f"<b>REPORT ID:</b> QC-{hash(filename)%100000:05d}<br/><b>DATE:</b> {timestamp}", ParagraphStyle('RightH', fontName='Helvetica', fontSize=8.5, leading=11, alignment=2, textColor=colors.HexColor("#475569")))
        ]
    ]
    header_table = Table(header_data, colWidths=[4.4*inch, 3.0*inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(header_table)
    elements.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT, spaceBefore=2, spaceAfter=6))
    
    # Executive Summary Card
    summary_data = [
        [
            Paragraph(f"<b>INSPECTION VERDICT:</b>", ParagraphStyle('Lbl', fontName='Helvetica-Bold', fontSize=11, textColor=PRIMARY)),
            Paragraph(f"<font color='{verdict_color.hexval()}'><b>{verdict} — {grade}</b></font>", ParagraphStyle('Vect', fontName='Helvetica-Bold', fontSize=12, alignment=2))
        ]
    ]
    summary_table = Table(summary_data, colWidths=[3.5*inch, 3.9*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), verdict_bg),
        ('BOX', (0,0), (-1,-1), 1.2, verdict_color),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 6))
    
    # Quality Engineer Approval Block (Requirement)
    app_is_approved = ("APPROVED" in quality_engineer_status and "NOT APPROVED" not in quality_engineer_status)
    app_is_rework = ("REWORK" in quality_engineer_status or "CONDITIONAL" in quality_permission)
    app_bg = colors.HexColor("#DCFCE7") if app_is_approved else (colors.HexColor("#FEF3C7") if app_is_rework else colors.HexColor("#FEE2E2"))
    app_border = PASS_COLOR if app_is_approved else (REWORK_COLOR if app_is_rework else FAIL_COLOR)
    
    approval_data = [
        [
            Paragraph("<b>Quality Engineer Sign-Off:</b>", cell_bold),
            Paragraph(f"<font color='{app_border.hexval()}'><b>{quality_engineer_status}</b></font>", ParagraphStyle('AppVal', fontName='Helvetica-Bold', fontSize=9, textColor=app_border))
        ],
        [
            Paragraph("<b>Production Permission:</b>", cell_bold),
            Paragraph(f"<font color='{app_border.hexval()}'><b>{quality_permission}</b></font>", ParagraphStyle('PermVal', fontName='Helvetica-Bold', fontSize=8.5, textColor=app_border))
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
        ('PADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(t_approval)
    elements.append(Spacer(1, 8))
    
    # Images Section (Original, Heatmap, Polar Ribbon)
    elements.append(Paragraph("Visual Evidence & Polar Discontinuity Analysis", h2_style))
    elements.append(Spacer(1, 3))
    
    def pil_to_rl(pil_img, w_inch, h_inch):
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        return RLImage(buf, width=w_inch*inch, height=h_inch*inch)
    
    rl_orig = pil_to_rl(orig_img_pil, 2.1, 2.1)
    rl_heat = pil_to_rl(heatmap_img_pil, 2.1, 2.1)
    rl_polar = pil_to_rl(polar_img_pil, 2.1, 2.1)
    
    img_table_data = [
        [rl_orig, rl_heat, rl_polar],
        [
            Paragraph("<b>(a) Input Top-View</b>", ParagraphStyle('C1', fontName='Helvetica-Bold', fontSize=7.5, alignment=1)),
            Paragraph("<b>(b) Spatial Anomaly Map</b>", ParagraphStyle('C2', fontName='Helvetica-Bold', fontSize=7.5, alignment=1)),
            Paragraph("<b>(c) Polar Unwrapped Strip</b>", ParagraphStyle('C3', fontName='Helvetica-Bold', fontSize=7.5, alignment=1))
        ]
    ]
    img_table = Table(img_table_data, colWidths=[2.45*inch, 2.45*inch, 2.45*inch])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,0), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0,0), (-1,-1), 2),
    ]))
    elements.append(img_table)
    elements.append(Spacer(1, 8))
    
    # Metrics Table
    elements.append(Paragraph("Quantitative Metric Triage & Industrial Action", h2_style))
    elements.append(Spacer(1, 3))
    
    metrics_data = [
        [Paragraph("Sample Filename", cell_bold), Paragraph(filename, cell_normal)],
        [Paragraph("Input Compatibility Status", cell_bold), Paragraph(f"<b>{input_compatibility_status}</b> ({input_compatibility_score:.1f}%) — <i>{inspection_confidence}</i>", cell_normal)],
        [Paragraph("Human Review Required", cell_bold), Paragraph(f"<font color='{FAIL_COLOR.hexval()}'><b>YES — Engineering Validation Required</b></font>" if human_review_required else f"<font color='{PASS_COLOR.hexval()}'><b>NO (Automated Inspection Valid)</b></font>", cell_normal)],
        [Paragraph("Operating QA Standard", cell_bold), Paragraph(f"<b>{qa_standard}</b> (Threshold: {threshold:.4f})", cell_normal)],
        [Paragraph("Anomaly Matching Score", cell_bold), Paragraph(f"{anomaly_score:.4f}", cell_normal)],
        [Paragraph("Defective Surface Area Ratio (DSAR)", cell_bold), Paragraph(f"<b>{dsar_percent:.2f}%</b>", cell_normal)],
        [Paragraph("Anatomical Defect Zone", cell_bold), Paragraph(f"<b>{defect_zone}</b>", cell_normal)],
        [Paragraph("Classified Defect Subtype", cell_bold), Paragraph(defect_type, cell_normal)],
        [Paragraph("Grading Classification", cell_bold), Paragraph(f"<b>{grade}</b>", cell_normal)],
        [Paragraph("Overall Quality Score", cell_bold), Paragraph(f"<b>{quality_score:.1f} / 100</b>", cell_normal)],
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
    elements.append(Spacer(1, 8))
    
    # Footer Signoff
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#94A3B8"), spaceBefore=2, spaceAfter=4))
    
    footer_data = [
        [
            Paragraph("<b>Automated Vision Inspection Unit — Glass Bottle Domain</b><br/>PatchCore Deep Feature Engine (MVTec AD Validated)", ParagraphStyle('F1', fontName='Helvetica', fontSize=7, textColor=colors.HexColor("#64748B"))),
            Paragraph(f"<b>Quality Engineer Sign-Off:</b> {quality_engineer_name}<br/><i>[ Decision: {quality_engineer_status[:35]} ]</i>", ParagraphStyle('F2', fontName='Helvetica', fontSize=7, alignment=2, textColor=colors.HexColor("#64748B")))
        ]
    ]
    footer_table = Table(footer_data, colWidths=[3.7*inch, 3.7*inch])
    footer_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(footer_table)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def generate_batch_qa_pdf_report(
    batch_id: str,
    total_inspected: int,
    defect_count: int,
    defect_rate: float,
    critical_defect_rate: float,
    batch_quality_score: float,
    avg_risk_score: float,
    results: list,
    timestamp: str = None,
    qa_standard: str = "ISO 2859-1 / DIN EN ISO 9001:2015 Glass Quality Standard",
    quality_engineer_name: str = "Lead Quality Engineer",
    quality_engineer_notes: str = ""
) -> bytes:
    """
    Generates a publication-ready single/multi-page PDF Batch Quality Inspection & Audit Report for Glass Bottles.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine batch release permission
    if batch_quality_score >= 85.0 and defect_rate <= 5.0:
        batch_status = "APPROVED — LOT RELEASED (Pass)"
        batch_permission = "PERMISSION GRANTED — BATCH RELEASED FOR COMMERCIAL BOTTLING"
        verdict_color = colors.HexColor("#16A34A")
        verdict_bg = colors.HexColor("#DCFCE7")
    elif batch_quality_score >= 65.0:
        batch_status = "CONDITIONAL — REWORK & 100% RE-INSPECTION REQUIRED"
        batch_permission = "PERMISSION CONDITIONAL — HOLD LOT & ROUTE TO RE-INSPECTION"
        verdict_color = colors.HexColor("#D97706")
        verdict_bg = colors.HexColor("#FEF3C7")
    else:
        batch_status = "REJECTED — LOT QUARANTINED (High Defect Density)"
        batch_permission = "PERMISSION DENIED — IMMEDIATE LOT QUARANTINE & SCRAP AUDIT"
        verdict_color = colors.HexColor("#DC2626")
        verdict_bg = colors.HexColor("#FEE2E2")

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
    PRIMARY = colors.HexColor("#0F172A")
    ACCENT = colors.HexColor("#2563EB")
    cell_bold = ParagraphStyle('CB', fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=PRIMARY)
    cell_normal = ParagraphStyle('CN', fontName='Helvetica', fontSize=8.5, leading=11, textColor=PRIMARY)

    elements = []

    # Title Block
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=17, leading=21, textColor=PRIMARY)
    header_data = [
        [
            Paragraph("<b>BATCH QUALITY AUDIT & LOT RELEASE REPORT</b><br/><font size=8 color='#64748B'>GLASS BOTTLE PRODUCTION LINE INSPECTION INTELLIGENCE</font>", title_style),
            Paragraph(f"<b>BATCH ID:</b> {batch_id}<br/><b>DATE:</b> {timestamp}", ParagraphStyle('RightH', fontName='Helvetica', fontSize=8.5, leading=11, alignment=2, textColor=colors.HexColor("#475569")))
        ]
    ]
    header_table = Table(header_data, colWidths=[4.4*inch, 3.0*inch])
    header_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    elements.append(header_table)
    elements.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT, spaceBefore=2, spaceAfter=6))

    # Executive Summary Card
    summary_data = [
        [
            Paragraph("<b>LOT QUALITY DECISION:</b>", ParagraphStyle('Lbl', fontName='Helvetica-Bold', fontSize=10.5, textColor=PRIMARY)),
            Paragraph(f"<font color='{verdict_color.hexval()}'><b>{batch_status}</b></font>", ParagraphStyle('Vect', fontName='Helvetica-Bold', fontSize=11, alignment=2))
        ]
    ]
    summary_table = Table(summary_data, colWidths=[3.2*inch, 4.2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), verdict_bg),
        ('BOX', (0,0), (-1,-1), 1.2, verdict_color),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 6))

    # Quality Engineer Sign-Off Block
    approval_data = [
        [Paragraph("<b>Quality Engineer Sign-Off:</b>", cell_bold), Paragraph(f"<font color='{verdict_color.hexval()}'><b>{batch_status}</b></font>", cell_bold)],
        [Paragraph("<b>Production Permission:</b>", cell_bold), Paragraph(f"<font color='{verdict_color.hexval()}'><b>{batch_permission}</b></font>", cell_bold)],
        [Paragraph("<b>Certified Inspector & Time:</b>", cell_bold), Paragraph(f"{quality_engineer_name} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Timestamp:</b> {timestamp}", cell_normal)],
        [Paragraph("<b>Operating QA Standard:</b>", cell_bold), Paragraph(qa_standard, cell_normal)]
    ]
    if quality_engineer_notes:
        approval_data.append([Paragraph("<b>Audit Notes:</b>", cell_bold), Paragraph(f"<i>{quality_engineer_notes}</i>", cell_normal)])

    t_approval = Table(approval_data, colWidths=[2.2*inch, 5.2*inch])
    t_approval.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(t_approval)
    elements.append(Spacer(1, 8))

    # KPI Metrics Table
    elements.append(Paragraph("<b>📊 Lot Quality Performance Indicators (KPIs)</b>", ParagraphStyle('H2', fontName='Helvetica-Bold', fontSize=10, textColor=PRIMARY)))
    elements.append(Spacer(1, 3))
    kpi_data = [
        [
            Paragraph("<b>Batch Quality Score (BQS)</b>", cell_bold),
            Paragraph(f"<b>{batch_quality_score:.1f} / 100</b>", ParagraphStyle('KPI1', fontName='Helvetica-Bold', fontSize=10, textColor=verdict_color)),
            Paragraph("<b>Total Containers Inspected</b>", cell_bold),
            Paragraph(f"<b>{total_inspected}</b>", cell_normal)
        ],
        [
            Paragraph("<b>Total Defective Units</b>", cell_bold),
            Paragraph(f"<b>{defect_count}</b>", cell_normal),
            Paragraph("<b>Defect Rate (%)</b>", cell_bold),
            Paragraph(f"<b>{defect_rate:.2f}%</b>", cell_bold)
        ],
        [
            Paragraph("<b>Critical Defect Rate</b>", cell_bold),
            Paragraph(f"<b>{critical_defect_rate:.2f}%</b>", cell_normal),
            Paragraph("<b>Average Risk Score</b>", cell_bold),
            Paragraph(f"<b>{avg_risk_score:.1f} / 100</b>", cell_normal)
        ]
    ]
    t_kpi = Table(kpi_data, colWidths=[2.0*inch, 1.7*inch, 2.0*inch, 1.7*inch])
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(t_kpi)
    elements.append(Spacer(1, 8))

    # Sample Inspection Log Table (Top items)
    elements.append(Paragraph("<b>📋 Lot Inspection Summary (Sample Log)</b>", ParagraphStyle('H2', fontName='Helvetica-Bold', fontSize=10, textColor=PRIMARY)))
    elements.append(Spacer(1, 3))

    table_header = [
        Paragraph("<b>#</b>", cell_bold),
        Paragraph("<b>Container File</b>", cell_bold),
        Paragraph("<b>Verdict</b>", cell_bold),
        Paragraph("<b>Grade</b>", cell_bold),
        Paragraph("<b>Score</b>", cell_bold),
        Paragraph("<b>Risk</b>", cell_bold),
        Paragraph("<b>Zone / Type</b>", cell_bold)
    ]
    log_rows = [table_header]

    for idx, item in enumerate(results[:20]):
        v = str(item.get("verdict", "GOOD"))
        v_col = colors.HexColor("#16A34A") if v == "GOOD" else colors.HexColor("#DC2626")
        log_rows.append([
            Paragraph(str(idx + 1), cell_normal),
            Paragraph(str(item.get("filename", item.get("Filename", "sample.png")))[:22], cell_normal),
            Paragraph(f"<font color='{v_col.hexval()}'><b>{v}</b></font>", cell_normal),
            Paragraph(str(item.get("grade", item.get("Grade", "Grade A")))[:18], cell_normal),
            Paragraph(f"{float(item.get('anomaly_score', item.get('Anomaly_Score', 0.0))):.4f}", cell_normal),
            Paragraph(str(item.get("risk_level", item.get("Risk_Level", "Low"))), cell_normal),
            Paragraph(str(item.get("defect_zone", item.get("Defect_Zone", item.get("Category", "Nominal"))))[:18], cell_normal)
        ])

    t_log = Table(log_rows, colWidths=[0.35*inch, 2.0*inch, 0.9*inch, 1.4*inch, 0.8*inch, 0.75*inch, 1.2*inch])
    t_log.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('GRID', (0,0), (-1,-1), 0.4, colors.HexColor("#CBD5E1")),
        ('PADDING', (0,0), (-1,-1), 2.5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(t_log)
    elements.append(Spacer(1, 8))

    # Footer
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#94A3B8"), spaceBefore=2, spaceAfter=4))
    footer_data = [
        [
            Paragraph("<b>Automated Vision Inspection Unit — Glass Bottle Batch Quality System</b><br/>PatchCore Deep Feature Engine (MVTec AD Validated)", ParagraphStyle('F1', fontName='Helvetica', fontSize=7, textColor=colors.HexColor("#64748B"))),
            Paragraph(f"<b>Quality Engineer Sign-Off:</b> {quality_engineer_name}<br/><i>[ Decision: {batch_status[:35]} ]</i>", ParagraphStyle('F2', fontName='Helvetica', fontSize=7, alignment=2, textColor=colors.HexColor("#64748B")))
        ]
    ]
    footer_table = Table(footer_data, colWidths=[3.7*inch, 3.7*inch])
    footer_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    elements.append(footer_table)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

