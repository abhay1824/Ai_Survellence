import io
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_incident_pdf(event_data: dict) -> bytes:
    """
    Generates an official Eagle Eye SOC Security Incident PDF Report using ReportLab.
    Returns bytes of the generated PDF file.
    """
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

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0F172A')
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748B')
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )

    bold_body_style = ParagraphStyle(
        'BoldBodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#0F172A')
    )

    xai_box_style = ParagraphStyle(
        'XAIBoxText',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#0F172A')
    )

    story = []

    # 1. Header Banner & Title
    event_id = event_data.get('event_id', 'UNKNOWN-ID')
    timestamp_raw = event_data.get('timestamp', datetime.utcnow().isoformat())
    try:
        dt = datetime.fromisoformat(timestamp_raw.replace('Z', '+00:00'))
        formatted_ts = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    except Exception:
        formatted_ts = str(timestamp_raw)

    camera_id = event_data.get('camera_id', 'CAM-01')
    threat_level = str(event_data.get('threat_level', 'MEDIUM')).upper()
    threat_score = event_data.get('threat_score', 50)
    event_type = event_data.get('event_type', 'Suspicious Activity')
    confidence = event_data.get('confidence', 0.85)

    # Color indicator for threat level
    badge_bg = colors.HexColor('#DC2626') if threat_level == 'CRITICAL' else \
               colors.HexColor('#EA580C') if threat_level == 'HIGH' else \
               colors.HexColor('#D97706') if threat_level == 'MEDIUM' else colors.HexColor('#16A34A')

    header_table_data = [
        [
            Paragraph("<b>EAGLE EYE SMART SURVEILLANCE SYSTEM</b><br/><font color='#64748B' size=8>SECURITY OPERATIONS CENTER • OFFICIAL INCIDENT AUDIT REPORT</font>", title_style),
            Paragraph(f"<font color='white'><b>CLASSIFICATION: {threat_level}</b></font>", ParagraphStyle('Badge', fontName='Helvetica-Bold', fontSize=9, alignment=1))
        ]
    ]

    header_table = Table(header_table_data, colWidths=[400, 140])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (1, 0), (1, 0), badge_bg),
        ('ALIGN', (1, 0), (1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (1, 0), (1, 0), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=10))

    # 2. Metadata Overview Table
    meta_data = [
        [
            Paragraph("<b>Incident ID:</b>", bold_body_style), Paragraph(str(event_id), body_style),
            Paragraph("<b>Timestamp:</b>", bold_body_style), Paragraph(formatted_ts, body_style)
        ],
        [
            Paragraph("<b>Camera Node:</b>", bold_body_style), Paragraph(str(camera_id), body_style),
            Paragraph("<b>Violation Class:</b>", bold_body_style), Paragraph(str(event_type), body_style)
        ],
        [
            Paragraph("<b>Threat Level:</b>", bold_body_style), Paragraph(f"<font color='{badge_bg.hexval()}'><b>{threat_level}</b></font>", body_style),
            Paragraph("<b>Threat Score:</b>", bold_body_style), Paragraph(f"<b>{threat_score} / 100</b>", body_style)
        ],
        [
            Paragraph("<b>Detection Confidence:</b>", bold_body_style), Paragraph(f"{round(confidence * 100, 1)}%", body_style),
            Paragraph("<b>Status:</b>", bold_body_style), Paragraph("ACTIVE AUDIT LOGGED", body_style)
        ]
    ]

    meta_table = Table(meta_data, colWidths=[100, 170, 100, 170])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#F1F5F9')),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # 3. Evidence Image Snapshot (If image path is available)
    image_path = event_data.get('image_path')
    if image_path:
        # Normalize local image path
        local_img = image_path.lstrip('/')
        if os.path.exists(local_img):
            try:
                story.append(Paragraph("EVIDENCE CAPTURE SNAPSHOT", section_heading))
                story.append(Spacer(1, 4))
                img = Image(local_img, width=6.8*inch, height=3.5*inch)
                story.append(img)
                story.append(Spacer(1, 10))
            except Exception as img_err:
                print(f"PDF Generator: Could not embed image {local_img}: {img_err}")

    # 4. XAI Structured Explanation Block
    xai = event_data.get('xai_explanation', {})
    story.append(Paragraph("EXPLAINABLE AI (XAI) EVIDENCE REASONING", section_heading))
    story.append(Spacer(1, 4))

    summary_text = xai.get('summary', f"Event {event_type} registered on {camera_id} with threat level {threat_level}.")
    evidence_list = xai.get('evidence_used', [])
    evidence_bullets = "<br/>".join([f"• {e}" for e in evidence_list]) if evidence_list else "• Object detections registered on frame buffer."
    
    beh = xai.get('behavioral_analysis', {})
    obs_actions = beh.get('observed_actions', 'Normal tracking movement observed.')
    state = beh.get('movement_state', 'Stationary')
    speed = beh.get('speed_m_s', 0.0)

    spat = xai.get('spatial_risk', {})
    nearby = spat.get('nearby_persons', [])
    nearby_text = ", ".join([f"ID {p.get('track_id')} ({p.get('distance_meters')}m)" for p in nearby]) if nearby else "No nearby individuals detected within 5.0m danger zone."

    rec_actions = xai.get('recommended_actions', ['Dispatch local patrol', 'Acknowledge alert'])
    actions_text = "<br/>".join([f"[  ] {a}" for a in rec_actions])

    xai_table_data = [
        [Paragraph("<b>ALERT SUMMARY</b>", bold_body_style)],
        [Paragraph(summary_text, body_style)],
        [Paragraph("<b>EVIDENCE IDENTIFIED</b>", bold_body_style)],
        [Paragraph(evidence_bullets, body_style)],
        [Paragraph("<b>BEHAVIORAL & MOVEMENT ANALYSIS</b>", bold_body_style)],
        [Paragraph(f"<b>Movement State:</b> {state} ({speed} m/s)<br/><b>Observed Action:</b> {obs_actions}", body_style)],
        [Paragraph("<b>SPATIAL PROXIMITY & DANGER ZONE</b>", bold_body_style)],
        [Paragraph(nearby_text, body_style)],
        [Paragraph("<b>RECOMMENDED STANDARD OPERATING PROCEDURES (SOP)</b>", bold_body_style)],
        [Paragraph(actions_text, body_style)]
    ]

    xai_table = Table(xai_table_data, colWidths=[540])
    xai_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
        ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#F1F5F9')),
        ('BACKGROUND', (0, 2), (0, 2), colors.HexColor('#F1F5F9')),
        ('BACKGROUND', (0, 4), (0, 4), colors.HexColor('#F1F5F9')),
        ('BACKGROUND', (0, 6), (0, 6), colors.HexColor('#F1F5F9')),
        ('BACKGROUND', (0, 8), (0, 8), colors.HexColor('#F1F5F9')),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(xai_table)
    story.append(Spacer(1, 14))

    # 5. Fight Detection Analysis Section
    fight_detected = event_data.get('fight_detected', False)
    fight_confidence = event_data.get('fight_confidence', 0.0)
    fight_participants = event_data.get('fight_participants', [])
    fight_duration = event_data.get('fight_duration_sec', 0.0)

    fight_header_bg = colors.HexColor('#DC2626') if fight_detected else colors.HexColor('#16A34A')
    fight_status_text = "DETECTED" if fight_detected else "NOT DETECTED"
    fight_conf_pct = f"{round(fight_confidence * 100, 1)}%" if fight_detected else "N/A"
    participants_str = ", ".join([f"Person-{p}" for p in fight_participants]) if fight_participants else "N/A"
    fight_score_contrib = (
        40 if fight_confidence > 0.95 else
        30 if fight_confidence > 0.90 else
        20 if fight_confidence >= 0.85 else 0
    ) if fight_detected else 0

    story.append(Paragraph("BEHAVIORAL EVENT SUMMARY — FIGHT DETECTION ANALYSIS", section_heading))
    story.append(Spacer(1, 4))

    fight_table_data = [
        [
            Paragraph(f"<font color='white'><b>FIGHT STATUS: {fight_status_text}</b></font>",
                      ParagraphStyle('FightBadge', fontName='Helvetica-Bold', fontSize=10, alignment=1)),
            Paragraph("", body_style)
        ],
        [
            Paragraph("<b>Fight Detected:</b>", bold_body_style),
            Paragraph(f"<font color='{'#DC2626' if fight_detected else '#16A34A'}'><b>{'YES' if fight_detected else 'NO'}</b></font>", body_style)
        ],
        [
            Paragraph("<b>Classification Confidence:</b>", bold_body_style),
            Paragraph(fight_conf_pct, body_style)
        ],
        [
            Paragraph("<b>Duration:</b>", bold_body_style),
            Paragraph(f"{fight_duration}s" if fight_detected and fight_duration > 0 else "N/A", body_style)
        ],
        [
            Paragraph("<b>Fight Participants (ByteTrack IDs):</b>", bold_body_style),
            Paragraph(participants_str, body_style)
        ],
        [
            Paragraph("<b>Threat Score Contribution:</b>", bold_body_style),
            Paragraph(f"+{fight_score_contrib} pts" if fight_score_contrib > 0 else "0 pts", body_style)
        ],
        [
            Paragraph("<b>Detection Method:</b>", bold_body_style),
            Paragraph("YOLO11 Classification — Temporal Confirmation (7/10 frames)", body_style)
        ],
        [
            Paragraph("<b>Safety Note:</b>", bold_body_style),
            Paragraph("Observations describe detected physical evidence only. No future crime predictions are made.", body_style)
        ],
    ]

    fight_table = Table(fight_table_data, colWidths=[200, 340])
    fight_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), fight_header_bg),
        ('SPAN', (0, 0), (-1, 0)),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 1), (-1, -1), 0.5, colors.HexColor('#F1F5F9')),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(fight_table)
    story.append(Spacer(1, 14))

    # 6. Accident Detection Analysis Section
    accident_detected = event_data.get('accident_detected', False)
    accident_confidence = event_data.get('accident_confidence', 0.0)

    accident_header_bg = colors.HexColor('#DC2626') if accident_detected else colors.HexColor('#16A34A')
    accident_status_text = "DETECTED" if accident_detected else "NOT DETECTED"
    accident_conf_pct = f"{round(accident_confidence * 100, 1)}%" if accident_detected else "N/A"
    accident_score_contrib = 35 if accident_detected else 0

    story.append(Paragraph("INCIDENT INTELLIGENCE — ACCIDENT & COLLISION ANALYSIS", section_heading))
    story.append(Spacer(1, 4))

    accident_table_data = [
        [
            Paragraph(f"<font color='white'><b>ACCIDENT STATUS: {accident_status_text}</b></font>",
                      ParagraphStyle('AccidentBadge', fontName='Helvetica-Bold', fontSize=10, alignment=1)),
            Paragraph("", body_style)
        ],
        [
            Paragraph("<b>Accident Detected:</b>", bold_body_style),
            Paragraph(f"<font color='{'#DC2626' if accident_detected else '#16A34A'}'><b>{'YES' if accident_detected else 'NO'}</b></font>", body_style)
        ],
        [
            Paragraph("<b>Detection Confidence:</b>", bold_body_style),
            Paragraph(accident_conf_pct, body_style)
        ],
        [
            Paragraph("<b>Threat Score Contribution:</b>", bold_body_style),
            Paragraph(f"+{accident_score_contrib} pts (Base: HIGH)" if accident_score_contrib > 0 else "0 pts", body_style)
        ],
        [
            Paragraph("<b>Detection Method:</b>", bold_body_style),
            Paragraph("YOLO11s Object Detection (Threshold >= 70%)", body_style)
        ],
        [
            Paragraph("<b>Safety & Compliance Note:</b>", bold_body_style),
            Paragraph("Observations strictly document detected collision evidence. Non-predictive safety protocols enforced.", body_style)
        ],
    ]

    accident_table = Table(accident_table_data, colWidths=[200, 340])
    accident_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), accident_header_bg),
        ('SPAN', (0, 0), (-1, 0)),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 1), (-1, -1), 0.5, colors.HexColor('#F1F5F9')),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(accident_table)
    story.append(Spacer(1, 14))

    # 7. Operator Sign-off & Audit Block
    story.append(KeepTogether([
        Paragraph("SECURITY OPERATOR SIGN-OFF & AUDIT VERIFICATION", section_heading),
        Spacer(1, 4),
        Table([
            [
                Paragraph("<b>Reviewing SOC Officer Name:</b> ___________________________", body_style),
                Paragraph("<b>Badge / ID Number:</b> _____________", body_style)
            ],
            [
                Paragraph("<b>Officer Action Taken:</b> [  ] Escalated  [  ] Dispatched  [  ] False Positive", body_style),
                Paragraph("<b>Officer Signature:</b> _________________", body_style)
            ],
            [
                Paragraph(f"<font color='#64748B' size=7>Report auto-generated by Eagle Eye XAI Engine on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC. Non-predictive safety rules enforced.</font>", body_style),
                Paragraph("<font color='#64748B' size=7>Page 1 of 1</font>", ParagraphStyle('RightSmall', fontName='Helvetica', fontSize=7, alignment=2))
            ]
        ], colWidths=[340, 200], style=[
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FFFFFF')),
            ('PADDING', (0, 0), (-1, -1), 6),
        ])
    ]))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def generate_pdf_report(start_date=None, end_date=None, events=None, stats=None) -> io.BytesIO:
    """
    Generates a full Eagle Eye Security Intelligence Summary PDF Report using ReportLab.
    Returns an io.BytesIO buffer positioned at start.
    """
    if events is None:
        events = []
    if stats is None:
        stats = {
            "total_events": len(events),
            "critical_count": sum(1 for e in events if e.get("threat_level") == "CRITICAL"),
            "high_count": sum(1 for e in events if e.get("threat_level") == "HIGH"),
            "medium_count": sum(1 for e in events if e.get("threat_level") == "MEDIUM"),
            "low_count": sum(1 for e in events if e.get("threat_level") == "LOW"),
        }

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
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0F172A')
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )

    bold_body_style = ParagraphStyle(
        'BoldBodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#0F172A')
    )

    story = []

    # Title & Header
    header_table_data = [
        [
            Paragraph("<b>EAGLE EYE SMART SURVEILLANCE SYSTEM</b><br/><font color='#64748B' size=8>SECURITY INTELLIGENCE SUMMARY REPORT</font>", title_style),
            Paragraph(f"<font color='white'><b>GEN TIME: {datetime.utcnow().strftime('%H:%M UTC')}</b></font>", ParagraphStyle('Badge', fontName='Helvetica-Bold', fontSize=9, alignment=1))
        ]
    ]

    header_table = Table(header_table_data, colWidths=[400, 140])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#1E293B')),
        ('ALIGN', (1, 0), (1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (1, 0), (1, 0), 6),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=10))

    # Summary Statistics Block
    stats_data = [
        [
            Paragraph("<b>Total Logged Events:</b>", bold_body_style), Paragraph(str(stats.get('total_events', 0)), body_style),
            Paragraph("<b>Critical Threats:</b>", bold_body_style), Paragraph(f"<font color='#DC2626'><b>{stats.get('critical_count', 0)}</b></font>", body_style)
        ],
        [
            Paragraph("<b>High Threats:</b>", bold_body_style), Paragraph(f"<font color='#EA580C'><b>{stats.get('high_count', 0)}</b></font>", body_style),
            Paragraph("<b>Medium Threats:</b>", bold_body_style), Paragraph(f"<font color='#D97706'><b>{stats.get('medium_count', 0)}</b></font>", body_style)
        ],
        [
            Paragraph("<b>Low Threats:</b>", bold_body_style), Paragraph(f"<font color='#16A34A'><b>{stats.get('low_count', 0)}</b></font>", body_style),
            Paragraph("<b>Date Filter:</b>", bold_body_style), Paragraph(f"{start_date or 'ALL'} to {end_date or 'NOW'}", body_style)
        ]
    ]

    stats_table = Table(stats_data, colWidths=[110, 160, 110, 160])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#F1F5F9')),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 10))

    # Events Table
    story.append(Paragraph("DETAILED AUDIT LOG", section_heading))
    story.append(Spacer(1, 4))

    events_table_data = [
        [
            Paragraph("<b>Event ID</b>", bold_body_style),
            Paragraph("<b>Timestamp</b>", bold_body_style),
            Paragraph("<b>Camera</b>", bold_body_style),
            Paragraph("<b>Type</b>", bold_body_style),
            Paragraph("<b>Threat Level</b>", bold_body_style),
            Paragraph("<b>Score</b>", bold_body_style)
        ]
    ]

    for ev in events[:50]:  # Limit to 50 items per report
        level = str(ev.get("threat_level", "LOW")).upper()
        lvl_color = '#DC2626' if level == 'CRITICAL' else '#EA580C' if level == 'HIGH' else '#D97706' if level == 'MEDIUM' else '#16A34A'
        ts = str(ev.get("timestamp", ""))[:19].replace("T", " ")
        events_table_data.append([
            Paragraph(str(ev.get("event_id", "")), body_style),
            Paragraph(ts, body_style),
            Paragraph(str(ev.get("camera_id", "")), body_style),
            Paragraph(str(ev.get("event_type", "")), body_style),
            Paragraph(f"<font color='{lvl_color}'><b>{level}</b></font>", body_style),
            Paragraph(str(ev.get("threat_score", 0)), body_style)
        ])

    if len(events_table_data) == 1:
        events_table_data.append([Paragraph("No events recorded for specified parameters.", body_style), "", "", "", "", ""])

    events_table = Table(events_table_data, colWidths=[100, 110, 75, 135, 75, 45])
    events_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(events_table)

    doc.build(story)
    buffer.seek(0)
    return buffer

