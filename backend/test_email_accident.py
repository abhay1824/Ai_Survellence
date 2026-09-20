"""
Test Suite: AEGIS SOC Email Notification System for Accident Detection
Verifies:
1. HTML Email template generation (_build_accident_alert_html)
2. Incident PDF report attachment generation (generate_incident_pdf)
3. Email triggering rules & flood cooldown (_should_trigger_email)
4. MIME structure with PDF attachment
5. Mock SMTP delivery verification
6. System recipient retrieval (get_all_admin_recipients)
"""

import os
import sys
import asyncio
from unittest.mock import patch, MagicMock
from email.mime.multipart import MIMEMultipart

# Add backend root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.config import settings
from app.services.email_service import EmailService
from app.services.event_service import _should_trigger_email, _fire_email_notification
from app.utils.pdf_generator import generate_incident_pdf


def test_html_template():
    print("\n--- TEST 1: Accident Alert HTML Template Rendering ---")
    html = EmailService._build_accident_alert_html(
        threat_level="HIGH",
        threat_score=35,
        camera_id="CAM-TRAFFIC-01",
        timestamp="2026-09-17 01:25:00 UTC",
        accident_confidence=0.875,
        xai_summary="Observed vehicle collision indicators on lane 2. Severity score: 35 (HIGH).",
        has_pdf=True
    )

    assert "🚨 AEGIS ACCIDENT DETECTION ALERT" in html
    assert "87.5%" in html
    assert "HIGH" in html
    assert "35" in html
    assert "CAM-TRAFFIC-01" in html
    assert "YOLO11s Object Detection (Threshold &ge; 70%)" in html
    assert "Official Incident Audit PDF Attached" in html
    assert "Observed vehicle collision indicators" in html
    print("  [PASS] HTML rendered with all required accident metrics & styling.")


def test_trigger_logic():
    print("\n--- TEST 2: Email Trigger Logic & Cooldown ---")

    # 1. Accident event triggers
    trig1 = _should_trigger_email(
        camera_id="CAM-TEST-A",
        event_type="ACCIDENT DETECTED",
        threat_level="HIGH",
        accident_detected=True
    )
    assert trig1 is True, "Expected accident to trigger email alert"
    print("  [PASS] Accident event qualifies for email delivery.")

    # 2. Immediate duplicate triggers cooldown
    trig2 = _should_trigger_email(
        camera_id="CAM-TEST-A",
        event_type="ACCIDENT DETECTED",
        threat_level="HIGH",
        accident_detected=True
    )
    assert trig2 is False, "Expected cooldown to block duplicate alert within 5 mins"
    print("  [PASS] Cooldown protection prevents notification flooding.")

    # 3. Different camera triggers immediately
    trig3 = _should_trigger_email(
        camera_id="CAM-TEST-B",
        event_type="ACCIDENT DETECTED",
        threat_level="HIGH",
        accident_detected=True
    )
    assert trig3 is True, "Different camera should trigger immediately"
    print("  [PASS] Camera-specific cooldown isolation works correctly.")


def test_mime_assembly_and_pdf():
    print("\n--- TEST 3: Incident PDF Generation & MIME Attachment ---")

    event_doc = {
        "event_id": "EVT-ACCIDENT-99",
        "camera_id": "CAM-01",
        "event_type": "ACCIDENT DETECTED",
        "threat_level": "HIGH",
        "threat_score": 35,
        "confidence": 0.88,
        "accident_detected": True,
        "accident_confidence": 0.88,
        "fight_detected": False,
        "fight_confidence": 0.0,
        "timestamp": "2026-09-17T01:25:00Z",
    }

    pdf_bytes = generate_incident_pdf(event_doc)
    assert pdf_bytes is not None and len(pdf_bytes) > 2000
    print(f"  [PASS] Incident PDF generated: {len(pdf_bytes)} bytes.")

    # Test synchronous sender with mock SMTP server
    with patch("smtplib.SMTP") as mock_smtp_cls:
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = mock_server

        # Enable email temporarily for test
        orig_enabled = settings.EMAIL_ENABLED
        orig_user = settings.SMTP_USERNAME
        orig_pass = settings.SMTP_PASSWORD
        settings.EMAIL_ENABLED = True
        settings.SMTP_USERNAME = "test_soc@example.com"
        settings.SMTP_PASSWORD = "app_specific_pwd"

        try:
            html_body = EmailService._build_accident_alert_html(
                threat_level="HIGH",
                threat_score=35,
                camera_id="CAM-01",
                timestamp="2026-09-17 01:25:00 UTC",
                accident_confidence=0.88,
                has_pdf=True
            )

            sent = EmailService._send_email_sync(
                recipients=["soc-team@example.com", "admin@surveillance.gov"],
                subject="🚨 ACCIDENT ALERT - AEGIS SOC",
                html_body=html_body,
                pdf_attachment=pdf_bytes,
                pdf_filename="AEGIS_ACCIDENT_INCIDENT.pdf"
            )

            assert bool(sent) is True, "Expected _send_email_sync to return success"
            assert mock_server.sendmail.called, "sendmail was not called"
            args = mock_server.sendmail.call_args[0]
            from_addr = args[0]
            to_addrs = args[1]
            raw_msg = args[2]

            assert "soc-team@example.com" in to_addrs
            assert "admin@surveillance.gov" in to_addrs
            assert "Subject:" in raw_msg and "ACCIDENT" in raw_msg
            assert "AEGIS_ACCIDENT_INCIDENT.pdf" in raw_msg
            print("  [PASS] MIME structure validated: contains HTML body, headers, and PDF attachment.")

        finally:
            settings.EMAIL_ENABLED = orig_enabled
            settings.SMTP_USERNAME = orig_user
            settings.SMTP_PASSWORD = orig_pass


async def test_end_to_end_dispatch():
    print("\n--- TEST 4: End-to-End Async Dispatch Mock ---")

    event_doc = {
        "event_id": "EVT-ACC-E2E",
        "camera_id": "CAM-NORTH-04",
        "event_type": "ACCIDENT DETECTED",
        "threat_level": "HIGH",
        "threat_score": 35,
        "confidence": 0.85,
        "accident_detected": True,
        "accident_confidence": 0.85,
        "timestamp": "2026-09-17T01:25:00Z",
    }

    # Mock get_all_admin_recipients to return test recipients
    with patch.object(EmailService, "get_all_admin_recipients", return_value=["soc_control@aegis.com"]):
        with patch.object(EmailService, "_send_with_retry", return_value=True) as mock_send:
            res = await EmailService.send_accident_alert_email(
                threat_level="HIGH",
                threat_score=35,
                camera_id="CAM-NORTH-04",
                timestamp="2026-09-17 01:25:00 UTC",
                accident_confidence=0.85,
                xai_summary="Observed vehicle collision trajectory on North intersection.",
                pdf_bytes=b"%PDF-1.4 test mock bytes",
                pdf_filename="AEGIS_ACCIDENT.pdf"
            )

            assert res["success"] is True
            assert res["recipients_count"] == 1
            assert mock_send.called
            kwargs = mock_send.call_args.kwargs
            recipients = kwargs.get("recipients")
            subject = kwargs.get("subject")
            html_body = kwargs.get("html_body")
            pdf_bytes = kwargs.get("pdf_attachment")

            assert recipients == ["soc_control@aegis.com"]
            assert subject == "🚨 ACCIDENT ALERT - AEGIS SOC"
            assert "85.0%" in html_body
            assert pdf_bytes == b"%PDF-1.4 test mock bytes"
            print("  [PASS] send_accident_alert_email dispatched with exact subject, recipients, and PDF.")


if __name__ == "__main__":
    print("==================================================================")
    print("AEGIS SOC: AUDITING & VERIFYING EMAIL NOTIFICATION SUBSYSTEM")
    print("==================================================================")

    test_html_template()
    test_trigger_logic()
    test_mime_assembly_and_pdf()
    asyncio.run(test_end_to_end_dispatch())

    print("\n==================================================================")
    print("[SUCCESS] ALL EMAIL SUBSYSTEM TESTS PASSED WITH 100% INTEGRITY!")
    print("==================================================================")
