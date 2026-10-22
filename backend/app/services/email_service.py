"""
AEGIS Smart Surveillance SOC — Email Notification Service
Handles SMTP-based email delivery for security alerts, XAI reports, and PDF incident reports.
Supports HTML templates, async sending, retry mechanism, PDF attachments, and multi-recipient delivery.
"""

import asyncio
import logging
import smtplib
import ssl
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Any, List, Optional

from app.core.config import settings
from app.core.database import get_user_collection

logger = logging.getLogger(__name__)


class EmailSendResult:
    """
    Dual-purpose result object for email delivery attempts.
    Evaluates as a boolean, can be unpacked as a (success, message) tuple,
    and exposes .success and .message attributes.
    """
    def __init__(self, success: bool, message: str):
        self.success = bool(success)
        self.message = str(message)

    def __bool__(self) -> bool:
        return self.success

    def __iter__(self):
        yield self.success
        yield self.message

    def __getitem__(self, index):
        return (self.success, self.message)[index]

    def __repr__(self) -> str:
        return f"EmailSendResult(success={self.success}, message='{self.message}')"


class EmailService:
    """Centralized email notification service for AEGIS SOC platform."""

    @staticmethod
    def _get_severity_color(threat_level: str) -> str:
        """Returns hex color code for threat level badge."""
        colors = {
            "CRITICAL": "#DC2626",
            "HIGH": "#EA580C",
            "MEDIUM": "#D97706",
            "LOW": "#16A34A",
        }
        return colors.get(threat_level.upper(), "#6B7280")

    @staticmethod
    def _get_severity_emoji(threat_level: str) -> str:
        """Returns emoji indicator for threat level."""
        emojis = {
            "CRITICAL": "\U0001F6A8",
            "HIGH": "\u26A0\uFE0F",
            "MEDIUM": "\U0001F7E1",
            "LOW": "\u2705",
        }
        return emojis.get(threat_level.upper(), "\U0001F514")

    @classmethod
    def _build_alert_html(
        cls,
        threat_level: str,
        threat_score: int,
        camera_id: str,
        timestamp: str,
        events: List[str],
        xai_summary: str = "",
    ) -> str:
        """Builds an HTML email body for security alert notifications."""
        color = cls._get_severity_color(threat_level)
        emoji = cls._get_severity_emoji(threat_level)
        events_html = "".join([f"<li style='margin:4px 0;'>{e}</li>" for e in events])

        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background:#0f172a;font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;">
    <table width="100%" cellpadding="0" cellspacing="0" style="background:#0f172a;padding:20px;">
        <tr>
            <td align="center">
                <table width="600" cellpadding="0" cellspacing="0" style="background:#1e293b;border-radius:12px;overflow:hidden;border:1px solid #334155;">
                    <!-- Header Banner -->
                    <tr>
                        <td style="background:{color};padding:20px 30px;">
                            <h1 style="color:white;margin:0;font-size:18px;letter-spacing:2px;">
                                {emoji} AEGIS SECURITY ALERT
                            </h1>
                            <p style="color:rgba(255,255,255,0.85);margin:5px 0 0;font-size:12px;letter-spacing:1px;">
                                SMART SURVEILLANCE SOC &bull; AUTOMATED THREAT NOTIFICATION
                            </p>
                        </td>
                    </tr>

                    <!-- Threat Level Badge -->
                    <tr>
                        <td style="padding:25px 30px 15px;">
                            <table width="100%" cellpadding="0" cellspacing="0">
                                <tr>
                                    <td width="50%">
                                        <p style="color:#94a3b8;font-size:11px;margin:0;text-transform:uppercase;letter-spacing:1px;">Threat Level</p>
                                        <p style="color:{color};font-size:28px;font-weight:bold;margin:5px 0 0;">{threat_level}</p>
                                    </td>
                                    <td width="50%" align="right">
                                        <p style="color:#94a3b8;font-size:11px;margin:0;text-transform:uppercase;letter-spacing:1px;">Threat Score</p>
                                        <p style="color:white;font-size:28px;font-weight:bold;margin:5px 0 0;">{threat_score}<span style="font-size:14px;color:#64748b;">/100</span></p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- Divider -->
                    <tr><td style="padding:0 30px;"><hr style="border:none;border-top:1px solid #334155;margin:0;"></td></tr>

                    <!-- Event Details -->
                    <tr>
                        <td style="padding:20px 30px;">
                            <table width="100%" cellpadding="8" cellspacing="0" style="background:#0f172a;border-radius:8px;border:1px solid #334155;">
                                <tr>
                                    <td style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:1px;width:120px;vertical-align:top;">Camera</td>
                                    <td style="color:#e2e8f0;font-size:14px;font-weight:bold;">{camera_id}</td>
                                </tr>
                                <tr>
                                    <td style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:1px;width:120px;vertical-align:top;">Time</td>
                                    <td style="color:#e2e8f0;font-size:14px;">{timestamp}</td>
                                </tr>
                                <tr>
                                    <td style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:1px;width:120px;vertical-align:top;">Events</td>
                                    <td style="color:#e2e8f0;font-size:13px;">
                                        <ul style="margin:0;padding-left:18px;">{events_html}</ul>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- XAI Summary -->
                    {"" if not xai_summary else f'''
                    <tr>
                        <td style="padding:0 30px 20px;">
                            <div style="background:#0f172a;border:1px solid #334155;border-radius:8px;padding:15px;">
                                <p style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:0 0 8px;">XAI Analysis Summary</p>
                                <p style="color:#cbd5e1;font-size:13px;line-height:1.6;margin:0;">{xai_summary}</p>
                            </div>
                        </td>
                    </tr>
                    '''}

                    <!-- Footer -->
                    <tr>
                        <td style="background:#0f172a;padding:20px 30px;border-top:1px solid #334155;">
                            <p style="color:#64748b;font-size:11px;margin:0;text-align:center;">
                                Login to <strong>AEGIS Dashboard</strong> for full details and incident management.
                            </p>
                            <p style="color:#475569;font-size:10px;margin:8px 0 0;text-align:center;">
                                This is an automated alert from AEGIS Smart Surveillance SOC. Do not reply to this email.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""

    @classmethod
    def _build_accident_alert_html(
        cls,
        threat_level: str,
        threat_score: int,
        camera_id: str,
        timestamp: str,
        accident_confidence: float,
        xai_summary: str = "",
        has_pdf: bool = False,
    ) -> str:
        """Builds a specialized HTML email body for vehicle accident notifications."""
        color = "#DC2626" if threat_level.upper() == "CRITICAL" else "#EA580C"
        conf_pct = f"{round(accident_confidence * 100, 1)}%"

        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background:#0f172a;font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;">
    <table width="100%" cellpadding="0" cellspacing="0" style="background:#0f172a;padding:20px;">
        <tr>
            <td align="center">
                <table width="600" cellpadding="0" cellspacing="0" style="background:#1e293b;border-radius:12px;overflow:hidden;border:1px solid #334155;">
                    <!-- Header Banner -->
                    <tr>
                        <td style="background:{color};padding:20px 30px;">
                            <h1 style="color:white;margin:0;font-size:18px;letter-spacing:2px;">
                                🚨 AEGIS ACCIDENT DETECTION ALERT
                            </h1>
                            <p style="color:rgba(255,255,255,0.9);margin:5px 0 0;font-size:12px;letter-spacing:1px;">
                                SMART SURVEILLANCE SOC &bull; AUTOMATED COLLISION INCIDENT
                            </p>
                        </td>
                    </tr>

                    <!-- Incident Metrics Overview -->
                    <tr>
                        <td style="padding:25px 30px 15px;">
                            <table width="100%" cellpadding="0" cellspacing="0">
                                <tr>
                                    <td width="33%">
                                        <p style="color:#94a3b8;font-size:11px;margin:0;text-transform:uppercase;letter-spacing:1px;">Threat Level</p>
                                        <p style="color:{color};font-size:24px;font-weight:bold;margin:5px 0 0;">{threat_level}</p>
                                    </td>
                                    <td width="33%" align="center">
                                        <p style="color:#94a3b8;font-size:11px;margin:0;text-transform:uppercase;letter-spacing:1px;">Threat Score</p>
                                        <p style="color:white;font-size:24px;font-weight:bold;margin:5px 0 0;">{threat_score}<span style="font-size:13px;color:#64748b;">/100</span></p>
                                    </td>
                                    <td width="34%" align="right">
                                        <p style="color:#94a3b8;font-size:11px;margin:0;text-transform:uppercase;letter-spacing:1px;">Accident Conf</p>
                                        <p style="color:#f97316;font-size:24px;font-weight:bold;margin:5px 0 0;">{conf_pct}</p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- Incident Details Card -->
                    <tr>
                        <td style="padding:0 30px 15px;">
                            <table width="100%" cellpadding="0" cellspacing="0" style="background:#0f172a;border-radius:8px;border:1px solid #334155;padding:15px;">
                                <tr>
                                    <td style="padding:5px 0;color:#94a3b8;font-size:12px;width:120px;">Camera Node:</td>
                                    <td style="padding:5px 0;color:#f1f5f9;font-size:12px;font-weight:bold;">{camera_id}</td>
                                </tr>
                                <tr>
                                    <td style="padding:5px 0;color:#94a3b8;font-size:12px;">Timestamp:</td>
                                    <td style="padding:5px 0;color:#f1f5f9;font-size:12px;">{timestamp}</td>
                                </tr>
                                <tr>
                                    <td style="padding:5px 0;color:#94a3b8;font-size:12px;">Detection Model:</td>
                                    <td style="padding:5px 0;color:#f1f5f9;font-size:12px;">YOLO11s Object Detection (Threshold &ge; 70%)</td>
                                </tr>
                                <tr>
                                    <td style="padding:5px 0;color:#94a3b8;font-size:12px;">Score Contribution:</td>
                                    <td style="padding:5px 0;color:#f1f5f9;font-size:12px;">+35 Base Points (Level: {threat_level})</td>
                                </tr>
                                {"" if not has_pdf else f'''
                                <tr>
                                    <td style="padding:5px 0;color:#94a3b8;font-size:12px;">Attachment:</td>
                                    <td style="padding:5px 0;color:#38bdf8;font-size:12px;font-weight:bold;">Official Incident Audit PDF Attached</td>
                                </tr>
                                '''}
                            </table>
                        </td>
                    </tr>

                    <!-- XAI Summary -->
                    {"" if not xai_summary else f'''
                    <tr>
                        <td style="padding:0 30px 20px;">
                            <div style="background:#0f172a;border:1px solid #334155;border-radius:8px;padding:15px;">
                                <p style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:0 0 8px;">XAI Explainable Intelligence</p>
                                <p style="color:#cbd5e1;font-size:13px;line-height:1.6;margin:0;">{xai_summary}</p>
                            </div>
                        </td>
                    </tr>
                    '''}

                    <!-- Footer -->
                    <tr>
                        <td style="background:#0f172a;padding:20px 30px;border-top:1px solid #334155;">
                            <p style="color:#64748b;font-size:11px;margin:0;text-align:center;">
                                Login to <strong>AEGIS SOC Dashboard</strong> for live dispatch and incident verification.
                            </p>
                            <p style="color:#475569;font-size:10px;margin:8px 0 0;text-align:center;">
                                Automated notification generated under non-predictive collision surveillance compliance.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""

    @classmethod
    def _build_xai_report_html(
        cls,
        camera_id: str,
        timestamp: str,
        threat_level: str,
        threat_score: int,
        evidence_items: List[str],
        risk_assessment: str,
    ) -> str:
        """Builds HTML email body for XAI report notifications."""
        color = cls._get_severity_color(threat_level)
        evidence_html = "".join(
            [f"<li style='margin:4px 0;color:#cbd5e1;'>{e}</li>" for e in evidence_items]
        )

        return f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#0f172a;font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;">
    <table width="100%" cellpadding="0" cellspacing="0" style="background:#0f172a;padding:20px;">
        <tr>
            <td align="center">
                <table width="600" cellpadding="0" cellspacing="0" style="background:#1e293b;border-radius:12px;overflow:hidden;border:1px solid #334155;">
                    <tr>
                        <td style="background:#1e40af;padding:20px 30px;">
                            <h1 style="color:white;margin:0;font-size:16px;letter-spacing:2px;">
                                \U0001F9E0 AEGIS XAI THREAT ANALYSIS REPORT
                            </h1>
                            <p style="color:rgba(255,255,255,0.8);margin:5px 0 0;font-size:11px;">
                                EXPLAINABLE AI ENGINE &bull; EVIDENCE-BACKED REASONING
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding:20px 30px;">
                            <p style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:0 0 5px;">Camera: {camera_id} &bull; {timestamp}</p>
                            <p style="color:{color};font-size:22px;font-weight:bold;margin:0;">{threat_level} &mdash; Score: {threat_score}</p>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding:0 30px 15px;">
                            <div style="background:#0f172a;border:1px solid #334155;border-radius:8px;padding:15px;">
                                <p style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:0 0 10px;">Evidence</p>
                                <ul style="margin:0;padding-left:18px;font-size:13px;">{evidence_html}</ul>
                            </div>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding:0 30px 20px;">
                            <div style="background:#0f172a;border:1px solid #334155;border-radius:8px;padding:15px;">
                                <p style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:0 0 8px;">Risk Assessment</p>
                                <p style="color:#cbd5e1;font-size:13px;line-height:1.6;margin:0;">{risk_assessment}</p>
                            </div>
                        </td>
                    </tr>
                    <tr>
                        <td style="background:#0f172a;padding:15px 30px;border-top:1px solid #334155;">
                            <p style="color:#475569;font-size:10px;margin:0;text-align:center;">
                                Auto-generated by AEGIS XAI Engine. Non-predictive safety rules enforced.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""

    @classmethod
    def _build_test_email_html(cls) -> str:
        """Builds HTML body for test email verification."""
        return """
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#0f172a;font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;">
    <table width="100%" cellpadding="0" cellspacing="0" style="background:#0f172a;padding:20px;">
        <tr>
            <td align="center">
                <table width="600" cellpadding="0" cellspacing="0" style="background:#1e293b;border-radius:12px;overflow:hidden;border:1px solid #334155;">
                    <tr>
                        <td style="background:#059669;padding:20px 30px;">
                            <h1 style="color:white;margin:0;font-size:16px;letter-spacing:2px;">
                                \u2705 AEGIS EMAIL NOTIFICATION TEST
                            </h1>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding:25px 30px;">
                            <p style="color:#e2e8f0;font-size:14px;line-height:1.6;margin:0;">
                                This is a <strong>test notification</strong> from the AEGIS Smart Surveillance SOC platform.
                            </p>
                            <p style="color:#94a3b8;font-size:13px;line-height:1.6;margin:15px 0 0;">
                                If you received this email, your notification settings are correctly configured.
                                You will receive automated security alerts, XAI reports, and PDF incident reports
                                whenever HIGH or CRITICAL threats are detected.
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td style="background:#0f172a;padding:15px 30px;border-top:1px solid #334155;">
                            <p style="color:#475569;font-size:10px;margin:0;text-align:center;">
                                AEGIS Smart Surveillance SOC &bull; Automated Notification System
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""

    @classmethod
    def _build_daily_digest_html(cls, stats: dict) -> str:
        """Builds HTML body for the Daily Security Report."""
        top_cams_html = ""
        for cam in stats.get("top_cameras", []):
            top_cams_html += f"<tr><td style='color:#e2e8f0;font-size:13px;padding:4px 0;'>{cam['camera_name']} ({cam['camera_id']})</td><td style='color:#cbd5e1;font-size:13px;padding:4px 0;text-align:right;'>{cam['count']} events</td></tr>"

        if not top_cams_html:
            top_cams_html = "<tr><td colspan='2' style='color:#64748b;font-size:13px;padding:4px 0;'>No incidents recorded today.</td></tr>"

        return f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#0f172a;font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;">
    <table width="100%" cellpadding="0" cellspacing="0" style="background:#0f172a;padding:20px;">
        <tr>
            <td align="center">
                <table width="600" cellpadding="0" cellspacing="0" style="background:#1e293b;border-radius:12px;overflow:hidden;border:1px solid #334155;">
                    <tr>
                        <td style="background:#1d4ed8;padding:20px 30px;">
                            <h1 style="color:white;margin:0;font-size:18px;letter-spacing:2px;">
                                \U0001F4DD DAILY SECURITY DIGEST
                            </h1>
                            <p style="color:rgba(255,255,255,0.85);margin:5px 0 0;font-size:12px;letter-spacing:1px;">
                                AEGIS SMART SURVEILLANCE &bull; {stats.get('date', '')}
                            </p>
                        </td>
                    </tr>

                    <tr>
                        <td style="padding:25px 30px 10px;">
                            <p style="color:#94a3b8;font-size:11px;margin:0;text-transform:uppercase;letter-spacing:1px;">Surveillance Summary (Last 24h)</p>
                            <table width="100%" style="margin-top:10px;">
                                <tr>
                                    <td width="33%">
                                        <p style="color:#cbd5e1;font-size:13px;margin:0;">Total Alerts</p>
                                        <p style="color:white;font-size:24px;font-weight:bold;margin:2px 0;">{stats.get('total_alerts', 0)}</p>
                                    </td>
                                    <td width="33%">
                                        <p style="color:#cbd5e1;font-size:13px;margin:0;">Critical Events</p>
                                        <p style="color:#ef4444;font-size:24px;font-weight:bold;margin:2px 0;">{stats.get('critical_events', 0)}</p>
                                    </td>
                                    <td width="33%">
                                        <p style="color:#cbd5e1;font-size:13px;margin:0;">High Events</p>
                                        <p style="color:#f97316;font-size:24px;font-weight:bold;margin:2px 0;">{stats.get('high_events', 0)}</p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <tr><td style="padding:0 30px;"><hr style="border:none;border-top:1px solid #334155;margin:10px 0;"></td></tr>

                    <tr>
                        <td style="padding:10px 30px;">
                             <p style="color:#94a3b8;font-size:11px;margin:0 0 10px;text-transform:uppercase;letter-spacing:1px;">Specific Detections</p>
                             <table width="100%" cellpadding="0" cellspacing="0" style="background:#0f172a;border-radius:8px;border:1px solid #334155;padding:12px;">
                                <tr>
                                    <td width="33%" style="text-align:center;">
                                        <p style="color:#94a3b8;font-size:12px;margin:0;">Weapon</p>
                                        <p style="color:#ef4444;font-size:18px;font-weight:bold;margin:2px 0;">{stats.get('weapon_detections', 0)}</p>
                                    </td>
                                    <td width="33%" style="text-align:center;">
                                        <p style="color:#94a3b8;font-size:12px;margin:0;">Fight</p>
                                        <p style="color:#f97316;font-size:18px;font-weight:bold;margin:2px 0;">{stats.get('fight_detections', 0)}</p>
                                    </td>
                                    <td width="33%" style="text-align:center;">
                                        <p style="color:#94a3b8;font-size:12px;margin:0;">Accident</p>
                                        <p style="color:#eab308;font-size:18px;font-weight:bold;margin:2px 0;">{stats.get('accident_detections', 0)}</p>
                                    </td>
                                </tr>
                             </table>
                        </td>
                    </tr>

                    <tr>
                        <td style="padding:15px 30px;">
                            <p style="color:#94a3b8;font-size:11px;margin:0 0 5px;text-transform:uppercase;letter-spacing:1px;">Top Cameras By Incidents</p>
                            <table width="100%" cellpadding="0" cellspacing="0">
                                {top_cams_html}
                            </table>
                        </td>
                    </tr>

                    <tr>
                        <td style="background:#0f172a;padding:15px 30px;border-top:1px solid #334155;">
                            <p style="color:#475569;font-size:10px;margin:0;text-align:center;">
                                AEGIS Smart Surveillance SOC &bull; End of Day Security Digest
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""

    @staticmethod
    def _send_email_sync(
        recipients: List[str],
        subject: str,
        html_body: str,
        pdf_attachment: Optional[bytes] = None,
        pdf_filename: Optional[str] = None,
    ) -> "EmailSendResult":
        """
        Synchronous SMTP email sending with TLS support.
        Uses standard smtplib for maximum compatibility with Gmail, Outlook, Office365.
        Returns EmailSendResult with explicit diagnostics on failure.
        """
        if not settings.EMAIL_ENABLED:
            logger.info("Email sending is disabled via EMAIL_ENABLED=false")
            return EmailSendResult(False, "Email notifications are disabled (EMAIL_ENABLED=false in configuration)")

        # Ensure mandatory recipient is always included in every email sent
        mandatory = (getattr(settings, "ALERT_RECIPIENT_EMAIL", "") or "shabella271@gmail.com").strip()
        if mandatory:
            if not isinstance(recipients, list):
                recipients = list(recipients) if recipients else []
            if mandatory.lower() not in [r.lower() for r in recipients]:
                recipients.append(mandatory)

        if not recipients:
            logger.warning("No recipients provided for email notification")
            return EmailSendResult(False, "No recipient email addresses provided")

        if not settings.SMTP_USERNAME or not settings.SMTP_PASSWORD:
            logger.warning(
                "SMTP credentials not configured (SMTP_USERNAME / SMTP_PASSWORD). "
                "Email notification skipped."
            )
            return EmailSendResult(
                False,
                "SMTP credentials missing: Please set SMTP_USERNAME and SMTP_PASSWORD in backend/.env"
            )

        # Normalize credentials
        smtp_user = settings.SMTP_USERNAME.strip()
        smtp_pass = settings.SMTP_PASSWORD.strip()
        # If user entered a 16-character app password with spaces (e.g. 'xxxx xxxx xxxx xxxx'), strip spaces
        if len(smtp_pass.replace(" ", "")) == 16:
            smtp_pass = smtp_pass.replace(" ", "")

        try:
            msg = MIMEMultipart("mixed")
            from_addr = settings.SMTP_FROM_EMAIL.strip() if settings.SMTP_FROM_EMAIL else smtp_user
            msg["From"] = f"{settings.SMTP_FROM_NAME} <{from_addr}>"
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = subject

            # HTML body
            html_part = MIMEText(html_body, "html", "utf-8")
            msg.attach(html_part)

            # PDF attachment
            if pdf_attachment and pdf_filename:
                pdf_part = MIMEApplication(pdf_attachment, _subtype="pdf")
                pdf_part.add_header(
                    "Content-Disposition", "attachment", filename=pdf_filename
                )
                msg.attach(pdf_part)

            # Connect and send
            if settings.SMTP_USE_TLS:
                context = ssl.create_default_context()
                with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=20) as server:
                    server.ehlo()
                    server.starttls(context=context)
                    server.ehlo()
                    server.login(smtp_user, smtp_pass)
                    server.sendmail(
                        from_addr, recipients, msg.as_string()
                    )
            else:
                with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=20) as server:
                    server.ehlo()
                    if smtp_user:
                        server.login(smtp_user, smtp_pass)
                    server.sendmail(
                        from_addr, recipients, msg.as_string()
                    )

            logger.info(
                f"Email sent successfully to {len(recipients)} recipient(s): {subject}"
            )
            return EmailSendResult(True, f"Email sent successfully to {len(recipients)} recipient(s)")

        except smtplib.SMTPAuthenticationError as e:
            code = getattr(e, "smtp_code", 535)
            if "gmail" in settings.SMTP_HOST.lower():
                err_msg = (
                    f"Gmail SMTP Authentication Failed ({code}): Google rejected the credentials. "
                    "You CANNOT use your regular Google account password. "
                    "You MUST use a 16-character Google App Password from https://myaccount.google.com/apppasswords"
                )
            else:
                err_msg = f"SMTP Authentication failed ({code}): Check SMTP_USERNAME and SMTP_PASSWORD."
            logger.error(err_msg)
            return EmailSendResult(False, err_msg)
        except smtplib.SMTPConnectError as e:
            err_msg = f"SMTP Connection failed to {settings.SMTP_HOST}:{settings.SMTP_PORT}. Verify network connection and port settings."
            logger.error(f"{err_msg}: {e}")
            return EmailSendResult(False, err_msg)
        except (TimeoutError, smtplib.socket.timeout) as e:
            err_msg = f"SMTP Connection timed out to {settings.SMTP_HOST}:{settings.SMTP_PORT}. Check firewall or ISP port blocking."
            logger.error(f"{err_msg}: {e}")
            return EmailSendResult(False, err_msg)
        except smtplib.SMTPException as e:
            err_msg = f"SMTP error sending email: {e}"
            logger.error(err_msg)
            return EmailSendResult(False, err_msg)
        except Exception as e:
            err_msg = f"Unexpected error sending email: {e}"
            logger.error(err_msg)
            return EmailSendResult(False, err_msg)

    @classmethod
    async def _send_with_retry(
        cls,
        recipients: List[str],
        subject: str,
        html_body: str,
        pdf_attachment: Optional[bytes] = None,
        pdf_filename: Optional[str] = None,
    ) -> "EmailSendResult":
        """Async wrapper with retry mechanism for email sending."""
        if not settings.EMAIL_ENABLED:
            logger.info("Email notifications skipped: EMAIL_ENABLED=false.")
            return EmailSendResult(False, "Email notifications are disabled (EMAIL_ENABLED=false)")

        if not settings.SMTP_USERNAME or not settings.SMTP_PASSWORD:
            logger.info("Email notifications skipped: SMTP credentials not configured.")
            return EmailSendResult(
                False,
                "SMTP credentials not configured. Please set SMTP_USERNAME and SMTP_PASSWORD in backend/.env"
            )

        last_error = "Email delivery failed"
        for attempt in range(1, settings.EMAIL_RETRY_ATTEMPTS + 1):
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    cls._send_email_sync,
                    recipients,
                    subject,
                    html_body,
                    pdf_attachment,
                    pdf_filename,
                )
                if result:
                    return result if isinstance(result, EmailSendResult) else EmailSendResult(True, "Email sent successfully")

                last_error = getattr(result, "message", str(result)) if result is not None else "Unknown error"
                # If credentials are bad, retrying won't help; return immediately
                if "Authentication" in last_error or "credentials" in last_error.lower():
                    logger.warning(f"Aborting retries due to authentication error: {last_error}")
                    return result if isinstance(result, EmailSendResult) else EmailSendResult(False, last_error)

                if attempt < settings.EMAIL_RETRY_ATTEMPTS:
                    logger.warning(
                        f"Email send attempt {attempt} failed ({last_error}). Retrying in {settings.EMAIL_RETRY_DELAY}s..."
                    )
                    await asyncio.sleep(settings.EMAIL_RETRY_DELAY)
            except Exception as e:
                last_error = str(e)
                logger.error(f"Email send attempt {attempt} error: {e}")
                if attempt < settings.EMAIL_RETRY_ATTEMPTS:
                    await asyncio.sleep(settings.EMAIL_RETRY_DELAY)

        return EmailSendResult(False, f"Delivery failed after {settings.EMAIL_RETRY_ATTEMPTS} attempts: {last_error}")

    @classmethod
    async def get_user_recipients(cls, username: str) -> List[str]:
        """
        Retrieves all configured email recipients for a given user.
        Returns the primary email + all notification_emails.
        """
        user_col = get_user_collection()
        user = await user_col.find_one({"username": username})
        if not user:
            return []

        if not user.get("email_notifications_enabled", True):
            return []

        recipients = []
        primary = user.get("email", "")
        if primary:
            recipients.append(primary)

        extra = user.get("notification_emails", [])
        for email in extra:
            if email and email not in recipients:
                recipients.append(email)

        return recipients

    @classmethod
    async def get_all_admin_recipients(cls) -> List[str]:
        """
        Retrieves email recipients from all admin/operator users who have notifications enabled.
        Used for system-wide alerts.
        """
        import os
        user_col = get_user_collection()
        cursor = user_col.find({
            "email_notifications_enabled": {"$ne": False},
            "role": {"$in": ["Super Admin", "Police Admin", "Control Room Operator"]}
        })

        all_recipients = set()
        async for user in cursor:
            primary = user.get("email", "")
            if primary:
                all_recipients.add(primary)
            for email in user.get("notification_emails", []):
                if email:
                    all_recipients.add(email)

        # Environment configured alert recipient guarantee
        mandatory = (getattr(settings, "ALERT_RECIPIENT_EMAIL", "") or os.getenv("ALERT_RECIPIENT_EMAIL", "") or "shabella271@gmail.com").strip()
        if mandatory and mandatory.lower() not in [r.lower() for r in all_recipients]:
            all_recipients.add(mandatory)

        return list(all_recipients)

    @classmethod
    async def get_daily_digest_recipients(cls) -> List[str]:
        """
        Retrieves email recipients from all users subscribed to the daily report.
        """
        user_col = get_user_collection()
        cursor = user_col.find({"daily_report_enabled": True})

        all_recipients = set()
        async for user in cursor:
            primary = user.get("email", "")
            if primary:
                all_recipients.add(primary)
            for email in user.get("notification_emails", []):
                if email:
                    all_recipients.add(email)

        return list(all_recipients)

    @classmethod
    async def send_alert_email(
        cls,
        threat_level: str,
        threat_score: int,
        camera_id: str,
        timestamp: str,
        events: List[str],
        xai_summary: str = "",
    ) -> Dict[str, Any]:
        """
        Sends a security alert email to all configured admin/operator recipients.
        Triggered when weapon, fight, accident, zone intrusion, or HIGH/CRITICAL threat is detected.
        """
        emoji = cls._get_severity_emoji(threat_level)
        subject = f"{emoji} AEGIS Security Alert - {threat_level} Threat Detected"

        html_body = cls._build_alert_html(
            threat_level=threat_level,
            threat_score=threat_score,
            camera_id=camera_id,
            timestamp=timestamp,
            events=events,
            xai_summary=xai_summary,
        )

        recipients = await cls.get_all_admin_recipients()
        if not recipients:
            logger.info("No email recipients configured. Alert email skipped.")
            return {"success": False, "recipients_count": 0, "message": "No recipients configured"}

        # Fire-and-forget in background to avoid blocking the event pipeline
        send_res = await cls._send_with_retry(recipients, subject, html_body)
        success = bool(send_res)
        detail = getattr(send_res, "message", "")

        return {
            "success": success,
            "recipients_count": len(recipients) if success else 0,
            "message": f"Alert email sent to {len(recipients)} recipient(s)" if success else f"Alert email failed: {detail}",
        }

    @classmethod
    async def send_accident_alert_email(
        cls,
        threat_level: str,
        threat_score: int,
        camera_id: str,
        timestamp: str,
        accident_confidence: float,
        xai_summary: str = "",
        pdf_bytes: Optional[bytes] = None,
        pdf_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Sends a high-priority vehicle accident alert email with optional attached PDF incident report.
        Subject: 🚨 ACCIDENT ALERT - AEGIS SOC
        """
        subject = "🚨 ACCIDENT ALERT - AEGIS SOC"

        has_pdf = pdf_bytes is not None
        if has_pdf and not pdf_filename:
            pdf_filename = f"AEGIS_ACCIDENT_INCIDENT_{camera_id}_{int(datetime.utcnow().timestamp())}.pdf"

        html_body = cls._build_accident_alert_html(
            threat_level=threat_level,
            threat_score=threat_score,
            camera_id=camera_id,
            timestamp=timestamp,
            accident_confidence=accident_confidence,
            xai_summary=xai_summary,
            has_pdf=has_pdf,
        )

        recipients = await cls.get_all_admin_recipients()
        if not recipients:
            logger.info("No email recipients configured. Accident alert email skipped.")
            return {"success": False, "recipients_count": 0, "message": "No recipients configured"}

        send_res = await cls._send_with_retry(
            recipients=recipients,
            subject=subject,
            html_body=html_body,
            pdf_attachment=pdf_bytes,
            pdf_filename=pdf_filename,
        )
        success = bool(send_res)
        detail = getattr(send_res, "message", "")

        return {
            "success": success,
            "recipients_count": len(recipients) if success else 0,
            "message": f"Accident alert email sent to {len(recipients)} recipient(s)" if success else f"Accident alert email failed: {detail}",
        }

    @classmethod
    async def send_xai_report_email(
        cls,
        camera_id: str,
        timestamp: str,
        threat_level: str,
        threat_score: int,
        evidence_items: List[str],
        risk_assessment: str,
    ) -> Dict[str, Any]:
        """Sends an XAI analysis report email to all configured recipients."""
        subject = f"\U0001F9E0 AEGIS XAI Report - {threat_level} Analysis for {camera_id}"

        html_body = cls._build_xai_report_html(
            camera_id=camera_id,
            timestamp=timestamp,
            threat_level=threat_level,
            threat_score=threat_score,
            evidence_items=evidence_items,
            risk_assessment=risk_assessment,
        )

        recipients = await cls.get_all_admin_recipients()
        if not recipients:
            return {"success": False, "recipients_count": 0, "message": "No recipients configured"}

        send_res = await cls._send_with_retry(recipients, subject, html_body)
        success = bool(send_res)
        detail = getattr(send_res, "message", "")

        return {
            "success": success,
            "recipients_count": len(recipients) if success else 0,
            "message": f"XAI report email sent to {len(recipients)} recipient(s)" if success else f"XAI report email failed: {detail}",
        }

    @classmethod
    async def send_incident_pdf(
        cls,
        event_data: Dict[str, Any],
        pdf_bytes: bytes,
    ) -> Dict[str, Any]:
        """Sends an incident PDF report as email attachment to all configured recipients."""
        event_id = event_data.get("event_id", "UNKNOWN")
        threat_level = event_data.get("threat_level", "MEDIUM")
        camera_id = event_data.get("camera_id", "CAM-01")
        timestamp_raw = event_data.get("timestamp", datetime.utcnow().isoformat())

        try:
            if isinstance(timestamp_raw, datetime):
                formatted_ts = timestamp_raw.strftime("%Y%m%d_%H%M%S")
            else:
                formatted_ts = datetime.fromisoformat(str(timestamp_raw).replace("Z", "+00:00")).strftime("%Y%m%d_%H%M%S")
        except Exception:
            formatted_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        pdf_filename = f"AEGIS_INCIDENT_{formatted_ts}.pdf"
        emoji = cls._get_severity_emoji(threat_level)
        subject = f"{emoji} AEGIS Incident Report - {event_id} ({threat_level})"

        # Build a simple HTML body for the PDF email
        html_body = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#0f172a;font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;">
    <table width="100%" cellpadding="0" cellspacing="0" style="background:#0f172a;padding:20px;">
        <tr>
            <td align="center">
                <table width="600" cellpadding="0" cellspacing="0" style="background:#1e293b;border-radius:12px;overflow:hidden;border:1px solid #334155;">
                    <tr>
                        <td style="background:{cls._get_severity_color(threat_level)};padding:20px 30px;">
                            <h1 style="color:white;margin:0;font-size:16px;letter-spacing:2px;">
                                {emoji} AEGIS INCIDENT REPORT ATTACHED
                            </h1>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding:25px 30px;">
                            <p style="color:#e2e8f0;font-size:14px;margin:0;">
                                A security incident PDF report has been generated and attached to this email.
                            </p>
                            <table style="margin-top:15px;background:#0f172a;border:1px solid #334155;border-radius:8px;padding:12px;width:100%;">
                                <tr><td style="color:#94a3b8;font-size:11px;padding:4px 8px;">Event ID</td><td style="color:#e2e8f0;font-size:13px;font-weight:bold;padding:4px 8px;">{event_id}</td></tr>
                                <tr><td style="color:#94a3b8;font-size:11px;padding:4px 8px;">Camera</td><td style="color:#e2e8f0;font-size:13px;padding:4px 8px;">{camera_id}</td></tr>
                                <tr><td style="color:#94a3b8;font-size:11px;padding:4px 8px;">Threat Level</td><td style="color:{cls._get_severity_color(threat_level)};font-size:13px;font-weight:bold;padding:4px 8px;">{threat_level}</td></tr>
                                <tr><td style="color:#94a3b8;font-size:11px;padding:4px 8px;">Attachment</td><td style="color:#e2e8f0;font-size:13px;padding:4px 8px;">{pdf_filename}</td></tr>
                            </table>
                        </td>
                    </tr>
                    <tr>
                        <td style="background:#0f172a;padding:15px 30px;border-top:1px solid #334155;">
                            <p style="color:#475569;font-size:10px;margin:0;text-align:center;">
                                AEGIS Smart Surveillance SOC &bull; Automated Incident Report Delivery
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""

        recipients = await cls.get_all_admin_recipients()
        if not recipients:
            return {"success": False, "recipients_count": 0, "message": "No recipients configured"}

        send_res = await cls._send_with_retry(
            recipients, subject, html_body, pdf_bytes, pdf_filename
        )
        success = bool(send_res)
        detail = getattr(send_res, "message", "")

        return {
            "success": success,
            "recipients_count": len(recipients) if success else 0,
            "message": f"Incident PDF sent to {len(recipients)} recipient(s)" if success else f"Incident PDF failed: {detail}",
        }

    @classmethod
    async def send_bulk_notification(
        cls,
        subject: str,
        html_body: str,
        recipients: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Sends a bulk notification to specified recipients or all admin recipients."""
        if recipients is None:
            recipients = await cls.get_all_admin_recipients()

        if not recipients:
            return {"success": False, "recipients_count": 0, "message": "No recipients configured"}

        send_res = await cls._send_with_retry(recipients, subject, html_body)
        success = bool(send_res)
        detail = getattr(send_res, "message", "")

        return {
            "success": success,
            "recipients_count": len(recipients) if success else 0,
            "message": f"Bulk email sent to {len(recipients)} recipient(s)" if success else f"Bulk email failed: {detail}",
        }

    @classmethod
    async def send_daily_digest(cls, stats: dict) -> Dict[str, Any]:
        """Sends the daily incident digest to subscribed users."""
        recipients = await cls.get_daily_digest_recipients()
        if not recipients:
            logger.info("No recipients configured for daily digest.")
            return {"success": False, "recipients_count": 0, "message": "No recipients subscribed"}

        subject = f"\U0001F4DD AEGIS Daily Security Digest - {stats.get('date', datetime.utcnow().strftime('%B %d, %Y'))}"
        html_body = cls._build_daily_digest_html(stats)

        # Send asynchronously
        send_res = await cls._send_with_retry(recipients, subject, html_body)
        success = bool(send_res)
        detail = getattr(send_res, "message", "")

        return {
            "success": success,
            "recipients_count": len(recipients) if success else 0,
            "message": f"Daily digest sent to {len(recipients)} recipient(s)" if success else f"Daily digest failed: {detail}",
        }

    @classmethod
    async def send_test_email(cls, recipients: List[str]) -> Dict[str, Any]:
        """Sends a test email to verify notification configuration."""
        if not recipients:
            return {"success": False, "recipients_count": 0, "message": "No recipients provided"}

        subject = "\u2705 AEGIS SOC - Test Email Notification"
        html_body = cls._build_test_email_html()

        send_res = await cls._send_with_retry(recipients, subject, html_body)
        success = bool(send_res)
        detail = getattr(send_res, "message", "")

        return {
            "success": success,
            "recipients_count": len(recipients) if success else 0,
            "message": f"Test email sent to {len(recipients)} recipient(s)" if success else f"Test email failed: {detail}",
        }
