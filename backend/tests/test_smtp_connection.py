"""
AEGIS Smart Surveillance SOC — SMTP Connection & Diagnostic Utility
Run this script to test and verify your email notification configuration:

    python test_smtp_connection.py [optional_recipient_email]

Examples:
    python test_smtp_connection.py
    python test_smtp_connection.py operator@myagency.gov
"""

import sys
import os
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

import os
import sys
import socket
import ssl
import smtplib
from datetime import datetime

# Configure utf-8 encoding for Windows terminals if needed
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add parent directory to sys.path so app modules can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.services.email_service import EmailService


def print_banner(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def run_diagnostics(target_email: str = None):
    print_banner("AEGIS SOC — SMTP EMAIL DIAGNOSTIC TOOL")

    print(f"\n[1] Configuration Check:")
    print(f"  - SMTP_HOST       : {settings.SMTP_HOST}")
    print(f"  - SMTP_PORT       : {settings.SMTP_PORT}")
    print(f"  - SMTP_USE_TLS    : {settings.SMTP_USE_TLS}")
    print(f"  - SMTP_FROM_NAME  : {settings.SMTP_FROM_NAME}")
    print(f"  - SMTP_FROM_EMAIL : {settings.SMTP_FROM_EMAIL or '(using username)'}")
    print(f"  - EMAIL_ENABLED   : {settings.EMAIL_ENABLED}")

    username = settings.SMTP_USERNAME.strip() if settings.SMTP_USERNAME else ""
    password = settings.SMTP_PASSWORD.strip() if settings.SMTP_PASSWORD else ""

    if username:
        print(f"  - SMTP_USERNAME   : {username}")
    else:
        print(f"  - SMTP_USERNAME   : [MISSING] Not configured in backend/.env")

    if password:
        masked = password[:2] + "*" * (len(password) - 4) + password[-2:] if len(password) > 4 else "***"
        print(f"  - SMTP_PASSWORD   : [CONFIGURED] ({masked}, {len(password)} chars)")
    else:
        print(f"  - SMTP_PASSWORD   : [MISSING] Not configured in backend/.env")

    # If credentials are not set, print instructions and exit
    if not username or not password:
        print_banner(">> SETUP REQUIRED: MISSING SMTP CREDENTIALS <<")
        print("To enable email notifications, configure your credentials in backend/.env:\n")
        print("FOR GMAIL USERS (RECOMMENDED):")
        print("  1. Go to your Google Account: https://myaccount.google.com/")
        print("  2. Select 'Security' and ensure '2-Step Verification' is turned ON.")
        print("  3. Open: https://myaccount.google.com/apppasswords")
        print("  4. Enter 'AEGIS SOC' as the app name and click 'Create'.")
        print("  5. Google will generate a 16-character password (e.g. 'abcd efgh ijkl mnop').")
        print("  6. Open 'backend/.env' and set:")
        print("       SMTP_USERNAME=your_email@gmail.com")
        print("       SMTP_PASSWORD=your_16_char_app_password")
        print("       SMTP_FROM_EMAIL=your_email@gmail.com")
        print("\n  NOTE: Your regular Gmail password will NOT work due to Google security policies.")
        print("=" * 65 + "\n")
        return False

    # Check 2: Network Connectivity (Socket Level)
    print(f"\n[2] Testing Network Socket to {settings.SMTP_HOST}:{settings.SMTP_PORT}...")
    try:
        sock = socket.create_connection((settings.SMTP_HOST, settings.SMTP_PORT), timeout=10)
        sock.close()
        print(f"  [PASS] Network connection successful (port {settings.SMTP_PORT} is reachable).")
    except socket.gaierror as e:
        print(f"  [FAIL] DNS lookup failed for host '{settings.SMTP_HOST}': {e}")
        return False
    except (socket.timeout, TimeoutError):
        print(f"  [FAIL] Connection timed out to {settings.SMTP_HOST}:{settings.SMTP_PORT}.")
        print("         Check your internet connection or firewall rules (port 587 may be blocked).")
        return False
    except Exception as e:
        print(f"  [FAIL] Connection failed: {e}")
        return False

    # Check 3: SMTP Handshake & TLS
    print(f"\n[3] Testing SMTP Handshake & TLS Security...")
    try:
        server = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15)
        server.ehlo()
        if settings.SMTP_USE_TLS:
            context = ssl.create_default_context()
            server.starttls(context=context)
            server.ehlo()
            print("  [PASS] STARTTLS handshake succeeded (encrypted TLS channel established).")
    except Exception as e:
        print(f"  [FAIL] SMTP Handshake/TLS failed: {e}")
        return False

    # Check 4: Authentication
    print(f"\n[4] Testing SMTP Authentication for '{username}'...")
    # Normalize 16-char app password if needed
    clean_pass = password.replace(" ", "") if len(password.replace(" ", "")) == 16 else password

    try:
        server.login(username, clean_pass)
        print("  [PASS] SMTP Authentication SUCCESSFUL! Server accepted credentials.")
    except smtplib.SMTPAuthenticationError as e:
        code = getattr(e, "smtp_code", 535)
        print(f"  [FAIL] SMTP Authentication FAILED (Code {code}): {e}")
        if "gmail" in settings.SMTP_HOST.lower():
            print("\n  [!] GMAIL APP PASSWORD REQUIRED:")
            print("      Google rejected your password. In modern Gmail accounts,")
            print("      you CANNOT use your regular account login password with SMTP.")
            print("      You MUST generate a 16-character App Password at:")
            print("      https://myaccount.google.com/apppasswords")
        server.quit()
        return False
    except Exception as e:
        print(f"  [FAIL] Authentication error: {e}")
        server.quit()
        return False

    # Check 5: Optional Test Email Send
    recipient = target_email or username
    if recipient:
        print(f"\n[5] Sending Test Email to '{recipient}'...")
        try:
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            msg = MIMEMultipart("mixed")
            from_addr = settings.SMTP_FROM_EMAIL.strip() if settings.SMTP_FROM_EMAIL else username
            msg["From"] = f"{settings.SMTP_FROM_NAME} <{from_addr}>"
            msg["To"] = recipient
            msg["Subject"] = "AEGIS SOC - Live SMTP Verification Test"

            now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            html_body = f"""
            <div style="background:#0f172a;color:#f8fafc;padding:25px;border-radius:12px;font-family:sans-serif;">
                <h2 style="color:#22c55e;margin-top:0;">AEGIS SMTP Live Test Successful</h2>
                <p>This verification email confirms that the AEGIS Smart Surveillance email notification subsystem is fully operational.</p>
                <div style="background:#1e293b;padding:15px;border-radius:8px;margin-top:15px;">
                    <p style="margin:4px 0;color:#94a3b8;font-size:13px;"><strong>Host:</strong> {settings.SMTP_HOST}:{settings.SMTP_PORT}</p>
                    <p style="margin:4px 0;color:#94a3b8;font-size:13px;"><strong>Authenticated As:</strong> {username}</p>
                    <p style="margin:4px 0;color:#94a3b8;font-size:13px;"><strong>Timestamp:</strong> {now_str}</p>
                </div>
            </div>
            """
            msg.attach(MIMEText(html_body, "html", "utf-8"))
            server.sendmail(from_addr, [recipient], msg.as_string())
            print(f"  [PASS] Test email DELIVERED to '{recipient}'!")
        except Exception as e:
            print(f"  [FAIL] Error sending test email: {e}")
        finally:
            server.quit()

    print_banner("SMTP DIAGNOSTIC COMPLETE - ALL CHECKS PASSED")
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    success = run_diagnostics(target)
    sys.exit(0 if success else 1)
