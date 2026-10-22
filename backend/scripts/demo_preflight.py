import os
import sys
import shutil
import socket
import smtplib
import ssl
import argparse
from pathlib import Path
import cv2
import numpy as np
from pymongo import MongoClient

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

from app.core.config import settings

def check_mongodb():
    """Check if MongoDB is reachable and return (passed, detail)"""
    try:
        mongo_uri = getattr(settings, "MONGO_URI", "mongodb://localhost:27017")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        return True, "MongoDB reachable and responding to ping"
    except Exception as e:
        return False, f"MongoDB unreachable: {e}"

def check_models_cuda():
    """Check if all 4 models load on cuda"""
    try:
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            return False, "CUDA is not available on this system"

        # 1. Person model
        person_path = repo_root / getattr(settings, "PERSON_MODEL_PATH", "yolo11s.pt")
        person_model = YOLO(str(person_path)).to("cuda")

        # 2. Weapon model
        weapon_path = repo_root / settings.WEAPON_MODEL_PATH
        weapon_model = YOLO(str(weapon_path)).to("cuda")

        # 3. Fight model
        fight_path = repo_root / getattr(settings, "FIGHT_MODEL_PATH", "runs/classify/fight_classifier/weights/best.pt")
        fight_model = YOLO(str(fight_path)).to("cuda")

        # 4. Accident model
        acc_path = repo_root / getattr(settings, "ACCIDENT_MODEL_PATH", "runs/detect/accident_detector/weights/best.pt")
        accident_model = YOLO(str(acc_path)).to("cuda")

        return True, "All 4 models successfully loaded on CUDA"
    except Exception as e:
        return False, f"Failed loading models on CUDA: {e}"

def check_camera_0():
    """Check if camera 0 opens with mean brightness > 40 after discarding first 15 frames."""
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "Camera 0 could not be opened (no webcam device or camera in use)"
        # Discard the first 15 frames to allow auto-exposure/gain to settle
        frame = None
        for _ in range(15):
            ret, frame = cap.read()
            if not ret:
                break
        if frame is None or frame.size == 0:
            return False, "Camera 0 opened but failed to read frame after 15 frames"
        mean_b = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        if mean_b > 40.0:
            return True, f"Camera 0 operational (mean brightness: {mean_b:.1f} > 40 after 15-frame auto-exposure)"
        else:
            return False, f"Camera 0 brightness too low: {mean_b:.1f} <= 40 (underexposed or lens covered)"
    except Exception as e:
        return False, f"Camera 0 check error: {e}"
    finally:
        if cap is not None:
            cap.release()

def check_sample_videos():
    """Check if the 4 sample_videos exist"""
    sample_dir = repo_root / "sample_videos"
    expected = ["accident.mp4", "fight.mp4", "normal.mp4", "weapon.mp4"]
    missing = []
    for vid in expected:
        p = sample_dir / vid
        if not p.exists() or p.stat().st_size == 0:
            missing.append(vid)
    if not missing:
        return True, f"All 4 sample videos exist in {sample_dir.name}/ ({', '.join(expected)})"
    else:
        return False, f"Missing sample videos: {', '.join(missing)}"

def check_cameras_seeded():
    """Check if 5 cameras are seeded in MongoDB surveillance_db"""
    try:
        mongo_uri = getattr(settings, "MONGO_URI", "mongodb://localhost:27017")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        db = client[getattr(settings, "MONGO_DB_NAME", "surveillance_db")]
        count = db.cameras.count_documents({})
        if count == 5:
            return True, f"5 cameras seeded in {db.name}.cameras"
        else:
            return False, f"Expected 5 seeded cameras, found {count}"
    except Exception as e:
        return False, f"Database camera check error: {e}"

def check_zones_cam001():
    """Check if zones for CAM-001 == 0"""
    try:
        mongo_uri = getattr(settings, "MONGO_URI", "mongodb://localhost:27017")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        db = client[getattr(settings, "MONGO_DB_NAME", "surveillance_db")]
        
        # Check zones collection and cameras document
        z_coll_count = db.zones.count_documents({"camera_id": "CAM-001"})
        cam_doc = db.cameras.find_one({"camera_id": "CAM-001"})
        cam_zones = len(cam_doc.get("zones", [])) if cam_doc else 0

        total_zones = z_coll_count + cam_zones
        if total_zones == 0:
            return True, "0 restricted zones assigned to CAM-001"
        else:
            return False, f"Found {total_zones} zone(s) assigned to CAM-001 (expected 0)"
    except Exception as e:
        return False, f"Zone check error: {e}"

def check_email_smtp():
    """Check EMAIL_ENABLED and test SMTP login, returning True, False, or 'WARN'."""
    try:
        enabled = getattr(settings, "EMAIL_ENABLED", False)
        if not enabled:
            return False, "EMAIL_ENABLED is False in configuration"

        smtp_host = getattr(settings, "SMTP_HOST", "")
        smtp_port = getattr(settings, "SMTP_PORT", 587)
        smtp_user = getattr(settings, "SMTP_USERNAME", "").strip()
        smtp_pass = getattr(settings, "SMTP_PASSWORD", "").strip()
        if len(smtp_pass.replace(" ", "")) == 16:
            smtp_pass = smtp_pass.replace(" ", "")

        if not smtp_host or not smtp_user or not smtp_pass:
            return False, "SMTP configuration missing (host, username, or password empty)"

        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.ehlo()
            if getattr(settings, "SMTP_USE_TLS", True):
                server.starttls(context=context)
                server.ehlo()
            server.login(smtp_user, smtp_pass)
        return True, f"SMTP login successful ({smtp_host}:{smtp_port} as {smtp_user})"
    except (socket.gaierror, smtplib.SMTPConnectError) as e:
        return "WARN", f"DNS/Network resolution failed ({e}): OFFLINE, email will not work"
    except Exception as e:
        err_str = str(e).lower()
        if "getaddrinfo failed" in err_str or "11001" in err_str or "temporary failure in name resolution" in err_str:
            return "WARN", "DNS resolution failed: OFFLINE, email will not work"
        return False, f"SMTP login test failed: {e}"

def check_media_dirs_writable():
    """Check if screenshots/ and event_clips/ are writable"""
    try:
        sc_dir = repo_root / "screenshots"
        ec_dir = repo_root / "event_clips"
        sc_dir.mkdir(parents=True, exist_ok=True)
        ec_dir.mkdir(parents=True, exist_ok=True)

        test_f1 = sc_dir / ".preflight_test"
        test_f2 = ec_dir / ".preflight_test"

        test_f1.write_text("test")
        test_f1.unlink()

        test_f2.write_text("test")
        test_f2.unlink()

        return True, "Both screenshots/ and event_clips/ exist and are writable"
    except Exception as e:
        return False, f"Media directories write test failed: {e}"

def check_ports_free():
    """Check if ports 8000 and 5173 are free"""
    def is_port_free(port: int) -> bool:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(0.5)
            res = s.connect_ex(("127.0.0.1", port))
            return res != 0  # connect failed -> port free
        finally:
            s.close()

    free_8000 = is_port_free(8000)
    free_5173 = is_port_free(5173)

    if free_8000 and free_5173:
        return True, "Ports 8000 and 5173 are free"
    else:
        busy = []
        if not free_8000:
            busy.append("8000 (Backend)")
        if not free_5173:
            busy.append("5173 (Frontend)")
        return False, f"Ports currently in use: {', '.join(busy)}"

def check_frontend_build():
    """Check if frontend build exists"""
    dist_index = repo_root / "frontend" / "dist" / "index.html"
    dist_assets = repo_root / "frontend" / "dist" / "assets"
    if dist_index.exists() and dist_assets.exists() and any(dist_assets.iterdir()):
        size_kb = dist_index.stat().st_size / 1024
        return True, f"Frontend build exists at frontend/dist/ (index.html: {size_kb:.1f} KB)"
    else:
        return False, "Frontend build missing. Run 'npm run build' inside frontend/"

def handle_clear_events():
    """
    Handles --clear-events flag:
    Prompts user to type 'YES'.
    Moves media to screenshots_old/ and event_clips_old/.
    Deletes ONLY documents in events collection.
    Users, cameras, and zones untouched.
    """
    print("\n" + "!" * 80)
    print("ACTION REQUESTED: CLEAR EVENTS AND ARCHIVE MEDIA")
    print("!" * 80)
    print("This will:")
    print("  1. Move all files from screenshots/ to screenshots_old/")
    print("  2. Move all files from event_clips/ to event_clips_old/")
    print("  3. Delete ALL documents in MongoDB 'events' collection.")
    print("  4. Leave 'users', 'cameras', and 'zones' untouched.")
    print("!" * 80)

    conf = input("Are you sure you want to proceed? Type 'YES' to confirm: ").strip()
    if conf != "YES":
        print("[ABORTED] Confirmation did not match 'YES'. No files or records were changed.")
        return

    # 1. Archive screenshots
    sc_src = repo_root / "screenshots"
    sc_dst = repo_root / "screenshots_old"
    sc_dst.mkdir(parents=True, exist_ok=True)
    moved_sc = 0
    if sc_src.exists():
        for f in list(sc_src.iterdir()):
            if f.is_file():
                shutil.move(str(f), str(sc_dst / f.name))
                moved_sc += 1

    # 2. Archive event_clips
    ec_src = repo_root / "event_clips"
    ec_dst = repo_root / "event_clips_old"
    ec_dst.mkdir(parents=True, exist_ok=True)
    moved_ec = 0
    if ec_src.exists():
        for f in list(ec_src.iterdir()):
            if f.is_file():
                shutil.move(str(f), str(ec_dst / f.name))
                moved_ec += 1

    # 3. Delete ONLY events collection documents in MongoDB
    deleted_events = 0
    try:
        mongo_uri = getattr(settings, "MONGO_URI", "mongodb://localhost:27017")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        db = client[getattr(settings, "MONGO_DB_NAME", "surveillance_db")]
        res = db.events.delete_many({})
        deleted_events = res.deleted_count
    except Exception as e:
        print(f"[ERROR] Failed to delete events from database: {e}")

    print("\n✅ CLEAR EVENTS COMPLETED:")
    print(f"  - Moved {moved_sc} screenshots to screenshots_old/")
    print(f"  - Moved {moved_ec} event clips to event_clips_old/")
    print(f"  - Deleted {deleted_events} documents from 'events' collection.")
    print("  - 'users', 'cameras', and 'zones' were NOT touched.")

def main():
    parser = argparse.ArgumentParser(description="AEGIS SOC Demo Preflight System Check")
    parser.add_argument("--clear-events", action="store_true", help="Archive media and clear events collection in database")
    args = parser.parse_args()

    if args.clear_events:
        handle_clear_events()
        return

    print("=" * 80)
    print("AEGIS SOC DEMO PREFLIGHT VERIFICATION")
    print("=" * 80)

    checks = [
        ("MongoDB reachable", check_mongodb),
        ("4 models load on cuda", check_models_cuda),
        ("camera 0 opens with mean brightness > 40", check_camera_0),
        ("4 sample_videos exist", check_sample_videos),
        ("5 cameras seeded", check_cameras_seeded),
        ("zones for CAM-001 == 0", check_zones_cam001),
        ("EMAIL_ENABLED and SMTP login test", check_email_smtp),
        ("screenshots/ and event_clips/ writable", check_media_dirs_writable),
        ("ports 8000 and 5173 free", check_ports_free),
        ("frontend build exists", check_frontend_build)
    ]

    all_passed = True
    print(f"\n{'Check Name':<45} | {'Result':<8} | {'Details'}")
    print("-" * 80)

    for name, fn in checks:
        passed, detail = fn()
        if passed == "WARN":
            status_str = "WARN"
        elif passed is True:
            status_str = "PASS"
        else:
            status_str = "FAIL"
            all_passed = False
        print(f"{name:<45} | {status_str:<8} | {detail}")

    print("=" * 80)
    if all_passed:
        print("🟢 ALL PREFLIGHT CHECKS PASSED — SYSTEM READY FOR DEMO!")
    else:
        print("🔴 ONE OR MORE PREFLIGHT CHECKS FAILED — REVIEW DETAILS ABOVE.")
    print("=" * 80)

if __name__ == "__main__":
    main()
