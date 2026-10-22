import os
import sys
import csv
import cv2
import asyncio
from pathlib import Path
from unittest.mock import patch
from motor.motor_asyncio import AsyncIOMotorClient

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.core import database
from app.services.ai_service import AIService
from app.services import event_service
from app.services.email_service import EmailService

NORMAL_VIDEO = repo_root / "sample_videos" / "normal.mp4"
NONFIGHT_DIR = repo_root / "RWF-2000" / "NonFight"
CSV_OUT = repo_root / "regression_results.csv"

THREAT_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
SCRATCH_DB_NAME = "scratch_regression_eval_db"

async def run_regression():
    print("=" * 125)
    print(f"RUNNING 24-CLIP REGRESSION TO CSV WITH REAL SCRATCH DB ({SCRATCH_DB_NAME}) & LOG_THREAT_EVENT")
    print("=" * 125)

    # 1. Setup real Motor client connected to SCRATCH database (not surveillance_db)
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    scratch_db = client[SCRATCH_DB_NAME]
    database.db_helper.client = client
    database.db_helper.db = scratch_db

    # Clean existing scratch DB if any
    await client.drop_database(SCRATCH_DB_NAME)

    # Use isolated temporary directories for test media so prod screenshots/ and event_clips/ are untouched
    import tempfile
    temp_media_dir = tempfile.TemporaryDirectory()
    temp_screenshots = Path(temp_media_dir.name) / "screenshots"
    temp_clips = Path(temp_media_dir.name) / "event_clips"
    temp_screenshots.mkdir(parents=True, exist_ok=True)
    temp_clips.mkdir(parents=True, exist_ok=True)

    orig_screenshots = settings.SCREENSHOTS_DIR
    orig_clips = settings.EVENT_CLIPS_DIR
    settings.SCREENSHOTS_DIR = str(temp_screenshots)
    settings.EVENT_CLIPS_DIR = str(temp_clips)
    os.environ["SCREENSHOTS_DIR"] = str(temp_screenshots)
    os.environ["EVENT_CLIPS_DIR"] = str(temp_clips)

    ai_service = AIService()

    # Prepare clip list
    nonfight_clips = sorted(list(NONFIGHT_DIR.glob("*.avi")))[:20]
    eval_clips = [("normal.mp4", NORMAL_VIDEO)] + [(c.name, c) for c in nonfight_clips]
    eval_clips.append(("accident.mp4", repo_root / "sample_videos" / "accident.mp4"))
    eval_clips.append(("fight.mp4", repo_root / "sample_videos" / "fight.mp4"))
    eval_clips.append(("weapon.mp4", repo_root / "sample_videos" / "weapon.mp4"))

    # Seed cameras in scratch DB so log_threat_event camera queries succeed
    for clip_idx, (clip_name, _) in enumerate(eval_clips, 1):
        cam_id = f"eval_cam_{clip_idx:02d}"
        await scratch_db["cameras"].insert_one({
            "camera_id": cam_id,
            "name": clip_name,
            "location": "Evaluation Test Area",
            "latitude": 12.9716,
            "longitude": 77.5946,
            "status": "ONLINE",
            "stream_url": ""
        })

    # Track email calls
    email_calls = []
    async def mock_send_email(**kwargs):
        email_calls.append(kwargs)
        return True

    async def mock_send_accident_email(**kwargs):
        email_calls.append(kwargs)
        return True

    rows = []
    knife_analysis_0Nq = []

    # Enable email setting so _should_trigger_email processes qualifying alerts
    orig_email_enabled = getattr(settings, "EMAIL_ENABLED", True)
    settings.EMAIL_ENABLED = True

    try:
        with patch.object(EmailService, "send_alert_email", side_effect=mock_send_email), \
             patch.object(EmailService, "send_accident_alert_email", side_effect=mock_send_accident_email):

            for clip_idx, (clip_name, clip_path) in enumerate(eval_clips, 1):
                cam_id = f"eval_cam_{clip_idx:02d}"
                ai_service.cleanup_camera(cam_id)
                event_service._email_cooldown_cache.clear()
                email_calls.clear()

                cap = cv2.VideoCapture(str(clip_path))
                if not cap.isOpened():
                    print(f"Error opening {clip_path}")
                    continue

                frame_idx = 0
                threat_levels = []
                raw_pistol_count = 0
                raw_knife_count = 0
                alerts_fired = []

                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    frame_idx += 1

                    res = ai_service.process_frame(
                        frame=frame.copy(),
                        camera_id=cam_id,
                        camera_name=clip_name,
                        gps_coords=(12.9716, 77.5946)
                    )
                    annotated_frame = res[0]
                    dets = res[1]
                    highest_threat = res[2]
                    score = res[3]
                    alerts = res[4]
                    correlated = res[5]
                    xai_exp = res[7]
                    fight_res = res[8]
                    accident_res = res[9]

                    threat_levels.append(highest_threat)

                    for d in dets:
                        cls_lower = d.get("class", "").lower()
                        conf = d.get("confidence", 0.0)
                        if cls_lower in ["pistol", "gun"]:
                            raw_pistol_count += 1
                        elif cls_lower in ["knife", "scissors"]:
                            raw_knife_count += 1
                            if "0NqPsSHdri8_0" in clip_name:
                                knife_analysis_0Nq.append({
                                    "frame": frame_idx,
                                    "class": d.get("class"),
                                    "conf": round(conf, 4),
                                    "bbox": d.get("bbox"),
                                    "score": score,
                                    "threat_level": highest_threat,
                                    "active_events": [a.get("event_type") for a in alerts]
                                })

                    # Pass each alert through real event_service.log_threat_event (same as stream.py)
                    for alert in alerts:
                        alerts_fired.append(alert.get("event_type"))
                        await event_service.log_threat_event(
                            camera_id=cam_id,
                            event_type=alert["event_type"],
                            threat_level=alert["threat_level"],
                            threat_score=score,
                            confidence=alert["confidence"],
                            frame=annotated_frame,
                            frame_buffer=ai_service.frame_buffers.get(cam_id, []),
                            person_id=alert.get("person_id"),
                            zone_id=alert.get("zone_id"),
                            duration=alert.get("duration"),
                            correlated_events=correlated,
                            xai_explanation=xai_exp,
                            fight_detected=fight_res.get("fight_detected", False),
                            fight_confidence=fight_res.get("fight_confidence", 0.0),
                            fight_participants=fight_res.get("fight_participants", []),
                            fight_duration_sec=fight_res.get("fight_duration_sec", 0.0),
                            accident_detected=accident_res.get("accident_detected", False),
                            accident_confidence=accident_res.get("accident_confidence", 0.0)
                        )

                cap.release()

                # Allow any fire-and-forget email tasks to finish executing
                await asyncio.sleep(0.05)

                max_threat = max(threat_levels, key=lambda l: THREAT_ORDER.get(l, 0)) if threat_levels else "LOW"
                high_frames = sum(1 for l in threat_levels if l == "HIGH")
                crit_frames = sum(1 for l in threat_levels if l == "CRITICAL")

                row_data = {
                    "clip_name": clip_name,
                    "frames": frame_idx,
                    "max_threat": max_threat,
                    "high_frames": high_frames,
                    "critical_frames": crit_frames,
                    "raw_pistol_dets": raw_pistol_count,
                    "raw_knife_dets": raw_knife_count,
                    "alerts": "; ".join(sorted(list(set(alerts_fired)))) if alerts_fired else "None",
                    "emails": len(email_calls)
                }
                rows.append(row_data)

        # Write to CSV
        fieldnames = ["clip_name", "frames", "max_threat", "high_frames", "critical_frames", "raw_pistol_dets", "raw_knife_dets", "alerts", "emails"]
        with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nWrote {len(rows)} rows to {CSV_OUT}")
        print("\n" + "=" * 125)
        print(f"{'#':<3} | {'Clip Name':<24} | {'Frames':<6} | {'Max Threat':<10} | {'HIGH':<5} | {'CRIT':<5} | {'Pistol':<6} | {'Knife':<6} | {'Alerts':<35} | {'Emails'}")
        print("-" * 125)
        for idx, r in enumerate(rows, 1):
            print(f"{idx:<3} | {r['clip_name']:<24} | {r['frames']:<6} | {r['max_threat']:<10} | {r['high_frames']:<5} | {r['critical_frames']:<5} | {r['raw_pistol_dets']:<6} | {r['raw_knife_dets']:<6} | {r['alerts']:<35} | {r['emails']}")
        print("=" * 125)

        print("\n" + "=" * 80)
        print("ANALYSIS FOR 0NqPsSHdri8_0.avi RAW KNIFE FRAMES:")
        print("=" * 80)
        if not knife_analysis_0Nq:
            print("No raw knife detections observed in 0NqPsSHdri8_0.avi!")
        else:
            for k in knife_analysis_0Nq:
                print(f"Frame {k['frame']:3d}: class={k['class']} conf={k['conf']:.3f} | Threat={k['threat_level']} (Score={k['score']}) | Alerts={k['active_events']}")

    finally:
        settings.EMAIL_ENABLED = orig_email_enabled
        settings.SCREENSHOTS_DIR = orig_screenshots
        settings.EVENT_CLIPS_DIR = orig_clips
        temp_media_dir.cleanup()
        # Drop scratch database
        await client.drop_database(SCRATCH_DB_NAME)
        client.close()

if __name__ == "__main__":
    asyncio.run(run_regression())
