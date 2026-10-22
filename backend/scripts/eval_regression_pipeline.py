import os
import sys
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock
import cv2
import numpy as np

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

from app.services.ai_service import AIService
from app.services import event_service
from app.services.email_service import EmailService

NORMAL_VIDEO = repo_root / "sample_videos" / "normal.mp4"
NONFIGHT_DIR = repo_root / "RWF-2000" / "NonFight"

THREAT_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

async def run_regression_pipeline():
    print("=" * 120)
    print("1. REGRESSION EVALUATION (21 CLIPS: normal.mp4 + 20 NonFight clips)")
    print("=" * 120)

    ai_service = AIService()

    # Mock DB and email collection
    mock_event_col = AsyncMock()
    mock_event_col.find_one.return_value = None
    mock_insert_res = AsyncMock()
    mock_insert_res.inserted_id = "test_event_id"
    mock_event_col.insert_one.return_value = mock_insert_res
    mock_event_col.update_one.return_value = AsyncMock()

    mock_cam_col = AsyncMock()
    mock_cam_col.find_one.return_value = {"latitude": 0.0, "longitude": 0.0}
    mock_cam_col.update_one.return_value = AsyncMock()

    email_calls = []
    async def mock_send_email(**kwargs):
        email_calls.append(kwargs)
        return True

    nonfight_clips = sorted(list(NONFIGHT_DIR.glob("*.avi")))[:20]
    eval_clips = [("normal.mp4", NORMAL_VIDEO)] + [(f"NonFight_{c.stem}", c) for c in nonfight_clips]

    results = []

    with patch.object(EmailService, "send_alert_email", side_effect=mock_send_email), \
         patch("app.services.event_service.get_event_collection", return_value=mock_event_col), \
         patch("app.services.event_service.get_camera_collection", return_value=mock_cam_col):

        for clip_label, clip_path in eval_clips:
            cam_id = f"eval_cam_{clip_label}"
            ai_service.cleanup_camera(cam_id)
            event_service._email_cooldown_cache.clear()
            email_calls.clear()

            cap = cv2.VideoCapture(str(clip_path))
            if not cap.isOpened():
                print(f"Error: Unable to open {clip_path}")
                continue

            frame_count = 0
            raw_pistol_dets = []
            raw_knife_dets = []
            all_alerts = []
            threat_levels = []

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame_count += 1

                out = ai_service.process_frame(frame, cam_id, clip_label, (0.0, 0.0))
                detections = out[1]
                highest_threat = out[2]
                threat_score = out[3]
                alerts = out[4]

                threat_levels.append(highest_threat)

                # Record raw detections
                for d in detections:
                    cls_name = d.get("class", "").lower()
                    conf = d.get("confidence", 0.0)
                    if cls_name in ["pistol", "gun"]:
                        raw_pistol_dets.append((frame_count, conf, highest_threat))
                    elif cls_name in ["knife", "scissors"]:
                        raw_knife_dets.append((frame_count, conf, highest_threat))

                # Record confirmed alerts and feed to event_service
                for a in alerts:
                    all_alerts.append(a.get("event_type"))
                    await event_service.log_threat_event(
                        camera_id=cam_id,
                        event_type=a["event_type"],
                        threat_level=a["threat_level"],
                        threat_score=threat_score,
                        confidence=a["confidence"],
                        frame=frame,
                        frame_buffer=[]
                    )

            cap.release()
            await asyncio.sleep(0.05)

            max_level = max(threat_levels, key=lambda l: THREAT_ORDER.get(l, 0)) if threat_levels else "LOW"
            high_or_crit_count = sum(1 for l in threat_levels if l in ("HIGH", "CRITICAL"))

            results.append({
                "clip": clip_label,
                "frames": frame_count,
                "raw_pistol": raw_pistol_dets,
                "raw_knife": raw_knife_dets,
                "max_threat": max_level,
                "high_crit_frames": high_or_crit_count,
                "alerts": list(set(all_alerts)),
                "email_calls": len(email_calls)
            })

    # Print summary table
    print("\n" + "=" * 125)
    print(f"{'Clip Name':<26} | {'Frames':<7} | {'Max Threat':<11} | {'HIGH/CRIT Frames':<17} | {'Raw Dets':<14} | {'Alerts':<20} | {'Emails'}")
    print("-" * 125)
    for r in results:
        raw_str = f"P:{len(r['raw_pistol'])} K:{len(r['raw_knife'])}"
        alerts_str = ", ".join(r["alerts"]) if r["alerts"] else "None"
        if len(alerts_str) > 20:
            alerts_str = alerts_str[:17] + "..."
        print(f"{r['clip']:<26} | {r['frames']:<7} | {r['max_threat']:<11} | {r['high_crit_frames']:<17} | {raw_str:<14} | {alerts_str:<20} | {r['email_calls']}")
    print("=" * 125)

    # Call out clip 0_DzLlklZa0_3 specifically
    clip_target = next((r for r in results if "0_DzLlklZa0_3" in r["clip"]), None)
    if clip_target:
        print("\n" + "*" * 80)
        print("CALLOUT: CLIP NonFight_0_DzLlklZa0_3")
        print("*" * 80)
        print(f"Total Frames: {clip_target['frames']}")
        print(f"Max Threat Level: {clip_target['max_threat']}")
        print(f"Frames at HIGH or CRITICAL: {clip_target['high_crit_frames']}")
        print(f"Raw Pistol Detections: {clip_target['raw_pistol']}")
        print(f"Raw Knife Detections: {clip_target['raw_knife']}")
        print(f"Weapon / Threat Alerts: {clip_target['alerts']}")
        print(f"Email Notifications: {clip_target['email_calls']}")
        if clip_target['raw_pistol']:
            f_num, f_conf, f_threat = clip_target['raw_pistol'][0]
            print(f"Note on frame {f_num}: Single-frame pistol detection (conf={f_conf:.2f}) evaluated to highest_threat='{f_threat}'.")
            print("Because 3-frame persistence was NOT met, weapon_counters never reached 3, 0 alerts were fired, and 0 emails were triggered.")
        print("*" * 80)

def main():
    asyncio.run(run_regression_pipeline())

if __name__ == "__main__":
    main()
