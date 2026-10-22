import sys
import os
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

import asyncio
import time
import json
import urllib.request
import urllib.parse
import websockets

def get_token():
    login_data = urllib.parse.urlencode({"username": "admin", "password": "admin123"}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/auth/login",
        data=login_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        return data["access_token"]

THREAT_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

async def monitor_camera(camera_id: str, token: str, duration_sec: float = 20.0):
    uri = f"ws://127.0.0.1:8000/api/ws/camera/{camera_id}?token={token}"
    frames_log = []
    first_alert_frame = {}
    distinct_alerts = set()
    max_threat = "LOW"

    print(f"\n[{camera_id}] Connecting to WebSocket stream for {duration_sec}s...")
    try:
        async with websockets.connect(uri) as ws:
            start_time = time.time()
            frame_idx = 0
            while (time.time() - start_time) < duration_sec:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                except asyncio.TimeoutError:
                    break

                frame_idx += 1
                data = json.loads(msg)
                
                person_count = data.get("person_count", 0)
                dets = data.get("detections", [])
                weapon_dets = [d.get("class") for d in dets if d.get("class", "").lower() in ("pistol", "knife", "gun")]
                fight_det = data.get("fight_detected", False)
                accident_det = data.get("accident_detected", False)
                highest_threat = data.get("highest_threat", "LOW")
                alerts = data.get("alerts", [])
                alert_types = [a.get("event_type") for a in alerts]

                for a_type in alert_types:
                    if a_type not in first_alert_frame:
                        first_alert_frame[a_type] = frame_idx
                    distinct_alerts.add(a_type)

                if THREAT_ORDER.get(highest_threat, 0) > THREAT_ORDER.get(max_threat, 0):
                    max_threat = highest_threat

                frames_log.append({
                    "frame": frame_idx,
                    "persons": person_count,
                    "weapons": weapon_dets,
                    "fight": fight_det,
                    "accident": accident_det,
                    "threat": highest_threat,
                    "alerts": alert_types
                })

                # Print every 20 frames or when an alert occurs
                if frame_idx % 25 == 0 or alert_types:
                    alerts_str = f" | ALERTS: {alert_types}" if alert_types else ""
                    print(f"  Frame {frame_idx:3d}: P={person_count} | Weap={weapon_dets} | Fight={fight_det} | Acc={accident_det} | Threat={highest_threat}{alerts_str}")

    except Exception as e:
        print(f"[{camera_id}] Connection error: {e}")

    return {
        "camera_id": camera_id,
        "frames_received": len(frames_log),
        "max_threat": max_threat,
        "distinct_alerts": list(distinct_alerts),
        "first_alert_frame": first_alert_frame
    }

async def run_multi_frame_test():
    print("=" * 100)
    print("ITEM 6: MULTI-FRAME LIVE CAMERA EVALUATION (20s per camera, CAM-001..CAM-005)")
    print("=" * 100)

    try:
        token = get_token()
        print("Obtained authentication token successfully.")
    except Exception as e:
        print(f"Auth token error (ensure uvicorn server is running): {e}")
        return

    cameras = ["CAM-001", "CAM-002", "CAM-003", "CAM-004", "CAM-005"]
    summary = []

    for cam in cameras:
        res = await monitor_camera(cam, token, duration_sec=20.0)
        summary.append(res)

    print("\n" + "=" * 100)
    print(f"{'Camera ID':<10} | {'Frames':<8} | {'Max Threat':<12} | {'Distinct Alerts and First Frame Appeared'}")
    print("-" * 100)
    for s in summary:
        alerts_detail = []
        for a in s["distinct_alerts"]:
            f_num = s["first_alert_frame"].get(a, "?")
            alerts_detail.append(f"{a} (frame {f_num})")
        alerts_str = ", ".join(alerts_detail) if alerts_detail else "None"
        print(f"{s['camera_id']:<10} | {s['frames_received']:<8} | {s['max_threat']:<12} | {alerts_str}")
    print("=" * 100)

    # Verification checks
    cam2 = next((s for s in summary if s["camera_id"] == "CAM-002"), None)
    cam4 = next((s for s in summary if s["camera_id"] == "CAM-004"), None)

    print("\n[VERIFICATION CHECKS]")
    if cam2 and any("FIGHT" in a for a in cam2["distinct_alerts"]):
        print(f"[PASS] CAM-002: Fight alert confirmed: {[a for a in cam2['distinct_alerts'] if 'FIGHT' in a]}")
    else:
        print(f"[FAIL] CAM-002: Fight alert was NOT observed during 20s window! Alerts: {cam2['distinct_alerts'] if cam2 else 'None'}")

    if cam4 and any("WEAPON" in a for a in cam4["distinct_alerts"]):
        print(f"[PASS] CAM-004: Weapon alert confirmed: {[a for a in cam4['distinct_alerts'] if 'WEAPON' in a]}")
    else:
        print(f"[FAIL] CAM-004: Weapon alert was NOT observed during 20s window! Alerts: {cam4['distinct_alerts'] if cam4 else 'None'}")

if __name__ == "__main__":
    asyncio.run(run_multi_frame_test())
