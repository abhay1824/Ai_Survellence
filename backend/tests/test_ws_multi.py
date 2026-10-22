import sys
import os
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

import asyncio
import websockets
import json
import time
import urllib.request
import urllib.parse
from datetime import datetime

THREAT_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

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

async def test_camera_ws_20s(camera_id: str, token: str):
    uri = f"ws://127.0.0.1:8000/api/ws/camera/{camera_id}?token={token}"
    print(f"\nConnecting to {camera_id} for 20 seconds...")
    
    frame_count = 0
    threat_levels = []
    threat_scores = []
    distinct_alerts = {}  # {alert_type: first_frame_idx}
    unattended_events = []
    unattended_detections = []
    critical_frames = []

    start_time = time.time()
    try:
        async with websockets.connect(uri) as ws:
            while time.time() - start_time < 20.0:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    break

                frame_count += 1
                data = json.loads(msg)

                highest_threat = data.get("highest_threat", "LOW")
                score = data.get("threat_score", 0)
                threat_levels.append(highest_threat)
                threat_scores.append(score)

                if highest_threat == "CRITICAL":
                    critical_frames.append(frame_count)

                alerts = data.get("alerts", [])
                for a in alerts:
                    ev_type = a.get("event_type", "")
                    if ev_type not in distinct_alerts:
                        distinct_alerts[ev_type] = frame_count
                    if "UNATTENDED" in ev_type.upper():
                        unattended_events.append((frame_count, ev_type))

                dets = data.get("detections", [])
                for d in dets:
                    if d.get("is_unattended", False):
                        unattended_detections.append((frame_count, d.get("class")))

        max_threat = max(threat_levels, key=lambda l: THREAT_ORDER.get(l, 0)) if threat_levels else "NONE"
        max_score = max(threat_scores) if threat_scores else 0

        return {
            "camera_id": camera_id,
            "status": "SUCCESS",
            "frames": frame_count,
            "max_threat": max_threat,
            "max_score": max_score,
            "distinct_alerts": distinct_alerts,
            "unattended_events": unattended_events,
            "unattended_detections": unattended_detections,
            "critical_frames": critical_frames
        }

    except Exception as e:
        return {
            "camera_id": camera_id,
            "status": f"ERROR: {e}",
            "frames": frame_count,
            "max_threat": "ERROR",
            "max_score": 0,
            "distinct_alerts": {},
            "unattended_events": [],
            "unattended_detections": [],
            "critical_frames": []
        }

async def test_all_cameras():
    print("=" * 115)
    print("LIVE 20-SECOND WEBSOCKET RE-CHECK (CAM-001 through CAM-005)")
    print("=" * 115)

    try:
        token = get_token()
        print("Obtained auth token successfully.")
    except Exception as e:
        print(f"Failed to obtain token: {e}")
        return

    cameras = ["CAM-001", "CAM-002", "CAM-003", "CAM-004", "CAM-005"]
    results = []

    for cam in cameras:
        res = await test_camera_ws_20s(cam, token)
        results.append(res)

    print("\n" + "=" * 115)
    print(f"{'Camera ID':<10} | {'Frames':<7} | {'Max Threat':<10} | {'Max Score':<10} | {'Distinct Alerts (First Frame)':<40} | {'Unattended Events'}")
    print("-" * 115)
    for r in results:
        alerts_str = ", ".join([f"{k} (f{v})" for k, v in r["distinct_alerts"].items()]) or "None"
        unatt_str = ", ".join([f"{e[1]} (f{e[0]})" for e in r["unattended_events"]]) or "None"
        print(f"{r['camera_id']:<10} | {r['frames']:<7} | {r['max_threat']:<10} | {r['max_score']:<10} | {alerts_str:<40} | {unatt_str}")
    print("=" * 115)

    print("\n--- VALIDATION CRITERIA CHECKS ---")
    cam3 = next(r for r in results if r["camera_id"] == "CAM-003")
    cam4 = next(r for r in results if r["camera_id"] == "CAM-004")

    print(f"CAM-003: Max Threat={cam3['max_threat']}, Critical Frames={len(cam3['critical_frames'])}, Unattended Events={len(cam3['unattended_events'])}")
    assert cam3["max_threat"] != "CRITICAL", f"CAM-003 reached CRITICAL!"
    assert len(cam3["critical_frames"]) == 0, f"CAM-003 hit CRITICAL on frames {cam3['critical_frames']}"
    assert not any("PISTOL" in e[1].upper() and "UNATTENDED" in e[1].upper() for e in cam3["unattended_events"]), "CAM-003 had unattended pistol event!"
    print(">>> PASS: CAM-003 has NO unattended pistol and NO CRITICAL.")

    print(f"CAM-004: Max Threat={cam4['max_threat']}, Unattended Events={len(cam4['unattended_events'])}, Unattended Detections={len(cam4['unattended_detections'])}")
    assert not any("UNATTENDED" in e[1].upper() for e in cam4["unattended_events"]), "CAM-004 had an unattended event!"
    assert len(cam4["unattended_detections"]) == 0, f"CAM-004 knife flipped to unattended in {len(cam4['unattended_detections'])} frames!"
    print(">>> PASS: CAM-004 has NO flip to unattended while held.")

if __name__ == "__main__":
    asyncio.run(test_all_cameras())
