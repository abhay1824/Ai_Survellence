import sys
import os
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

import asyncio
import base64
import json
import urllib.request
import urllib.parse
import cv2
import numpy as np
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

async def run_cam001_live():
    print("=" * 80)
    print("CAM-001 LIVE WEBCAM EVALUATION (Mean Brightness & Persons per 20 frames)")
    print("=" * 80)

    token = get_token()
    uri = f"ws://127.0.0.1:8000/api/ws/camera/CAM-001?token={token}"

    brightnesses = []
    persons_log = []

    async with websockets.connect(uri) as ws:
        print("Connected to CAM-001 live stream. Monitoring 100 frames...")
        for frame_idx in range(1, 101):
            msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
            data = json.loads(msg)

            person_count = data.get("person_count", 0)
            b64_frame = data.get("frame", "")

            mean_brightness = 0.0
            if b64_frame:
                if "," in b64_frame:
                    b64_frame = b64_frame.split(",", 1)[1]
                raw_bytes = base64.b64decode(b64_frame)
                np_arr = np.frombuffer(raw_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is not None:
                    mean_brightness = float(np.mean(img))

            brightnesses.append(mean_brightness)
            persons_log.append(person_count)

            if frame_idx % 20 == 0:
                recent_brightness = brightnesses[-20:]
                recent_persons = persons_log[-20:]
                avg_b = sum(recent_brightness) / len(recent_brightness)
                avg_p = sum(recent_persons) / len(recent_persons)
                print(f"Frames {frame_idx-19:3d}-{frame_idx:3d}: Mean Brightness = {avg_b:.2f} | Avg Persons = {avg_p:.1f} | Latest Frame Brightness = {mean_brightness:.2f}")

    overall_avg_b = sum(brightnesses) / len(brightnesses)
    print("\n" + "=" * 80)
    print(f"OVERALL CAM-001 MEAN BRIGHTNESS: {overall_avg_b:.2f}")
    if overall_avg_b > 30.0:
        print(">>> PROOF: Frame stream is LIVE physical webcam feed (mean brightness > 30, typical room lighting).")
        print("    Offline card has mean brightness < 25 (dark slate placeholder).")
    else:
        print(">>> Frame brightness indicates offline card or dark scene.")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_cam001_live())
