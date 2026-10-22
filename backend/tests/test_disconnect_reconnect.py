import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure backend dir is on path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.ai_service import AIService

print("================================================================================")
print("DISCONNECT / RECONNECT & IDLE TIMEOUT VERIFICATION TEST")
print("================================================================================")

service = AIService()

# -------------------------------------------------------------------------
# Test 1: WebSocket Disconnect with cleanup_camera() (stream.py finally block)
# -------------------------------------------------------------------------
print("\n--- Subtest 1: Mid-fight Disconnect with cleanup_camera() ---")
cam_id = "CAM-002-DISCONNECT"
service.cleanup_camera(cam_id)

repo_root = backend_dir.parent
video_path = repo_root / "RWF-2000" / "Fight" / "-1l5631l3fg_0.avi"
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print(f"Error opening {video_path}")
    sys.exit(1)

fight_achieved = False
frame_idx = 0
last_frame = None

print("Streaming fight.mp4 until fight is detected...")
while frame_idx < 150:
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    last_frame = frame.copy()
    
    res = service.process_frame(
        frame=frame,
        camera_id=cam_id,
        camera_name="Cam 2",
        gps_coords=(16.83, 75.71)
    )
    fight_res = res[8]
    if fight_res.get("fight_detected", False):
        print(f"  Fight successfully detected at frame {frame_idx} (conf={fight_res['fight_confidence']:.2f}, active={fight_res['fight_detected']})")
        fight_achieved = True
        break
    frame_idx += 1

cap.release()

if not fight_achieved:
    print("FAILED: Could not trigger initial fight on fight.mp4")
    sys.exit(1)

print("\nSimulating WebSocket Disconnect mid-fight:")
print("  Calling service.cleanup_camera(cam_id) as in stream.py finally block...")
service.cleanup_camera(cam_id)

print("  Waiting 10 seconds before client reconnects...")
time.sleep(10.0)

print("\nClient reconnects: Processing First Frame on reconnected stream...")
# Even if the reconnected stream sends a frame (e.g. calm frame or white frame or offline placeholder)
white_img = np.full((480, 640, 3), 255, dtype=np.uint8)
res_reconnect = service.process_frame(
    frame=white_img,
    camera_id=cam_id,
    camera_name="Cam 2",
    gps_coords=(16.83, 75.71)
)
f_det_1 = res_reconnect[8].get("fight_detected", False)
raw_f_1 = res_reconnect[8].get("raw_fight", False)
events_1 = [e.get("event_type") for e in res_reconnect[4]]

print(f"  Frame 1 on Reconnect: fight_detected={f_det_1}, raw_fight={raw_f_1}, events={events_1}")
if not f_det_1:
    print("  ✅ PASS: fight_detected is False on the first frame after reconnect!")
else:
    print("  ❌ FAIL: fight_detected is True on the first frame!")

# -------------------------------------------------------------------------
# Test 2: Wall-Clock Idle Timeout (>5 seconds) WITHOUT explicit cleanup
# -------------------------------------------------------------------------
print("\n--- Subtest 2: Wall-Clock Idle Timeout (>5s) WITHOUT cleanup_camera() ---")
cam_idle_id = "CAM-002-IDLE"
service.cleanup_camera(cam_idle_id)

# Populate fight buffers directly
cap2 = cv2.VideoCapture(str(video_path))
f_idx = 0
while f_idx < 150:
    ret, frame = cap2.read()
    if not ret or frame is None:
        break
    res = service.process_frame(
        frame=frame,
        camera_id=cam_idle_id,
        camera_name="Cam 2 Idle",
        gps_coords=(16.83, 75.71)
    )
    if res[8].get("fight_detected", False):
        print(f"  Fight active on {cam_idle_id} at frame {f_idx}")
        break
    f_idx += 1
cap2.release()

print("  Simulating 6.0 seconds stream freeze / pause (NO cleanup called)...")
# Manually adjust the last processed time to 6.0 seconds in the past to test wall-clock expiry
service._last_fight_process_time[cam_idle_id] = time.time() - 6.0

print("  Processing frame after 6s idle gap...")
res_after_idle = service.process_frame(
    frame=white_img,
    camera_id=cam_idle_id,
    camera_name="Cam 2 Idle",
    gps_coords=(16.83, 75.71)
)
f_det_idle = res_after_idle[8].get("fight_detected", False)
print(f"  Frame after 6s idle: fight_detected={f_det_idle}, raw_fight={res_after_idle[8].get('raw_fight')}")
if not f_det_idle:
    print("  ✅ PASS: Wall-clock idle timeout cleared buffers and fight_detected is False on frame 1!")
else:
    print("  ❌ FAIL: fight_detected remained True after 6s idle gap!")

print("\n" + "="*80)
print("ALL DISCONNECT & IDLE TIMEOUT TESTS COMPLETED")
print("="*80)
