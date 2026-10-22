import sys
import os
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

import sys
import time
import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

from app.services.ai_service import AIService

ai_service = AIService()
cap = cv2.VideoCapture(0)
camera_id = "CAM-001"

print("Capturing 100 frames from real webcam 0...")
fight_detected_count = 0
raw_fight_count = 0
person_count_sum = 0
events_seen = set()

# Also save a sample frame to disk to inspect its brightness and content
first_frame = None

for i in range(100):
    ret, frame = cap.read()
    if not ret:
        print(f"Frame {i+1}: Failed to read frame from webcam!")
        continue
    
    if first_frame is None:
        first_frame = frame.copy()
        cv2.imwrite("test_webcam_frame.jpg", first_frame)
        print(f"Webcam frame size: {frame.shape}, mean brightness: {np.mean(frame):.2f}, stddev: {np.std(frame):.2f}")
    
    annotated, dets, highest_threat, score, active_events, corr, p_count, xai, fight_res, acc_res = ai_service.process_frame(
        frame, camera_id, "Cam 1", (16.83, 75.71)
    )
    raw_fight = fight_res.get("raw_fight", False)
    f_det = fight_res.get("fight_detected", False)
    if raw_fight:
        raw_fight_count += 1
    if f_det:
        fight_detected_count += 1
    person_count_sum += p_count
    for e in active_events:
        events_seen.add(e.get("event_type"))
        
    buf = list(ai_service.fight_buffers.get(camera_id, []))
    pg = list(ai_service.person_gate_buffers.get(camera_id, []))
    
    person_confs = [round(d.get("confidence", 0), 2) for d in dets if d.get("class", "").lower() == "person"]
    
    if i < 10 or i % 10 == 0 or raw_fight or f_det or p_count > 0 or len(active_events) > 0:
        print(f"Frame {i+1:3d} | Persons: {p_count} {person_confs} | "
              f"raw_fight: {raw_fight} ({fight_res.get('fight_confidence', 0):.2f}) | "
              f"f_det: {f_det} | "
              f"buf_sum: {sum(1 for f,_ in buf if f)}/{len(buf)} | "
              f"gate_sum: {sum(1 for g in pg if g)}/{len(pg)} | "
              f"events: {[e.get('event_type') for e in active_events]}")

cap.release()
print(f"\nReal Webcam Summary across 100 frames:")
print(f"  raw_fight: {raw_fight_count}/100")
print(f"  fight_detected: {fight_detected_count}/100")
print(f"  total persons detected: {person_count_sum}")
print(f"  all active event types: {events_seen}")
