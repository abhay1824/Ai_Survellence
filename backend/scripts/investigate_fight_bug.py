import sys
import os
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

import sys
import os
import time
import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

from app.services.ai_service import AIService
from app.routers.stream import generate_offline_frame

ai_service = AIService()

print("=== AI Service Model Check ===")
print("Person model:", ai_service.person_model)
print("Weapon model:", ai_service.model)
print("Fight model:", ai_service.fight_model)

def run_test_suite(name, get_frame_fn, camera_id):
    print(f"\n========================================================")
    print(f"--- Testing 100 Frames: {name} (camera_id={camera_id}) ---")
    print(f"========================================================")
    
    fight_detected_count = 0
    raw_fight_count = 0
    total_persons = 0
    active_event_types = set()
    
    for i in range(100):
        frame = get_frame_fn(i)
        res = ai_service.process_frame(frame, camera_id, name, (16.83, 75.71))
        annotated, dets, highest_threat, score, active_events, corr, p_count, xai, fight_res, acc_res = res
        
        raw_fight = fight_res.get("raw_fight", False)
        f_det = fight_res.get("fight_detected", False)
        if raw_fight:
            raw_fight_count += 1
        if f_det:
            fight_detected_count += 1
        total_persons += p_count
        
        for e in active_events:
            active_event_types.add(e.get("event_type"))
            
        buf = list(ai_service.fight_buffers.get(camera_id, []))
        pg = list(ai_service.person_gate_buffers.get(camera_id, []))
        
        # Log first 5 frames, milestone frames, or any frame where something triggers
        if i < 5 or i in [25, 50, 75, 99] or raw_fight or f_det or p_count > 0 or len(active_events) > 0:
            print(f"Frame {i+1:3d} | Persons: {p_count} | raw_fight: {raw_fight} ({fight_res.get('fight_confidence', 0):.2f}) | "
                  f"f_det: {f_det} | buf_fight_count: {sum(1 for f,_ in buf if f)}/{len(buf)} | "
                  f"gate_persons_seen: {sum(1 for g in pg if g)}/{len(pg)} | "
                  f"events: {[e.get('event_type') for e in active_events]}")

    print(f"\nSummary for {name}:")
    print(f"  raw_fight: {raw_fight_count}/100")
    print(f"  fight_detected: {fight_detected_count}/100")
    print(f"  total persons detected across 100 frames: {total_persons}")
    print(f"  all active event types seen: {active_event_types}")

# 1. Plain White
white_img = np.full((480, 640, 3), 255, dtype=np.uint8)
run_test_suite("Plain White", lambda i: white_img.copy(), "CAM-TEST-WHITE")

# 2. Plain Gray
gray_img = np.full((480, 640, 3), 128, dtype=np.uint8)
run_test_suite("Plain Gray", lambda i: gray_img.copy(), "CAM-TEST-GRAY")

# 3. Offline Placeholder Frame
run_test_suite("Offline Placeholder", lambda i: generate_offline_frame("Cam 1", i), "CAM-TEST-OFFLINE")

# 4. Check Raw Person Model on Blank Frames
print("\n========================================================")
print("--- Direct Person Model Check on Blank Frames ---")
print("========================================================")
for label, img in [("White", white_img), ("Gray", gray_img), ("Offline", generate_offline_frame("Cam 1", 1))]:
    if ai_service.person_model is not None:
        preds = ai_service.person_model.predict(img, classes=[0], verbose=False)
        boxes = preds[0].boxes if preds else None
        b_count = len(boxes) if boxes is not None else 0
        confs = [float(b.conf[0]) for b in boxes] if b_count > 0 else []
        print(f"Person model on {label}: {b_count} boxes found. Confidences: {confs}")
    else:
        print("Person model is None!")

# 5. Check Raw Fight Model on Blank Frames (directly without gate)
print("\n========================================================")
print("--- Direct Fight Model Check on Blank Frames (No Person Gate) ---")
print("========================================================")
for label, img in [("White", white_img), ("Gray", gray_img), ("Offline", generate_offline_frame("Cam 1", 1))]:
    if ai_service.fight_model is not None:
        preds = ai_service.fight_model.predict([img], verbose=False)
        if preds and preds[0].probs is not None:
            probs = preds[0].probs
            names = preds[0].names
            top1_idx = int(probs.top1)
            top1_conf = float(probs.top1conf)
            top1_name = names.get(top1_idx, "").lower()
            all_scores = {names[i]: float(probs.data[i]) for i in range(len(names))}
            print(f"Fight model on {label}: top1='{top1_name}' ({top1_conf:.4f}), all={all_scores}")
        else:
            print(f"Fight model on {label}: no probs returned")
    else:
        print("Fight model is None!")

# 6. Test Cooldown / Buffer clearing after a real fight
print("\n========================================================")
print("--- Test Cooldown & Buffer Clearing After Simulated Fight ---")
print("========================================================")
cam_cooldown_id = "CAM-TEST-COOLDOWN"
# Step A: Feed 10 frames with raw_fight override = True and 2 qualified persons
person_det_dummy = [
    {"bbox": [10, 10, 100, 200], "class": "Person", "confidence": 0.85, "track_id": 1, "threat_level": "LOW"},
    {"bbox": [120, 10, 210, 200], "class": "Person", "confidence": 0.82, "track_id": 2, "threat_level": "LOW"}
]

print("Simulating 10 fight frames...")
for f in range(10):
    t_now = time.time() + f * 0.067
    f_res = ai_service.run_fight_classification(
        white_img, cam_cooldown_id, person_det_dummy, t_now,
        raw_fight_override=True, fight_conf_override=0.95
    )
    print(f"Fight frame {f+1}: active={f_res['fight_detected']}, started={f_res['fight_started']}, dur={f_res['fight_duration_sec']}")

print("\nNow feeding 40 calm/blank frames (0 persons, raw_fight=False)...")
for f in range(40):
    t_now = time.time() + (10 + f) * 0.067
    f_res = ai_service.run_fight_classification(
        white_img, cam_cooldown_id, [], t_now,
        raw_fight_override=None
    )
    dur_state = ai_service.fight_duration[cam_cooldown_id]
    buf = list(ai_service.fight_buffers[cam_cooldown_id])
    pg = list(ai_service.person_gate_buffers[cam_cooldown_id])
    print(f"Calm frame {f+1:2d} | active={str(f_res['fight_detected']):5s} | ended={str(f_res['fight_ended']):5s} | "
          f"non_fight_consec={dur_state['non_fight_consecutive']:2d} | "
          f"buf_fight_sum={sum(1 for x,_ in buf if x)} | gate_sum={sum(1 for g in pg if g)}")
