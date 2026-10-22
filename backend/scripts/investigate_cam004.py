import os
import cv2
import json
from pathlib import Path

# Add backend directory to sys.path
import sys
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.ai_service import AIService

def run_weapon_flip_investigation():
    print("=" * 80)
    print("INVESTIGATION 2: CAM-004 FLIP ON sample_videos/weapon.mp4 (3 loops)")
    print("=" * 80)

    video_path = str(backend_dir.parent / "sample_videos" / "weapon.mp4")
    ai_service = AIService()
    camera_id = "CAM_WEAPON_FLIP_INVESTIGATE"

    # Read all frames of weapon.mp4 into memory first
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_video_frames = len(frames)
    print(f"Loaded {total_video_frames} frames from {video_path}. Running 3 loops ({total_video_frames * 3} total frames)...")

    global_frame = 0
    flip_events = []

    for loop in range(1, 4):
        print(f"\n--- LOOP {loop} (frames {global_frame + 1} to {global_frame + total_video_frames}) ---")
        for local_idx, frame in enumerate(frames):
            global_frame += 1
            h, w = frame.shape[:2]

            (
                annotated_frame,
                detections,
                highest_threat,
                total_score,
                active_events,
                correlated,
                person_count,
                xai_exp,
                fight_res,
                accident_res
            ) = ai_service.process_frame(
                camera_id=camera_id,
                frame=frame.copy(),
                camera_name="CAM-004",
                gps_coords=(12.9716, 77.5946)
            )

            person_dets = [d for d in detections if d.get("class", "").lower() == "person"]
            weapon_dets = [d for d in detections if d.get("class", "").lower() in ("knife", "pistol", "gun", "scissors")]

            alert_types = [e.get("event_type") for e in active_events]

            # Print frames 1 to 30 or any frame where weapon is present
            if local_idx < 30 or weapon_dets:
                p_info = [f"conf={p['confidence']:.2f},trk={p.get('track_id')}" for p in person_dets]
                w_info = [f"{w_det['class']}:conf={w_det['confidence']:.2f},trk={w_det.get('track_id')},unatt={w_det.get('is_unattended')},dur={w_det.get('unattended_duration', 0):.1f}s" for w_det in weapon_dets]
                
                # Check unattended state
                state_for_cam = ai_service.unattended_weapon_state.get(camera_id, {})
                
                print(f"L{loop}-F{local_idx+1:2d} (G{global_frame:3d}): Persons={len(person_dets)} ({p_info}) | Weap={w_info} | Events={alert_types} | StateKeys={list(state_for_cam.keys())}")
                
                for w_det in weapon_dets:
                    if w_det.get("is_unattended"):
                        flip_events.append((loop, local_idx + 1, global_frame, w_det, alert_types))

    print("\n" + "=" * 80)
    print("UNATTENDED FLIP OCCURRENCES:")
    for f in flip_events[:15]:
        print(f"  Loop {f[0]}, Frame {f[1]} (Global {f[2]}): {f[3]['class']} trk={f[3].get('track_id')} | Events: {f[4]}")

if __name__ == "__main__":
    run_weapon_flip_investigation()
