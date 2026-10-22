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

def inspect():
    video_path = str(backend_dir.parent / "sample_videos" / "weapon.mp4")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    ai = AIService()
    camera_id = "CAM-004"

    # Let's test two scenarios:
    # 1. Direct sequential frames
    # 2. CameraWorker simulation where the same frame is read multiple times or at CameraWorker cadence
    print("Direct sequential feed:")
    for idx, f in enumerate(frames):
        res = ai.process_frame(camera_id=camera_id, frame=f.copy(), camera_name="CAM-004", gps_coords=(0,0))
        dets = res[1]
        evts = res[4]
        p_dets = [d for d in dets if d['class'] == 'Person']
        w_dets = [d for d in dets if d['class'] in ('Knife', 'Pistol')]
        w_state = ai.unattended_weapon_state.get(camera_id, {})
        if idx in range(12, 20):
            print(f"Frame {idx+1:2d}: P={len(p_dets)} {[(p['bbox'], p['confidence']) for p in p_dets]} | W={[(w['bbox'], w['confidence'], w.get('track_id'), w.get('is_unattended')) for w in w_dets]} | State={w_state}")

if __name__ == "__main__":
    inspect()
