import os
import sys
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

# Ensure backend dir is on path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.services.ai_service import AIService

repo_root = backend_dir.parent
FIGHT_DIR = repo_root / "RWF-2000" / "Fight"
NONFIGHT_DIR = repo_root / "RWF-2000" / "NonFight"

fight_clips = sorted(list(FIGHT_DIR.glob("*.avi")))[:20]
nonfight_clips = sorted(list(NONFIGHT_DIR.glob("*.avi")))[:20]

print("================================================================================")
print(f"40-CLIP FIGHT REGRESSION (Config 2: min_persons={settings.FIGHT_MIN_PERSONS_TO_RUN}, cooldown=8)")
print("================================================================================")

service = AIService()

def eval_clip_list(clips, is_fight: bool):
    detected_count = 0
    first_frames = []
    
    for c_idx, clip_path in enumerate(clips):
        cam_id = f"CAM-EVAL-{clip_path.stem}"
        service.cleanup_camera(cam_id)
        
        cap = cv2.VideoCapture(str(clip_path))
        frame_idx = 0
        clip_ever_detected = False
        first_confirm = None
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            res = service.process_frame(
                frame=frame,
                camera_id=cam_id,
                camera_name="Eval Cam",
                gps_coords=(0.0, 0.0)
            )
            fight_res = res[8]
            if fight_res.get("fight_detected", False):
                clip_ever_detected = True
                if first_confirm is None:
                    first_confirm = frame_idx
                    
            frame_idx += 1
            
        cap.release()
        service.cleanup_camera(cam_id)
        
        status_str = f"CONFIRMED @ frame {first_confirm}" if clip_ever_detected else "CLEAN (No Fight)"
        clip_label = "FIGHT" if is_fight else "NON-FIGHT"
        print(f"  [{clip_label} {c_idx+1:2d}/20] {clip_path.name:<20} ({frame_idx:3d} frames) -> {status_str}")
        
        if clip_ever_detected:
            detected_count += 1
            if first_confirm is not None:
                first_frames.append(first_confirm)
                
    return detected_count, first_frames

print("\n--- Evaluating 20 Fight Clips ---")
fight_detected, fight_frames = eval_clip_list(fight_clips, is_fight=True)

print("\n--- Evaluating 20 NonFight Clips ---")
nonfight_detected, _ = eval_clip_list(nonfight_clips, is_fight=False)

f_total = len(fight_clips)
nf_total = len(nonfight_clips)
f_recall = (fight_detected / f_total) * 100
nf_false = (nonfight_detected / nf_total) * 100
avg_frames = (sum(fight_frames) / len(fight_frames)) if fight_frames else 0.0

print("\n" + "="*80)
print("REGRESSION SUMMARY (Config 2)")
print("="*80)
print(f"Fight Recall:              {fight_detected}/{f_total} ({f_recall:.1f}%)")
print(f"NonFight False Alarms:     {nonfight_detected}/{nf_total} ({nf_false:.1f}%)")
print(f"Average Frames to Confirm: {avg_frames:.1f} frames (~{avg_frames/15.0:.2f}s @ 15fps)")
print("="*80)
