import os
import cv2
import time
from pathlib import Path
import sys

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.camera_manager import camera_manager
from app.services.ai_service import AIService

def simulate():
    # Start CAM-004 worker with weapon.mp4
    video_path = str(backend_dir.parent / "sample_videos" / "weapon.mp4")
    camera_manager.start_camera(camera_id="CAM-004", camera_name="Weapon Cam", source_url=video_path)
    
    # Wait 2 seconds so worker is running and looping
    time.sleep(2.0)
    
    ai_service = AIService()
    camera_id = "CAM-004"
    
    print("Simulating stream.py loop for 100 iterations on CAM-004:")
    for frame_idx in range(1, 101):
        frame = camera_manager.get_frame(camera_id)
        if frame is None:
            time.sleep(0.02)
            continue
            
        (
            annotated_frame,
            detections,
            highest_threat,
            total_score,
            active_events,
            correlated,
            person_count,
            xai_explanation,
            fight_result,
            accident_result
        ) = ai_service.process_frame(camera_id=camera_id, frame=frame.copy(), camera_name="CAM-004", gps_coords=(0,0))
        
        weapon_dets = [d for d in detections if d.get("class") in ("Knife", "Pistol")]
        person_dets = [d for d in detections if d.get("class") == "Person"]
        event_types = [e.get("event_type") for e in active_events]
        
        # Check if unattended weapon alert fired
        unattended_alerts = [e for e in event_types if "UNATTENDED" in e]
        if unattended_alerts or (frame_idx in range(14, 20)):
            w_info = [(w['class'], w.get('track_id'), w.get('is_unattended'), w.get('confidence'), w.get('bbox')) for w in weapon_dets]
            p_info = [(p.get('track_id'), p.get('confidence'), p.get('bbox')) for p in person_dets]
            state = dict(ai_service.unattended_weapon_state.get(camera_id, {}))
            print(f"Frame {frame_idx:3d}: P={p_info} | W={w_info} | Events={event_types} | State={state}")
            
        time.sleep(0.02)
        
    camera_manager.stop_all()

if __name__ == "__main__":
    simulate()
