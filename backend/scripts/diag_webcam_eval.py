import os
import sys
from pathlib import Path
import cv2

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

from app.services.ai_service import AIService

TESTS_DIR = repo_root / "webcam_tests"

def evaluate_clip(clip_path: Path, service: AIService, idx: int):
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"Could not open {clip_path}")
        return
        
    camera_id = f"CAM-TEST-WC-{idx}"
    frame_count = 0
    persons_seen = []
    weapons_seen = []
    fight_ever_detected = False
    max_fight_conf = 0.0
    alerts_seen = []
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        (
            annotated_frame,
            detections,
            threat_level,
            total_score,
            active_events,
            correlated_events,
            person_count,
            xai_explanation,
            fight_result,
            accident_result
        ) = service.process_frame(
            frame=frame,
            camera_id=camera_id,
            camera_name=f"Webcam Test {clip_path.name}",
            gps_coords=(0.0, 0.0)
        )
        
        persons_seen.append(person_count)
        
        # Check weapon detections in detections list
        h, w = frame.shape[:2]
        for d in detections:
            d_cls = d.get("class", "").lower()
            if d_cls in ["pistol", "knife", "gun"]:
                bbox = d.get("bbox", [0, 0, 0, 0])
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                area_pct = round(((bw * bh) / float(w * h)) * 100, 2)
                weapons_seen.append({
                    "frame": frame_count,
                    "class": d["class"],
                    "conf": round(d["confidence"], 4),
                    "area_pct": area_pct
                })
                
        if fight_result.get("fight_detected"):
            fight_ever_detected = True
        if fight_result.get("fight_confidence", 0.0) > max_fight_conf:
            max_fight_conf = fight_result.get("fight_confidence")
            
        for evt in active_events:
            etype = evt.get("event_type")
            if etype not in alerts_seen:
                alerts_seen.append(etype)
                
        frame_count += 1
        
    cap.release()
    service.cleanup_camera(camera_id)
    
    avg_persons = round(sum(persons_seen) / len(persons_seen), 1) if persons_seen else 0.0
    min_persons = min(persons_seen) if persons_seen else 0
    max_persons = max(persons_seen) if persons_seen else 0
    
    print("="*85)
    print(f"EVALUATION RESULT: {clip_path.name} ({frame_count} frames)")
    print("="*85)
    print(f"  - Persons:              Min={min_persons}, Max={max_persons}, Avg={avg_persons}")
    print(f"  - Fight Status:         Detected={fight_ever_detected} (Max Conf={max_fight_conf:.4f})")
    print(f"  - Active Alerts Fired:  {alerts_seen if alerts_seen else 'None (0 false alarms)'}")
    
    if weapons_seen:
        print(f"  - Weapon Detections ({len(weapons_seen)} occurrences):")
        for w in weapons_seen[:10]:
            print(f"      Frame {w['frame']:3d}: {w['class']} (conf={w['conf']:.4f}, box area={w['area_pct']}%)")
        if len(weapons_seen) > 10:
            print(f"      ... and {len(weapons_seen) - 10} more.")
    else:
        print("  - Weapon Detections:    0 (clean)")
    print()

def main():
    if not TESTS_DIR.exists():
        print(f"Directory {TESTS_DIR} does not exist.")
        return
        
    clips = sorted(list(TESTS_DIR.glob("*.mp4")) + list(TESTS_DIR.glob("*.avi")))
    if not clips:
        print(f"No video clips found in {TESTS_DIR}.")
        print("Record a clip first using: python backend/record_webcam_clip.py <name> <seconds>")
        return
        
    print(f"Found {len(clips)} clips in {TESTS_DIR}. Loading AIService on CUDA...")
    service = AIService()
    
    for i, clip_path in enumerate(clips):
        evaluate_clip(clip_path, service, i+1)

if __name__ == "__main__":
    main()
