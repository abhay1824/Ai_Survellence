import os
import cv2
import json
import numpy as np
from pathlib import Path

# Add backend directory to sys.path
import sys
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.ai_service import AIService
from app.services.event_service import _should_trigger_email
from app.core.config import settings

def run_accident_investigation():
    print("=" * 80)
    print("INVESTIGATION 1: CAM-003 FALSE PISTOL ON sample_videos/accident.mp4")
    print("=" * 80)

    video_path = str(backend_dir.parent / "sample_videos" / "accident.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    ai_service = AIService()
    camera_id = "CAM_ACCIDENT_INVESTIGATE"

    out_dir = backend_dir.parent / "debug_crops"
    out_dir.mkdir(exist_ok=True)

    frame_idx = 0
    saved_frames = 0
    saved_frame_paths = []
    
    pistol_detections_log = []
    high_events = []
    email_triggers = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        h, w = frame.shape[:2]
        frame_area = float(w * h)

        # Process frame
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
            camera_name="CAM-003",
            gps_coords=(12.9716, 77.5946)
        )

        # Extract weapon detections
        weapon_dets = [d for d in detections if d.get("class", "").lower() in ("pistol", "gun", "knife", "scissors")]
        
        # Log any weapon detections
        if weapon_dets or active_events:
            for d in weapon_dets:
                bx1, by1, bx2, by2 = d["bbox"]
                box_area = (bx2 - bx1) * (by2 - by1)
                area_pct = (box_area / frame_area) * 100.0
                
                log_entry = {
                    "frame": frame_idx,
                    "persons": person_count,
                    "class": d["class"],
                    "conf": round(d["confidence"], 4),
                    "area_pct": round(area_pct, 3),
                    "track_id": d.get("track_id"),
                    "is_unattended": d.get("is_unattended"),
                    "threat_level": d.get("threat_level"),
                    "highest_threat": highest_threat,
                    "score": total_score,
                    "active_events": [e.get("event_type") for e in active_events]
                }
                pistol_detections_log.append(log_entry)
                print(f"Frame {frame_idx:3d} | Persons: {person_count} | Weap: {d['class']} conf={d['confidence']:.3f} area={area_pct:.2f}% trk={d.get('track_id')} | Unatt: {d.get('is_unattended')} | Threat: {highest_threat} ({total_score}) | Events: {[e.get('event_type') for e in active_events]}")

                # Save 3 annotated frames
                if saved_frames < 3 and "pistol" in d["class"].lower():
                    crop_name = f"accident_pistol_f{frame_idx}.jpg"
                    crop_path = out_dir / crop_name
                    cv2.imwrite(str(crop_path), annotated_frame)
                    saved_frame_paths.append((str(crop_path), d, frame_idx))
                    saved_frames += 1

        # Check active_events for unattended weapon and emails
        for evt in active_events:
            e_type = evt.get("event_type", "")
            t_level = evt.get("threat_level", "")
            if "UNATTENDED" in e_type:
                if t_level == "HIGH":
                    high_events.append((frame_idx, e_type, t_level))
            
            # Check if email would trigger
            # Note: accident_detected might trigger email for ACCIDENT, but we check if UNATTENDED weapon would trigger email
            triggers_email = _should_trigger_email(camera_id, e_type, t_level, accident_detected=False)
            if "UNATTENDED" in e_type and triggers_email:
                email_triggers.append((frame_idx, e_type, t_level))

    cap.release()

    print("\n" + "=" * 80)
    print("CAM-003 SUMMARY RESULTS:")
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Total frames with weapon detections: {len(pistol_detections_log)}")
    print(f"Did unattended pistol event reach HIGH?: {len(high_events) > 0} ({len(high_events)} frames)")
    if high_events:
        print(f"  First frame reaching HIGH: {high_events[0]}")
    print(f"Did any UNATTENDED weapon email trigger?: {len(email_triggers) > 0} ({len(email_triggers)} calls)")
    print(f"Saved {len(saved_frame_paths)} annotated frames:")
    for p, d, f_num in saved_frame_paths:
        print(f"  Frame {f_num}: {p} (bbox={d['bbox']}, conf={d['confidence']:.3f})")

    # Part B: Run weapon model on no-person footage (acc/ test images, accident_frames)
    print("\n" + "=" * 80)
    print("NO-PERSON FOOTAGE EVALUATION (acc/ and accident_frames)")
    print("=" * 80)

    acc_dirs = [
        backend_dir.parent / "acc" / "test" / "images",
        backend_dir.parent / "acc" / "valid" / "images",
        backend_dir.parent / "accident_frames"
    ]

    image_paths = []
    for ad in acc_dirs:
        if ad.exists():
            for f in ad.glob("*.jpg"):
                image_paths.append(f)
            for f in ad.glob("*.png"):
                image_paths.append(f)

    print(f"Found {len(image_paths)} frames across accident/road datasets.")
    
    # Test up to 100 frames to avoid overly long run
    sample_images = image_paths[:100]
    false_weapon_detections = []

    for img_p in sample_images:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        iarea = float(iw * ih)

        # Run weapon model directly
        results = ai_service.model(img, conf=0.35, verbose=False)
        for r in results:
            for b in r.boxes:
                cls_id = int(b.cls[0])
                cls_name = ai_service.model.names.get(cls_id, str(cls_id))
                conf = float(b.conf[0])
                xyxy = [float(c) for c in b.xyxy[0]]
                bw = xyxy[2] - xyxy[0]
                bh = xyxy[3] - xyxy[1]
                area_pct = ((bw * bh) / iarea) * 100.0

                if cls_name.lower() in ("pistol", "gun", "knife", "scissors"):
                    false_weapon_detections.append({
                        "file": img_p.name,
                        "class": cls_name,
                        "conf": round(conf, 4),
                        "area_pct": round(area_pct, 2),
                        "bbox": [round(x, 1) for x in xyxy]
                    })

    print(f"Tested {len(sample_images)} no-person road frames.")
    print(f"False weapon detections found: {len(false_weapon_detections)}")
    for fwd in false_weapon_detections[:15]:
        print(f"  {fwd['file']}: {fwd['class']} conf={fwd['conf']} area={fwd['area_pct']}% bbox={fwd['bbox']}")

if __name__ == "__main__":
    run_accident_investigation()
