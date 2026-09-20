import os
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

backend_dir = Path(__file__).resolve().parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

from app.core.config import settings

CROPS_DIR = repo_root / "debug_crops" / "normal"
CROPS_DIR.mkdir(parents=True, exist_ok=True)

NONFIGHT_CROPS_DIR = repo_root / "debug_crops" / "nonfight"
NONFIGHT_CROPS_DIR.mkdir(parents=True, exist_ok=True)

WEAPON_MODEL_PATH = repo_root / "OD-WeaponDetection/Weapons and similar handled objects/Sohas_weapon-Detection-YOLOv5/runs/detect/weapon6cls/weights/best.pt"
PERSON_MODEL_PATH = repo_root / "yolo11s.pt"

device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 or True else "cpu"
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading models on {device}...")
weapon_model = YOLO(str(WEAPON_MODEL_PATH)).to(device)
person_model = YOLO(str(PERSON_MODEL_PATH)).to(device)

def run_clip(video_path: Path, max_frames=None, save_crops_dir=None, clip_label=""):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return []

    detections = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        # Run weapon model at low threshold (0.25) to capture all potential detections
        w_results = weapon_model.predict(frame, conf=0.25, verbose=False)
        p_results = person_model.predict(frame, classes=[0], conf=0.35, verbose=False)

        person_boxes = []
        if p_results and p_results[0].boxes is not None:
            for pbox in p_results[0].boxes:
                person_boxes.append(pbox.xyxy[0].cpu().numpy().tolist())

        if w_results and w_results[0].boxes is not None:
            for box in w_results[0].boxes:
                cls_idx = int(box.cls[0])
                cls_name = weapon_model.names[cls_idx].lower()
                conf = float(box.conf[0])
                if cls_name in ["pistol", "knife"]:
                    xyxy = [int(v) for v in box.xyxy[0].cpu().numpy()]
                    
                    # Check overlap / proximity with any person
                    # Proximity: min distance between weapon center and person box
                    wx_center = (xyxy[0] + xyxy[2]) / 2.0
                    wy_center = (xyxy[1] + xyxy[3]) / 2.0
                    
                    overlap = False
                    for pbox in person_boxes:
                        # Overlap: intersection of weapon box and person box
                        ix1 = max(xyxy[0], pbox[0])
                        iy1 = max(xyxy[1], pbox[1])
                        ix2 = min(xyxy[2], pbox[2])
                        iy2 = min(xyxy[3], pbox[3])
                        if ix1 < ix2 and iy1 < iy2:
                            overlap = True
                            break

                    det_info = {
                        "clip": clip_label,
                        "frame": frame_idx,
                        "class": cls_name,
                        "conf": round(conf, 4),
                        "bbox": xyxy,
                        "person_count": len(person_boxes),
                        "overlaps_person": overlap
                    }
                    detections.append(det_info)

                    # Save crop
                    if save_crops_dir:
                        h, w = frame.shape[:2]
                        pad = 20
                        cx1 = max(0, xyxy[0] - pad)
                        cy1 = max(0, xyxy[1] - pad)
                        cx2 = min(w, xyxy[2] + pad)
                        cy2 = min(h, xyxy[3] + pad)
                        crop = frame[cy1:cy2, cx1:cx2].copy()
                        # Draw bounding box inside crop
                        cv2.rectangle(
                            crop,
                            (xyxy[0] - cx1, xyxy[1] - cy1),
                            (xyxy[2] - cx1, xyxy[3] - cy1),
                            (0, 0, 255),
                            2
                        )
                        label_txt = f"{cls_name} {conf:.2f}"
                        cv2.putText(
                            crop,
                            label_txt,
                            (xyxy[0] - cx1, max(15, xyxy[1] - cy1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1
                        )
                        crop_filename = f"{clip_label}_f{frame_idx}_{cls_name}_{int(conf*100)}.jpg"
                        cv2.imwrite(str(save_crops_dir / crop_filename), crop)

        frame_idx += 1

    cap.release()
    return detections

def main():
    print("=================================================================")
    print("TEST 1: sample_videos/normal.mp4 (every frame)")
    print("=================================================================")
    normal_video = repo_root / "sample_videos" / "normal.mp4"
    normal_dets = run_clip(normal_video, save_crops_dir=CROPS_DIR, clip_label="normal")
    print(f"Total pistol/knife detections in normal.mp4: {len(normal_dets)}")
    for d in normal_dets:
        print(f"  Frame {d['frame']:3d}: class={d['class']}, conf={d['conf']:.4f}, bbox={d['bbox']}, persons={d['person_count']}, overlaps_person={d['overlaps_person']}")

    print("\n=================================================================")
    print("TEST 2: 20 clips from RWF-2000/NonFight (first 30 frames each)")
    print("=================================================================")
    nonfight_dir = repo_root / "RWF-2000" / "NonFight"
    all_clips = sorted(list(nonfight_dir.glob("*.avi")))[:20]
    total_nonfight_dets = []
    for i, clip in enumerate(all_clips):
        clip_name = clip.stem
        dets = run_clip(clip, max_frames=30, save_crops_dir=NONFIGHT_CROPS_DIR, clip_label=f"nonfight_{i}_{clip_name[:8]}")
        if dets:
            print(f"Clip {i+1}/20 [{clip.name}]: {len(dets)} detections")
            for d in dets:
                print(f"  Frame {d['frame']:3d}: class={d['class']}, conf={d['conf']:.4f}, bbox={d['bbox']}, persons={d['person_count']}, overlaps_person={d['overlaps_person']}")
            total_nonfight_dets.extend(dets)
        else:
            print(f"Clip {i+1}/20 [{clip.name}]: 0 weapon detections")

    print(f"\nTotal pistol/knife detections across 20 NonFight clips (600 frames): {len(total_nonfight_dets)}")

    print("\n=================================================================")
    print("TEST 3: sample_videos/weapon.mp4 (every frame)")
    print("=================================================================")
    weapon_video = repo_root / "sample_videos" / "weapon.mp4"
    weapon_dets = run_clip(weapon_video, clip_label="weapon")
    print(f"Total pistol/knife detections in weapon.mp4: {len(weapon_dets)}")
    for d in weapon_dets:
        print(f"  Frame {d['frame']:3d}: class={d['class']}, conf={d['conf']:.4f}, bbox={d['bbox']}, persons={d['person_count']}, overlaps_person={d['overlaps_person']}")

if __name__ == "__main__":
    main()
