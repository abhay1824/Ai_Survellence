import os
import sys
import glob
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

WEAPON_MODEL_PATH = repo_root / "OD-WeaponDetection/Weapons and similar handled objects/Sohas_weapon-Detection-YOLOv5/runs/detect/weapon6cls/weights/best.pt"
PERSON_MODEL_PATH = repo_root / "yolo11s.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading models on {device}...")
weapon_model = YOLO(str(WEAPON_MODEL_PATH)).to(device)
person_model = YOLO(str(PERSON_MODEL_PATH)).to(device)

PISTOL_IMG_DIR = repo_root / "Dataaset2" / "Pistol detection" / "Weapons"
KNIFE_IMG_DIR = repo_root / "Dataaset2" / "Knife_detection" / "Images"

NORMAL_VIDEO = repo_root / "sample_videos" / "normal.mp4"
NONFIGHT_DIR = repo_root / "RWF-2000" / "NonFight"

PISTOL_THRESHOLDS = [0.35, 0.45, 0.55, 0.58]
KNIFE_THRESHOLDS = [0.35, 0.50, 0.60]

def violates_size_ceiling(bbox, frame_shape):
    h, w = frame_shape[:2]
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    area_ratio = (bw * bh) / float(w * h)
    w_ratio = bw / float(w)
    h_ratio = bh / float(h)
    return (area_ratio > 0.15 or h_ratio > 0.40 or w_ratio > 0.40)

def evaluate_static_images():
    print("\n" + "="*80)
    print("1. EVALUATING TRUE WEAPON IMAGES (100 Pistols, 100 Knives)")
    print("="*80)
    
    pistol_files = sorted(list(PISTOL_IMG_DIR.glob("*.jpg")))[:100]
    knife_files = sorted(list(KNIFE_IMG_DIR.glob("*.jpg")))[:100]
    
    print(f"Found {len(pistol_files)} pistol images and {len(knife_files)} knife images.")
    
    # Pre-extract all detections so we can evaluate different thresholds instantly
    pistol_data = [] # list of list of (cls_name, conf, bbox, frame_shape, has_person)
    for p_path in pistol_files:
        img = cv2.imread(str(p_path))
        if img is None: continue
        w_res = weapon_model.predict(img, conf=0.20, verbose=False)
        p_res = person_model.predict(img, classes=[0], conf=0.35, verbose=False)
        has_person = bool(p_res and p_res[0].boxes is not None and len(p_res[0].boxes) > 0)
        
        dets = []
        if w_res and w_res[0].boxes is not None:
            for b in w_res[0].boxes:
                cname = weapon_model.names[int(b.cls[0])].lower()
                conf = float(b.conf[0])
                bbox = [int(v) for v in b.xyxy[0].cpu().numpy()]
                dets.append((cname, conf, bbox))
        pistol_data.append((dets, img.shape, has_person))
        
    knife_data = []
    for k_path in knife_files:
        img = cv2.imread(str(k_path))
        if img is None: continue
        w_res = weapon_model.predict(img, conf=0.20, verbose=False)
        p_res = person_model.predict(img, classes=[0], conf=0.35, verbose=False)
        has_person = bool(p_res and p_res[0].boxes is not None and len(p_res[0].boxes) > 0)
        
        dets = []
        if w_res and w_res[0].boxes is not None:
            for b in w_res[0].boxes:
                cname = weapon_model.names[int(b.cls[0])].lower()
                conf = float(b.conf[0])
                bbox = [int(v) for v in b.xyxy[0].cpu().numpy()]
                dets.append((cname, conf, bbox))
        knife_data.append((dets, img.shape, has_person))
        
    return pistol_data, knife_data

def evaluate_false_positive_videos():
    print("\n" + "="*80)
    print("2. EXTRACTING FALSE POSITIVE VIDEO DETECTIONS (normal.mp4 + 20 NonFight clips)")
    print("="*80)
    
    # We will extract per-frame (cname, conf, bbox, shape, has_person)
    video_frames = [] # list of lists per clip
    
    # 1. normal.mp4 (all 150 frames)
    cap = cv2.VideoCapture(str(NORMAL_VIDEO))
    normal_frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None: break
        w_res = weapon_model.predict(frame, conf=0.20, verbose=False)
        p_res = person_model.predict(frame, classes=[0], conf=0.35, verbose=False)
        has_person = bool(p_res and p_res[0].boxes is not None and len(p_res[0].boxes) > 0)
        dets = []
        if w_res and w_res[0].boxes is not None:
            for b in w_res[0].boxes:
                cname = weapon_model.names[int(b.cls[0])].lower()
                conf = float(b.conf[0])
                bbox = [int(v) for v in b.xyxy[0].cpu().numpy()]
                dets.append((cname, conf, bbox))
        normal_frames.append((dets, frame.shape, has_person))
    cap.release()
    video_frames.append(("normal.mp4", normal_frames))
    print(f"Extracted {len(normal_frames)} frames from normal.mp4")
    
    # 2. 20 NonFight clips (30 frames each)
    nf_clips = sorted(list(NONFIGHT_DIR.glob("*.avi")))[:20]
    for c_idx, clip_path in enumerate(nf_clips):
        cap = cv2.VideoCapture(str(clip_path))
        c_frames = []
        f_count = 0
        while f_count < 30:
            ret, frame = cap.read()
            if not ret or frame is None: break
            w_res = weapon_model.predict(frame, conf=0.20, verbose=False)
            p_res = person_model.predict(frame, classes=[0], conf=0.35, verbose=False)
            has_person = bool(p_res and p_res[0].boxes is not None and len(p_res[0].boxes) > 0)
            dets = []
            if w_res and w_res[0].boxes is not None:
                for b in w_res[0].boxes:
                    cname = weapon_model.names[int(b.cls[0])].lower()
                    conf = float(b.conf[0])
                    bbox = [int(v) for v in b.xyxy[0].cpu().numpy()]
                    dets.append((cname, conf, bbox))
            c_frames.append((dets, frame.shape, has_person))
            f_count += 1
        cap.release()
        video_frames.append((clip_path.name, c_frames))
    print(f"Extracted 20 NonFight clips (30 frames each = {len(video_frames)-1} clips)")
    
    return video_frames

def main():
    pistol_data, knife_data = evaluate_static_images()
    video_frames = evaluate_false_positive_videos()
    
    print("\n" + "="*85)
    print("TABLE 1: TRUE PISTOL RECALL (100 IMAGES)")
    print("="*85)
    print(f"{'Threshold':<12} | {'Recall (No Ceiling)':<22} | {'Recall (With Size Ceiling)':<25}")
    print("-" * 65)
    for p_thresh in PISTOL_THRESHOLDS:
        # Without ceiling
        hits_no_ceil = 0
        hits_ceil = 0
        for dets, shape, _ in pistol_data:
            has_hit_no_ceil = any(cname == "pistol" and conf >= p_thresh for cname, conf, bbox in dets)
            has_hit_ceil = any(cname == "pistol" and conf >= p_thresh and not violates_size_ceiling(bbox, shape) for cname, conf, bbox in dets)
            if has_hit_no_ceil: hits_no_ceil += 1
            if has_hit_ceil: hits_ceil += 1
            
        print(f"{p_thresh:<12} | {hits_no_ceil}/100 ({hits_no_ceil:.1f}%)          | {hits_ceil}/100 ({hits_ceil:.1f}%)")

    print("\n" + "="*85)
    print("TABLE 2: TRUE KNIFE RECALL (100 IMAGES)")
    print("="*85)
    print(f"{'Threshold':<12} | {'Recall (No Ceiling)':<22} | {'Recall (With Size Ceiling)':<25}")
    print("-" * 65)
    for k_thresh in KNIFE_THRESHOLDS:
        hits_no_ceil = 0
        hits_ceil = 0
        for dets, shape, _ in knife_data:
            has_hit_no_ceil = any(cname == "knife" and conf >= k_thresh for cname, conf, bbox in dets)
            has_hit_ceil = any(cname == "knife" and conf >= k_thresh and not violates_size_ceiling(bbox, shape) for cname, conf, bbox in dets)
            if has_hit_no_ceil: hits_no_ceil += 1
            if has_hit_ceil: hits_ceil += 1
            
        print(f"{k_thresh:<12} | {hits_no_ceil}/100 ({hits_no_ceil:.1f}%)          | {hits_ceil}/100 ({hits_ceil:.1f}%)")

    print("\n" + "="*85)
    print("TABLE 3: FALSE DETECTIONS ON normal.mp4 (150 frames) & 20 NonFight CLIPS (600 frames)")
    print("="*85)
    print(f"{'Config (Pistol / Knife / Ceiling)':<35} | {'normal.mp4 Raw / Confirmed':<28} | {'20 NonFight Raw / Confirmed':<28}")
    print("-" * 95)
    
    configs_to_test = [
        ("p>=0.35, k>=0.35, No Ceiling", 0.35, 0.35, False),
        ("p>=0.35, k>=0.35, With Ceiling", 0.35, 0.35, True),
        ("p>=0.45, k>=0.50, No Ceiling", 0.45, 0.50, False),
        ("p>=0.45, k>=0.50, With Ceiling", 0.45, 0.50, True),
        ("p>=0.55, k>=0.50, With Ceiling", 0.55, 0.50, True),
        ("p>=0.58, k>=0.60, No Ceiling", 0.58, 0.60, False),
        ("p>=0.58, k>=0.60, With Ceiling", 0.58, 0.60, True),
    ]
    
    for label, p_th, k_th, use_ceil in configs_to_test:
        # Evaluate normal.mp4
        normal_frames = video_frames[0][1]
        raw_normal_hits = 0
        normal_confirmed_alerts = 0
        consec_normal = {"pistol": 0, "knife": 0}
        
        for dets, shape, _ in normal_frames:
            detected_sec = set()
            for cname, conf, bbox in dets:
                if cname == "pistol" and conf >= p_th:
                    if not use_ceil or not violates_size_ceiling(bbox, shape):
                        detected_sec.add("pistol")
                elif cname == "knife" and conf >= k_th:
                    if not use_ceil or not violates_size_ceiling(bbox, shape):
                        detected_sec.add("knife")
            if detected_sec:
                raw_normal_hits += 1
            for sec_cls in ["pistol", "knife"]:
                if sec_cls in detected_sec:
                    consec_normal[sec_cls] += 1
                    if consec_normal[sec_cls] == 3:
                        normal_confirmed_alerts += 1
                else:
                    consec_normal[sec_cls] = 0
                    
        # Evaluate 20 NonFight clips
        total_raw_nf = 0
        total_confirmed_nf = 0
        for clip_name, c_frames in video_frames[1:]:
            consec_nf = {"pistol": 0, "knife": 0}
            for dets, shape, _ in c_frames:
                detected_sec = set()
                for cname, conf, bbox in dets:
                    if cname == "pistol" and conf >= p_th:
                        if not use_ceil or not violates_size_ceiling(bbox, shape):
                            detected_sec.add("pistol")
                    elif cname == "knife" and conf >= k_th:
                        if not use_ceil or not violates_size_ceiling(bbox, shape):
                            detected_sec.add("knife")
                if detected_sec:
                    total_raw_nf += 1
                for sec_cls in ["pistol", "knife"]:
                    if sec_cls in detected_sec:
                        consec_nf[sec_cls] += 1
                        if consec_nf[sec_cls] == 3:
                            total_confirmed_nf += 1
                    else:
                        consec_nf[sec_cls] = 0
                        
        print(f"{label:<35} | {raw_normal_hits:3d} frames / {normal_confirmed_alerts:2d} alerts     | {total_raw_nf:3d} frames / {total_confirmed_nf:2d} alerts")

    print("\n" + "="*85)
    print("TABLE 4: 'NO PERSON IN SCENE' ANALYSIS (True Weapon Datasets)")
    print("="*85)
    # Check how many images have a person detected
    p_with_person = sum(1 for _, _, hp in pistol_data if hp)
    p_no_person = len(pistol_data) - p_with_person
    k_with_person = sum(1 for _, _, hp in knife_data if hp)
    k_no_person = len(knife_data) - k_with_person
    
    print(f"Pistol Dataset (100 images): {p_with_person} images with person, {p_no_person} images WITHOUT person")
    print(f"Knife Dataset (100 images):  {k_with_person} images with person, {k_no_person} images WITHOUT person")
    print("\nImpact Comparison on True Weapons:")
    print("  Policy A (Strict Rejection if No Person):")
    print(f"    - Automatically drops {p_no_person}/100 ({p_no_person}%) true pistol images!")
    print(f"    - Automatically drops {k_no_person}/100 ({k_no_person}%) true knife images!")
    print("  Policy B (Downgrade to MEDIUM / No Email if No Person):")
    print(f"    - Preserves 100% of detected weapons in database / live HUD as 'UNATTENDED WEAPON'.")
    print(f"    - Caps threat score at MEDIUM (<60), suppresses siren and email dispatch.")
    print("="*85)

if __name__ == "__main__":
    main()
