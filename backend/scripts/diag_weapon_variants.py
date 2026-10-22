import os
import sys
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

P_THRESH = 0.45
K_THRESH = 0.50

def check_variant(variant_num: int, wbox: list[int], frame_shape: tuple, person_boxes: list[list[float]]) -> bool:
    """
    Returns True if weapon box passes the ceiling (i.e. NOT rejected), False if rejected.
    """
    h, w = frame_shape[:2]
    bw = wbox[2] - wbox[0]
    bh = wbox[3] - wbox[1]
    w_area = bw * bh
    f_area = float(w * h)
    area_ratio = w_area / f_area
    w_ratio = bw / float(w)
    h_ratio = bh / float(h)
    
    if variant_num == 1:
        # Variant 1: area > 0.15 or h > 0.40 or w > 0.40
        if area_ratio > 0.15 or h_ratio > 0.40 or w_ratio > 0.40:
            return False
        return True
        
    elif variant_num == 2:
        # Variant 2: area > 0.25 only
        if area_ratio > 0.25:
            return False
        return True
        
    elif variant_num == 3:
        # Variant 3: person-relative: reject if weapon box area > 50% of overlapping person box;
        # if no person overlaps, use variant 2's area limit (area > 0.25)
        overlapping_person_areas = []
        for pbox in person_boxes:
            ix1 = max(wbox[0], pbox[0])
            iy1 = max(wbox[1], pbox[1])
            ix2 = min(wbox[2], pbox[2])
            iy2 = min(wbox[3], pbox[3])
            if ix1 < ix2 and iy1 < iy2:
                # overlap exists
                pw = pbox[2] - pbox[0]
                ph = pbox[3] - pbox[1]
                overlapping_person_areas.append(pw * ph)
                
        if overlapping_person_areas:
            max_p_area = max(overlapping_person_areas)
            if w_area > 0.50 * max_p_area:
                return False
            return True
        else:
            # no person overlaps, use variant 2 limit
            if area_ratio > 0.25:
                return False
            return True
    return True

def preextract_images():
    print("Extracting 100 pistol images and 100 knife images...")
    pistol_files = sorted(list(PISTOL_IMG_DIR.glob("*.jpg")))[:100]
    knife_files = sorted(list(KNIFE_IMG_DIR.glob("*.jpg")))[:100]
    
    pistols = []
    for p in pistol_files:
        img = cv2.imread(str(p))
        if img is None: continue
        w_res = weapon_model.predict(img, conf=0.20, verbose=False)
        p_res = person_model.predict(img, classes=[0], conf=0.35, verbose=False)
        p_boxes = [b.xyxy[0].cpu().numpy().tolist() for b in p_res[0].boxes] if (p_res and p_res[0].boxes is not None) else []
        w_dets = []
        if w_res and w_res[0].boxes is not None:
            for b in w_res[0].boxes:
                cname = weapon_model.names[int(b.cls[0])].lower()
                conf = float(b.conf[0])
                bbox = [int(v) for v in b.xyxy[0].cpu().numpy()]
                w_dets.append((cname, conf, bbox))
        pistols.append((w_dets, img.shape, p_boxes))
        
    knives = []
    for k in knife_files:
        img = cv2.imread(str(k))
        if img is None: continue
        w_res = weapon_model.predict(img, conf=0.20, verbose=False)
        p_res = person_model.predict(img, classes=[0], conf=0.35, verbose=False)
        p_boxes = [b.xyxy[0].cpu().numpy().tolist() for b in p_res[0].boxes] if (p_res and p_res[0].boxes is not None) else []
        w_dets = []
        if w_res and w_res[0].boxes is not None:
            for b in w_res[0].boxes:
                cname = weapon_model.names[int(b.cls[0])].lower()
                conf = float(b.conf[0])
                bbox = [int(v) for v in b.xyxy[0].cpu().numpy()]
                w_dets.append((cname, conf, bbox))
        knives.append((w_dets, img.shape, p_boxes))
        
    return pistols, knives

def preextract_videos():
    print("Extracting normal.mp4 and 20 NonFight clips...")
    video_data = []
    
    # normal.mp4
    cap = cv2.VideoCapture(str(NORMAL_VIDEO))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None: break
        w_res = weapon_model.predict(frame, conf=0.20, verbose=False)
        p_res = person_model.predict(frame, classes=[0], conf=0.35, verbose=False)
        p_boxes = [b.xyxy[0].cpu().numpy().tolist() for b in p_res[0].boxes] if (p_res and p_res[0].boxes is not None) else []
        w_dets = []
        if w_res and w_res[0].boxes is not None:
            for b in w_res[0].boxes:
                cname = weapon_model.names[int(b.cls[0])].lower()
                conf = float(b.conf[0])
                bbox = [int(v) for v in b.xyxy[0].cpu().numpy()]
                w_dets.append((cname, conf, bbox))
        frames.append((w_dets, frame.shape, p_boxes))
    cap.release()
    video_data.append(("normal.mp4", frames))
    
    # 20 NonFight clips
    nf_clips = sorted(list(NONFIGHT_DIR.glob("*.avi")))[:20]
    for p in nf_clips:
        cap = cv2.VideoCapture(str(p))
        frames = []
        count = 0
        while count < 30:
            ret, frame = cap.read()
            if not ret or frame is None: break
            w_res = weapon_model.predict(frame, conf=0.20, verbose=False)
            p_res = person_model.predict(frame, classes=[0], conf=0.35, verbose=False)
            p_boxes = [b.xyxy[0].cpu().numpy().tolist() for b in p_res[0].boxes] if (p_res and p_res[0].boxes is not None) else []
            w_dets = []
            if w_res and w_res[0].boxes is not None:
                for b in w_res[0].boxes:
                    cname = weapon_model.names[int(b.cls[0])].lower()
                    conf = float(b.conf[0])
                    bbox = [int(v) for v in b.xyxy[0].cpu().numpy()]
                    w_dets.append((cname, conf, bbox))
            frames.append((w_dets, frame.shape, p_boxes))
            count += 1
        cap.release()
        video_data.append((p.name, frames))
        
    return video_data

def evaluate_variant(variant_num, pistols, knives, video_data):
    # 1. Pistol recall
    p_hits = 0
    for w_dets, shape, p_boxes in pistols:
        valid_hit = False
        for cname, conf, bbox in w_dets:
            if cname == "pistol" and conf >= P_THRESH:
                if check_variant(variant_num, bbox, shape, p_boxes):
                    valid_hit = True
                    break
        if valid_hit: p_hits += 1
        
    # 2. Knife recall
    k_hits = 0
    for w_dets, shape, p_boxes in knives:
        valid_hit = False
        for cname, conf, bbox in w_dets:
            if cname == "knife" and conf >= K_THRESH:
                if check_variant(variant_num, bbox, shape, p_boxes):
                    valid_hit = True
                    break
        if valid_hit: k_hits += 1
        
    # 3. normal.mp4
    normal_frames = video_data[0][1]
    norm_raw = 0
    norm_alerts = 0
    consec = {"pistol": 0, "knife": 0}
    for w_dets, shape, p_boxes in normal_frames:
        frame_sec = set()
        for cname, conf, bbox in w_dets:
            if cname == "pistol" and conf >= P_THRESH and check_variant(variant_num, bbox, shape, p_boxes):
                frame_sec.add("pistol")
            elif cname == "knife" and conf >= K_THRESH and check_variant(variant_num, bbox, shape, p_boxes):
                frame_sec.add("knife")
        if frame_sec: norm_raw += 1
        for sec in ["pistol", "knife"]:
            if sec in frame_sec:
                consec[sec] += 1
                if consec[sec] == 3: norm_alerts += 1
            else:
                consec[sec] = 0
                
    # 4. 20 NonFight clips
    nf_raw = 0
    nf_alerts = 0
    for _, c_frames in video_data[1:]:
        consec_nf = {"pistol": 0, "knife": 0}
        for w_dets, shape, p_boxes in c_frames:
            frame_sec = set()
            for cname, conf, bbox in w_dets:
                if cname == "pistol" and conf >= P_THRESH and check_variant(variant_num, bbox, shape, p_boxes):
                    frame_sec.add("pistol")
                elif cname == "knife" and conf >= K_THRESH and check_variant(variant_num, bbox, shape, p_boxes):
                    frame_sec.add("knife")
            if frame_sec: nf_raw += 1
            for sec in ["pistol", "knife"]:
                if sec in frame_sec:
                    consec_nf[sec] += 1
                    if consec_nf[sec] == 3: nf_alerts += 1
                else:
                    consec_nf[sec] = 0
                    
    return {
        "p_recall": p_hits,
        "k_recall": k_hits,
        "norm_raw": norm_raw,
        "norm_alerts": norm_alerts,
        "nf_raw": nf_raw,
        "nf_alerts": nf_alerts
    }

def main():
    pistols, knives = preextract_images()
    video_data = preextract_videos()
    
    variant_names = [
        (1, "Variant 1: area > 0.15 or h > 0.40 or w > 0.40"),
        (2, "Variant 2: area > 0.25 only"),
        (3, "Variant 3: person-relative (w_area > 50% p_area, else area > 0.25)")
    ]
    
    print("\n" + "="*105)
    print("WEAPON SIZE CEILING VARIANTS EVALUATION (pistol >= 0.45, knife >= 0.50)")
    print("="*105)
    print(f"{'Ceiling Variant':<40} | {'100 Pistols':<12} | {'100 Knives':<11} | {'normal.mp4 (Raw/3-Fr)':<20} | {'20 NonFight (Raw/3-Fr)':<20}")
    print("-" * 115)
    
    for v_num, v_label in variant_names:
        res = evaluate_variant(v_num, pistols, knives, video_data)
        p_str = f"{res['p_recall']}/100 ({res['p_recall']}%)"
        k_str = f"{res['k_recall']}/100 ({res['k_recall']}%)"
        norm_str = f"{res['norm_raw']} frames / {res['norm_alerts']} alerts"
        nf_str = f"{res['nf_raw']} frames / {res['nf_alerts']} alerts"
        print(f"{v_label:<40} | {p_str:<12} | {k_str:<11} | {norm_str:<20} | {nf_str:<20}")
        
    print("="*105)

if __name__ == "__main__":
    main()
