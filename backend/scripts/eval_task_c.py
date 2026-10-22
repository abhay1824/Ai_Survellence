import os
import sys
import math
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

WEAPON_MODEL_PATH = repo_root / "OD-WeaponDetection" / "Weapons and similar handled objects" / "Sohas_weapon-Detection-YOLOv5" / "runs" / "detect" / "weapon6cls" / "weights" / "best.pt"
PERSON_MODEL_PATH = repo_root / "yolo11s.pt"

weapon_model = YOLO(str(WEAPON_MODEL_PATH)).to(device)
person_model = YOLO(str(PERSON_MODEL_PATH)).to(device)

# 1. 5 Background frames from distinct NonFight clips
BG_CLIP_PATHS = [
    repo_root / "RWF-2000" / "NonFight" / "-1l5631l3fg_0.avi",  # Indoor lobby
    repo_root / "RWF-2000" / "NonFight" / "0_DzLlklZa0_0.avi",  # Street / road
    repo_root / "RWF-2000" / "NonFight" / "0AvTZRYx_0.avi",     # Indoor corridor
    repo_root / "RWF-2000" / "NonFight" / "0H2s9UJcNJ0_0.avi",  # Outdoor courtyard
    repo_root / "RWF-2000" / "NonFight" / "0vasozvJYvk_0.avi"   # Outdoor walkway
]

FRAME_W = 640
FRAME_H = 480
FRAME_AREA = FRAME_W * FRAME_H  # 307,200 px

def load_background_frames():
    bg_frames = []
    for p in BG_CLIP_PATHS:
        cap = cv2.VideoCapture(str(p))
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            bg_frames.append(cv2.resize(frame, (FRAME_W, FRAME_H)))
        else:
            raise RuntimeError(f"Failed to load background frame from {p}")
    return bg_frames

# 2. Dataset loader: 100 pistols and 100 knives
def load_dataset_crops():
    # Pistols
    p_dir = repo_root / "Dataaset2" / "Pistol detection"
    p_labels = sorted(list((p_dir / "labels").glob("*.txt")))
    pistol_crops = []
    for lbl in p_labels:
        img_path = p_dir / "Weapons" / f"{lbl.stem}.jpg"
        if not img_path.exists():
            continue
        try:
            lines = lbl.read_text().strip().splitlines()
            if not lines:
                continue
            # Parse bounding boxes
            best_crop = None
            max_a = 0
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = max(0, int((xc - w / 2) * iw))
                    y1 = max(0, int((yc - h / 2) * ih))
                    x2 = min(iw, int((xc + w / 2) * iw))
                    y2 = min(ih, int((yc + h / 2) * ih))
                    if x2 > x1 + 5 and y2 > y1 + 5:
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_a:
                            max_a = area
                            best_crop = img[y1:y2, x1:x2]
            if best_crop is not None and best_crop.size > 0:
                pistol_crops.append(best_crop)
                if len(pistol_crops) >= 100:
                    break
        except Exception:
            continue

    # Knives
    k_dir = repo_root / "Dataaset2" / "Knife_detection"
    k_labels = sorted(list((k_dir / "labels").glob("*.txt")))
    knife_crops = []
    for lbl in k_labels:
        img_path = None
        for ext in [".JPG", ".jpg", ".jpeg", ".png"]:
            cand = k_dir / "Images" / f"{lbl.stem}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue
        try:
            lines = lbl.read_text().strip().splitlines()
            if not lines:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]
            best_crop = None
            max_a = 0
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = max(0, int((xc - w / 2) * iw))
                    y1 = max(0, int((yc - h / 2) * ih))
                    x2 = min(iw, int((xc + w / 2) * iw))
                    y2 = min(ih, int((yc + h / 2) * ih))
                    if x2 > x1 + 5 and y2 > y1 + 5:
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_a:
                            max_a = area
                            best_crop = img[y1:y2, x1:x2]
            if best_crop is not None and best_crop.size > 0:
                knife_crops.append(best_crop)
                if len(knife_crops) >= 100:
                    break
        except Exception:
            continue

    print(f"Loaded {len(pistol_crops)} pistol crops and {len(knife_crops)} knife crops.")
    return pistol_crops, knife_crops

def check_hit(pred_boxes, target_cls, gt_box, conf_thresh):
    """
    Hit condition:
    pred class matches target_cls,
    pred conf >= conf_thresh,
    and pred center (cx, cy) is strictly inside gt_box [x1, y1, x2, y2].
    """
    for b in pred_boxes:
        cls_name = b["class"]
        conf = b["conf"]
        if cls_name == target_cls and conf >= conf_thresh:
            cx, cy = b["center"]
            if gt_box[0] <= cx <= gt_box[2] and gt_box[1] <= cy <= gt_box[3]:
                return True
    return False

def evaluate_scale_and_resolution(bg_frames, crops, target_cls, conf_thresh, scale_pcts, imgsz_list):
    """
    Evaluates weapon recall across scales and image resolutions.
    Each crop is tested across all 5 backgrounds at each scale.
    """
    results = {}
    for imgsz in imgsz_list:
        results[imgsz] = {}
        for s in scale_pcts:
            results[imgsz][s] = 0

    total_tests_per_condition = len(crops) * len(bg_frames)

    for c_idx, crop in enumerate(crops):
        ch, cw = crop.shape[:2]
        aspect = max(0.1, cw / max(1, ch))

        for s in scale_pcts:
            target_area = s * FRAME_AREA
            th = int(math.sqrt(target_area / aspect))
            tw = int(th * aspect)
            tw = max(8, min(FRAME_W - 20, tw))
            th = max(8, min(FRAME_H - 20, th))
            resized_crop = cv2.resize(crop, (tw, th))

            for bg_idx, bg in enumerate(bg_frames):
                # Paste weapon in center-right quadrant
                px = (FRAME_W - tw) // 2
                py = (FRAME_H - th) // 2
                comp = bg.copy()
                comp[py:py+th, px:px+tw] = resized_crop
                gt_box = [px, py, px+tw, py+th]

                for imgsz in imgsz_list:
                    # Run inference
                    preds = weapon_model.predict(comp, imgsz=imgsz, conf=0.25, verbose=False)
                    boxes = []
                    if preds and preds[0].boxes is not None:
                        for b in preds[0].boxes:
                            cname = preds[0].names[int(b.cls[0])].lower()
                            cf = float(b.conf[0])
                            bx1, by1, bx2, by2 = [float(coord) for coord in b.xyxy[0]]
                            boxes.append({
                                "class": cname,
                                "conf": cf,
                                "center": ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
                            })
                    if check_hit(boxes, target_cls, gt_box, conf_thresh):
                        results[imgsz][s] += 1

    return results, total_tests_per_condition

def evaluate_person_crop_pass(bg_frames, crops, target_cls, conf_thresh, s=0.05):
    """
    Person-crop pass at 5% scale:
    - Finds person in background frame with person_model.
    - Pastes weapon inside/overlapping person box.
    - Expands person box by 30%.
    - Crops expanded person box, resizes long edge to 640.
    - Runs weapon model at imgsz=640 on crop.
    - Maps detections back to full frame and checks hit.
    """
    total_hits = 0
    total_tests = 0

    for c_idx, crop in enumerate(crops):
        ch, cw = crop.shape[:2]
        aspect = max(0.1, cw / max(1, ch))
        target_area = s * FRAME_AREA
        th = int(math.sqrt(target_area / aspect))
        tw = int(th * aspect)
        tw = max(8, min(FRAME_W - 20, tw))
        th = max(8, min(FRAME_H - 20, th))
        resized_crop = cv2.resize(crop, (tw, th))

        for bg in bg_frames:
            # Detect persons in bg
            pres = person_model.predict(bg, classes=[0], conf=0.35, verbose=False)
            person_boxes = []
            if pres and pres[0].boxes is not None:
                for b in pres[0].boxes:
                    bx1, by1, bx2, by2 = [int(coord) for coord in b.xyxy[0]]
                    person_boxes.append((bx1, by1, bx2, by2))

            if not person_boxes:
                # If no person in bg frame, use center as fallback region
                person_boxes = [(160, 80, 480, 400)]

            # Pick largest person
            person_boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
            pbx1, pby1, pbx2, pby2 = person_boxes[0]

            # Paste weapon on person (e.g. torso/hand region)
            pw = pbx2 - pbx1
            ph = pby2 - pby1
            paste_x = max(0, min(FRAME_W - tw, pbx1 + (pw - tw) // 2))
            paste_y = max(0, min(FRAME_H - th, pby1 + int(ph * 0.4)))

            comp = bg.copy()
            comp[paste_y:paste_y+th, paste_x:paste_x+tw] = resized_crop
            gt_box = [paste_x, paste_y, paste_x+tw, paste_y+th]

            # 30% expansion around person
            cx = (pbx1 + pbx2) / 2.0
            cy = (pby1 + pby2) / 2.0
            ew = pw * 1.3
            eh = ph * 1.3
            crop_x1 = max(0, int(cx - ew / 2.0))
            crop_y1 = max(0, int(cy - eh / 2.0))
            crop_x2 = min(FRAME_W, int(cx + ew / 2.0))
            crop_y2 = min(FRAME_H, int(cy + eh / 2.0))

            person_crop = comp[crop_y1:crop_y2, crop_x1:crop_x2]
            if person_crop.size == 0:
                continue

            # Resize long side to 640
            pch, pcw = person_crop.shape[:2]
            scale_fac = 640.0 / max(pch, pcw)
            nw = int(pcw * scale_fac)
            nh = int(pch * scale_fac)
            crop_resized = cv2.resize(person_crop, (nw, nh))

            preds = weapon_model.predict(crop_resized, imgsz=640, conf=0.25, verbose=False)
            boxes = []
            if preds and preds[0].boxes is not None:
                for b in preds[0].boxes:
                    cname = preds[0].names[int(b.cls[0])].lower()
                    cf = float(b.conf[0])
                    cbx1, cby1, cbx2, cby2 = [float(coord) for coord in b.xyxy[0]]
                    # Map back to full frame coords
                    f_bx1 = crop_x1 + (cbx1 / scale_fac)
                    f_by1 = crop_y1 + (cby1 / scale_fac)
                    f_bx2 = crop_x1 + (cbx2 / scale_fac)
                    f_by2 = crop_y1 + (cby2 / scale_fac)
                    boxes.append({
                        "class": cname,
                        "conf": cf,
                        "center": ((f_bx1 + f_bx2) / 2.0, (f_by1 + f_by2) / 2.0)
                    })

            total_tests += 1
            if check_hit(boxes, target_cls, gt_box, conf_thresh):
                total_hits += 1

    return total_hits, total_tests

def main():
    print("Loading 5 background frames from NonFight clips...")
    bg_frames = load_background_frames()

    print("\n" + "="*80)
    print("STEP 1: BASELINE UNPASTED CHECK ON 5 BACKGROUND FRAMES")
    print("="*80)
    for idx, bg in enumerate(bg_frames):
        p_name = BG_CLIP_PATHS[idx].name
        res = weapon_model.predict(bg, imgsz=640, conf=0.25, verbose=False)
        dets = []
        if res and res[0].boxes is not None:
            for b in res[0].boxes:
                cname = res[0].names[int(b.cls[0])].lower()
                conf = float(b.conf[0])
                dets.append(f"{cname} ({conf:.3f})")
        print(f"BG {idx+1} ({p_name:<20}): {len(dets)} detections -> {dets if dets else 'None (Clean)'}")

    print("\nLoading dataset crops...")
    pistol_crops, knife_crops = load_dataset_crops()

    # Scales
    scales = [0.02, 0.05, 0.10, 0.20]

    # Standard imgsz=640
    print("\n" + "="*80)
    print("STEP 2: CCTV SCALE EVALUATION (imgsz=640)")
    print("="*80)
    p_res_640, p_total = evaluate_scale_and_resolution(bg_frames, pistol_crops, "pistol", 0.45, scales, [640])
    k_res_640, k_total = evaluate_scale_and_resolution(bg_frames, knife_crops, "knife", 0.50, scales, [640])

    print(f"\nResults at imgsz=640 (Total tests per scale = {p_total}):")
    print(f"{'Scale':<12} | {'Pistol (conf>=0.45)':<25} | {'Knife (conf>=0.50)':<25}")
    print("-" * 68)
    for s in scales:
        p_hits = p_res_640[640][s]
        p_pct = (p_hits / p_total) * 100
        k_hits = k_res_640[640][s]
        k_pct = (k_hits / k_total) * 100
        print(f"{int(s*100):2d}% of frame | {p_hits:3d}/{p_total} ({p_pct:5.1f}%)            | {k_hits:3d}/{k_total} ({k_pct:5.1f}%)")

    # Higher resolution (imgsz=960 and imgsz=1280) on 2% and 5%
    print("\n" + "="*80)
    print("STEP 3: RESOLUTION SCALING (imgsz=960 & 1280 on 2% and 5% scales)")
    print("="*80)
    small_scales = [0.02, 0.05]
    p_res_hires, _ = evaluate_scale_and_resolution(bg_frames, pistol_crops, "pistol", 0.45, small_scales, [960, 1280])
    k_res_hires, _ = evaluate_scale_and_resolution(bg_frames, knife_crops, "knife", 0.50, small_scales, [960, 1280])

    print(f"\nResolution Scaling Comparison (Total tests = {p_total}):")
    print(f"{'Class':<8} | {'Scale':<6} | {'imgsz=640':<18} | {'imgsz=960':<18} | {'imgsz=1280':<18}")
    print("-" * 76)
    for s in small_scales:
        p640 = f"{p_res_640[640][s]}/{p_total} ({p_res_640[640][s]/p_total*100:.1f}%)"
        p960 = f"{p_res_hires[960][s]}/{p_total} ({p_res_hires[960][s]/p_total*100:.1f}%)"
        p1280 = f"{p_res_hires[1280][s]}/{p_total} ({p_res_hires[1280][s]/p_total*100:.1f}%)"
        print(f"{'Pistol':<8} | {int(s*100):2d}%   | {p640:<18} | {p960:<18} | {p1280:<18}")

    for s in small_scales:
        k640 = f"{k_res_640[640][s]}/{k_total} ({k_res_640[640][s]/k_total*100:.1f}%)"
        k960 = f"{k_res_hires[960][s]}/{k_total} ({k_res_hires[960][s]/k_total*100:.1f}%)"
        k1280 = f"{k_res_hires[1280][s]}/{k_total} ({k_res_hires[1280][s]/k_total*100:.1f}%)"
        print(f"{'Knife':<8} | {int(s*100):2d}%   | {k640:<18} | {k960:<18} | {k1280:<18}")

    # Person-crop pass at 5% scale
    print("\n" + "="*80)
    print("STEP 4: PERSON-CROP PASS (30% expanded box, imgsz=640 crop) AT 5% SCALE")
    print("="*80)
    p_crop_hits, p_crop_total = evaluate_person_crop_pass(bg_frames, pistol_crops, "pistol", 0.45, s=0.05)
    k_crop_hits, k_crop_total = evaluate_person_crop_pass(bg_frames, knife_crops, "knife", 0.50, s=0.05)

    p_crop_pct = (p_crop_hits / p_crop_total) * 100 if p_crop_total else 0.0
    k_crop_pct = (k_crop_hits / k_crop_total) * 100 if k_crop_total else 0.0

    print(f"Pistol @ 5% with Person Crop Pass: {p_crop_hits}/{p_crop_total} ({p_crop_pct:.1f}%) vs Full-Frame 640: {p_res_640[640][0.05]/p_total*100:.1f}%")
    print(f"Knife  @ 5% with Person Crop Pass: {k_crop_hits}/{k_crop_total} ({k_crop_pct:.1f}%) vs Full-Frame 640: {k_res_640[640][0.05]/k_total*100:.1f}%")
    print("="*80)

if __name__ == "__main__":
    main()
