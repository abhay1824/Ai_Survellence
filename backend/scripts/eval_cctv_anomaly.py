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
weapon_model = YOLO(str(WEAPON_MODEL_PATH)).to(device)

BG_CLIP_PATHS = [
    repo_root / "RWF-2000" / "NonFight" / "-1l5631l3fg_0.avi",  # BG 0: Indoor lobby
    repo_root / "RWF-2000" / "NonFight" / "0_DzLlklZa0_0.avi",  # BG 1: Street / road
    repo_root / "RWF-2000" / "NonFight" / "0AvTZRYx_0.avi",     # BG 2: Indoor corridor
    repo_root / "RWF-2000" / "NonFight" / "0H2s9UJcNJ0_0.avi",  # BG 3: Outdoor courtyard
    repo_root / "RWF-2000" / "NonFight" / "0vasozvJYvk_0.avi"   # BG 4: Outdoor walkway
]

FRAME_W = 640
FRAME_H = 480
FRAME_AREA = FRAME_W * FRAME_H

OUT_DIR = repo_root / "debug_crops" / "cctv_composites"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
                knife_crops.append(best_crop)
                if len(knife_crops) >= 100:
                    break
        except Exception:
            continue

    print(f"Loaded {len(pistol_crops)} pistol crops and {len(knife_crops)} knife crops.")
    return pistol_crops, knife_crops

def run_eval():
    bg_frames = load_background_frames()
    pistol_crops, knife_crops = load_dataset_crops()

    # Per-background and per-crop tracking:
    # hit_record[cls][scale][crop_idx][bg_idx] = bool
    scales = [0.02, 0.05, 0.10, 0.20]
    conf_pistol = 0.45
    conf_knife = 0.50

    p_records = {s: np.zeros((len(pistol_crops), len(bg_frames)), dtype=bool) for s in scales}
    k_records = {s: np.zeros((len(knife_crops), len(bg_frames)), dtype=bool) for s in scales}

    # Store detection info for anomaly inspection: (comp_img, gt_box, preds)
    knife_anomalies_5_vs_20 = []
    pistol_anomalies_2_vs_5 = []

    print("\nRunning Pistol Evaluation across scales...")
    for c_idx, crop in enumerate(pistol_crops):
        ch, cw = crop.shape[:2]
        aspect = max(0.1, cw / max(1, ch))

        for s in scales:
            target_area = s * FRAME_AREA
            th = int(math.sqrt(target_area / aspect))
            tw = int(th * aspect)
            tw = max(8, min(FRAME_W - 20, tw))
            th = max(8, min(FRAME_H - 20, th))
            resized_crop = cv2.resize(crop, (tw, th))

            for bg_idx, bg in enumerate(bg_frames):
                px = (FRAME_W - tw) // 2
                py = (FRAME_H - th) // 2
                comp = bg.copy()
                comp[py:py+th, px:px+tw] = resized_crop
                gt_box = [px, py, px+tw, py+th]

                preds = weapon_model.predict(comp, imgsz=640, conf=0.25, verbose=False)
                hit = False
                det_list = []
                if preds and preds[0].boxes is not None:
                    for b in preds[0].boxes:
                        cname = preds[0].names[int(b.cls[0])].lower()
                        cf = float(b.conf[0])
                        bx1, by1, bx2, by2 = [float(coord) for coord in b.xyxy[0]]
                        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                        det_list.append((cname, cf, [bx1, by1, bx2, by2]))
                        if cname in ["pistol", "gun"] and cf >= conf_pistol:
                            if gt_box[0] <= cx <= gt_box[2] and gt_box[1] <= cy <= gt_box[3]:
                                hit = True
                p_records[s][c_idx, bg_idx] = hit

                # Collect potential anomaly samples: hit at 2% but missed at 5%
                if s == 0.02 and hit:
                    # Will check after 5% is evaluated
                    pass

    # Find pistol 2% hit vs 5% miss
    for c_idx in range(len(pistol_crops)):
        for bg_idx in range(len(bg_frames)):
            if p_records[0.02][c_idx, bg_idx] and not p_records[0.05][c_idx, bg_idx]:
                pistol_anomalies_2_vs_5.append((c_idx, bg_idx))

    print("Running Knife Evaluation across scales...")
    for c_idx, crop in enumerate(knife_crops):
        ch, cw = crop.shape[:2]
        aspect = max(0.1, cw / max(1, ch))

        for s in scales:
            target_area = s * FRAME_AREA
            th = int(math.sqrt(target_area / aspect))
            tw = int(th * aspect)
            tw = max(8, min(FRAME_W - 20, tw))
            th = max(8, min(FRAME_H - 20, th))
            resized_crop = cv2.resize(crop, (tw, th))

            for bg_idx, bg in enumerate(bg_frames):
                px = (FRAME_W - tw) // 2
                py = (FRAME_H - th) // 2
                comp = bg.copy()
                comp[py:py+th, px:px+tw] = resized_crop
                gt_box = [px, py, px+tw, py+th]

                preds = weapon_model.predict(comp, imgsz=640, conf=0.25, verbose=False)
                hit = False
                det_list = []
                if preds and preds[0].boxes is not None:
                    for b in preds[0].boxes:
                        cname = preds[0].names[int(b.cls[0])].lower()
                        cf = float(b.conf[0])
                        bx1, by1, bx2, by2 = [float(coord) for coord in b.xyxy[0]]
                        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                        det_list.append((cname, cf, [bx1, by1, bx2, by2]))
                        if cname in ["knife", "scissors"] and cf >= conf_knife:
                            if gt_box[0] <= cx <= gt_box[2] and gt_box[1] <= cy <= gt_box[3]:
                                hit = True
                k_records[s][c_idx, bg_idx] = hit

    # Find knife 5% hit vs 20% miss
    for c_idx in range(len(knife_crops)):
        for bg_idx in range(len(bg_frames)):
            if k_records[0.05][c_idx, bg_idx] and not k_records[0.20][c_idx, bg_idx]:
                knife_anomalies_5_vs_20.append((c_idx, bg_idx))

    # --- 1. REPORT PER-BACKGROUND RECALL ---
    print("\n" + "=" * 90)
    print("PER-BACKGROUND RECALL ACROSS 5 BACKGROUNDS")
    print("=" * 90)
    bg_names = ["BG 0 (Indoor Lobby)", "BG 1 (Street/Road)", "BG 2 (Corridor)", "BG 3 (Courtyard)", "BG 4 (Walkway)"]

    print("\n[PISTOL RECALL BY BACKGROUND]")
    print(f"{'Background':<25} | {'2% Scale':<12} | {'5% Scale':<12} | {'10% Scale':<12} | {'20% Scale':<12}")
    print("-" * 75)
    for bg_idx in range(5):
        r2 = p_records[0.02][:, bg_idx].mean() * 100
        r5 = p_records[0.05][:, bg_idx].mean() * 100
        r10 = p_records[0.10][:, bg_idx].mean() * 100
        r20 = p_records[0.20][:, bg_idx].mean() * 100
        print(f"{bg_names[bg_idx]:<25} | {r2:5.1f}%      | {r5:5.1f}%      | {r10:5.1f}%      | {r20:5.1f}%")
    print("-" * 75)
    print(f"{'OVERALL AVERAGE':<25} | {p_records[0.02].mean()*100:5.1f}%      | {p_records[0.05].mean()*100:5.1f}%      | {p_records[0.10].mean()*100:5.1f}%      | {p_records[0.20].mean()*100:5.1f}%")

    print("\n[KNIFE RECALL BY BACKGROUND]")
    print(f"{'Background':<25} | {'2% Scale':<12} | {'5% Scale':<12} | {'10% Scale':<12} | {'20% Scale':<12}")
    print("-" * 75)
    for bg_idx in range(5):
        r2 = k_records[0.02][:, bg_idx].mean() * 100
        r5 = k_records[0.05][:, bg_idx].mean() * 100
        r10 = k_records[0.10][:, bg_idx].mean() * 100
        r20 = k_records[0.20][:, bg_idx].mean() * 100
        print(f"{bg_names[bg_idx]:<25} | {r2:5.1f}%      | {r5:5.1f}%      | {r10:5.1f}%      | {r20:5.1f}%")
    print("-" * 75)
    print(f"{'OVERALL AVERAGE':<25} | {k_records[0.02].mean()*100:5.1f}%      | {k_records[0.05].mean()*100:5.1f}%      | {k_records[0.10].mean()*100:5.1f}%      | {k_records[0.20].mean()*100:5.1f}%")

    # --- 2. BOOTSTRAP 95% CONFIDENCE INTERVALS ---
    print("\n" + "=" * 90)
    print("95% BOOTSTRAP CONFIDENCE INTERVALS (10,000 Resamples of 100 Image Crops)")
    print("=" * 90)
    np.random.seed(42)
    B = 10000
    n_crops = 100

    # For Knife: Recall(20%) - Recall(5%)
    # Average recall per crop across the 5 backgrounds
    k_crop_recall_5 = k_records[0.05].mean(axis=1)  # shape (100,)
    k_crop_recall_20 = k_records[0.20].mean(axis=1) # shape (100,)
    k_diffs = k_crop_recall_20 - k_crop_recall_5

    boot_k_diffs = np.zeros(B)
    for b in range(B):
        idx = np.random.choice(n_crops, size=n_crops, replace=True)
        boot_k_diffs[b] = k_diffs[idx].mean() * 100

    k_ci_low, k_ci_high = np.percentile(boot_k_diffs, [2.5, 97.5])
    k_mean_diff = k_diffs.mean() * 100
    print(f"Knife Recall Difference [20% Scale - 5% Scale]:")
    print(f"  Observed Difference: {k_mean_diff:+.2f}%")
    print(f"  95% Bootstrap CI: [{k_ci_low:+.2f}%, {k_ci_high:+.2f}%]")
    if k_ci_low <= 0 <= k_ci_high:
        print("  -> Contains 0: The difference is NOT statistically significant (consistent with random noise).")
    else:
        print("  -> Does NOT contain 0: The difference is statistically significant.")

    # For Pistol: Recall(2%) - Recall(5%)
    p_crop_recall_2 = p_records[0.02].mean(axis=1)  # shape (100,)
    p_crop_recall_5 = p_records[0.05].mean(axis=1)  # shape (100,)
    p_diffs = p_crop_recall_2 - p_crop_recall_5

    boot_p_diffs = np.zeros(B)
    for b in range(B):
        idx = np.random.choice(n_crops, size=n_crops, replace=True)
        boot_p_diffs[b] = p_diffs[idx].mean() * 100

    p_ci_low, p_ci_high = np.percentile(boot_p_diffs, [2.5, 97.5])
    p_mean_diff = p_diffs.mean() * 100
    print(f"\nPistol Recall Difference [2% Scale - 5% Scale]:")
    print(f"  Observed Difference: {p_mean_diff:+.2f}%")
    print(f"  95% Bootstrap CI: [{p_ci_low:+.2f}%, {p_ci_high:+.2f}%]")
    if p_ci_low <= 0 <= p_ci_high:
        print("  -> Contains 0: The difference is NOT statistically significant (consistent with random noise).")
    else:
        print("  -> Does NOT contain 0: The difference is statistically significant.")

    # --- 3. SAVE 10 COMPOSITE SAMPLES WITH BOUNDING BOXES DRAWN ---
    print("\n" + "=" * 90)
    print("SAVING 10 COMPOSITE IMAGES WITH BOUNDING BOXES DRAWN FOR EACH ANOMALY")
    print("=" * 90)

    # Save 10 Knife samples (comparing 5% vs 20%)
    print(f"Found {len(knife_anomalies_5_vs_20)} cases where Knife was detected at 5% but missed at 20%.")
    saved_k = 0
    for c_idx, bg_idx in knife_anomalies_5_vs_20[:10]:
        crop = knife_crops[c_idx]
        bg = bg_frames[bg_idx]
        ch, cw = crop.shape[:2]
        aspect = max(0.1, cw / max(1, ch))

        # Build 20% scale composite
        s = 0.20
        target_area = s * FRAME_AREA
        th = int(math.sqrt(target_area / aspect))
        tw = int(th * aspect)
        tw = max(8, min(FRAME_W - 20, tw))
        th = max(8, min(FRAME_H - 20, th))
        resized_crop = cv2.resize(crop, (tw, th))
        px = (FRAME_W - tw) // 2
        py = (FRAME_H - th) // 2
        comp = bg.copy()
        comp[py:py+th, px:px+tw] = resized_crop
        gt_box = [px, py, px+tw, py+th]

        preds = weapon_model.predict(comp, imgsz=640, conf=0.25, verbose=False)
        det_img = comp.copy()
        # Draw GT in Blue
        cv2.rectangle(det_img, (px, py), (px+tw, py+th), (255, 0, 0), 2)
        cv2.putText(det_img, f"GT Knife (20% scale, {tw}x{th})", (px, max(15, py - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Draw model predictions in Green (or Red if other class)
        pred_desc = []
        if preds and preds[0].boxes is not None:
            for b in preds[0].boxes:
                cname = preds[0].names[int(b.cls[0])]
                cf = float(b.conf[0])
                bx1, by1, bx2, by2 = [int(coord) for coord in b.xyxy[0]]
                color = (0, 255, 0) if cname.lower() in ["knife", "scissors"] else (0, 0, 255)
                cv2.rectangle(det_img, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(det_img, f"{cname} {cf:.2f}", (bx1, max(15, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                pred_desc.append(f"{cname}:{cf:.2f}")

        out_path = OUT_DIR / f"knife_20pct_miss_crop{c_idx}_bg{bg_idx}.jpg"
        cv2.imwrite(str(out_path), det_img)
        saved_k += 1
        print(f"Saved Knife 20% sample {saved_k}: {out_path.name} | GT: {tw}x{th} | Dets: {pred_desc}")

    # Save 10 Pistol samples (comparing 2% vs 5%)
    print(f"\nFound {len(pistol_anomalies_2_vs_5)} cases where Pistol was detected at 2% but missed at 5%.")
    saved_p = 0
    for c_idx, bg_idx in pistol_anomalies_2_vs_5[:10]:
        crop = pistol_crops[c_idx]
        bg = bg_frames[bg_idx]
        ch, cw = crop.shape[:2]
        aspect = max(0.1, cw / max(1, ch))

        # Build 5% scale composite where it was missed
        s = 0.05
        target_area = s * FRAME_AREA
        th = int(math.sqrt(target_area / aspect))
        tw = int(th * aspect)
        tw = max(8, min(FRAME_W - 20, tw))
        th = max(8, min(FRAME_H - 20, th))
        resized_crop = cv2.resize(crop, (tw, th))
        px = (FRAME_W - tw) // 2
        py = (FRAME_H - th) // 2
        comp = bg.copy()
        comp[py:py+th, px:px+tw] = resized_crop
        gt_box = [px, py, px+tw, py+th]

        preds = weapon_model.predict(comp, imgsz=640, conf=0.25, verbose=False)
        det_img = comp.copy()
        # Draw GT in Blue
        cv2.rectangle(det_img, (px, py), (px+tw, py+th), (255, 0, 0), 2)
        cv2.putText(det_img, f"GT Pistol (5% scale, {tw}x{th})", (px, max(15, py - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        pred_desc = []
        if preds and preds[0].boxes is not None:
            for b in preds[0].boxes:
                cname = preds[0].names[int(b.cls[0])]
                cf = float(b.conf[0])
                bx1, by1, bx2, by2 = [int(coord) for coord in b.xyxy[0]]
                color = (0, 255, 0) if cname.lower() in ["pistol", "gun"] else (0, 0, 255)
                cv2.rectangle(det_img, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(det_img, f"{cname} {cf:.2f}", (bx1, max(15, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                pred_desc.append(f"{cname}:{cf:.2f}")

        out_path = OUT_DIR / f"pistol_5pct_miss_crop{c_idx}_bg{bg_idx}.jpg"
        cv2.imwrite(str(out_path), det_img)
        saved_p += 1
        print(f"Saved Pistol 5% sample {saved_p}: {out_path.name} | GT: {tw}x{th} | Dets: {pred_desc}")

if __name__ == "__main__":
    run_eval()
