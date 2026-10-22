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
PISTOL_IMG_DIR = repo_root / "Dataaset2" / "Pistol detection" / "Weapons"
KNIFE_IMG_DIR = repo_root / "Dataaset2" / "Knife_detection" / "Images"
NONFIGHT_CLIP = repo_root / "RWF-2000" / "NonFight" / "0_DzLlklZa0_0.avi"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading weapon model on {device}...")
weapon_model = YOLO(str(WEAPON_MODEL_PATH)).to(device)

def get_background_frame():
    cap = cv2.VideoCapture(str(NONFIGHT_CLIP))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        # fallback to neutral indoor CCTV colored canvas
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    else:
        frame = cv2.resize(frame, (640, 480))
    return frame

def scale_and_paste(bg: np.ndarray, fg: np.ndarray, target_area_ratio: float):
    bh, bw = bg.shape[:2]
    total_area = float(bw * bh)
    target_area = target_area_ratio * total_area
    
    fh, fw = fg.shape[:2]
    current_area = float(fw * fh)
    if current_area <= 0:
        return bg.copy()
        
    scale = (target_area / current_area) ** 0.5
    nw = max(10, min(bw - 20, int(fw * scale)))
    nh = max(10, min(bh - 20, int(fh * scale)))
    
    resized_fg = cv2.resize(fg, (nw, nh), interpolation=cv2.INTER_AREA)
    
    composite = bg.copy()
    # Place in middle third of scene
    x_offset = (bw - nw) // 2
    y_offset = (bh - nh) // 2
    
    composite[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized_fg
    return composite

def evaluate_cctv_scale(image_files: list[Path], target_cls: str):
    scales = [0.05, 0.10, 0.20]
    thresholds = [0.35, 0.45, 0.55]
    
    # results[scale][thresh] = hit_count
    results = {s: {th: 0 for th in thresholds} for s in scales}
    bg = get_background_frame()
    
    total = len(image_files)
    print(f"Evaluating {total} images for class '{target_cls}' across scales 5%, 10%, 20%...")
    
    for idx, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        for s in scales:
            composite = scale_and_paste(bg, img, s)
            w_res = weapon_model.predict(composite, conf=0.30, verbose=False)
            
            # Find max confidence for target class
            max_conf = 0.0
            if w_res and w_res[0].boxes is not None:
                for b in w_res[0].boxes:
                    cname = weapon_model.names[int(b.cls[0])].lower()
                    if cname == target_cls:
                        c_conf = float(b.conf[0])
                        if c_conf > max_conf:
                            max_conf = c_conf
                            
            for th in thresholds:
                if max_conf >= th:
                    results[s][th] += 1
                    
    return results, total

def main():
    pistol_files = sorted(list(PISTOL_IMG_DIR.glob("*.jpg")))[:100]
    knife_files = sorted(list(KNIFE_IMG_DIR.glob("*.jpg")))[:100]
    
    p_results, p_total = evaluate_cctv_scale(pistol_files, "pistol")
    k_results, k_total = evaluate_cctv_scale(knife_files, "knife")
    
    print("\n" + "="*85)
    print("PISTOL & KNIFE RECALL AT REALISTIC CCTV SCALES (640x480 NonFight Background)")
    print("="*85)
    print(f"{'Class':<8} | {'Scale (% Frame Area)':<22} | {'Conf >= 0.35':<16} | {'Conf >= 0.45':<16} | {'Conf >= 0.55':<16}")
    print("-" * 85)
    
    # Pistols
    for s in [0.05, 0.10, 0.20]:
        pct_label = f"~{int(s*100)}% scale"
        r35 = f"{p_results[s][0.35]}/{p_total} ({p_results[s][0.35]/p_total*100:.1f}%)"
        r45 = f"{p_results[s][0.45]}/{p_total} ({p_results[s][0.45]/p_total*100:.1f}%)"
        r55 = f"{p_results[s][0.55]}/{p_total} ({p_results[s][0.55]/p_total*100:.1f}%)"
        print(f"{'Pistol':<8} | {pct_label:<22} | {r35:<16} | {r45:<16} | {r55:<16}")
        
    print("-" * 85)
    
    # Knives
    for s in [0.05, 0.10, 0.20]:
        pct_label = f"~{int(s*100)}% scale"
        r35 = f"{k_results[s][0.35]}/{k_total} ({k_results[s][0.35]/k_total*100:.1f}%)"
        r45 = f"{k_results[s][0.45]}/{k_total} ({k_results[s][0.45]/k_total*100:.1f}%)"
        r55 = f"{k_results[s][0.55]}/{k_total} ({k_results[s][0.55]/k_total*100:.1f}%)"
        print(f"{'Knife':<8} | {pct_label:<22} | {r35:<16} | {r45:<16} | {r55:<16}")
        
    print("="*85)

if __name__ == "__main__":
    main()
