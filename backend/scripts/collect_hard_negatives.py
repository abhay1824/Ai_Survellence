import os
import sys
import cv2
import shutil
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

from app.services.ai_service import AIService

def collect_hard_negatives():
    print("=" * 80)
    print("COLLECTING HARD NEGATIVES")
    print("=" * 80)

    ai_service = AIService()
    model = ai_service.model
    assert model is not None, "Weapon model is not loaded"

    out_img_dir = repo_root / "hard_negatives" / "images"
    out_lbl_dir = repo_root / "hard_negatives" / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    collected_count = 0
    saved_files = set()

    def save_hard_negative(frame, base_name):
        nonlocal collected_count
        img_filename = f"{base_name}.jpg"
        lbl_filename = f"{base_name}.txt"
        
        img_dest = out_img_dir / img_filename
        lbl_dest = out_lbl_dir / lbl_filename
        
        cv2.imwrite(str(img_dest), frame)
        # Empty YOLO label file for hard negative
        with open(lbl_dest, "w", encoding="utf-8") as f:
            pass
        
        collected_count += 1
        saved_files.add(base_name)

    # 1. CAM-003 (sample_videos/accident.mp4) frames 30-58
    acc_vid_path = repo_root / "sample_videos" / "accident.mp4"
    if acc_vid_path.exists():
        cap = cv2.VideoCapture(str(acc_vid_path))
        f_idx = 0
        cam003_saved = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            f_idx += 1
            if 30 <= f_idx <= 58:
                save_hard_negative(frame, f"cam003_accident_frame_{f_idx:03d}")
                cam003_saved += 1
        cap.release()
        print(f"[CAM-003] Saved {cam003_saved} frames (frames 30 to 58).")

    # 2. NonFight clips (detect any false pistol/knife at conf >= 0.35)
    nonfight_dir = repo_root / "RWF-2000" / "NonFight"
    nonfight_saved = 0
    if nonfight_dir.exists():
        nonfight_clips = sorted(list(nonfight_dir.glob("*.avi")))[:20]
        for clip in nonfight_clips:
            cap = cv2.VideoCapture(str(clip))
            f_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                f_idx += 1
                res = model(frame, conf=0.35, verbose=False)
                has_false_weapon = False
                for r in res:
                    for b in r.boxes:
                        c_id = int(b.cls[0])
                        c_name = model.names.get(c_id, "").lower()
                        if c_name in ("pistol", "gun", "knife", "scissors"):
                            has_false_weapon = True
                            break
                    if has_false_weapon:
                        break
                if has_false_weapon:
                    save_hard_negative(frame, f"nonfight_{clip.stem}_f{f_idx:03d}")
                    nonfight_saved += 1
            cap.release()
        print(f"[NonFight Clips] Saved {nonfight_saved} false weapon frames.")

    # 3. normal.mp4
    normal_vid_path = repo_root / "sample_videos" / "normal.mp4"
    normal_saved = 0
    if normal_vid_path.exists():
        cap = cv2.VideoCapture(str(normal_vid_path))
        f_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            f_idx += 1
            res = model(frame, conf=0.35, verbose=False)
            has_false_weapon = False
            for r in res:
                for b in r.boxes:
                    c_id = int(b.cls[0])
                    c_name = model.names.get(c_id, "").lower()
                    if c_name in ("pistol", "gun", "knife", "scissors"):
                        has_false_weapon = True
                        break
                if has_false_weapon:
                    break
            if has_false_weapon:
                save_hard_negative(frame, f"normal_mp4_f{f_idx:03d}")
                normal_saved += 1
        cap.release()
        print(f"[normal.mp4] Saved {normal_saved} false weapon frames.")

    # 4. Road frames (acc/ and accident_frames)
    road_dirs = [
        repo_root / "acc" / "test" / "images",
        repo_root / "acc" / "valid" / "images",
        repo_root / "accident_frames"
    ]
    road_saved = 0
    for rdir in road_dirs:
        if rdir.exists():
            for img_path in list(rdir.glob("*.jpg")) + list(rdir.glob("*.png")):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                res = model(img, conf=0.35, verbose=False)
                has_false_weapon = False
                for r in res:
                    for b in r.boxes:
                        c_id = int(b.cls[0])
                        c_name = model.names.get(c_id, "").lower()
                        if c_name in ("pistol", "gun", "knife", "scissors"):
                            has_false_weapon = True
                            break
                    if has_false_weapon:
                        break
                if has_false_weapon:
                    save_hard_negative(img, f"road_{img_path.stem}")
                    road_saved += 1
    print(f"[Road Frames] Saved {road_saved} false weapon frames.")

    print("\n" + "=" * 80)
    print(f"HARD NEGATIVES COLLECTION COMPLETE: Total {collected_count} frames collected.")
    print(f"Images directory: {out_img_dir} ({len(list(out_img_dir.glob('*.jpg')))} files)")
    print(f"Labels directory: {out_lbl_dir} ({len(list(out_lbl_dir.glob('*.txt')))} files)")
    print("=" * 80)

if __name__ == "__main__":
    collect_hard_negatives()
