import os
import sys
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

from app.services.ai_service import AIService

def parse_clip_info(filename):
    stem = Path(filename).stem
    # 1. cam003_accident_frame_030
    m = re.match(r'^(cam003_accident)_frame_(\d+)$', stem)
    if m:
        return m.group(1), int(m.group(2))
    # 2. nonfight_<clip>_f<frame>
    m = re.match(r'^(nonfight_.*)_f(\d+)$', stem)
    if m:
        return m.group(1), int(m.group(2))
    # 3. normal_mp4_f<frame>
    m = re.match(r'^(normal_mp4)_f(\d+)$', stem)
    if m:
        return m.group(1), int(m.group(2))
    # 4. road video cutter
    m = re.match(r'^road_(Accidents-online-video-cutter[^-]*).*', stem)
    if m:
        # try to find frame number in filename e.g. -28_jpg
        fm = re.search(r'-(\d+)_jpg', stem)
        f_idx = int(fm.group(1)) if fm else 0
        return f"road_{m.group(1)}", f_idx
    # 5. road image prefixes
    parts = stem.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}", 0
    return "road_misc", 0

def main():
    img_dir = repo_root / "hard_negatives" / "images"
    sheet_dir = repo_root / "hard_negatives" / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    
    ai_service = AIService()
    model = ai_service.model
    assert model is not None, "Weapon model not loaded"

    files = sorted(os.listdir(img_dir))
    print(f"Total frames to process: {len(files)}")

    records = []
    annotated_thumbs = []

    # Assign whole clips to train/val 70/30 split
    clip_frames = defaultdict(list)
    for f in files:
        clip_name, f_idx = parse_clip_info(f)
        clip_frames[clip_name].append(f)

    # Deterministic 70/30 split by clip
    # Sort clips by frame count descending, then alternate/greedily assign to balance frame counts
    sorted_clips = sorted(clip_frames.keys(), key=lambda c: len(clip_frames[c]), reverse=True)
    
    train_clips = set()
    val_clips = set()
    train_frame_count = 0
    val_frame_count = 0
    target_train_ratio = 0.70

    for clip in sorted_clips:
        count = len(clip_frames[clip])
        total_curr = train_frame_count + val_frame_count
        if total_curr == 0 or (train_frame_count / (total_curr + count)) < target_train_ratio:
            train_clips.add(clip)
            train_frame_count += count
        else:
            val_clips.add(clip)
            val_frame_count += count

    print(f"Split Summary: Train clips={len(train_clips)} ({train_frame_count} frames, {train_frame_count/len(files)*100:.1f}%), Val clips={len(val_clips)} ({val_frame_count} frames, {val_frame_count/len(files)*100:.1f}%)")

    thumb_w, thumb_h = 400, 300

    for i, fname in enumerate(files):
        fpath = img_dir / fname
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        h_orig, w_orig = img.shape[:2]
        clip_name, f_idx = parse_clip_info(fname)
        split = "train" if clip_name in train_clips else "val"

        # Run inference
        res = model(img, conf=0.35, verbose=False)
        best_box = None
        best_conf = 0.0
        best_cls = "none"
        best_area_pct = 0.0

        for r in res:
            for b in r.boxes:
                c_id = int(b.cls[0])
                c_name = model.names.get(c_id, "").lower()
                conf = float(b.conf[0])
                xyxy = b.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                box_area = (x2 - x1) * (y2 - y1)
                area_pct = (box_area / (w_orig * h_orig)) * 100.0
                
                # We record weapon detections or highest conf detection
                if c_name in ("pistol", "knife", "gun", "scissors") or conf > best_conf:
                    if conf > best_conf:
                        best_conf = conf
                        best_cls = c_name
                        best_box = (int(x1), int(y1), int(x2), int(y2))
                        best_area_pct = area_pct

        records.append({
            "source_clip": clip_name,
            "frame_index": f_idx,
            "filename": fname,
            "detected_class": best_cls,
            "confidence": round(best_conf, 3),
            "box_area_pct": round(best_area_pct, 2),
            "split": split
        })

        # Create thumbnail with drawings
        annotated = img.copy()
        if best_box is not None:
            bx1, by1, bx2, by2 = best_box
            # Draw rectangle in bright magenta
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (255, 0, 255), 3)
            # Label
            lbl = f"{best_cls.upper()} {best_conf:.2f} ({best_area_pct:.1f}%)"
            t_size = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated, (bx1, max(0, by1 - 25)), (bx1 + t_size[0] + 6, max(0, by1)), (255, 0, 255), -1)
            cv2.putText(annotated, lbl, (bx1 + 3, max(0, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Resize to thumbnail
        thumb = cv2.resize(annotated, (thumb_w, thumb_h - 40)) # Leave 40px for bottom caption
        
        # Add footer banner
        footer = np.zeros((40, thumb_w, 3), dtype=np.uint8)
        caption1 = f"#{i+1}: {clip_name}"
        caption2 = f"F:{f_idx} [{split.upper()}] {best_cls}:{best_conf:.2f}"
        cv2.putText(footer, caption1[:32], (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(footer, caption2, (5, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        tile = np.vstack([thumb, footer])
        annotated_thumbs.append(tile)

    # Save manifest
    df = pd.DataFrame(records)
    manifest_path = repo_root / "hard_negatives" / "manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"Saved manifest to {manifest_path} ({len(df)} rows)")

    # Build contact sheets (~30 per sheet -> 5 cols x 6 rows = 30)
    cols = 5
    rows = 6
    per_sheet = cols * rows
    total_sheets = (len(annotated_thumbs) + per_sheet - 1) // per_sheet

    for s_idx in range(total_sheets):
        start_i = s_idx * per_sheet
        end_i = min(start_i + per_sheet, len(annotated_thumbs))
        sheet_tiles = annotated_thumbs[start_i:end_i]
        
        # Pad if less than per_sheet
        while len(sheet_tiles) < per_sheet:
            blank = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
            sheet_tiles.append(blank)

        grid_rows = []
        for r in range(rows):
            r_tiles = sheet_tiles[r * cols:(r + 1) * cols]
            grid_rows.append(np.hstack(r_tiles))
        
        sheet_img = np.vstack(grid_rows)
        out_sheet = sheet_dir / f"sheet_{s_idx+1:02d}.jpg"
        cv2.imwrite(str(out_sheet), sheet_img)
        print(f"Saved contact sheet: {out_sheet} (frames {start_i+1} to {end_i})")

    print("\nContact sheets generation finished successfully!")

if __name__ == "__main__":
    main()
