import os
import sys
import csv
import time
import math
import shutil
from pathlib import Path
import cv2
import numpy as np

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.services.ai_service import AIService

DEMO_DIR = repo_root / "demo_images"
CATEGORIES = ["pistol", "knife", "fight", "accident"]

def make_phone_ready(img: np.ndarray, max_dim: int = 1600) -> np.ndarray:
    """
    Phone-ready form: longest side scaled to about 1600 px using Lanczos interpolation,
    aspect ratio unchanged, no cropping, no added borders.
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest == 0:
        return img
    scale = float(max_dim) / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_LANCZOS4 if scale >= 1.0 else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp)

def build_simulation_frames(phone_img: np.ndarray) -> dict:
    """
    Creates 4 phone-in-front-of-webcam simulation frames (640x480):
      (a) the image full frame (aspect-preserved fit inside 640x480 centered on dark background)
      (b) scaled to 60% centered on a dark 640x480 frame
      (c) 40% centered on a dark 640x480 frame
      (d) 60% with Gaussian blur sigma 1.5 and brightness -25
    """
    h, w = phone_img.shape[:2]
    DARK_BG = (15, 15, 15)

    # (a) Full frame fit into 640x480
    frame_a = np.full((480, 640, 3), DARK_BG, dtype=np.uint8)
    s_a = min(640.0 / w, 480.0 / h)
    wa, ha = max(1, int(round(w * s_a))), max(1, int(round(h * s_a)))
    res_a = cv2.resize(phone_img, (wa, ha), interpolation=cv2.INTER_AREA if s_a < 1.0 else cv2.INTER_LANCZOS4)
    xoa = (640 - wa) // 2
    yoa = (480 - ha) // 2
    frame_a[yoa:yoa+ha, xoa:xoa+wa] = res_a

    # (b) Scaled to 60% centered on dark 640x480 frame
    frame_b = np.full((480, 640, 3), DARK_BG, dtype=np.uint8)
    s_b = min(640.0 / w, 480.0 / h) * 0.60
    wb, hb = max(1, int(round(w * s_b))), max(1, int(round(h * s_b)))
    res_b = cv2.resize(phone_img, (wb, hb), interpolation=cv2.INTER_AREA if s_b < 1.0 else cv2.INTER_LANCZOS4)
    xob = (640 - wb) // 2
    yob = (480 - hb) // 2
    frame_b[yob:yob+hb, xob:xob+wb] = res_b

    # (c) Scaled to 40% centered on dark 640x480 frame
    frame_c = np.full((480, 640, 3), DARK_BG, dtype=np.uint8)
    s_c = min(640.0 / w, 480.0 / h) * 0.40
    wc, hc = max(1, int(round(w * s_c))), max(1, int(round(h * s_c)))
    res_c = cv2.resize(phone_img, (wc, hc), interpolation=cv2.INTER_AREA if s_c < 1.0 else cv2.INTER_LANCZOS4)
    xoc = (640 - wc) // 2
    yoc = (480 - hc) // 2
    frame_c[yoc:yoc+hc, xoc:xoc+wc] = res_c

    # (d) 60% with Gaussian blur sigma 1.5 and brightness -25
    frame_d = cv2.GaussianBlur(frame_b, (0, 0), sigmaX=1.5, sigmaY=1.5)
    frame_d = np.clip(frame_d.astype(np.int16) - 25, 0, 255).astype(np.uint8)

    return {
        "a": frame_a,
        "b": frame_b,
        "c": frame_c,
        "d": frame_d
    }

def evaluate_image_simulations(ai_service: AIService, img_input, category: str, stem: str = "img") -> dict:
    """
    Evaluates a phone-ready image under 4 simulations repeated 15 times per simulation.
    Checks if the RIGHT alert fires for the category.
    """
    if isinstance(img_input, np.ndarray):
        img = img_input
    else:
        stem = getattr(img_input, "stem", stem)
        img = None
        for _ in range(5):
            try:
                with open(str(img_input), "rb") as f:
                    data = f.read()
                if data:
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        break
            except Exception:
                pass
            time.sleep(0.05)
    if img is None:
        return {
            "sims_passed": 0,
            "sim_details": {},
            "alert_type": "NONE",
            "frame_fired": "-",
            "threat_level": "LOW",
            "max_confidence": 0.0,
            "person_count": 0
        }

    sim_frames = build_simulation_frames(img)
    target_keyword = {
        "pistol": "PISTOL",
        "knife": "KNIFE",
        "fight": "FIGHT",
        "accident": "ACCIDENT"
    }[category]

    sims_passed = 0
    sim_details = {}
    best_alert_type = "NONE"
    earliest_frame_fired = 999
    max_threat_level = "LOW"
    max_confidence = 0.0
    recorded_person_count = 0

    threat_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

    for sim_key in ["a", "b", "c", "d"]:
        sim_frame = sim_frames[sim_key]
        cam_id = f"eval_{category}_{stem}_{sim_key}"
        ai_service.cleanup_camera(cam_id)

        sim_fired = False
        sim_fired_frame = None
        sim_alert_type = "NONE"
        sim_threat = "LOW"
        sim_conf = 0.0
        sim_persons = 0

        for frame_idx in range(1, 16):
            out = ai_service.process_frame(
                sim_frame,
                cam_id,
                f"Demo {category.capitalize()}",
                (0.0, 0.0)
            )
            # out = (annotated_frame, detections, threat_level, total_score, active_events, correlated_events, person_count, xai, fight_res, acc_res)
            active_events = out[4]
            current_threat = out[2]
            current_persons = out[6]

            # Check for right alert
            matching_events = [e for e in active_events if target_keyword in e.get("event_type", "").upper()]
            if matching_events:
                sim_fired = True
                sim_fired_frame = frame_idx
                sim_alert_type = matching_events[0].get("event_type", "")
                sim_threat = current_threat
                sim_conf = max(e.get("confidence", 0.0) for e in matching_events)
                sim_persons = current_persons
                break

            # Early break optimization on frame 1: if static frame has zero candidate evidence, future identical frames cannot fire
            if frame_idx == 1:
                dets = out[1]
                has_cand_evidence = False
                if category in ("pistol", "knife"):
                    has_cand_evidence = any(target_keyword in d.get("class", "").upper() for d in dets)
                elif category == "accident":
                    has_cand_evidence = (out[9].get("accident_confidence", 0.0) >= 0.70 or len(out[9].get("boxes", [])) > 0)
                elif category == "fight":
                    has_cand_evidence = (out[8].get("raw_fight", False) and current_persons >= 2)
                if not has_cand_evidence:
                    break

        ai_service.cleanup_camera(cam_id)

        sim_details[sim_key] = {
            "passed": sim_fired,
            "frame_fired": sim_fired_frame,
            "alert_type": sim_alert_type,
            "threat_level": sim_threat,
            "confidence": sim_conf,
            "person_count": sim_persons
        }

        if sim_fired:
            sims_passed += 1
            if sim_fired_frame < earliest_frame_fired:
                earliest_frame_fired = sim_fired_frame
                best_alert_type = sim_alert_type
                recorded_person_count = sim_persons
            if threat_order.get(sim_threat, 0) > threat_order.get(max_threat_level, 0):
                max_threat_level = sim_threat
            if sim_conf > max_confidence:
                max_confidence = sim_conf

    return {
        "sims_passed": sims_passed,
        "sim_details": sim_details,
        "alert_type": best_alert_type if sims_passed > 0 else "NONE",
        "frame_fired": earliest_frame_fired if sims_passed > 0 else "-",
        "threat_level": max_threat_level,
        "max_confidence": round(max_confidence, 4),
        "person_count": recorded_person_count
    }

def gather_candidates(ai_service: AIService) -> dict:
    """
    Gathers 40 candidate image paths per category:
      - Pistol: 20 from Dataaset2 + 20 from Sohas test set
      - Knife: 20 from Dataaset2 + 20 from Sohas test set
      - Fight: 40 from fight_frames (train/val Fight) with >= 2 persons detected (preferring 2-4 persons)
      - Accident: 40 from acc/test/images
    """
    print("\n[demo_prep] Gathering candidate images...")
    candidates = {"pistol": [], "knife": [], "fight": [], "accident": []}

    # 1. Pistol & Knife from Sohas test set
    sohas_img = repo_root / "OD-WeaponDetection" / "Weapons and similar handled objects" / "Sohas_weapon-Detection-YOLOv5" / "obj_train_data" / "images" / "test"
    sohas_lbl = repo_root / "OD-WeaponDetection" / "Weapons and similar handled objects" / "Sohas_weapon-Detection-YOLOv5" / "obj_train_data" / "labels" / "test"

    sohas_pistols = []
    sohas_knives = []

    if sohas_lbl.exists():
        for lf in sorted(sohas_lbl.glob("*.txt")):
            try:
                with open(lf, "r", encoding="utf-8") as f:
                    classes = [int(l.split()[0]) for l in f if l.strip()]
            except Exception:
                continue
            img_p = sohas_img / f"{lf.stem}.jpg"
            if not img_p.exists():
                continue
            if 0 in classes and len(sohas_pistols) < 25:
                sohas_pistols.append(img_p)
            if 2 in classes and len(sohas_knives) < 25:
                sohas_knives.append(img_p)

    # 2. Pistol & Knife from Dataaset2 (screened for actual detections)
    d2_pistol_dir = repo_root / "Dataaset2" / "Pistol detection" / "Weapons"
    d2_knife_dir = repo_root / "Dataaset2" / "Knife_detection" / "Images"

    print("[demo_prep] Screening Dataaset2 pistols & knives...")
    d2_pistols = []
    if d2_pistol_dir.exists():
        for p in sorted(list(d2_pistol_dir.glob("*.jpg"))):
            im = cv2.imread(str(p))
            if im is None:
                continue
            res = ai_service.model(im, verbose=False)
            if res and res[0].boxes is not None:
                if any(ai_service.model.names[int(b.cls[0])].lower() == "pistol" and float(b.conf[0]) >= 0.45 for b in res[0].boxes):
                    d2_pistols.append(p)
                    if len(d2_pistols) >= 25:
                        break

    d2_knives = []
    if d2_knife_dir.exists():
        for p in sorted(list(d2_knife_dir.glob("*.jpg"))):
            im = cv2.imread(str(p))
            if im is None:
                continue
            res = ai_service.model(im, verbose=False)
            if res and res[0].boxes is not None:
                if any(ai_service.model.names[int(b.cls[0])].lower() == "knife" and float(b.conf[0]) >= 0.50 for b in res[0].boxes):
                    d2_knives.append(p)
                    if len(d2_knives) >= 25:
                        break

    # Combine to get 40 pistol candidates (20 Sohas + 20 D2)
    candidates["pistol"] = (sohas_pistols[:20] + d2_pistols[:20])[:40]

    # Combine to get 40 knife candidates (20 Sohas + 20 D2)
    candidates["knife"] = (sohas_knives[:20] + d2_knives[:20])[:40]

    # 3. Accident candidates from acc/test/images (screened for conf >= 0.70)
    print("[demo_prep] Screening accident candidates...")
    acc_dir = repo_root / "acc" / "test" / "images"
    acc_selected = []
    if acc_dir.exists():
        for p in sorted(list(acc_dir.glob("*.jpg"))):
            im = cv2.imread(str(p))
            if im is None:
                continue
            res = ai_service.accident_model(im, verbose=False)
            if res and res[0].boxes is not None:
                if any(float(b.conf[0]) >= 0.70 for b in res[0].boxes):
                    acc_selected.append(p)
                    if len(acc_selected) >= 40:
                        break
    candidates["accident"] = acc_selected[:40]

    # 4. Fight candidates from fight_frames (preferring 2 clearly separate people, 2-4 persons, and fight conf >= 0.75)
    fight_val_dir = repo_root / "fight_frames" / "val" / "Fight"
    fight_train_dir = repo_root / "fight_frames" / "train" / "Fight"
    all_fight_files = []
    if fight_val_dir.exists():
        all_fight_files.extend(sorted(list(fight_val_dir.glob("*.jpg"))))
    if fight_train_dir.exists():
        all_fight_files.extend(sorted(list(fight_train_dir.glob("*.jpg"))))

    print("[demo_prep] Screening fight frames for 2 clearly separate people & high fight conf...")
    fight_selected = []
    for fp in all_fight_files:
        im = cv2.imread(str(fp))
        if im is None:
            continue
        res_p = ai_service.person_model(im, verbose=False)
        if not res_p or res_p[0].boxes is None:
            continue
        p_boxes = [b for b in res_p[0].boxes if float(b.conf[0]) >= 0.35]
        if 2 <= len(p_boxes) <= 4:
            res_f = ai_service.fight_model.predict([im], verbose=False)
            if res_f and res_f[0].probs is not None:
                top1_name = res_f[0].names.get(int(res_f[0].probs.top1), "").lower()
                top1_conf = float(res_f[0].probs.top1conf)
                if top1_name == "fight" and top1_conf >= 0.75:
                    # Check that person detection survives downscaling (sim b: 384x288)
                    im_small = cv2.resize(im, (384, 288))
                    res_ps = ai_service.person_model(im_small, verbose=False)
                    if res_ps and res_ps[0].boxes is not None:
                        ps_boxes = [b for b in res_ps[0].boxes if float(b.conf[0]) >= 0.35]
                        if len(ps_boxes) >= 2:
                            fight_selected.append(fp)
                            if len(fight_selected) >= 40:
                                break

    candidates["fight"] = fight_selected[:40]

    # If fewer than 40, fallback to any >=2 persons
    if len(fight_selected) < 40:
        for fp in all_fight_files:
            if fp in fight_selected:
                continue
            im = cv2.imread(str(fp))
            if im is None:
                continue
            res_p = ai_service.person_model(im, verbose=False)
            if res_p and res_p[0].boxes is not None:
                p_boxes = [b for b in res_p[0].boxes if float(b.conf[0]) >= 0.35]
                if len(p_boxes) >= 2:
                    fight_selected.append(fp)
                    if len(fight_selected) >= 40:
                        break

    candidates["fight"] = fight_selected[:40]

    for cat in CATEGORIES:
        print(f"  - {cat}: {len(candidates[cat])} candidates")

    return candidates

def generate_contact_sheet(image_paths: list, labels: list, output_path: Path):
    """
    Creates a single contact sheet of all 12 images (3 cols x 4 rows),
    each thumbnail labeled with its file name.
    """
    thumb_w, thumb_h = 480, 360
    header_h = 40
    cell_w = thumb_w
    cell_h = thumb_h + header_h

    cols = 3
    rows = 4
    sheet = np.full((rows * cell_h, cols * cell_w, 3), (25, 25, 28), dtype=np.uint8)

    for idx, (img_path, label) in enumerate(zip(image_paths, labels)):
        r = idx // cols
        c = idx % cols
        x_start = c * cell_w
        y_start = r * cell_h

        img = cv2.imread(str(img_path))
        if img is not None:
            # Letterbox into thumb_w x thumb_h
            h, w = img.shape[:2]
            scale = min(thumb_w / float(w), thumb_h / float(h))
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

            thumb_canvas = np.full((thumb_h, thumb_w, 3), (18, 18, 20), dtype=np.uint8)
            xo = (thumb_w - nw) // 2
            yo = (thumb_h - nh) // 2
            thumb_canvas[yo:yo+nh, xo:xo+nw] = resized

            # Place header
            header = np.full((header_h, cell_w, 3), (35, 35, 40), dtype=np.uint8)
            cv2.putText(
                header,
                label,
                (15, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            sheet[y_start:y_start+header_h, x_start:x_start+cell_w] = header
            sheet[y_start+header_h:y_start+cell_h, x_start:x_start+cell_w] = thumb_canvas

            # Outer border around cell
            cv2.rectangle(
                sheet,
                (x_start, y_start),
                (x_start + cell_w - 1, y_start + cell_h - 1),
                (60, 60, 65),
                1
            )

    cv2.imwrite(str(output_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"[demo_prep] Saved contact sheet: {output_path}")

def main():
    print("=" * 80)
    print("DEMO IMAGE PREPARATION & SIMULATION RANKING (demo_prep.py)")
    print("=" * 80)
    print("NOTE: This script performs phone-in-front-of-webcam simulation ranking.")
    print("      This is only a ranking aid, not proof.")
    print("=" * 80)

    # 1. Create directory structure
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    for cat in CATEGORIES:
        (DEMO_DIR / cat).mkdir(parents=True, exist_ok=True)

    cand_cache = DEMO_DIR / "_candidate_cache"
    cand_cache.mkdir(parents=True, exist_ok=True)

    ai_service = AIService()

    # 2. Gather candidates
    candidates = gather_candidates(ai_service)

    # 3. For each category, convert candidates to phone-ready format, then rank via the 4 simulations
    ranked_results = {}

    for cat in CATEGORIES:
        print(f"\n[demo_prep] Evaluating candidates for category: {cat.upper()}...")
        cat_cands = candidates[cat]
        scored_candidates = []

        cat_cache = cand_cache / cat
        cat_cache.mkdir(parents=True, exist_ok=True)

        for idx, src_path in enumerate(cat_cands, 1):
            raw_img = cv2.imread(str(src_path))
            if raw_img is None:
                continue

            # Convert to phone-ready format
            phone_ready_img = make_phone_ready(raw_img, max_dim=1600)

            # Save in phone-ready form in candidate cache with NO spaces in file name
            safe_stem = src_path.stem.replace(" ", "_")
            phone_ready_path = cat_cache / f"cand_{idx}_{safe_stem}.jpg"
            cv2.imwrite(str(phone_ready_path), phone_ready_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            # Run simulations on exact phone-ready image
            eval_res = evaluate_image_simulations(ai_service, phone_ready_img, cat, stem=f"{src_path.stem}_{idx}")
            eval_res["src_path"] = src_path
            eval_res["phone_ready_img"] = phone_ready_img
            scored_candidates.append(eval_res)

            print(f"  [{cat} {idx:02d}/{len(cat_cands):02d}] {src_path.name} -> Passed: {eval_res['sims_passed']}/4 | "
                  f"Fired Frame: {eval_res['frame_fired']} | Conf: {eval_res['max_confidence']:.4f} | "
                  f"Level: {eval_res['threat_level']} | Persons: {eval_res['person_count']}")

        # Rank candidates:
        # Primary: sims_passed descending
        # Secondary: max_confidence descending
        # Tertiary: frame_fired ascending (lower is faster)
        def sort_key(c):
            ff = c["frame_fired"] if isinstance(c["frame_fired"], (int, float)) else 999
            return (-c["sims_passed"], -c["max_confidence"], ff)

        scored_candidates.sort(key=sort_key)
        ranked_results[cat] = scored_candidates

    # 4. Save top 3 per category to demo_images/<cat>/<cat>_<rank>.jpg
    print("\n" + "=" * 80)
    print("SAVING WINNING DEMO IMAGES (Top 3 per category)")
    print("=" * 80)

    top_12_records = []
    weak_categories = []

    for cat in CATEGORIES:
        top_3 = ranked_results[cat][:3]
        passed_ge_3 = sum(1 for c in top_3 if c["sims_passed"] >= 3)
        if passed_ge_3 < 3:
            weak_categories.append((cat, passed_ge_3, [c["sims_passed"] for c in top_3]))

        for rank_idx, cand in enumerate(top_3, 1):
            dest_filename = f"{cat}_{rank_idx}.jpg"
            dest_path = DEMO_DIR / cat / dest_filename

            # Save to destination
            phone_img = cand["phone_ready_img"]
            cv2.imwrite(str(dest_path), phone_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            # IMPORTANT: Re-run all 4 simulations directly on the final phone image
            final_eval = evaluate_image_simulations(ai_service, phone_img, cat, stem=f"final_{dest_filename}")

            record = {
                "file_name": str(Path(cat) / dest_filename),
                "full_path": str(dest_path.resolve()),
                "source_dataset_path": str(cand["src_path"].resolve()),
                "category": cat,
                "rank": rank_idx,
                "simulations_passed": f"{final_eval['sims_passed']}/4",
                "alert_type": final_eval["alert_type"],
                "frame_fired": final_eval["frame_fired"],
                "threat_level": final_eval["threat_level"],
                "max_confidence": final_eval["max_confidence"],
                "person_count": final_eval["person_count"]
            }
            top_12_records.append(record)
            print(f"Saved: {dest_path} (from {cand['src_path'].name})")

    # Clean up candidate cache
    try:
        shutil.rmtree(cand_cache)
    except Exception as e:
        print(f"Notice: could not delete temp cache: {e}")

    # 5. Write demo_images/manifest.csv
    manifest_path = DEMO_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file name",
            "source dataset path",
            "category",
            "simulations passed (of 4)",
            "alert type",
            "frame fired",
            "threat level",
            "max confidence",
            "person count"
        ])
        for r in top_12_records:
            writer.writerow([
                r["file_name"],
                r["source_dataset_path"],
                r["category"],
                r["simulations_passed"],
                r["alert_type"],
                r["frame_fired"],
                r["threat_level"],
                r["max_confidence"],
                r["person_count"]
            ])
    print(f"\n[demo_prep] Written manifest: {manifest_path}")

    # 6. Write demo_images/README.txt
    readme_path = DEMO_DIR / "README.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("AEGIS AI SURVEILLANCE - DEMO IMAGES GUIDE\n")
        f.write("=" * 80 + "\n\n")
        f.write("NOTE: This is only a ranking aid, not proof. Live camera lighting, reflection, and distance\n")
        f.write("      will affect recognition. Always check framing during presentation.\n\n")
        f.write("PRESENTATION ORDER (SHOW IN THIS EXACT SEQUENCE):\n")
        f.write("--------------------------------------------------\n")
        f.write("1. knife\\knife_1.jpg\n")
        f.write("   - What to show: Primary knife image on phone screen.\n")
        f.write("   - Expected Alert: WEAPON DETECTED (KNIFE) (HIGH) or UNATTENDED WEAPON DETECTED (KNIFE)\n\n")
        f.write("2. pistol\\pistol_1.jpg\n")
        f.write("   - What to show: Primary pistol image on phone screen.\n")
        f.write("   - Expected Alert: WEAPON DETECTED (PISTOL) (CRITICAL)\n")
        f.write("   - CRITICAL OPERATOR NOTE: Pistol images work best when I am in the shot so the alert is CRITICAL!\n")
        f.write("     If no person is detected in frame, system treats it as unattended weapon (starting at MEDIUM).\n")
        f.write("     Holding the phone near your body/chest immediately qualifies attended status -> CRITICAL alert!\n\n")
        f.write("3. accident\\accident_1.jpg\n")
        f.write("   - What to show: Primary vehicle collision image.\n")
        f.write("   - Expected Alert: ACCIDENT DETECTED (HIGH)\n")
        f.write("   - Confirms within 2 frames.\n\n")
        f.write("4. fight\\fight_1.jpg\n")
        f.write("   - What to show: Primary physical altercation frame showing two distinct participants.\n")
        f.write("   - Expected Alert: FIGHT STARTED / FIGHT ONGOING (HIGH)\n")
        f.write("   - Requires temporal buffer (6/10 frames) and >=2 persons in scene.\n\n")
        f.write("BACKUPS (IF FIRST IMAGE FAILS TO CATCH ANGLE/GLARE):\n")
        f.write("--------------------------------------------------\n")
        f.write("Knife Backups:    knife\\knife_2.jpg, knife\\knife_3.jpg\n")
        f.write("Pistol Backups:   pistol\\pistol_2.jpg, pistol\\pistol_3.jpg\n")
        f.write("Accident Backups: accident\\accident_2.jpg, accident\\accident_3.jpg\n")
        f.write("Fight Backups:    fight\\fight_2.jpg, fight\\fight_3.jpg\n\n")
        f.write("ALL 12 IMAGES SPECIFICATION:\n")
        f.write("- Format: JPEG Quality 95\n")
        f.write("- Scaling: Longest side scaled to ~1600 px (Lanczos interpolation)\n")
        f.write("- Geometry: Aspect ratio strictly unchanged, no cropping, no added borders\n")
        f.write("- Ranking: Evaluated directly on the phone-ready files across 4 webcam distance/lighting simulations.\n")
    print(f"[demo_prep] Written README: {readme_path}")

    # 7. Write demo_images/_all_12.jpg contact sheet
    # Ordering for contact sheet:
    # knife_1, pistol_1, accident_1
    # fight_1, knife_2, pistol_2
    # accident_2, fight_2, knife_3
    # pistol_3, accident_3, fight_3
    # Or in category order: pistol 1-3, knife 1-3, fight 1-3, accident 1-3
    sheet_paths = []
    sheet_labels = []
    for r in top_12_records:
        sheet_paths.append(Path(r["full_path"]))
        sheet_labels.append(f"{r['file_name']} ({r['simulations_passed']})")

    contact_sheet_path = DEMO_DIR / "_all_12.jpg"
    generate_contact_sheet(sheet_paths, sheet_labels, contact_sheet_path)

    # 8. Print the 12 full file paths and manifest table
    print("\n" + "=" * 100)
    print("THE 12 DEMO IMAGE FILE PATHS:")
    print("=" * 100)
    for r in top_12_records:
        print(f"  {r['file_name']:<25} -> {r['full_path']}")

    print("\n" + "=" * 100)
    print("MANIFEST TABLE:")
    print("=" * 100)
    print(f"{'File Name':<20} | {'Passed':<8} | {'Alert Type':<32} | {'Frame':<6} | {'Level':<9} | {'Max Conf':<9} | {'Persons':<8}")
    print("-" * 100)
    for r in top_12_records:
        print(f"{r['file_name']:<20} | {r['simulations_passed']:<8} | {r['alert_type']:<32} | {str(r['frame_fired']):<6} | {r['threat_level']:<9} | {r['max_confidence']:<9.4f} | {r['person_count']:<8}")
    print("=" * 100)

    # 9. Weak category report
    if weak_categories:
        print("\n[WARNING] WEAK CATEGORIES DETECTED (Fewer than 3 images passed >= 3/4 simulations):")
        for cat, count, passed_list in weak_categories:
            print(f"  - Category '{cat}': only {count} images passed >= 3 of 4 simulations. Passed scores: {passed_list}")
    else:
        print("\n✅ All categories have 3 images passing at least 3 of 4 simulations.")

if __name__ == "__main__":
    main()
