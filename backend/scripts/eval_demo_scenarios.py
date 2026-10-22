import os
import sys
import random
from pathlib import Path
import cv2
import numpy as np

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

from app.services.ai_service import AIService

DEMO_DIR = repo_root / "demo_images"
WEAPON_IMAGES = [
    ("pistol", DEMO_DIR / "pistol" / "pistol_1.jpg"),
    ("pistol", DEMO_DIR / "pistol" / "pistol_2.jpg"),
    ("pistol", DEMO_DIR / "pistol" / "pistol_3.jpg"),
    ("knife", DEMO_DIR / "knife" / "knife_1.jpg"),
    ("knife", DEMO_DIR / "knife" / "knife_2.jpg"),
    ("knife", DEMO_DIR / "knife" / "knife_3.jpg"),
]

FIGHT_IMAGES = [
    DEMO_DIR / "fight" / "fight_1.jpg",
    DEMO_DIR / "fight" / "fight_2.jpg",
    DEMO_DIR / "fight" / "fight_3.jpg",
]

def load_presenter_frame() -> np.ndarray:
    vid_path = repo_root / "webcam_tests" / "test_single_person.mp4"
    cap = cv2.VideoCapture(str(vid_path))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("Failed to read presenter frame from webcam_tests/test_single_person.mp4")
    return frame

def make_alone_frame(img: np.ndarray, target_w: int = 640, target_h: int = 480, scale_factor: float = 0.60, offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
    """Places image centered (with optional offset) on a dark canvas."""
    canvas = np.full((target_h, target_w, 3), (15, 15, 15), dtype=np.uint8)
    h, w = img.shape[:2]
    s = min(target_w / float(w), target_h / float(h)) * scale_factor
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    xo = max(0, min(target_w - nw, (target_w - nw) // 2 + offset_x))
    yo = max(0, min(target_h - nh, (target_h - nh) // 2 + offset_y))
    canvas[yo:yo+nh, xo:xo+nw] = resized
    return canvas

def make_attended_frame(weapon_img: np.ndarray, presenter_frame: np.ndarray) -> np.ndarray:
    """Pastes weapon image at chest height (40% scale) on presenter frame."""
    canvas = presenter_frame.copy()
    h, w = canvas.shape[:2]
    wh, ww = weapon_img.shape[:2]

    # Scale weapon to 40% of standard dimension
    scale = (w * 0.40) / float(max(ww, wh))
    nw, nh = max(1, int(round(ww * scale))), max(1, int(round(wh * scale)))
    resized = cv2.resize(weapon_img, (nw, nh), interpolation=cv2.INTER_AREA)

    # Presenter chest location in test_single_person.mp4: around x=260-380, y=220-340
    cx = 320
    cy = 280
    x1 = max(0, cx - nw // 2)
    y1 = max(0, cy - nh // 2)
    x2 = min(w, x1 + nw)
    y2 = min(h, y1 + nh)

    # Paste onto canvas
    canvas[y1:y2, x1:x2] = resized[:y2-y1, :x2-x1]
    return canvas

def make_fight_pasted_frame(fight_img: np.ndarray, presenter_frame: np.ndarray) -> np.ndarray:
    """Pastes fight image at 40% scale next to presenter in presenter frame."""
    canvas = presenter_frame.copy()
    h, w = canvas.shape[:2]
    fh, fw = fight_img.shape[:2]

    # Scale fight image to 40% of canvas width
    scale = (w * 0.40) / float(fw)
    nw, nh = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
    resized = cv2.resize(fight_img, (nw, nh), interpolation=cv2.INTER_AREA)

    # Paste on left side next to presenter (e.g. x=20, y=100)
    x1 = 20
    y1 = max(0, (h - nh) // 2)
    x2 = min(w, x1 + nw)
    y2 = min(h, y1 + nh)

    canvas[y1:y2, x1:x2] = resized[:y2-y1, :x2-x1]
    return canvas

def main():
    print("=" * 100)
    print("DEMO SCENARIOS EVALUATION: ITEMS 2 & 3")
    print("=" * 100)

    ai_service = AIService()
    presenter_frame = load_presenter_frame()

    # -------------------------------------------------------------
    # ITEM 2: WEAPON ATTENDED VS UNATTENDED & SHAKE TEST
    # -------------------------------------------------------------
    print("\n" + "=" * 100)
    print("ITEM 2: WEAPON EVALUATION (ATTENDED VS UNATTENDED)")
    print("=" * 100)

    results_item2 = []
    shake_results = []

    for category, img_path in WEAPON_IMAGES:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error loading {img_path}")
            continue

        target_sec = category.upper()

        # (i) Image alone (unattended) - run for 35 frames to observe frame 15 (MEDIUM) and frame 30 (HIGH)
        cam_alone = f"eval_weapon_alone_{img_path.stem}"
        ai_service.cleanup_camera(cam_alone)

        alone_frame = make_alone_frame(img, scale_factor=0.60)
        alone_fired = False
        alone_fired_frame = None
        alone_alert_type = "NONE"
        alone_level = "LOW"
        alone_high_frame = None

        for fidx in range(1, 36):
            out = ai_service.process_frame(alone_frame, cam_alone, "Weapon Alone", (0.0, 0.0))
            active_events = out[4]
            current_threat = out[2]

            matching = [e for e in active_events if target_sec in e.get("event_type", "").upper()]
            if matching and not alone_fired:
                alone_fired = True
                alone_fired_frame = fidx
                alone_alert_type = matching[0].get("event_type", "")
                alone_level = current_threat
            if matching and current_threat == "HIGH" and alone_high_frame is None:
                alone_high_frame = fidx
                alone_level = "HIGH"

        ai_service.cleanup_camera(cam_alone)

        # (ii) Image pasted on presenter frame (attended) - 15 frames
        cam_att = f"eval_weapon_att_{img_path.stem}"
        ai_service.cleanup_camera(cam_att)

        att_frame = make_attended_frame(img, presenter_frame)
        att_fired = False
        att_fired_frame = None
        att_alert_type = "NONE"
        att_level = "LOW"

        for fidx in range(1, 16):
            out = ai_service.process_frame(att_frame, cam_att, "Weapon Attended", (0.0, 0.0))
            active_events = out[4]
            current_threat = out[2]

            matching = [e for e in active_events if target_sec in e.get("event_type", "").upper() and not e.get("is_unattended", False)]
            if matching and not att_fired:
                att_fired = True
                att_fired_frame = fidx
                att_alert_type = matching[0].get("event_type", "")
                att_level = current_threat
                break

        ai_service.cleanup_camera(cam_att)

        results_item2.append({
            "image": img_path.name,
            "category": category,
            "alone_alert": alone_alert_type,
            "alone_fired_frame": alone_fired_frame if alone_fired else "-",
            "alone_level": alone_level,
            "alone_high_frame": alone_high_frame if alone_high_frame else "-",
            "att_alert": att_alert_type,
            "att_fired_frame": att_fired_frame if att_fired else "-",
            "att_level": att_level
        })

        # (iii) Handheld shake test on unattended weapon (5-10 px jitter per frame)
        cam_shake = f"eval_weapon_shake_{img_path.stem}"
        ai_service.cleanup_camera(cam_shake)

        random.seed(42)
        shake_fired_med = False
        shake_med_frame = None
        shake_fired_high = False
        shake_high_frame = None
        max_disp_seen = 0.0

        # Accumulated jitter: simulates hand drift/shake moving 5-10 px per frame
        cur_x = 0
        cur_y = 0
        for fidx in range(1, 36):
            step_dx = random.choice([-8, -6, -5, 5, 6, 8])
            step_dy = random.choice([-8, -6, -5, 5, 6, 8])
            cur_x += step_dx
            cur_y += step_dy

            shaken_frame = make_alone_frame(img, scale_factor=0.60, offset_x=cur_x, offset_y=cur_y)
            out = ai_service.process_frame(shaken_frame, cam_shake, "Weapon Shake", (0.0, 0.0))
            active_events = out[4]
            current_threat = out[2]

            # Inspect tracker displacement
            if cam_shake in ai_service.unattended_weapon_state:
                for k, s in ai_service.unattended_weapon_state[cam_shake].items():
                    disp = s.get("max_displacement", 0.0)
                    if disp > max_disp_seen:
                        max_disp_seen = disp

            matching = [e for e in active_events if target_sec in e.get("event_type", "").upper()]
            if matching and not shake_fired_med:
                shake_fired_med = True
                shake_med_frame = fidx
            if matching and current_threat == "HIGH" and not shake_fired_high:
                shake_fired_high = True
                shake_high_frame = fidx

        ai_service.cleanup_camera(cam_shake)

        shake_results.append({
            "image": img_path.name,
            "category": category,
            "max_displacement": round(max_disp_seen, 1),
            "promoted_medium": shake_fired_med,
            "medium_frame": shake_med_frame if shake_fired_med else "-",
            "promoted_high": shake_fired_high,
            "high_frame": shake_high_frame if shake_fired_high else "-"
        })

    # Print Item 2 Table
    print("\n" + "-" * 110)
    print("ITEM 2 (i & ii): ATTENDED VS UNATTENDED SUMMARY TABLE")
    print("-" * 110)
    print(f"{'Image':<18} | {'Alone Alert':<36} | {'Alone Fr':<9} | {'Alone Lvl':<10} | {'Attended Alert':<30} | {'Att Fr':<7} | {'Att Lvl':<8}")
    print("-" * 110)
    for r in results_item2:
        print(f"{r['image']:<18} | {r['alone_alert']:<36} | {str(r['alone_fired_frame']):<9} | {r['alone_level']:<10} | {r['att_alert']:<30} | {str(r['att_fired_frame']):<7} | {r['att_level']:<8}")
    print("-" * 110)

    # Print Shake Test Table
    print("\n" + "-" * 90)
    print("ITEM 2 (iii): UNATTENDED WEAPON WITH 5-10 PX/FRAME HANDHELD SHAKE")
    print("-" * 90)
    print(f"{'Image':<18} | {'Max Disp (px)':<15} | {'Promoted to MEDIUM?':<22} | {'Promoted to HIGH?':<18}")
    print("-" * 90)
    for sr in shake_results:
        print(f"{sr['image']:<18} | {sr['max_displacement']:<15} | {str(sr['promoted_medium']) + ' (Frame ' + str(sr['medium_frame']) + ')':<22} | {str(sr['promoted_high']) + ' (Frame ' + str(sr['high_frame']) + ')':<18}")
    print("-" * 90)


    # -------------------------------------------------------------
    # ITEM 3: FIGHT IMAGES PASTED NEXT TO PRESENTER
    # -------------------------------------------------------------
    print("\n" + "=" * 100)
    print("ITEM 3: FIGHT IMAGES PASTED NEXT TO PRESENTER FRAME")
    print("=" * 100)

    item3_summary = []

    for f_idx, f_path in enumerate(FIGHT_IMAGES, 1):
        f_img = cv2.imread(str(f_path))
        if f_img is None:
            continue

        cam_fight = f"eval_fight_pasted_{f_path.stem}"
        ai_service.cleanup_camera(cam_fight)

        pasted_frame = make_fight_pasted_frame(f_img, presenter_frame)

        frame_logs = []
        confirmed_frame = None

        for fidx in range(1, 16):
            out = ai_service.process_frame(pasted_frame, cam_fight, "Fight Pasted", (0.0, 0.0))
            active_events = out[4]
            current_threat = out[2]
            p_count = out[6]
            fight_res = out[8]

            raw_fight_prob = fight_res.get("fight_confidence", 0.0)
            raw_fight_bool = fight_res.get("raw_fight", False)
            raw_frames_in_buf = fight_res.get("raw_fight_frames", 0)

            matching_fight = [e for e in active_events if "FIGHT" in e.get("event_type", "").upper()]
            if matching_fight and confirmed_frame is None:
                confirmed_frame = fidx

            frame_logs.append({
                "frame": fidx,
                "persons": p_count,
                "raw_fight": raw_fight_bool,
                "raw_prob": raw_fight_prob,
                "buf_count": raw_frames_in_buf,
                "confirmed": confirmed_frame is not None
            })

        ai_service.cleanup_camera(cam_fight)

        avg_p = sum(fl["persons"] for fl in frame_logs) / len(frame_logs)
        avg_prob = sum(fl["raw_prob"] for fl in frame_logs) / len(frame_logs)

        item3_summary.append({
            "file": f_path.name,
            "avg_persons": avg_p,
            "raw_fight_prob": frame_logs[0]["raw_prob"] if frame_logs else 0.0,
            "frames_to_confirm": confirmed_frame if confirmed_frame is not None else "Not Confirmed",
            "frame_logs": frame_logs
        })

        print(f"\n--- Detailed Per-Frame Log for {f_path.name} ---")
        print(f"{'Frame':<6} | {'Persons':<8} | {'Raw Fight':<10} | {'Fight Conf':<12} | {'Buf Fight Frames':<18} | {'Confirmed?'}")
        print("-" * 75)
        for fl in frame_logs:
            print(f"{fl['frame']:<6} | {fl['persons']:<8} | {str(fl['raw_fight']):<10} | {fl['raw_prob']:<12.4f} | {fl['buf_count']:<18} | {fl['confirmed']}")

    print("\n" + "=" * 100)
    print("ITEM 3 SUMMARY TABLE: FIGHT IMAGES NEXT TO PRESENTER")
    print("=" * 100)
    print(f"{'Fight Image':<20} | {'Person Count':<14} | {'Raw Fight Prob':<16} | {'Frames Until Confirmation'}")
    print("-" * 100)
    for s in item3_summary:
        print(f"{s['file']:<20} | {s['avg_persons']:<14.1f} | {s['raw_fight_prob']:<16.4f} | {s['frames_to_confirm']}")
    print("=" * 100)

if __name__ == "__main__":
    main()
