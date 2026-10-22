import os
import sys
import math
from pathlib import Path
import cv2
import numpy as np

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

from app.services.ai_service import AIService

DEMO_DIR = repo_root / "demo_images"
IMAGE_FILES = [
    ("pistol", DEMO_DIR / "pistol" / "pistol_1.jpg"),
    ("pistol", DEMO_DIR / "pistol" / "pistol_2.jpg"),
    ("pistol", DEMO_DIR / "pistol" / "pistol_3.jpg"),
    ("knife", DEMO_DIR / "knife" / "knife_1.jpg"),
    ("knife", DEMO_DIR / "knife" / "knife_2.jpg"),
    ("knife", DEMO_DIR / "knife" / "knife_3.jpg"),
    ("fight", DEMO_DIR / "fight" / "fight_1.jpg"),
    ("fight", DEMO_DIR / "fight" / "fight_2.jpg"),
    ("fight", DEMO_DIR / "fight" / "fight_3.jpg"),
    ("accident", DEMO_DIR / "accident" / "accident_1.jpg"),
    ("accident", DEMO_DIR / "accident" / "accident_2.jpg"),
    ("accident", DEMO_DIR / "accident" / "accident_3.jpg"),
]

def make_aspect_fit_frame(img: np.ndarray, target_w: int = 640, target_h: int = 480) -> np.ndarray:
    """Aspect-fit the image onto a dark 640x480 frame."""
    canvas = np.full((target_h, target_w, 3), (15, 15, 15), dtype=np.uint8)
    h, w = img.shape[:2]
    s = min(target_w / float(w), target_h / float(h))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LANCZOS4)
    xo = (target_w - nw) // 2
    yo = (target_h - nh) // 2
    canvas[yo:yo+nh, xo:xo+nw] = resized
    return canvas

def add_gaussian_noise(frame: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Adds fresh Gaussian noise (sigma in 1-2) to the frame."""
    noise = np.random.normal(0, sigma, frame.shape).astype(np.float32)
    noisy = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def main():
    print("=" * 110)
    print("HELD-STILL TEST: 200 FRAMES PER DEMO IMAGE WITH FRESH GAUSSIAN NOISE (SIGMA 1-2)")
    print("=" * 110)

    ai_service = AIService()
    all_results = []

    np.random.seed(42)

    for cat, img_path in IMAGE_FILES:
        if not img_path.exists():
            print(f"File missing: {img_path}")
            continue

        raw_img = cv2.imread(str(img_path))
        if raw_img is None:
            print(f"Could not load image: {img_path}")
            continue

        base_frame = make_aspect_fit_frame(raw_img, 640, 480)
        base_brightness = float(np.mean(base_frame))
        base_stddev = float(np.std(base_frame))

        cam_id = f"held_still_{img_path.stem}"
        ai_service.cleanup_camera(cam_id)

        # Track first appearance of any distinct event or health status
        first_seen_events = {}  # {event_type: first_frame_idx}
        seen_event_levels = {}  # {event_type: max_threat_level}
        freeze_frames = []
        tamper_frames = []
        low_light_frames = []
        final_threat_level = "LOW"
        final_score = 0
        detections_seen = set()

        mse_list = []

        for fidx in range(1, 201):
            # Sigma between 1.0 and 2.0 (averaging 1.5)
            sigma = 1.0 + (fidx % 11) * 0.10
            noisy_frame = add_gaussian_noise(base_frame, sigma=sigma)

            # Measure brightness & check for low light
            b_val = float(np.mean(noisy_frame))
            if b_val < 40.0:
                low_light_frames.append(fidx)

            # Run through real process_frame
            out = ai_service.process_frame(
                noisy_frame,
                cam_id,
                f"Held Still {img_path.stem}",
                (0.0, 0.0)
            )
            # out = (annotated_frame, detections, threat_level, total_score, active_events, correlated_events, person_count, xai, fight_res, acc_res)
            dets = out[1]
            t_level = out[2]
            t_score = out[3]
            active_events = out[4]
            correlated_events = out[5]
            p_count = out[6]
            fight_res = out[8]
            acc_res = out[9]

            final_threat_level = t_level
            final_score = t_score

            for d in dets:
                detections_seen.add(f"{d['class']} ({d.get('threat_level', 'LOW')})")

            # Check active events
            for e in active_events:
                etype = e.get("event_type", "UNKNOWN")
                elevel = e.get("threat_level", "LOW")
                if etype not in first_seen_events:
                    first_seen_events[etype] = fidx
                    seen_event_levels[etype] = elevel

            # Check health buffers directly
            hbuf = ai_service.health_buffers.get(cam_id, {})
            if hbuf.get("freeze_counter", 0) > 40:
                if "CAMERA FREEZE DETECTED" not in first_seen_events:
                    first_seen_events["CAMERA FREEZE DETECTED"] = fidx
                    seen_event_levels["CAMERA FREEZE DETECTED"] = "MEDIUM"

            # Check if unattended weapon status exists in unattended_weapon_state
            if cam_id in ai_service.unattended_weapon_state:
                for k, w_state in ai_service.unattended_weapon_state[cam_id].items():
                    if w_state.get("is_unattended", False):
                        u_frames = w_state.get("consecutive_unattended_frames", 0)
                        raw_c = w_state.get("raw_cls", "")
                        status_tag = f"UNATTENDED_TRACKING ({raw_c.upper()})"
                        if status_tag not in first_seen_events:
                            first_seen_events[status_tag] = fidx
                            seen_event_levels[status_tag] = "INFO"

        ai_service.cleanup_camera(cam_id)

        all_results.append({
            "category": cat,
            "filename": img_path.name,
            "base_brightness": round(base_brightness, 1),
            "first_seen_events": first_seen_events,
            "seen_event_levels": seen_event_levels,
            "final_threat_level": final_threat_level,
            "final_score": final_score,
            "low_light_first_frame": low_light_frames[0] if low_light_frames else None,
            "detections_seen": sorted(list(detections_seen))
        })

    # Print Report
    print("\n" + "=" * 125)
    print("HELD-STILL 200-FRAME TEST RESULTS (PER-IMAGE BREAKDOWN)")
    print("=" * 125)

    for r in all_results:
        print(f"\n📁 [{r['category'].upper()}] {r['filename']} (Base Brightness: {r['base_brightness']}, Final Threat: {r['final_threat_level']} / Score: {r['final_score']})")
        print(f"   Detections Encountered: {', '.join(r['detections_seen']) if r['detections_seen'] else 'None'}")
        print(f"   Distinct Alerts & Camera-Health Statuses:")
        if r['first_seen_events']:
            for ev, fr in r['first_seen_events'].items():
                lvl = r['seen_event_levels'].get(ev, "N/A")
                print(f"     - [Frame {fr:3d}] {ev} (Level: {lvl})")
        else:
            print("     - Zero events fired")

        if r['low_light_first_frame'] is not None:
            print(f"     - [Health] Low Light (Mean < 40): Yes (from Frame {r['low_light_first_frame']}, Mean: {r['base_brightness']})")
        else:
            print(f"     - [Health] Low Light (Mean < 40): No (Well-lit, Mean: {r['base_brightness']})")

    # Summary Table
    print("\n" + "=" * 125)
    print(f"{'Image':<20} | {'Category':<10} | {'Alerts Fired (First Frame Seen)':<65} | {'Final Level'}")
    print("-" * 125)
    for r in all_results:
        ev_strs = []
        for ev, fr in r['first_seen_events'].items():
            ev_strs.append(f"{ev} (F{fr})")
        ev_display = "; ".join(ev_strs) if ev_strs else "None"
        if len(ev_display) > 63:
            ev_display = ev_display[:60] + "..."
        print(f"{r['filename']:<20} | {r['category']:<10} | {ev_display:<65} | {r['final_threat_level']}")
    print("=" * 125)

if __name__ == "__main__":
    main()
