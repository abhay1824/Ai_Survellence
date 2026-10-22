import os
import sys
from pathlib import Path
import cv2

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

from app.services.ai_service import AIService
ai_service = AIService()

FIGHT_DIR = repo_root / "RWF-2000" / "Fight"
NONFIGHT_DIR = repo_root / "RWF-2000" / "NonFight"

def run_eval(clip_paths, is_fight=True):
    results = []
    total_detected = 0
    confirm_frames = []

    label = "FIGHT" if is_fight else "NON-FIGHT"
    print(f"\n==========================================")
    print(f"RUNNING REAL AIService ON {len(clip_paths)} {label} CLIPS")
    print(f"==========================================")

    for idx, clip_path in enumerate(clip_paths):
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            print(f"Failed to open {clip_path.name}")
            continue

        camera_id = f"eval_cam_{'f' if is_fight else 'nf'}_{idx}_{clip_path.stem}"
        # Ensure fresh state
        ai_service.cleanup_camera(camera_id)
        if camera_id in ai_service.fight_buffers:
            del ai_service.fight_buffers[camera_id]
        if camera_id in ai_service.person_gate_buffers:
            del ai_service.person_gate_buffers[camera_id]
        if camera_id in ai_service.fight_duration:
            del ai_service.fight_duration[camera_id]

        frame_idx = 0
        first_confirm = None
        ever_detected = False

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            out = ai_service.process_frame(
                frame=frame,
                camera_id=camera_id,
                camera_name=f"Cam {camera_id}",
                gps_coords=(0.0, 0.0),
                zones=[]
            )
            # out: (annotated_frame, detections, threat_level, total_score, active_events, correlated_events, person_count, xai_explanation, fight_result, accident_result)
            fight_result = out[8]
            is_detected = fight_result.get("fight_detected", False)
            if is_detected and not ever_detected:
                ever_detected = True
                first_confirm = frame_idx

            frame_idx += 1

        cap.release()
        ai_service.cleanup_camera(camera_id)

        if ever_detected:
            total_detected += 1
            confirm_frames.append(first_confirm)
            status_str = f"CONFIRMED @ frame {first_confirm}"
        else:
            status_str = "NO DETECTION"

        print(f"[{label} {idx+1:02d}/20] {clip_path.name:<25} ({frame_idx} frames) -> {status_str}")
        results.append({
            "clip": clip_path.name,
            "frames": frame_idx,
            "detected": ever_detected,
            "confirm_frame": first_confirm
        })

    return results, total_detected, confirm_frames

def main():
    fight_clips = sorted(list(FIGHT_DIR.glob("*.avi")))[:20]
    nonfight_clips = sorted(list(NONFIGHT_DIR.glob("*.avi")))[:20]

    f_results, f_det, f_frames = run_eval(fight_clips, is_fight=True)
    nf_results, nf_det, nf_frames = run_eval(nonfight_clips, is_fight=False)

    avg_f_frame = (sum(f_frames) / len(f_frames)) if f_frames else 0.0
    print("\n" + "="*80)
    print("REAL AISERVICE.PROCESS_FRAME EVALUATION SUMMARY (CONFIG 4)")
    print("="*80)
    print(f"Fight Recall:            {f_det}/{len(fight_clips)} ({f_det/len(fight_clips)*100:.1f}%)")
    print(f"NonFight False Conf:     {nf_det}/{len(nonfight_clips)} ({nf_det/len(nonfight_clips)*100:.1f}%)")
    print(f"Avg Frames to Confirm:   {avg_f_frame:.1f}")
    print("="*80)

    print("\nDetailed Per-Clip Confirmation Frames (Fight):")
    for r in f_results:
        cf = f"Frame {r['confirm_frame']}" if r['detected'] else "None (Missed)"
        print(f"  {r['clip']:<25}: {cf}")

    print("\nDetailed Per-Clip Confirmations (NonFight):")
    for r in nf_results:
        cf = f"FALSE CONFIRMATION at Frame {r['confirm_frame']}" if r['detected'] else "Clean (0 false alarms)"
        print(f"  {r['clip']:<25}: {cf}")

if __name__ == "__main__":
    main()
