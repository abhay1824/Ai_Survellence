import os
import sys
import time
from pathlib import Path
import cv2

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
output_dir = repo_root / "webcam_tests"
output_dir.mkdir(parents=True, exist_ok=True)

def record_clip(name: str, duration_sec: float):
    filename = f"{name}.mp4" if not name.endswith(".mp4") else name
    output_path = output_dir / filename
    
    print(f"Opening camera 0 to record '{filename}' for {duration_sec}s...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera 0!")
        sys.exit(1)
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or fps > 60:
        fps = 30.0
        
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"ERROR: Could not create VideoWriter for {output_path}")
        cap.release()
        sys.exit(1)
        
    start_time = time.time()
    frames_recorded = 0
    
    print(f"Recording started ({width}x{height} @ {fps:.1f} FPS)...")
    while time.time() - start_time < duration_sec:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        out.write(frame)
        frames_recorded += 1
        
    cap.release()
    out.release()
    elapsed = round(time.time() - start_time, 2)
    print(f"[SUCCESS] Saved RAW clip: {output_path}")
    print(f"   Recorded {frames_recorded} frames in {elapsed}s (avg {(frames_recorded/elapsed):.1f} FPS)")

if __name__ == "__main__":
    name_arg = sys.argv[1] if len(sys.argv) > 1 else "normal_sitting"
    sec_arg = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    record_clip(name_arg, sec_arg)
