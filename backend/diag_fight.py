import os
import sys
import difflib
from pathlib import Path
import cv2

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from app.services.ai_service import AIService

VIDEO_PATH = backend_dir.parent / "sample_videos" / "fight.mp4"
OUTPUT_FILE = backend_dir.parent / "fight_test_output.txt"

def run_fight_test(instance_id: int) -> list[str]:
    print(f"--- Running Fight Test (Instance #{instance_id}) ---")
    # Force new instance
    AIService._instance = None
    service = AIService()
    
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")
        
    camera_id = f"CAM-DIAG-{instance_id}"
    frame_idx = 0
    lines = []
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        (
            annotated_frame,
            detections,
            threat_level,
            total_score,
            active_events,
            correlated_events,
            person_count,
            xai_explanation,
            fight_result,
            accident_result
        ) = service.process_frame(
            frame=frame,
            camera_id=camera_id,
            camera_name="Fight Test Cam",
            gps_coords=(0.0, 0.0)
        )
        
        persons = person_count
        raw_fight = fight_result.get("raw_fight", False)
        buf_count = fight_result.get("raw_fight_frames", 0)
        fight_detected = fight_result.get("fight_detected", False)
        
        # Raw confidence for this frame from the temporal buffer
        buf = service.fight_buffers.get(camera_id, [])
        raw_conf = buf[-1][1] if buf else 0.0
        fight_confidence = raw_conf if raw_fight else (fight_result.get("fight_confidence", 0.0) if fight_detected else 0.0)
        
        line = f"frame: {frame_idx}, persons: {persons}, raw_fight: {raw_fight}, fight_confidence: {fight_confidence:.4f}, buffer_count: {buf_count}, fight_detected: {fight_detected}"
        lines.append(line)
        frame_idx += 1
        
    cap.release()
    service.cleanup_camera(camera_id)
    print(f"Instance #{instance_id} completed {len(lines)} frames.")
    return lines

def main():
    if not VIDEO_PATH.exists():
        print(f"Error: {VIDEO_PATH} does not exist!")
        sys.exit(1)
        
    # Run 1
    lines1 = run_fight_test(1)
    
    # Run 2
    lines2 = run_fight_test(2)
    
    # Diff outputs
    diff = list(difflib.unified_diff(lines1, lines2, fromfile="run1", tofile="run2", lineterm=""))
    if not diff:
        print("SUCCESS: Run 1 and Run 2 produced IDENTICAL outputs (0 differences).")
    else:
        print(f"WARNING: Differences found between Run 1 and Run 2 ({len(diff)} diff lines):")
        for d in diff:
            print(d)
            
    # Write to output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in lines1:
            f.write(line + "\n")
            
    print(f"Wrote {len(lines1)} lines to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
