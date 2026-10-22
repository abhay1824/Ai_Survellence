import cv2
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE = str(PROJECT_ROOT / "RWF-2000")
DEST = str(PROJECT_ROOT / "fight_frames")

for cls in ["Fight", "NonFight"]:
    src = os.path.join(SOURCE, cls)

    for video in os.listdir(src):
        video_path = os.path.join(src, video)

        cap = cv2.VideoCapture(video_path)

        frame_id = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_id % 10 == 0:
                save_dir = os.path.join(
                    DEST,
                    "train",
                    cls
                )

                os.makedirs(save_dir, exist_ok=True)

                name = f"{os.path.splitext(video)[0]}_{frame_id}.jpg"

                cv2.imwrite(
                    os.path.join(save_dir, name),
                    frame
                )

            frame_id += 1

        cap.release()

print("DONE")