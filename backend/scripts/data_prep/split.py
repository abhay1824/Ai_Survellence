import os
import random
import shutil
from pathlib import Path

# Root dataset folder
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROOT = str(PROJECT_ROOT / "fight_frames")

CLASSES = ["Fight", "NonFight"]

VAL_RATIO = 0.20  # 20% validation

for cls in CLASSES:

    train_dir = os.path.join(ROOT, "train", cls)
    val_dir = os.path.join(ROOT, "val", cls)

    os.makedirs(val_dir, exist_ok=True)

    images = [
        f for f in os.listdir(train_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    val_count = int(len(images) * VAL_RATIO)

    val_images = images[:val_count]

    print(f"\n{cls}")
    print(f"Total Images : {len(images)}")
    print(f"Moving to Val: {val_count}")

    for img in val_images:

        src = os.path.join(train_dir, img)
        dst = os.path.join(val_dir, img)

        shutil.move(src, dst)

print("\nDataset split completed successfully!")