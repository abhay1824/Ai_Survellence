import os
import sys
import shutil
import argparse
import pandas as pd
from pathlib import Path
from collections import Counter

# Set up paths
backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

ORIG_WEIGHTS = repo_root / "OD-WeaponDetection" / "Weapons and similar handled objects" / "Sohas_weapon-Detection-YOLOv5" / "runs" / "detect" / "weapon6cls" / "weights" / "best.pt"
BACKUP_WEIGHTS = ORIG_WEIGHTS.parent / "best_orig_backup.pt"

OBJ_TRAIN_DATA = repo_root / "OD-WeaponDetection" / "Weapons and similar handled objects" / "Sohas_weapon-Detection-YOLOv5" / "obj_train_data"
HN_DIR = repo_root / "hard_negatives"
MANIFEST_PATH = HN_DIR / "manifest.csv"

TARGET_DATASET_DIR = repo_root / "OD-WeaponDetection" / "Weapons and similar handled objects" / "Sohas_weapon-Detection-YOLOv5" / "dataset_weapon6cls_hn"
TARGET_YAML = TARGET_DATASET_DIR / "dataset.yaml"

def backup_weights():
    """Copy current best.pt to a safety backup before any training."""
    if not ORIG_WEIGHTS.exists():
        raise FileNotFoundError(f"Original weights not found: {ORIG_WEIGHTS}")
    
    if not BACKUP_WEIGHTS.exists():
        print(f"Creating backup of original weights: {ORIG_WEIGHTS} -> {BACKUP_WEIGHTS}")
        shutil.copy2(ORIG_WEIGHTS, BACKUP_WEIGHTS)
    else:
        print(f"Backup weights already exist: {BACKUP_WEIGHTS}")

def prepare_dataset(repeat_factor: int = 2):
    """
    Assemble fine-tuning dataset:
    - ALL 5,002 positive training images from obj_train_data
    - Training hard negatives repeated 2-3x (~8% background images)
    - 857 positive test images + held-out validation hard negatives
    """
    print("=" * 80)
    print("PREPARING FINE-TUNING DATASET")
    print("=" * 80)

    # 1. Create directories
    train_img_dir = TARGET_DATASET_DIR / "images" / "train"
    val_img_dir = TARGET_DATASET_DIR / "images" / "val"
    train_lbl_dir = TARGET_DATASET_DIR / "labels" / "train"
    val_lbl_dir = TARGET_DATASET_DIR / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Link/copy 5,002 positive training images and labels
    orig_train_imgs = OBJ_TRAIN_DATA / "images" / "train"
    orig_train_lbls = OBJ_TRAIN_DATA / "labels" / "train"
    pos_train_files = sorted(list(orig_train_imgs.glob("*.jpg")) + list(orig_train_imgs.glob("*.png")))
    print(f"Found {len(pos_train_files)} positive training images in {orig_train_imgs}")

    for img_path in pos_train_files:
        dest_img = train_img_dir / img_path.name
        if not dest_img.exists():
            try:
                os.link(str(img_path), str(dest_img))
            except Exception:
                shutil.copy2(str(img_path), str(dest_img))
        
        lbl_path = orig_train_lbls / f"{img_path.stem}.txt"
        if lbl_path.exists():
            dest_lbl = train_lbl_dir / lbl_path.name
            if not dest_lbl.exists():
                try:
                    os.link(str(lbl_path), str(dest_lbl))
                except Exception:
                    shutil.copy2(str(lbl_path), str(dest_lbl))

    # 3. Read manifest and add train hard negatives with repeat_factor
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    
    df = pd.read_csv(MANIFEST_PATH)
    train_negatives = df[df["split"] == "train"]
    val_negatives = df[df["split"] == "val"]
    print(f"Manifest: {len(train_negatives)} train negatives, {len(val_negatives)} val negatives")

    hn_img_dir = HN_DIR / "images"
    added_train_negs = 0

    for rep in range(repeat_factor):
        for _, row in train_negatives.iterrows():
            fname = row["filename"]
            stem = Path(fname).stem
            src_img = hn_img_dir / fname
            if not src_img.exists():
                continue
            
            target_fname = f"hn_rep{rep}_{fname}"
            dest_img = train_img_dir / target_fname
            dest_lbl = train_lbl_dir / f"hn_rep{rep}_{stem}.txt"

            if not dest_img.exists():
                shutil.copy2(str(src_img), str(dest_img))
            # Empty label file for background image
            if not dest_lbl.exists():
                with open(dest_lbl, "w", encoding="utf-8") as f:
                    pass
            added_train_negs += 1

    total_train = len(pos_train_files) + added_train_negs
    bg_ratio = (added_train_negs / total_train) * 100.0
    print(f"Added {added_train_negs} negative images to train (repeated {repeat_factor}x).")
    print(f"Total train images: {total_train} (Background: {bg_ratio:.1f}%)")

    # 4. Link/copy 857 positive val images and add held-out val negatives
    orig_val_imgs = OBJ_TRAIN_DATA / "images" / "test"
    orig_val_lbls = OBJ_TRAIN_DATA / "labels" / "test"
    pos_val_files = sorted(list(orig_val_imgs.glob("*.jpg")) + list(orig_val_imgs.glob("*.png")))
    print(f"Found {len(pos_val_files)} positive validation images in {orig_val_imgs}")

    for img_path in pos_val_files:
        dest_img = val_img_dir / img_path.name
        if not dest_img.exists():
            try:
                os.link(str(img_path), str(dest_img))
            except Exception:
                shutil.copy2(str(img_path), str(dest_img))
        
        lbl_path = orig_val_lbls / f"{img_path.stem}.txt"
        if lbl_path.exists():
            dest_lbl = val_lbl_dir / lbl_path.name
            if not dest_lbl.exists():
                try:
                    os.link(str(lbl_path), str(dest_lbl))
                except Exception:
                    shutil.copy2(str(lbl_path), str(dest_lbl))

    # Add held-out val negatives (1x, not repeated)
    added_val_negs = 0
    for _, row in val_negatives.iterrows():
        fname = row["filename"]
        stem = Path(fname).stem
        src_img = hn_img_dir / fname
        if not src_img.exists():
            continue
        dest_img = val_img_dir / fname
        dest_lbl = val_lbl_dir / f"{stem}.txt"
        if not dest_img.exists():
            shutil.copy2(str(src_img), str(dest_img))
        if not dest_lbl.exists():
            with open(dest_lbl, "w", encoding="utf-8") as f:
                pass
        added_val_negs += 1

    print(f"Added {added_val_negs} held-out validation negative images.")
    print(f"Total val images: {len(pos_val_files) + added_val_negs}")

    # 5. Write dataset.yaml
    yaml_content = f"""path: {TARGET_DATASET_DIR.as_posix()}
train: images/train
val: images/val

nc: 6
names:
  0: pistol
  1: smartphone
  2: knife
  3: monedero
  4: billete
  5: tarjeta
"""
    with open(TARGET_YAML, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"Wrote dataset config to: {TARGET_YAML}")

def main():
    parser = argparse.ArgumentParser(description="Prepare and fine-tune YOLO model with hard negatives.")
    parser.add_argument("--prepare-only", action="store_true", default=True, help="Only prepare dataset and safety backups (default)")
    parser.add_argument("--execute", action="store_true", help="Explicitly start training run")
    parser.add_argument("--epochs", type=int, default=30, help="Number of fine-tuning epochs (default: 30)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--repeat-negatives", type=int, default=2, help="Repeat factor for train negatives to reach ~8% bg (default: 2)")
    args = parser.parse_args()

    # Step 1: Backup
    backup_weights()

    # Step 2: Prepare dataset
    prepare_dataset(repeat_factor=args.repeat_negatives)

    # Step 3: Training guard
    if not args.execute:
        print("\n" + "=" * 80)
        print("FINE-TUNE PREPARATION COMPLETE. TRAINING NOT EXECUTED.")
        print("To launch fine-tuning after user approval, run:")
        print("  python backend/finetune_weapon.py --execute")
        print("=" * 80)
        return

    # Execution requires explicit flag
    print("\n" + "=" * 80)
    print("STARTING FINE-TUNING RUN: weapon6cls_hn")
    print("=" * 80)
    from ultralytics import YOLO
    model = YOLO(str(BACKUP_WEIGHTS))
    
    results = model.train(
        data=str(TARGET_YAML),
        name="weapon6cls_hn",
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        lr0=0.001,
        lrf=0.01,
        patience=10,
        device="0",
        workers=2,
        exist_ok=True,
        save=True
    )
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
