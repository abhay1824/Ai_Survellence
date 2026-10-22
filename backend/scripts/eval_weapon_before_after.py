import os
import sys
import argparse
import json
import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# Set up paths
backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

DEFAULT_WEIGHTS = repo_root / "OD-WeaponDetection" / "Weapons and similar handled objects" / "Sohas_weapon-Detection-YOLOv5" / "runs" / "detect" / "weapon6cls" / "weights" / "best.pt"
DATASET_YAML = repo_root / "OD-WeaponDetection" / "Weapons and similar handled objects" / "Sohas_weapon-Detection-YOLOv5" / "dataset.yaml"
DATAASET2_DIR = repo_root / "Dataaset2"
MANIFEST_PATH = repo_root / "hard_negatives" / "manifest.csv"

def eval_map50(model, yaml_path):
    """Evaluate per-class mAP50 on the 857 validation images."""
    print("\n" + "=" * 80)
    print("TASK 1: VALIDATION mAP50 (857 IMAGES)")
    print("=" * 80)
    
    val_results = model.val(
        data=str(yaml_path),
        split="val",
        imgsz=640,
        batch=8,
        conf=0.001,
        iou=0.6,
        device="0",
        verbose=False
    )
    
    class_names = ["pistol", "smartphone", "knife", "monedero", "billete", "tarjeta"]
    class_map50 = {}
    
    # Extract overall and per-class mAP50
    overall_map50 = float(val_results.box.map50)
    print(f"Overall mAP50: {overall_map50:.4f}")
    
    if hasattr(val_results.box, "maps") and len(val_results.box.maps) >= len(class_names):
        for idx, cname in enumerate(class_names):
            score = float(val_results.box.maps[idx])
            class_map50[cname] = round(score, 4)
            print(f"  {cname:<12}: mAP50 = {score:.4f}")
    else:
        for cname in class_names:
            class_map50[cname] = "N/A"
            
    return {"overall_map50": overall_map50, "per_class_map50": class_map50}

def eval_dataset2_recall(model, max_samples=None):
    """Evaluate pistol and knife recall on Dataaset2 at conf 0.45 and 0.50."""
    print("\n" + "=" * 80)
    print("TASK 2: RECALL ON Dataaset2 (CONF 0.45 & 0.50)")
    print("=" * 80)

    pistol_dir = DATAASET2_DIR / "Pistol detection" / "Weapons"
    knife_dir = DATAASET2_DIR / "Knife_detection" / "Images"

    results = {}

    for target_class, img_dir in [("pistol", pistol_dir), ("knife", knife_dir)]:
        if not img_dir.exists():
            print(f"Directory not found: {img_dir}")
            continue

        all_imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
        if max_samples and max_samples < len(all_imgs):
            all_imgs = all_imgs[:max_samples]

        total = len(all_imgs)
        detected_45 = 0
        detected_50 = 0

        print(f"Evaluating {target_class.upper()} images ({total} files)...")
        for img_p in all_imgs:
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            
            res = model(img, conf=0.40, verbose=False)
            found_45 = False
            found_50 = False

            for r in res:
                for b in r.boxes:
                    c_id = int(b.cls[0])
                    c_name = model.names.get(c_id, "").lower()
                    conf = float(b.conf[0])
                    
                    is_match = False
                    if target_class == "pistol" and c_name in ("pistol", "gun"):
                        is_match = True
                    elif target_class == "knife" and c_name in ("knife", "scissors"):
                        is_match = True

                    if is_match:
                        if conf >= 0.45:
                            found_45 = True
                        if conf >= 0.50:
                            found_50 = True
                if found_50:
                    break

            if found_45:
                detected_45 += 1
            if found_50:
                detected_50 += 1

        recall_45 = (detected_45 / total) * 100.0 if total > 0 else 0.0
        recall_50 = (detected_50 / total) * 100.0 if total > 0 else 0.0

        results[target_class] = {
            "total_images": total,
            "detected_at_0.45": detected_45,
            "recall_0.45": round(recall_45, 2),
            "detected_at_0.50": detected_50,
            "recall_0.50": round(recall_50, 2)
        }
        print(f"  {target_class.upper()} Recall @ 0.45: {recall_45:.2f}% ({detected_45}/{total})")
        print(f"  {target_class.upper()} Recall @ 0.50: {recall_50:.2f}% ({detected_50}/{total})")

    return results

def eval_held_out_negatives(model):
    """Evaluate false detections on held-out negative clips from manifest."""
    print("\n" + "=" * 80)
    print("TASK 3: FALSE DETECTIONS ON HELD-OUT NEGATIVE CLIPS")
    print("=" * 80)

    if not MANIFEST_PATH.exists():
        print(f"Manifest not found: {MANIFEST_PATH}")
        return {}

    df = pd.read_csv(MANIFEST_PATH)
    val_negatives = df[df["split"] == "val"]
    total_val = len(val_negatives)
    print(f"Evaluating {total_val} held-out validation negative frames...")

    hn_img_dir = repo_root / "hard_negatives" / "images"

    false_pistol_35 = 0
    false_knife_35 = 0
    false_pistol_50 = 0
    false_knife_50 = 0

    for _, row in val_negatives.iterrows():
        fpath = hn_img_dir / row["filename"]
        img = cv2.imread(str(fpath))
        if img is None:
            continue

        res = model(img, conf=0.35, verbose=False)
        for r in res:
            for b in r.boxes:
                c_id = int(b.cls[0])
                c_name = model.names.get(c_id, "").lower()
                conf = float(b.conf[0])

                if c_name in ("pistol", "gun"):
                    false_pistol_35 += 1
                    if conf >= 0.50:
                        false_pistol_50 += 1
                elif c_name in ("knife", "scissors"):
                    false_knife_35 += 1
                    if conf >= 0.50:
                        false_knife_50 += 1

    results = {
        "val_negative_frames": total_val,
        "false_pistol_conf_0.35": false_pistol_35,
        "false_pistol_conf_0.50": false_pistol_50,
        "false_knife_conf_0.35": false_knife_35,
        "false_knife_conf_0.50": false_knife_50
    }
    print(f"  False Pistol @ conf >= 0.35: {false_pistol_35} / @ conf >= 0.50: {false_pistol_50}")
    print(f"  False Knife  @ conf >= 0.35: {false_knife_35} / @ conf >= 0.50: {false_knife_50}")
    return results

def eval_24_clip_regression():
    """Run the 24-clip regression harness and summarize outputs."""
    print("\n" + "=" * 80)
    print("TASK 4: 24-CLIP REGRESSION HARNESS")
    print("=" * 80)

    import subprocess
    cmd = [sys.executable, "backend/run_regression_csv.py"]
    res = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)

    csv_path = repo_root / "regression_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"24-Clip Regression complete ({len(df)} clips evaluated):")
        acc = df[df["clip_name"] == "accident.mp4"].iloc[0]
        fight = df[df["clip_name"] == "fight.mp4"].iloc[0]
        wep = df[df["clip_name"] == "weapon.mp4"].iloc[0]
        nonfight_emails = df[df["clip_name"].str.endswith(".avi")]["emails"].sum()

        print(f"  accident.mp4: threat={acc['max_threat']}, emails={acc['emails']}")
        print(f"  fight.mp4   : threat={fight['max_threat']}, emails={fight['emails']}")
        print(f"  weapon.mp4  : threat={wep['max_threat']}, emails={wep['emails']}")
        print(f"  NonFight clips total emails: {nonfight_emails}")
        return {
            "accident_threat": acc["max_threat"],
            "fight_threat": fight["max_threat"],
            "weapon_threat": wep["max_threat"],
            "nonfight_emails": int(nonfight_emails)
        }
    return {}

def main():
    parser = argparse.ArgumentParser(description="Evaluate weapon model weights before/after fine-tuning.")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Path to weights to evaluate")
    parser.add_argument("--output-json", type=str, default=None, help="Save evaluation metrics to json")
    parser.add_argument("--sample-dataset2", type=int, default=None, help="Optionally sample N images from Dataset2")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    print("=" * 80)
    print(f"EVALUATING MODEL WEIGHTS: {weights_path}")
    print("=" * 80)

    model = YOLO(str(weights_path))

    metrics = {
        "weights": str(weights_path),
        "map50": eval_map50(model, DATASET_YAML),
        "dataset2_recall": eval_dataset2_recall(model, max_samples=args.sample_dataset2),
        "held_out_negatives": eval_held_out_negatives(model),
        "regression_24_clips": eval_24_clip_regression()
    }

    if args.output_json:
        out_path = Path(args.output_json)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved metrics report to: {out_path}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
