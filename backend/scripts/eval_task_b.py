import os
import sys
from pathlib import Path
from collections import deque
import cv2
import torch
from ultralytics import YOLO

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

FIGHT_DIR = repo_root / "RWF-2000" / "Fight"
NONFIGHT_DIR = repo_root / "RWF-2000" / "NonFight"
WEBCAM_DIR = repo_root / "webcam_tests"
FIGHT_MODEL_PATH = repo_root / "runs" / "classify" / "fight_classifier" / "weights" / "best.pt"
PERSON_MODEL_PATH = repo_root / "yolo11s.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading models on {device}...")
person_model = YOLO(str(PERSON_MODEL_PATH)).to(device)
fight_model = YOLO(str(FIGHT_MODEL_PATH)).to(device)

class Config2Sim:
    def __init__(self):
        self.person_win = deque(maxlen=5)
        self.fight_buf = deque(maxlen=10)
        self.confirmed = False
        self.first_frame = None

    def update(self, frame_idx, p_35, p_50, model_raw_fight):
        self.person_win.append(p_35 >= 2)
        gate_open = (sum(1 for h in self.person_win if h) >= 3)
        raw_fight = model_raw_fight if gate_open else False
        self.fight_buf.append(raw_fight)
        if len(self.fight_buf) >= 6 and sum(1 for f in self.fight_buf if f) >= 6 and gate_open:
            if not self.confirmed:
                self.confirmed = True
                self.first_frame = frame_idx

class Config4Sim:
    def __init__(self):
        self.person_win = deque(maxlen=10)
        self.fight_buf = deque(maxlen=10)
        self.confirmed = False
        self.first_frame = None

    def update(self, frame_idx, p_35, p_50, model_raw_fight):
        self.person_win.append(p_35 >= 2)
        can_run = (p_35 >= 1)
        raw_fight = model_raw_fight if can_run else False
        self.fight_buf.append(raw_fight)
        has_2p = (sum(1 for h in self.person_win if h) >= 3)
        if len(self.fight_buf) >= 6 and sum(1 for f in self.fight_buf if f) >= 6 and has_2p:
            if not self.confirmed:
                self.confirmed = True
                self.first_frame = frame_idx

class Config4bSim:
    def __init__(self):
        self.person_win = deque(maxlen=10)
        self.fight_buf = deque(maxlen=10)
        self.confirmed = False
        self.first_frame = None

    def update(self, frame_idx, p_35, p_50, model_raw_fight):
        # 4b uses conf >= 0.50 for the 2-person confirmation gate
        self.person_win.append(p_50 >= 2)
        can_run = (p_35 >= 1)
        raw_fight = model_raw_fight if can_run else False
        self.fight_buf.append(raw_fight)
        has_2p = (sum(1 for h in self.person_win if h) >= 3)
        if len(self.fight_buf) >= 6 and sum(1 for f in self.fight_buf if f) >= 6 and has_2p:
            if not self.confirmed:
                self.confirmed = True
                self.first_frame = frame_idx

def run_raw_classifier_and_configs():
    nonfight_clips = sorted(list(NONFIGHT_DIR.glob("*.avi")))[:20]
    webcam_clips = sorted(list(WEBCAM_DIR.glob("*.mp4")))
    fight_clips = sorted(list(FIGHT_DIR.glob("*.avi")))[:20]

    # Part 1: Raw Classifier on NonFight and Webcam clips
    print("\n" + "="*85)
    print("PART 1: RAW CLASSIFIER CHECK (20 NonFight Clips + Webcam Clips)")
    print("="*85)

    all_nf_and_webcam = [("NonFight", p) for p in nonfight_clips] + [("Webcam", p) for p in webcam_clips]
    
    total_eval_frames = 0
    total_raw_fight_frames = 0
    total_gated_run_frames = 0
    total_gated_fight_frames = 0

    print(f"{'Source':<10} | {'Clip Name':<28} | {'Frames':<6} | {'Raw Fight Frames (%)':<22} | {'Config 4 Ran (%)':<18}")
    print("-" * 90)

    for src, clip_path in all_nf_and_webcam:
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            continue

        c_frames = 0
        c_raw_fights = 0
        c_runs = 0
        c_gated_fights = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            c_frames += 1

            # Person detection (conf=0.25 to capture all candidates)
            p_res = person_model.predict(frame, classes=[0], conf=0.25, verbose=False)
            p_confs = []
            if p_res and p_res[0].boxes is not None:
                p_confs = [float(b.conf[0]) for b in p_res[0].boxes]

            p_35 = sum(1 for c in p_confs if c >= 0.35)
            p_50 = sum(1 for c in p_confs if c >= 0.50)

            # Raw fight classifier (UNCONDITIONAL)
            f_res = fight_model.predict([frame], verbose=False)
            model_is_fight = False
            fight_prob = 0.0
            if f_res and f_res[0].probs is not None:
                top1_idx = int(f_res[0].probs.top1)
                top1_name = f_res[0].names.get(top1_idx, "").lower()
                top1_conf = float(f_res[0].probs.top1conf)
                # find fight prob
                for k, v in f_res[0].names.items():
                    if v.lower() == "fight":
                        fight_prob = float(f_res[0].probs.data[k])
                        break
                if top1_name == "fight" and top1_conf >= 0.75:
                    model_is_fight = True

            if model_is_fight:
                c_raw_fights += 1

            can_run = (p_35 >= 1)
            if can_run:
                c_runs += 1
                if model_is_fight:
                    c_gated_fights += 1

        cap.release()
        total_eval_frames += c_frames
        total_raw_fight_frames += c_raw_fights
        total_gated_run_frames += c_runs
        total_gated_fight_frames += c_gated_fights

        raw_pct = (c_raw_fights / c_frames * 100) if c_frames else 0.0
        run_pct = (c_runs / c_frames * 100) if c_frames else 0.0
        print(f"{src:<10} | {clip_path.name:<28} | {c_frames:<6} | {c_raw_fights:3d}/{c_frames} ({raw_pct:5.1f}%)        | {c_runs:3d}/{c_frames} ({run_pct:5.1f}%)")

    overall_raw_pct = (total_raw_fight_frames / total_eval_frames * 100) if total_eval_frames else 0.0
    overall_run_pct = (total_gated_run_frames / total_eval_frames * 100) if total_eval_frames else 0.0
    print("-" * 90)
    print(f"TOTAL: {total_raw_fight_frames}/{total_eval_frames} ({overall_raw_pct:.2f}%) frames flagged as Fight by RAW classifier regardless of gating.")
    print(f"Config 4 gate allowed classifier to run on {total_gated_run_frames}/{total_eval_frames} ({overall_run_pct:.2f}%) frames.")
    print("="*90)

    # Part 2: Benchmark Config 2, Config 4, and Config 4b on 20 Fight + 20 NonFight
    print("\n" + "="*85)
    print("PART 2: CONFIG 2 vs CONFIG 4 vs CONFIG 4b BENCHMARK")
    print("="*85)

    configs = {
        "Config 2 (p>=0.35, 3-of-5 win, gate to run)": {"fight_hits": 0, "nf_hits": 0, "f_frames": []},
        "Config 4 (p>=0.35, 3-of-10 win, 1p to run)": {"fight_hits": 0, "nf_hits": 0, "f_frames": []},
        "Config 4b (p>=0.50, 3-of-10 win, 1p to run)": {"fight_hits": 0, "nf_hits": 0, "f_frames": []},
    }

    # Evaluate Fight Clips
    for p in fight_clips:
        cap = cv2.VideoCapture(str(p))
        c2 = Config2Sim()
        c4 = Config4Sim()
        c4b = Config4bSim()
        f_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            p_res = person_model.predict(frame, classes=[0], conf=0.25, verbose=False)
            p_confs = [float(b.conf[0]) for b in p_res[0].boxes] if (p_res and p_res[0].boxes is not None) else []
            p_35 = sum(1 for c in p_confs if c >= 0.35)
            p_50 = sum(1 for c in p_confs if c >= 0.50)

            f_res = fight_model.predict([frame], verbose=False)
            model_is_fight = False
            if f_res and f_res[0].probs is not None:
                top1_idx = int(f_res[0].probs.top1)
                top1_name = f_res[0].names.get(top1_idx, "").lower()
                top1_conf = float(f_res[0].probs.top1conf)
                if top1_name == "fight" and top1_conf >= 0.75:
                    model_is_fight = True

            c2.update(f_idx, p_35, p_50, model_is_fight)
            c4.update(f_idx, p_35, p_50, model_is_fight)
            c4b.update(f_idx, p_35, p_50, model_is_fight)
            f_idx += 1
        cap.release()

        if c2.confirmed:
            configs["Config 2 (p>=0.35, 3-of-5 win, gate to run)"]["fight_hits"] += 1
            configs["Config 2 (p>=0.35, 3-of-5 win, gate to run)"]["f_frames"].append(c2.first_frame)
        if c4.confirmed:
            configs["Config 4 (p>=0.35, 3-of-10 win, 1p to run)"]["fight_hits"] += 1
            configs["Config 4 (p>=0.35, 3-of-10 win, 1p to run)"]["f_frames"].append(c4.first_frame)
        if c4b.confirmed:
            configs["Config 4b (p>=0.50, 3-of-10 win, 1p to run)"]["fight_hits"] += 1
            configs["Config 4b (p>=0.50, 3-of-10 win, 1p to run)"]["f_frames"].append(c4b.first_frame)

    # Evaluate NonFight Clips
    for p in nonfight_clips:
        cap = cv2.VideoCapture(str(p))
        c2 = Config2Sim()
        c4 = Config4Sim()
        c4b = Config4bSim()
        f_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            p_res = person_model.predict(frame, classes=[0], conf=0.25, verbose=False)
            p_confs = [float(b.conf[0]) for b in p_res[0].boxes] if (p_res and p_res[0].boxes is not None) else []
            p_35 = sum(1 for c in p_confs if c >= 0.35)
            p_50 = sum(1 for c in p_confs if c >= 0.50)

            f_res = fight_model.predict([frame], verbose=False)
            model_is_fight = False
            if f_res and f_res[0].probs is not None:
                top1_idx = int(f_res[0].probs.top1)
                top1_name = f_res[0].names.get(top1_idx, "").lower()
                top1_conf = float(f_res[0].probs.top1conf)
                if top1_name == "fight" and top1_conf >= 0.75:
                    model_is_fight = True

            c2.update(f_idx, p_35, p_50, model_is_fight)
            c4.update(f_idx, p_35, p_50, model_is_fight)
            c4b.update(f_idx, p_35, p_50, model_is_fight)
            f_idx += 1
        cap.release()

        if c2.confirmed:
            configs["Config 2 (p>=0.35, 3-of-5 win, gate to run)"]["nf_hits"] += 1
        if c4.confirmed:
            configs["Config 4 (p>=0.35, 3-of-10 win, 1p to run)"]["nf_hits"] += 1
        if c4b.confirmed:
            configs["Config 4b (p>=0.50, 3-of-10 win, 1p to run)"]["nf_hits"] += 1

    header = f"{'Configuration':<45} | {'Fight Recall':<14} | {'NonFight False Conf':<20} | {'Avg Frames to Confirm':<22}"
    print(header)
    print("-" * len(header))
    for name, data in configs.items():
        f_hits = data["fight_hits"]
        f_rec = (f_hits / 20) * 100
        nf_hits = data["nf_hits"]
        nf_pct = (nf_hits / 20) * 100
        avg_f = (sum(data["f_frames"]) / len(data["f_frames"])) if data["f_frames"] else 0.0
        print(f"{name:<45} | {f_hits}/20 ({f_rec:4.1f}%)    | {nf_hits}/20 ({nf_pct:4.1f}%)         | {avg_f:<22.1f}")
    print("=" * len(header))

if __name__ == "__main__":
    run_raw_classifier_and_configs()
