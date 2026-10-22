import sys
import os
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

import os
import sys
import time
from pathlib import Path
from collections import deque
import cv2
import torch
from ultralytics import YOLO

repo_root = Path('.').resolve().parent
FIGHT_DIR = repo_root / "RWF-2000" / "Fight"
NONFIGHT_DIR = repo_root / "RWF-2000" / "NonFight"

PERSON_MODEL_PATH = repo_root / "yolo11s.pt"
FIGHT_MODEL_PATH = repo_root / "runs" / "classify" / "fight_classifier" / "weights" / "best.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

person_model = YOLO(str(PERSON_MODEL_PATH)).to(device)
fight_model = YOLO(str(FIGHT_MODEL_PATH)).to(device)

class Config2GateSimulator:
    def __init__(self, cooldown_thresh: int, min_persons_to_run: int = 2):
        self.cooldown_thresh = cooldown_thresh
        self.min_persons_to_run = min_persons_to_run
        self.reset()
        
    def reset(self):
        self.fight_buf = deque(maxlen=10)
        self.person_gate_buf = deque(maxlen=10)
        self.is_active = False
        self.non_fight_consecutive = 0
        self.first_confirm_frame = None
        self.ever_detected = False
        self.active_frame_count = 0
        self.incident_splits = 0  # Number of times active dropped and re-triggered
        
    def update(self, frame_idx: int, q_person_count: int, raw_model_fight: bool, model_conf: float):
        # Config 2: qualified persons >= 0.35
        has_2_persons = (q_person_count >= 2)
        self.person_gate_buf.append(has_2_persons)
        
        # Gate to run classifier
        can_run = (q_person_count >= self.min_persons_to_run)
        
        raw_fight = (raw_model_fight and can_run)
        fight_conf = model_conf if raw_fight else 0.0
        self.fight_buf.append((raw_fight, fight_conf))
        
        # Temporal confirmation: 6 of 10 fight frames AND >= 3 frames with >=2 persons in last 10
        buf_fights = sum(1 for f, _ in self.fight_buf if f)
        persons_seen_in_10 = sum(1 for h in self.person_gate_buf if h)
        
        buffer_confirmed = (
            len(self.fight_buf) >= 6 and
            buf_fights >= 6 and
            persons_seen_in_10 >= 3
        )
        
        if buffer_confirmed:
            self.non_fight_consecutive = 0
            if not self.is_active:
                if self.ever_detected:
                    self.incident_splits += 1
                self.is_active = True
                self.ever_detected = True
                if self.first_confirm_frame is None:
                    self.first_confirm_frame = frame_idx
        else:
            if self.is_active:
                self.non_fight_consecutive += 1
                if self.non_fight_consecutive >= self.cooldown_thresh:
                    self.is_active = False
                    self.non_fight_consecutive = 0
                    
        if self.is_active:
            self.active_frame_count += 1

def preextract_clip(clip_path: Path):
    cap = cv2.VideoCapture(str(clip_path))
    frames_data = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        # Person count with conf >= 0.35
        p_res = person_model.predict(frame, classes=[0], conf=0.35, verbose=False)
        q_persons = 0
        if p_res and p_res[0].boxes is not None:
            q_persons = len(p_res[0].boxes)
            
        # Fight classifier
        f_res = fight_model.predict([frame], verbose=False)
        model_is_fight = False
        model_conf = 0.0
        if f_res and f_res[0].probs is not None:
            top1_idx = int(f_res[0].probs.top1)
            top1_name = f_res[0].names.get(top1_idx, "").lower()
            top1_conf = float(f_res[0].probs.top1conf)
            if top1_name == "fight" and top1_conf >= 0.75:
                model_is_fight = True
                model_conf = top1_conf
                
        frames_data.append((q_persons, model_is_fight, model_conf))
    cap.release()
    return frames_data

def main():
    fight_clips = sorted(list(FIGHT_DIR.glob("*.avi")))[:20]
    nonfight_clips = sorted(list(NONFIGHT_DIR.glob("*.avi")))[:20]
    
    print(f"Pre-extracting 20 Fight clips...")
    fight_extracted = [preextract_clip(p) for p in fight_clips]
    print(f"Pre-extracting 20 NonFight clips...")
    nonfight_extracted = [preextract_clip(p) for p in nonfight_clips]
    
    thresholds = [5, 8, 10, 12, 15]
    
    print("\n" + "="*110)
    print("COOLDOWN THRESHOLD EVALUATION: 20 FIGHT + 20 NON-FIGHT CLIPS (Config 2: min_persons=2)")
    print("="*110)
    header = f"{'Cooldown Frames':<16} | {'Fight Recall':<14} | {'NonFight False Conf':<20} | {'Avg Frames Active':<18} | {'Splits / Drops':<15} | {'Drain Latency (Calm)':<20}"
    print(header)
    print("-" * len(header))
    
    for cd in thresholds:
        sim = Config2GateSimulator(cooldown_thresh=cd, min_persons_to_run=2)
        
        # Fight clips
        f_detected = 0
        total_active_frames = 0
        total_splits = 0
        for frames in fight_extracted:
            sim.reset()
            for idx, (qp, mf, mc) in enumerate(frames):
                sim.update(idx, qp, mf, mc)
            if sim.ever_detected:
                f_detected += 1
            total_active_frames += sim.active_frame_count
            total_splits += sim.incident_splits
            
        avg_active = total_active_frames / len(fight_clips)
        
        # Non-fight clips
        nf_detected = 0
        nf_active_frames = 0
        for frames in nonfight_extracted:
            sim.reset()
            for idx, (qp, mf, mc) in enumerate(frames):
                sim.update(idx, qp, mf, mc)
            if sim.ever_detected:
                nf_detected += 1
            nf_active_frames += sim.active_frame_count
            
        f_recall_str = f"{f_detected}/20 ({(f_detected/20)*100:.1f}%)"
        nf_false_str = f"{nf_detected}/20 ({(nf_detected/20)*100:.1f}%)"
        # Drain latency in calm frames is: 4 frames (buffer drain) + cd frames
        drain_str = f"{4 + cd} frames (~{(4+cd)/15:.1f}s @ 15fps)"
        
        print(f"{cd:<16} | {f_recall_str:<14} | {nf_false_str:<20} | {avg_active:<18.1f} | {total_splits:<15} | {drain_str:<20}")
        
    print("="*110)

if __name__ == "__main__":
    main()
