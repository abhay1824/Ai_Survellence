import os
import sys
import time
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
FIGHT_MODEL_PATH = repo_root / "runs" / "classify" / "fight_classifier" / "weights" / "best.pt"
PERSON_MODEL_PATH = repo_root / "yolo11s.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading models on device: {device}...")
person_model = YOLO(str(PERSON_MODEL_PATH)).to(device)
fight_model = YOLO(str(FIGHT_MODEL_PATH)).to(device)

class GateSimulator:
    def __init__(self, name: str, person_thresh: float, use_window: bool):
        self.name = name
        self.person_thresh = person_thresh
        self.use_window = use_window
        self.reset()
        
    def reset(self):
        self.window = deque(maxlen=5)
        self.fight_buf = deque(maxlen=10)
        self.is_active = False
        self.non_fight_consecutive = 0
        self.first_confirm_frame = None
        self.ever_detected = False
        
    def update(self, frame_idx: int, person_confs: list[float], model_is_fight: bool, model_conf: float):
        # 1. Qualified persons count
        q_count = sum(1 for c in person_confs if c >= self.person_thresh)
        has_2_persons = (q_count >= 2)
        
        # 2. Window update
        self.window.append(has_2_persons)
        
        # 3. Gate decision
        if self.use_window:
            gate_open = (sum(1 for h in self.window if h) >= 3)
        else:
            gate_open = has_2_persons
            
        # 4. Raw fight
        raw_fight = model_is_fight if gate_open else False
        raw_conf = model_conf if raw_fight else 0.0
        self.fight_buf.append((raw_fight, raw_conf))
        
        # 5. Temporal confirmation (6 of 10 in buffer and gate open)
        buf_fights = sum(1 for f, _ in self.fight_buf if f)
        if self.use_window:
            buffer_confirmed = (len(self.fight_buf) >= 6 and buf_fights >= 6 and gate_open)
        else:
            buffer_confirmed = (len(self.fight_buf) >= 6 and buf_fights >= 6 and has_2_persons)
            
        # 6. State machine
        if buffer_confirmed:
            self.non_fight_consecutive = 0
            if not self.is_active:
                self.is_active = True
                self.ever_detected = True
                if self.first_confirm_frame is None:
                    self.first_confirm_frame = frame_idx
        else:
            if self.is_active:
                self.non_fight_consecutive += 1
                if self.non_fight_consecutive >= 15:
                    self.is_active = False
                    self.non_fight_consecutive = 0

def evaluate_clips(clip_paths: list[Path], is_fight: bool):
    results = {
        "Config 1 (current: conf>=0.35, 1-frame)": {"detected": 0, "frames_to_confirm": []},
        "Config 2 (conf>=0.35, 3-of-5 window)": {"detected": 0, "frames_to_confirm": []},
        "Config 3 (conf>=0.52, 3-of-5 window)": {"detected": 0, "frames_to_confirm": []},
    }
    
    simulators = [
        ("Config 1 (current: conf>=0.35, 1-frame)", GateSimulator("config1", 0.35, False)),
        ("Config 2 (conf>=0.35, 3-of-5 window)", GateSimulator("config2", 0.35, True)),
        ("Config 3 (conf>=0.52, 3-of-5 window)", GateSimulator("config3", 0.52, True)),
    ]
    
    for c_idx, clip_path in enumerate(clip_paths):
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            print(f"Failed to open {clip_path}")
            continue
            
        for _, sim in simulators:
            sim.reset()
            
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            # Person detection (conf=0.30 to capture all candidates)
            p_res = person_model.predict(frame, classes=[0], conf=0.30, verbose=False)
            person_confs = []
            if p_res and p_res[0].boxes is not None:
                for b in p_res[0].boxes:
                    person_confs.append(float(b.conf[0]))
                    
            # Fight classification
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
                    
            for _, sim in simulators:
                sim.update(frame_idx, person_confs, model_is_fight, model_conf)
                
            frame_idx += 1
            
        cap.release()
        
        clip_type = "FIGHT" if is_fight else "NON-FIGHT"
        print(f"[{clip_type} {c_idx+1}/{len(clip_paths)}] {clip_path.name} ({frame_idx} frames):")
        for name, sim in simulators:
            status = f"CONFIRMED @ frame {sim.first_confirm_frame}" if sim.ever_detected else "NO"
            print(f"   {name}: {status}")
            if sim.ever_detected:
                results[name]["detected"] += 1
                results[name]["frames_to_confirm"].append(sim.first_confirm_frame)
                
    return results

def main():
    fight_clips = sorted(list(FIGHT_DIR.glob("*.avi")))[:20]
    nonfight_clips = sorted(list(NONFIGHT_DIR.glob("*.avi")))[:20]
    
    print(f"\nEvaluating {len(fight_clips)} Fight clips...")
    fight_results = evaluate_clips(fight_clips, is_fight=True)
    
    print(f"\nEvaluating {len(nonfight_clips)} NonFight clips...")
    nonfight_results = evaluate_clips(nonfight_clips, is_fight=False)
    
    print("\n" + "="*85)
    print("FIGHT GATE EVALUATION SUMMARY")
    print("="*85)
    header = f"{'Configuration':<42} | {'Fight Recall':<14} | {'NonFight False Conf':<20} | {'Avg Frames to Confirm':<22}"
    print(header)
    print("-" * len(header))
    
    for cfg in fight_results.keys():
        f_det = fight_results[cfg]["detected"]
        f_total = len(fight_clips)
        f_recall_pct = (f_det / f_total) * 100
        
        nf_det = nonfight_results[cfg]["detected"]
        nf_total = len(nonfight_clips)
        nf_false_pct = (nf_det / nf_total) * 100
        
        frames_list = fight_results[cfg]["frames_to_confirm"]
        avg_frames = (sum(frames_list) / len(frames_list)) if frames_list else 0.0
        
        print(f"{cfg:<42} | {f_det}/{f_total} ({f_recall_pct:.1f}%)   | {nf_det}/{nf_total} ({nf_false_pct:.1f}%)        | {avg_frames:<22.1f}")
        
    print("="*85)

if __name__ == "__main__":
    main()
