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
FIGHT_MODEL_PATH = repo_root / "runs" / "classify" / "fight_classifier" / "weights" / "best.pt"
PERSON_MODEL_PATH = repo_root / "yolo11s.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading models on {device}...")
person_model = YOLO(str(PERSON_MODEL_PATH)).to(device)
fight_model = YOLO(str(FIGHT_MODEL_PATH)).to(device)

MISSED_CLIPS = ["-1l5631l3fg_2.avi", "0NWz-01A2yk_0.avi", "0NWz-01A2yk_1.avi"]

def analyze_missed_clips():
    print("="*85)
    print("DETAILED ANALYSIS OF 3 MISSED FIGHT CLIPS")
    print("="*85)
    
    for clip_name in MISSED_CLIPS:
        clip_path = FIGHT_DIR / clip_name
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            print(f"Failed to open {clip_name}")
            continue
            
        max_persons = 0
        frames_ge_2_persons = 0
        frames_classifier_fight = 0
        max_fight_conf = 0.0
        total_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            total_frames += 1
            
            # Person detection (conf >= 0.35)
            p_res = person_model.predict(frame, classes=[0], conf=0.35, verbose=False)
            p_count = len(p_res[0].boxes) if (p_res and p_res[0].boxes is not None) else 0
            if p_count > max_persons:
                max_persons = p_count
            if p_count >= 2:
                frames_ge_2_persons += 1
                
            # Fight classification (unconditional on this frame to see pure model capability)
            f_res = fight_model.predict([frame], verbose=False)
            if f_res and f_res[0].probs is not None:
                top1_idx = int(f_res[0].probs.top1)
                top1_name = f_res[0].names.get(top1_idx, "").lower()
                top1_conf = float(f_res[0].probs.top1conf)
                
                # Check fight confidence
                # If top1 is fight, record confidence
                if top1_name == "fight":
                    if top1_conf > max_fight_conf:
                        max_fight_conf = top1_conf
                    if top1_conf >= 0.75:
                        frames_classifier_fight += 1
                else:
                    # check class probs directly for fight class
                    fight_idx = None
                    for k, v in f_res[0].names.items():
                        if v.lower() == "fight":
                            fight_idx = k
                            break
                    if fight_idx is not None:
                        conf_fight = float(f_res[0].probs.data[fight_idx])
                        if conf_fight > max_fight_conf:
                            max_fight_conf = conf_fight
                            
        cap.release()
        print(f"\nClip: {clip_name} (Total frames: {total_frames})")
        print(f"  - Max persons in a single frame:    {max_persons}")
        print(f"  - Frames with >= 2 persons:         {frames_ge_2_persons} / {total_frames} ({frames_ge_2_persons/total_frames*100:.1f}%)")
        print(f"  - Frames where classifier >= 0.75:  {frames_classifier_fight} / {total_frames} ({frames_classifier_fight/total_frames*100:.1f}%)")
        print(f"  - Max fight confidence across clip: {max_fight_conf:.4f}")

class Config2Sim:
    def __init__(self):
        self.person_win = deque(maxlen=5)
        self.fight_buf = deque(maxlen=10)
        self.confirmed = False
        self.first_frame = None
    def update(self, frame_idx, p_count, model_fight):
        self.person_win.append(p_count >= 2)
        gate_open = (sum(1 for h in self.person_win if h) >= 3)
        raw_fight = model_fight if gate_open else False
        self.fight_buf.append(raw_fight)
        if len(self.fight_buf) >= 6 and sum(1 for f in self.fight_buf if f) >= 6 and gate_open:
            if not self.confirmed:
                self.confirmed = True
                self.first_frame = frame_idx

class Config4Sim:
    """
    Config 4:
    - Run classifier whenever >= 1 person is detected.
    - Confirm only when 6/10 frames are Fight AND >= 2 persons were seen in at least 3 of the last 10 frames.
    """
    def __init__(self):
        self.person_win_10 = deque(maxlen=10) # records (p_count >= 2)
        self.fight_buf = deque(maxlen=10)
        self.confirmed = False
        self.first_frame = None
    def update(self, frame_idx, p_count, model_fight):
        self.person_win_10.append(p_count >= 2)
        # Gate to run classifier: at least 1 person
        classifier_allowed = (p_count >= 1)
        raw_fight = model_fight if classifier_allowed else False
        self.fight_buf.append(raw_fight)
        
        # Confirmation: 6/10 fight frames AND >= 2 persons in at least 3 of last 10 frames
        persons_confirmed = (sum(1 for h in self.person_win_10 if h) >= 3)
        fight_confirmed = (len(self.fight_buf) >= 6 and sum(1 for f in self.fight_buf if f) >= 6)
        
        if fight_confirmed and persons_confirmed:
            if not self.confirmed:
                self.confirmed = True
                self.first_frame = frame_idx

def evaluate_all_clips():
    fight_clips = sorted(list(FIGHT_DIR.glob("*.avi")))[:20]
    nonfight_clips = sorted(list(NONFIGHT_DIR.glob("*.avi")))[:20]
    
    print("\n" + "="*85)
    print("RUNNING COMPARISON: CONFIG 2 vs CONFIG 4 (20 Fight + 20 NonFight)")
    print("="*85)
    
    c2_fight_hits = 0
    c4_fight_hits = 0
    c2_fight_frames = []
    c4_fight_frames = []
    
    for c_idx, p in enumerate(fight_clips):
        cap = cv2.VideoCapture(str(p))
        c2 = Config2Sim()
        c4 = Config4Sim()
        f_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None: break
            p_res = person_model.predict(frame, classes=[0], conf=0.35, verbose=False)
            p_count = len(p_res[0].boxes) if (p_res and p_res[0].boxes is not None) else 0
            
            f_res = fight_model.predict([frame], verbose=False)
            top1_idx = int(f_res[0].probs.top1) if (f_res and f_res[0].probs is not None) else 0
            top1_name = f_res[0].names.get(top1_idx, "").lower()
            top1_conf = float(f_res[0].probs.top1conf) if (f_res and f_res[0].probs is not None) else 0.0
            model_fight = (top1_name == "fight" and top1_conf >= 0.75)
            
            c2.update(f_idx, p_count, model_fight)
            c4.update(f_idx, p_count, model_fight)
            f_idx += 1
        cap.release()
        
        if c2.confirmed:
            c2_fight_hits += 1
            c2_fight_frames.append(c2.first_frame)
        if c4.confirmed:
            c4_fight_hits += 1
            c4_fight_frames.append(c4.first_frame)
            
        print(f"[FIGHT {c_idx+1}/20] {p.name}: Config 2={'CONFIRMED @ '+str(c2.first_frame) if c2.confirmed else 'NO'} | Config 4={'CONFIRMED @ '+str(c4.first_frame) if c4.confirmed else 'NO'}")
        
    c2_nf_hits = 0
    c4_nf_hits = 0
    for c_idx, p in enumerate(nonfight_clips):
        cap = cv2.VideoCapture(str(p))
        c2 = Config2Sim()
        c4 = Config4Sim()
        f_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None: break
            p_res = person_model.predict(frame, classes=[0], conf=0.35, verbose=False)
            p_count = len(p_res[0].boxes) if (p_res and p_res[0].boxes is not None) else 0
            
            f_res = fight_model.predict([frame], verbose=False)
            top1_idx = int(f_res[0].probs.top1) if (f_res and f_res[0].probs is not None) else 0
            top1_name = f_res[0].names.get(top1_idx, "").lower()
            top1_conf = float(f_res[0].probs.top1conf) if (f_res and f_res[0].probs is not None) else 0.0
            model_fight = (top1_name == "fight" and top1_conf >= 0.75)
            
            c2.update(f_idx, p_count, model_fight)
            c4.update(f_idx, p_count, model_fight)
            f_idx += 1
        cap.release()
        if c2.confirmed: c2_nf_hits += 1
        if c4.confirmed: c4_nf_hits += 1
        
    print("\n" + "="*85)
    print("COMPARISON SUMMARY: CONFIG 2 vs CONFIG 4")
    print("="*85)
    print(f"{'Configuration':<45} | {'Fight Recall':<15} | {'NonFight False Conf':<20} | {'Avg Frames to Confirm':<22}")
    print("-" * 105)
    avg_c2 = (sum(c2_fight_frames)/len(c2_fight_frames)) if c2_fight_frames else 0.0
    avg_c4 = (sum(c4_fight_frames)/len(c4_fight_frames)) if c4_fight_frames else 0.0
    print(f"{'Config 2 (>=2 persons 3-of-5 window, gate at classifier)':<45} | {c2_fight_hits}/20 ({c2_fight_hits/20*100:.1f}%)    | {c2_nf_hits}/20 ({c2_nf_hits/20*100:.1f}%)         | {avg_c2:<22.1f}")
    print(f"{'Config 4 (>=1 person at classifier, >=2 in 3-of-10 at alert)':<45} | {c4_fight_hits}/20 ({c4_fight_hits/20*100:.1f}%)    | {c4_nf_hits}/20 ({c4_nf_hits/20*100:.1f}%)         | {avg_c4:<22.1f}")
    print("="*85)

if __name__ == "__main__":
    analyze_missed_clips()
    evaluate_all_clips()
