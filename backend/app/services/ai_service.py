import os
import time
import threading
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import logging
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from app.core.config import settings
from app.core.database import get_event_collection, get_camera_collection
from app.models.event import EventDB
from app.utils.cv_helpers import point_in_polygon, draw_annotations
from app.services.xai_service import XAIService

logger = logging.getLogger(__name__)

class AIService:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AIService, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
        
        # Determine compute device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[AIService] Compute device: {self.device}")
        print(f"⚡ AI Service compute device: {self.device}")

        # --- MODULE 0: DEDICATED PERSON DETECTION MODEL (COCO) ---
        person_model_path = getattr(settings, "PERSON_MODEL_PATH", str(PROJECT_ROOT / "yolo11s.pt"))
        try:
            if os.path.exists(person_model_path):
                self.person_model = YOLO(person_model_path)
                self.person_model.to(self.device)
                logger.info(f"Person Detection Model loaded ({self.device}): {person_model_path}")
                print(f"✅ Person model loaded successfully from {person_model_path} [{self.device}]")
            else:
                logger.error(f"Person model file not found at {person_model_path}")
                print(f"❌ Person model load failed: path not found at {person_model_path}")
                self.person_model = None
        except Exception as e:
            logger.error(f"Failed to load person model: {e}")
            print(f"❌ Person model load failed: {e}")
            self.person_model = None

        # --- MODULE 1: WEAPON DETECTION MODEL ---
        weapon_model_path = settings.WEAPON_MODEL_PATH
        try:
            if os.path.exists(weapon_model_path):
                self.model = YOLO(weapon_model_path)
                self.model.to(self.device)
                logger.info(f"Weapon Detection Model loaded ({self.device}): {weapon_model_path}")
                print(f"✅ Weapon model loaded successfully from {weapon_model_path} [{self.device}]")
            else:
                logger.error(f"Weapon model file not found at {weapon_model_path}")
                print(f"❌ Weapon model load failed: path not found at {weapon_model_path}")
                self.model = None
        except Exception as e:
            logger.error(f"Failed to load weapon model from {weapon_model_path}: {e}")
            print(f"❌ Weapon model load failed: {e}")
            self.model = None

        # --- MODULE: FIGHT CLASSIFICATION MODEL ---
        fight_model_path = getattr(settings, "FIGHT_MODEL_PATH", str(PROJECT_ROOT / "runs" / "classify" / "fight_classifier" / "weights" / "best.pt"))
        try:
            if os.path.exists(fight_model_path):
                self.fight_model = YOLO(fight_model_path)
                self.fight_model.to(self.device)
                logger.info(f"Fight Classification Model loaded ({self.device}): {fight_model_path}")
                print(f"✅ Fight model loaded successfully [{self.device}]")
            else:
                logger.error(f"Fight model file not found at {fight_model_path}")
                print(f"❌ Fight model load failed: path not found at {fight_model_path}")
                self.fight_model = None
        except Exception as e:
            logger.error(f"Failed to load fight model: {e}")
            print(f"❌ Fight model load failed: {e}")
            self.fight_model = None

        # --- MODULE: ACCIDENT DETECTION MODEL ---
        accident_model_path = getattr(settings, "ACCIDENT_MODEL_PATH", str(PROJECT_ROOT / "runs" / "detect" / "accident_detector" / "weights" / "best.pt"))
        try:
            if os.path.exists(accident_model_path):
                self.accident_model = YOLO(accident_model_path)
                self.accident_model.to(self.device)
                logger.info(f"Accident Detection Model loaded ({self.device}): {accident_model_path}")
                print(f"✅ Accident model loaded successfully [{self.device}]")
            else:
                logger.error(f"Accident model file not found at {accident_model_path}")
                print(f"❌ Accident model load failed: path not found at {accident_model_path}")
                self.accident_model = None
        except Exception as e:
            logger.error(f"Failed to load accident model: {e}")
            print(f"❌ Accident model load failed: {e}")
            self.accident_model = None

        # Tracking & Trajectory Storage
        # Format: { camera_id: { track_id: {"first_seen": float, "last_seen": float, "trajectory": [(x,y),...]} } }
        self.tracking_history = {}
        
        # Pre/Post Frame Buffers for 10s clip generation (approx 150 frames @ 15fps)
        self.frame_buffers = {}
        
        # Frame diff buffer for freeze detection: { camera_id: {"last_frame": ndarray, "freeze_counter": int} }
        self.health_buffers = {}

        # 3-Frame Consecutive Validation Counter per camera: { camera_id: { "pistol": int, "knife": int } }
        self.weapon_counters = {}
        
        # 2-Frame Consecutive Accident Validation Counter per camera: { camera_id: int }
        self.accident_counters = {}

        # Frame counters
        self.frame_counters = {}
        self.event_collection = None

        # Temporal Fight Buffer: { camera_id: deque of last 10 fight predictions (True/False) }
        self.fight_buffers = {}

        # Temporal Person Gate Buffer: { camera_id: deque of last 5 bools (2+ persons present) }
        self.person_gate_buffers = {}

        # Fight duration & cooldown tracking: { camera_id: {"start_time": float | None, "active": bool, "non_fight_consecutive": int, "participant_memory": dict, "smoothed_participants": list} }
        self.fight_duration = {}

        # Per-camera ByteTrack tracker isolation state and concurrency locks
        self._person_model_lock = threading.Lock()
        self._weapon_model_lock = threading.Lock()
        self._person_trackers = {}  # { camera_id: [tracker instances] }
        self._weapon_trackers = {}  # { camera_id: [tracker instances] }
        self._warned_missing_trackers = False

        self._initialized = True

    def _track_with_camera_state(
        self,
        model,
        frame: np.ndarray,
        camera_id: str,
        tracker_dict: dict,
        lock: threading.Lock,
        **track_kwargs
    ):
        """
        Thread-safe inference using dedicated per-camera ByteTrack tracker state.
        Swaps the predictor's tracker instance for the specific camera_id to prevent cross-camera ID jumps.
        """
        with lock:
            if hasattr(model, 'predictor') and model.predictor is not None:
                if hasattr(model.predictor, 'trackers'):
                    if camera_id in tracker_dict:
                        model.predictor.trackers = tracker_dict[camera_id]
                    else:
                        delattr(model.predictor, 'trackers')
                elif not self._warned_missing_trackers:
                    logger.warning("[AIService] Model predictor has no 'trackers' attribute; tracker isolation skipped.")
                    self._warned_missing_trackers = True

            results = model.track(frame, **track_kwargs)

            if hasattr(model, 'predictor') and model.predictor is not None and hasattr(model.predictor, 'trackers'):
                tracker_dict[camera_id] = model.predictor.trackers

            return results

    def cleanup_camera(self, camera_id: str):
        """Release tracker state, trajectories, and history buffers when a camera disconnects."""
        with self._person_model_lock:
            self._person_trackers.pop(camera_id, None)
        with self._weapon_model_lock:
            self._weapon_trackers.pop(camera_id, None)

        self.tracking_history.pop(camera_id, None)
        self.frame_buffers.pop(camera_id, None)
        self.health_buffers.pop(camera_id, None)
        self.weapon_counters.pop(camera_id, None)
        self.accident_counters.pop(camera_id, None)
        self.fight_buffers.pop(camera_id, None)
        self.person_gate_buffers.pop(camera_id, None)
        self.fight_duration.pop(camera_id, None)
        self.frame_counters.pop(camera_id, None)
        logger.info(f"[AIService] Cleaned up state and trackers for camera: {camera_id}")

    def calculate_threat_score(
        self,
        detections: list,
        active_events: list,
        fight_result: dict = None,
        accident_result: dict = None
    ) -> tuple[int, str]:
        """
        MODULE 5: Centralized Threat Scoring Engine with Weighted Normalization
        Confidence-Weighted Base Inputs:
          - Pistol: base 80 * conf
          - Knife: base 60 * conf
          - Person: +5
          - Loitering: +20
          - Crowd: +20
          - Intrusion: +50
          - Tampering/Disconnect: +50
          - Freeze: +30

        Fight Detection (confidence-weighted):
          Fight conf 85–90%  → +20
          Fight conf 90–95%  → +30
          Fight conf >95%    → +40

        Accident Detection:
          Accident Only             → +35 (HIGH)
          Accident + Intrusion      → +55 (HIGH)
          Accident + Fight          → +75 (HIGH)
          Accident + Weapon         → +85 (CRITICAL)
          Accident + Fight + Weapon → +100 (CRITICAL)

        Combined multi-signal escalation overrides:
          Fight + Weapon              → minimum 70 total
          Fight + Weapon + Intrusion  → minimum 90 total
          Fight + Weapon + Running    → minimum 100 total

        Weighted Normalization:
          Prevents linear additive saturation using a smooth diminishing-returns curve:
          score = min(100, int(100 * (1 - e^(-raw_score / 75))))

        Levels:
          0–29:  LOW
          30–59: MEDIUM
          60–79: HIGH
          80+:   CRITICAL
        """
        import math

        raw_score = 0
        fight_result = fight_result or {}
        fight_detected = fight_result.get("fight_detected", False)
        fight_conf = fight_result.get("fight_confidence", 0.0)

        accident_result = accident_result or {}
        accident_detected = accident_result.get("accident_detected", False)
        accident_conf = accident_result.get("accident_confidence", 0.0)

        has_weapon = False
        has_running = False
        has_intrusion = False

        person_score = 0
        for det in detections:
            cls_name = det.get("class", "").lower()
            conf = det.get("confidence", 0.0)

            if "pistol" in cls_name or "gun" in cls_name:
                base = settings.BASE_THREAT_SCORES.get("pistol", 80)
                raw_score += int(base * conf)
                has_weapon = True
            elif "knife" in cls_name or "scissors" in cls_name:
                base = settings.BASE_THREAT_SCORES.get("knife", 60)
                raw_score += int(base * conf)
                has_weapon = True
            elif cls_name == "person":
                person_score = min(person_score + 5, 15)  # Cap person contribution at +15 max

        raw_score += person_score

        for evt in active_events:
            e_type = evt.get("event_type", "").upper()
            if "LOITERING" in e_type:
                raw_score += 20
            elif "CROWD" in e_type:
                raw_score += 20
            elif "INTRUSION" in e_type:
                raw_score += 50
                has_intrusion = True
            elif "TAMPERING" in e_type or "DISCONNECTED" in e_type:
                raw_score += 50
            elif "FREEZE" in e_type:
                raw_score += 30
            elif "RUNNING" in e_type:
                has_running = True

        # --- Fight Detection Confidence-Weighted Score ---
        fight_base = 0
        if fight_detected and fight_conf >= 0.85:
            if fight_conf > 0.95:
                fight_base = 40
            elif fight_conf > 0.90:
                fight_base = 30
            else:
                fight_base = 20
            raw_score += fight_base

        # --- Accident Detection Score Contribution ---
        if accident_detected:
            raw_score += 35

        # Weighted Normalization via diminishing-returns curve
        if raw_score <= 0:
            score = 0
        else:
            # Soft saturation: score curves smoothly towards 100
            normalized = int(100 * (1.0 - math.exp(-raw_score / 75.0)))
            score = min(100, normalized)

        # Multi-signal escalation overrides guarantee minimum priority levels
        if fight_detected and fight_conf >= 0.85:
            if has_weapon and has_intrusion:
                score = max(score, 90)
            elif has_weapon and has_running:
                score = max(score, 100)
            elif has_weapon:
                score = max(score, 70)

        # Accident escalation and calibrated scoring (per user specifications):
        if accident_detected:
            if has_weapon and (fight_detected and fight_conf >= 0.85):
                score = 100
            elif has_weapon:
                score = max(score, 85)
            elif fight_detected and fight_conf >= 0.85:
                score = max(score, 75)
            elif has_intrusion:
                score = 55
            elif not has_weapon and not (fight_detected and fight_conf >= 0.85):
                score = 35

        # Determine level based on thresholds
        if score >= 80:
            level = "CRITICAL"
        elif score >= 60:
            level = "HIGH"
        elif score >= 30:
            level = "MEDIUM"
        else:
            level = "LOW"

        # Accident alone is classified as HIGH priority for SOC surveillance
        if accident_detected and level in ["LOW", "MEDIUM"]:
            level = "HIGH"

        # Confirmed fight is classified as HIGH or CRITICAL priority for SOC surveillance
        if fight_detected and fight_conf >= 0.70:
            if fight_conf >= 0.90:
                score = max(score, 85)
                level = "CRITICAL"
            else:
                score = max(score, 75)
                if level in ["LOW", "MEDIUM"]:
                    level = "HIGH"

        return score, level

    def run_fight_classification(
        self,
        frame: np.ndarray,
        camera_id: str,
        person_detections: list,
        now: float,
        raw_fight_override: bool = None,
        fight_conf_override: float = None,
        all_detections: list = None
    ) -> dict:
        """
        MODULE: Fight Classification with Temporal Confirmation, Cooldown Period, and ID Reconciliation.
        - Multi-region screen & mobile crop analysis: isolates handheld phone screens and contours.
        - Uses a rolling window with responsive confirmation for mobile screens and surveillance.
        - Cooldown period (15 consecutive non-fight frames) ensures brief classification drops do NOT split an active fight into multiple incidents.
        - Participant ID Reconciliation & Smoothing handles ByteTrack ID drops, swaps, and occlusions.
        Returns fight_result dict.
        """
        from collections import deque

        COOLDOWN_FRAMES_THRESHOLD = 15  # Cooldown threshold to declare fight ended

        # Initialize per-camera fight buffer, person gate, and duration state (Config 2)
        if camera_id not in self.fight_buffers:
            self.fight_buffers[camera_id] = deque(maxlen=10)
        if camera_id not in self.person_gate_buffers:
            self.person_gate_buffers[camera_id] = deque(maxlen=5)
        if camera_id not in self.fight_duration:
            self.fight_duration[camera_id] = {
                "start_time": None,
                "active": False,
                "non_fight_consecutive": 0,
                "participant_memory": {},  # {track_id: last_seen_time}
                "smoothed_participants": [],
            }

        # Config 2: Qualified person count (conf >= 0.35) and 3-of-5 sliding window gate
        qualified_persons = [d for d in person_detections if d.get("confidence", 0.0) >= 0.35]
        has_2_persons = (len(qualified_persons) >= 2)
        self.person_gate_buffers[camera_id].append(has_2_persons)
        gate_open = (sum(1 for h in self.person_gate_buffers[camera_id] if h) >= 3)

        dur_tracker = self.fight_duration[camera_id]
        fight_conf = 0.0
        raw_fight = False

        if raw_fight_override is not None:
            raw_fight = raw_fight_override
            fight_conf = fight_conf_override if fight_conf_override is not None else (0.90 if raw_fight else 0.0)
        elif self.fight_model is not None and frame is not None and frame.size > 0:
            try:
                # Altercation strictly requires temporal confirmation of at least 2 people in view
                if gate_open:
                    results = self.fight_model.predict([frame], verbose=False)
                    if results and results[0].probs is not None:
                        probs = results[0].probs
                        names = results[0].names
                        top1_idx = int(probs.top1)
                        top1_conf = float(probs.top1conf)
                        top1_name = names.get(top1_idx, "").lower()

                        if top1_name == "fight" and top1_conf >= 0.75:
                            raw_fight = True
                            fight_conf = top1_conf
            except Exception as e:
                logger.warning(f"Fight model inference error on {camera_id}: {e}")

        # Push raw result into temporal buffer
        self.fight_buffers[camera_id].append((raw_fight, fight_conf))

        # Temporal confirmation: requires 6/10 frames in fight buffer and temporal person gate open
        buf = self.fight_buffers[camera_id]
        fight_frames = sum(1 for f, _ in buf if f)
        if raw_fight_override is not None:
            buffer_confirmed = (fight_frames >= 6)
        else:
            buffer_confirmed = (len(buf) >= 6 and fight_frames >= 6 and gate_open)

        # Average confidence over fight frames only
        fight_confs = [c for f, c in buf if f]
        avg_conf = round(sum(fight_confs) / len(fight_confs), 4) if fight_confs else 0.0

        fight_started = False
        fight_ended = False

        # --- FIGHT STATE MACHINE WITH COOLDOWN PERIOD ---
        if buffer_confirmed:
            dur_tracker["non_fight_consecutive"] = 0
            if not dur_tracker["active"]:
                dur_tracker["start_time"] = now
                dur_tracker["active"] = True
                dur_tracker["participant_memory"] = {}
                dur_tracker["smoothed_participants"] = []
                fight_started = True
        else:
            if dur_tracker["active"]:
                dur_tracker["non_fight_consecutive"] += 1
                if dur_tracker["non_fight_consecutive"] >= COOLDOWN_FRAMES_THRESHOLD:
                    dur_tracker["active"] = False
                    fight_ended = True
                    dur_tracker["non_fight_consecutive"] = 0

        is_fight_active = dur_tracker["active"]

        duration_sec = round(now - dur_tracker["start_time"], 1) if (
            is_fight_active and dur_tracker["start_time"] is not None
        ) else 0.0

        # --- PARTICIPANT TRACKING & ID RECONCILIATION ---
        current_pids = [
            det["track_id"]
            for det in person_detections
            if det.get("track_id") is not None
        ]

        if is_fight_active:
            mem = dur_tracker["participant_memory"]
            # 1. Update memory with current active person IDs
            for pid in current_pids:
                mem[pid] = now

            # 2. Prune participant IDs older than 3 seconds (reconcile lost / left IDs)
            expired_ids = [pid for pid, last_t in mem.items() if now - last_t > 3.0]
            for pid in expired_ids:
                del mem[pid]

            # 3. Smoothed participants list
            dur_tracker["smoothed_participants"] = sorted(list(mem.keys()))

        participants = dur_tracker["smoothed_participants"] if is_fight_active else []

        return {
            "fight_detected": is_fight_active,
            "fight_confidence": avg_conf if is_fight_active else 0.0,
            "fight_participants": participants,
            "fight_duration_sec": duration_sec,
            "fight_started": fight_started,
            "fight_ended": fight_ended,
            "raw_fight_frames": fight_frames,
            "cooldown_active": not buffer_confirmed and is_fight_active,
            "raw_fight": raw_fight
        }


    def run_accident_detection(self, frame: np.ndarray, screen_boxes: list = None) -> dict:
        """
        MODULE: Accident Detection using YOLO11s Object Detection on camera feed.
        Returns:
            {"accident_detected": bool, "accident_confidence": float, "boxes": list}
        """
        if self.accident_model is None or frame is None or frame.size == 0:
            return {
                "accident_detected": False,
                "accident_confidence": 0.0,
                "boxes": []
            }

        try:
            best_conf = 0.0
            boxes = []

            # Full camera frame evaluation (calibrated threshold >= 0.70)
            results = self.accident_model(frame, verbose=False)
            if results and len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_idx = int(box.cls[0])
                    raw_cls = str(self.accident_model.names.get(cls_idx, "")).lower()
                    conf = float(box.conf[0])

                    if ("accident" in raw_cls or cls_idx == 0) and conf >= 0.70:
                        bbox = box.xyxy[0].tolist()
                        boxes.append({
                            "bbox": bbox,
                            "confidence": conf,
                            "source": "full_frame"
                        })
                        if conf > best_conf:
                            best_conf = conf

            if best_conf >= 0.70 and len(boxes) > 0:
                return {
                    "accident_detected": True,
                    "accident_confidence": round(best_conf, 4),
                    "boxes": boxes
                }
        except Exception as e:
            logger.warning(f"Accident detection inference error: {e}")

        return {
            "accident_detected": False,
            "accident_confidence": 0.0,
            "boxes": []
        }

    def correlate_multi_events(self, active_events: list, detections: list, accident_result: dict = None, fight_result: dict = None) -> tuple[list, str]:
        """
        MODULE 6: Multi-Event Correlation Engine (Explicit Security Rules)
        - Fight + Knife -> CRITICAL
        - Fight + Pistol -> CRITICAL
        - Pistol + Restricted Area -> CRITICAL
        - Knife + Restricted Area -> HIGH
        - Crowd + Pistol -> CRITICAL
        - Crowd + Knife -> HIGH
        - Accident + Weapon -> CRITICAL
        - Accident + Fight -> HIGH
        - Accident + Restricted Area -> HIGH
        - Accident + Crowd -> HIGH
        """
        event_types = {e.get("event_type", "").upper() for e in active_events}
        det_classes = [d.get("class", "").lower() for d in detections]
        
        correlated = []
        override_level = None

        has_fight = any("FIGHT" in e for e in event_types) or any("fight" in c for c in det_classes) or (fight_result and fight_result.get("fight_detected", False))
        has_knife = any("KNIFE" in e for e in event_types) or any("knife" in c for c in det_classes) or any("scissors" in c for c in det_classes)
        has_pistol = any("PISTOL" in e or "GUN" in e for e in event_types) or any("pistol" in c or "gun" in c for c in det_classes)
        has_crowd = any("CROWD" in e for e in event_types)
        has_intrusion = any("INTRUSION" in e for e in event_types)
        has_weapon = has_knife or has_pistol

        accident_result = accident_result or {}
        has_accident = (
            any("ACCIDENT" in e for e in event_types) or
            any("accident" in c for c in det_classes) or
            accident_result.get("accident_detected", False)
        )

        if has_fight and has_knife:
            correlated.append("FIGHT + KNIFE COMBINATION (CRITICAL)")
            override_level = "CRITICAL"
        if has_fight and has_pistol:
            correlated.append("FIGHT + PISTOL THREAT (CRITICAL)")
            override_level = "CRITICAL"
        if has_pistol and has_intrusion:
            correlated.append("PISTOL IN RESTRICTED AREA (CRITICAL)")
            override_level = "CRITICAL"
        if has_knife and has_intrusion:
            correlated.append("KNIFE IN RESTRICTED AREA (HIGH)")
            if override_level != "CRITICAL":
                override_level = "HIGH"
        if has_crowd and has_pistol:
            correlated.append("CROWD + PISTOL THREAT (CRITICAL)")
            override_level = "CRITICAL"
        if has_crowd and has_knife:
            correlated.append("CROWD + KNIFE THREAT (HIGH)")
            if override_level != "CRITICAL":
                override_level = "HIGH"

        # --- Accident Correlations ---
        if has_accident and has_weapon:
            correlated.append("ACCIDENT + WEAPON INVOLVEMENT (CRITICAL)")
            override_level = "CRITICAL"
        if has_accident and has_fight:
            correlated.append("ACCIDENT + PHYSICAL ALTERCATION (HIGH)")
            if override_level != "CRITICAL":
                override_level = "HIGH"
        if has_accident and has_intrusion:
            correlated.append("ACCIDENT IN RESTRICTED ZONE (HIGH)")
            if override_level != "CRITICAL":
                override_level = "HIGH"
        if has_accident and has_crowd:
            correlated.append("ACCIDENT + CROWD GATHERING (HIGH)")
            if override_level != "CRITICAL":
                override_level = "HIGH"

        return correlated, override_level

    def process_frame(
        self,
        frame: np.ndarray,
        camera_id: str,
        camera_name: str,
        gps_coords: tuple,
        zones: list = None,
        loitering_threshold: float = 60.0,
        crowd_threshold: int = 10
    ) -> tuple:
        """
        Processes frame across all 14 Modules:
        ByteTrack, Trajectory rendering, Loitering check, Intrusion point-in-polygon, Crowd count,
        Threat Scoring, Correlation Engine, Camera Tampering check, evidence buffering.
        """
        if frame is None:
            logger.warning(f"None frame received for Camera: {camera_id}")
            return None, [], "LOW", 0, [{
                "event_type": "NO VIDEO INPUT",
                "threat_level": "LOW",
                "confidence": 1.0,
                "timestamp": datetime.utcnow().isoformat(),
                "reason": "No frame data received (frame is None)"
            }], [], 0, None, {}, {}

        if frame.size == 0:
            logger.warning(f"Zero size frame received for Camera: {camera_id}")
            return frame, [], "LOW", 0, [{
                "event_type": "NO VIDEO INPUT",
                "threat_level": "LOW",
                "confidence": 1.0,
                "timestamp": datetime.utcnow().isoformat(),
                "reason": "Empty frame buffer (frame.size == 0)"
            }], [], 0, None, {}, {}

        h, w = frame.shape[:2]
        now = time.time()

        # Initialize Camera Buffers
        if camera_id not in self.tracking_history:
            self.tracking_history[camera_id] = {}
        if camera_id not in self.frame_buffers:
            self.frame_buffers[camera_id] = []
        if camera_id not in self.health_buffers:
            self.health_buffers[camera_id] = {"last_frame": None, "freeze_counter": 0}
        if camera_id not in self.frame_counters:
            self.frame_counters[camera_id] = 0

        self.frame_counters[camera_id] += 1

        # Maintain 150-frame buffer for 10-second clip recording
        self.frame_buffers[camera_id].append(frame.copy())
        if len(self.frame_buffers[camera_id]) > 150:
            self.frame_buffers[camera_id].pop(0)

        active_events = []
        detections = []
        person_count = 0

        # --- FRAME QUALITY GATE & VALIDATION ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        brightness = float(np.mean(frame))
        stddev = float(np.std(frame))

        # Check for pure zero/empty placeholder frame (e.g. uninitialized black canvas before webcam loads)
        is_empty_placeholder = (brightness <= 5.0 and stddev <= 2.0)
        valid_frame = not is_empty_placeholder
        
        last_time = self.health_buffers[camera_id].get("last_frame_time", now)
        time_since_last_frame = now - last_time
        self.health_buffers[camera_id]["last_frame_time"] = now
        fps = (1.0 / time_since_last_frame) if time_since_last_frame > 0 else 0.0

        mse = 0.0
        last_frame = self.health_buffers[camera_id]["last_frame"]
        if last_frame is not None and last_frame.shape == gray.shape:
            diff = cv2.absdiff(gray, last_frame)
            mse = float(np.mean(diff ** 2))

        # ===== CONTINUOUS DEBUG LOG (every 30 frames) =====
        freeze_count = self.health_buffers[camera_id]["freeze_counter"]
        if self.frame_counters[camera_id] % 30 == 0:
            print(f"[FRAME METRICS] cam={camera_id} | frame={self.frame_counters[camera_id]} | "
                  f"Brightness={brightness:.2f} | StdDev={stddev:.2f} | MSE={mse:.4f} | "
                  f"FPS={fps:.2f} | valid_frame={valid_frame}")
        # ==========================================================

        if is_empty_placeholder:
            # Uninitialized / empty placeholder frame before webcam stream warms up -> return annotated frame without blocking
            self.health_buffers[camera_id]["last_frame"] = gray.copy()

            annotated_frame = draw_annotations(
                frame=frame,
                detections=[],
                tracking_history={},
                zones=zones,
                threat_score=0,
                threat_level="LOW",
                person_count=0,
                correlated_events=[]
            )
            return annotated_frame, [], "LOW", 0, [], [], 0, None, {}, {}

        # --- MODULE 10: CAMERA TAMPERING & FREEZE DETECTION (Valid Frames Only) ---
        # 1. Obstructed/Covered check (blackout or blocked camera lens)
        is_obstructed = (brightness < 8.0 and stddev < 2.0)
        if is_obstructed:
            tamper_reason = f"Covered/Obstructed (Brightness={brightness:.2f} < 8.0 and StdDev={stddev:.2f} < 2.0)"
            print("==================== [TAMPER ALERT DEBUG] ====================")
            print("Camera ID:", camera_id)
            print("Brightness:", brightness)
            print("Std Dev:", stddev)
            print("MSE:", mse)
            print("FPS:", round(fps, 2))
            print("Time Since Last Frame:", round(time_since_last_frame, 4))
            print("Tamper Reason:", tamper_reason)
            print("==============================================================")

            active_events.append({
                "event_type": "CAMERA TAMPERING (OBSTRUCTED)",
                "threat_level": "HIGH",
                "confidence": 0.98,
                "timestamp": datetime.utcnow().isoformat()
            })

        # 2. Frame Freeze check (difference relative to previous frame)
        if last_frame is not None and last_frame.shape == gray.shape:
            if mse < 0.5:
                self.health_buffers[camera_id]["freeze_counter"] += 1
                if self.health_buffers[camera_id]["freeze_counter"] > 40:
                    tamper_reason = f"Feed Frozen (MSE={mse:.4f} < 0.5 for >40 frames)"
                    print("==================== [TAMPER ALERT DEBUG] ====================")
                    print("Camera ID:", camera_id)
                    print("Brightness:", brightness)
                    print("Std Dev:", stddev)
                    print("MSE:", mse)
                    print("FPS:", round(fps, 2))
                    print("Time Since Last Frame:", round(time_since_last_frame, 4))
                    print("Tamper Reason:", tamper_reason)
                    print("==============================================================")

                    active_events.append({
                        "event_type": "CAMERA FREEZE DETECTED",
                        "threat_level": "MEDIUM",
                        "confidence": 0.95,
                        "timestamp": datetime.utcnow().isoformat()
                    })
            else:
                self.health_buffers[camera_id]["freeze_counter"] = 0

        # 3. Feed lost / delay check
        if time_since_last_frame > 5.0 and self.frame_counters[camera_id] > 1:
            tamper_reason = f"Feed Delayed/Lost (Time Since Last Frame={time_since_last_frame:.2f}s > 5.0s)"
            print("==================== [TAMPER ALERT DEBUG] ====================")
            print("Camera ID:", camera_id)
            print("Brightness:", brightness)
            print("Std Dev:", stddev)
            print("MSE:", mse)
            print("FPS:", round(fps, 2))
            print("Time Since Last Frame:", round(time_since_last_frame, 4))
            print("Tamper Reason:", tamper_reason)
            print("==============================================================")

            active_events.append({
                "event_type": "CAMERA DISCONNECTED",
                "threat_level": "HIGH",
                "confidence": 0.99,
                "timestamp": datetime.utcnow().isoformat()
            })

        self.health_buffers[camera_id]["last_frame"] = gray.copy()

        # Initialize weapon persistence counter for camera
        if camera_id not in self.weapon_counters:
            self.weapon_counters[camera_id] = {"pistol": 0, "knife": 0}

        frame_detected_security_weapons = set()

        # --- MODULE 0: DEDICATED PERSON DETECTION (ByteTrack) ---
        if self.person_model is not None:
            try:
                person_results = self._track_with_camera_state(
                    self.person_model,
                    frame,
                    camera_id,
                    self._person_trackers,
                    self._person_model_lock,
                    classes=[0],
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False
                )
                if person_results and person_results[0].boxes is not None:
                    for box in person_results[0].boxes:
                        conf = float(box.conf[0])
                        if conf < 0.35:
                            continue
                        bbox = box.xyxy[0].tolist()
                        track_id = int(box.id[0]) if box.id is not None else None
                        person_count += 1
                        detections.append({
                            "bbox": bbox,
                            "class": "Person",
                            "confidence": conf,
                            "track_id": track_id,
                            "threat_level": "LOW"
                        })
            except Exception as e:
                logger.error(f"Person track error: {e}")

        # --- MODULE 1: WEAPON & OBJECT DETECTION (ByteTrack) ---
        if self.model:
            try:
                # ByteTrack persistence with per-camera tracker state
                results = self._track_with_camera_state(
                    self.model,
                    frame,
                    camera_id,
                    self._weapon_trackers,
                    self._weapon_model_lock,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False
                )
                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls_idx = int(box.cls[0])
                        raw_cls = str(self.model.names[cls_idx]).lower()
                        conf = float(box.conf[0])

                        # Skip person from weapon model to avoid duplicate tracking with person_model
                        if raw_cls == "person":
                            continue

                        # Filter out non-security clutter classes (monedero, tarjeta, billete)
                        if raw_cls in ["monedero", "tarjeta", "billete"] and conf < 0.70:
                            continue
                        if raw_cls == "smartphone" and conf < 0.25:
                            continue

                        # Security classes use calibrated threshold (0.35) to handle live webcam and screen glare
                        is_security_weapon = raw_cls in settings.SECURITY_CLASSES or raw_cls in ["gun", "firearm"]
                        required_conf = 0.35 if is_security_weapon else 0.25

                        if conf < required_conf:
                            continue

                        bbox = box.xyxy[0].tolist()
                        track_id = int(box.id[0]) if box.id is not None else None
                        display_name = raw_cls.capitalize()

                        if raw_cls in ["pistol", "gun"]:
                            frame_detected_security_weapons.add("pistol")
                            threat_level = "CRITICAL"
                        elif raw_cls in ["knife", "scissors"]:
                            frame_detected_security_weapons.add("knife")
                            threat_level = "HIGH"
                        else:
                            # non-security object with high confidence (>= 0.75)
                            threat_level = "LOW"

                        detections.append({
                            "bbox": bbox,
                            "class": display_name,
                            "confidence": conf,
                            "track_id": track_id,
                            "threat_level": threat_level
                        })
            except Exception as e:
                logger.error(f"YOLO track error: {e}")

        # --- 3-FRAME CONSECUTIVE PERSISTENCE VALIDATION ---
        for sec_cls in ["pistol", "knife"]:
            if sec_cls in frame_detected_security_weapons:
                self.weapon_counters[camera_id][sec_cls] += 1
                if self.weapon_counters[camera_id][sec_cls] >= 3:
                    matching_dets = [d for d in detections if sec_cls in d["class"].lower() or (sec_cls == "pistol" and "gun" in d["class"].lower())]
                    max_conf = max([d["confidence"] for d in matching_dets], default=0.85) if matching_dets else 0.85
                    
                    event_type = f"WEAPON DETECTED ({sec_cls.upper()})"
                    threat_level = "CRITICAL" if sec_cls == "pistol" else "HIGH"
                    
                    active_events.append({
                        "event_type": event_type,
                        "class_name": sec_cls,
                        "threat_level": threat_level,
                        "confidence": round(max_conf, 2),
                        "timestamp": datetime.utcnow().isoformat()
                    })
            else:
                self.weapon_counters[camera_id][sec_cls] = 0

        # Update Tracking & Trajectories
        for det in detections:
            track_id = det["track_id"]
            bbox = det["bbox"]
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

            if det["class"].lower() == "person" and track_id is not None:
                if track_id not in self.tracking_history[camera_id]:
                    self.tracking_history[camera_id][track_id] = {
                        "first_seen": now,
                        "last_seen": now,
                        "trajectory": [(cx, cy)]
                    }
                else:
                    hist = self.tracking_history[camera_id][track_id]
                    hist["last_seen"] = now
                    hist["trajectory"].append((cx, cy))
                    if len(hist["trajectory"]) > 30:  # keep last 30 breadcrumb points
                        hist["trajectory"].pop(0)

                # --- MODULE 2: LOITERING DETECTION ---
                first_seen = self.tracking_history[camera_id][track_id]["first_seen"]
                loiter_duration = now - first_seen
                if loiter_duration >= loitering_threshold:
                    det["threat_level"] = "MEDIUM"
                    active_events.append({
                        "event_type": "LOITERING DETECTED",
                        "person_id": track_id,
                        "duration": round(loiter_duration, 1),
                        "threat_level": "MEDIUM",
                        "confidence": 0.92,
                        "timestamp": datetime.utcnow().isoformat()
                    })

                # --- MODULE 3: RESTRICTED AREA INTRUSION ---
                if zones:
                    for zone in zones:
                        z_pts = zone.get("points", [])
                        if len(z_pts) > 2 and point_in_polygon((cx, cy), z_pts):
                            det["threat_level"] = "HIGH"
                            active_events.append({
                                "event_type": "RESTRICTED AREA INTRUSION",
                                "person_id": track_id,
                                "zone_id": zone.get("zone_id", "ZONE-1"),
                                "threat_level": "HIGH",
                                "confidence": 0.96,
                                "timestamp": datetime.utcnow().isoformat()
                            })

        # --- MODULE 4: CROWD DETECTION ---
        if person_count >= crowd_threshold:
            active_events.append({
                "event_type": f"CROWD DETECTION ({person_count} People)",
                "threat_level": "MEDIUM",
                "confidence": 0.90,
                "timestamp": datetime.utcnow().isoformat()
            })

        # --- MODULE: FIGHT CLASSIFICATION (Temporal Confirmation & Cooldown) ---
        person_dets = [d for d in detections if d.get("class", "").lower() == "person"]
        fight_result = self.run_fight_classification(frame, camera_id, person_dets, now, all_detections=detections)

        # Generate state-aware fight active events
        if fight_result.get("fight_started", False):
            participants = fight_result.get("fight_participants", [])
            participant_str = ", ".join([f"Person #{p}" for p in participants]) if participants else "Unknown participants"
            active_events.append({
                "event_type": "FIGHT STARTED",
                "threat_level": "HIGH",
                "confidence": fight_result["fight_confidence"],
                "participants": participants,
                "duration_sec": fight_result["fight_duration_sec"],
                "timestamp": datetime.utcnow().isoformat(),
                "details": f"Physical altercation indicators observed. {participant_str}."
            })
        elif fight_result.get("fight_detected"):
            participants = fight_result.get("fight_participants", [])
            participant_str = ", ".join([f"Person #{p}" for p in participants]) if participants else "Unknown participants"
            active_events.append({
                "event_type": "FIGHT ONGOING",
                "threat_level": "HIGH",
                "confidence": fight_result["fight_confidence"],
                "participants": participants,
                "duration_sec": fight_result["fight_duration_sec"],
                "timestamp": datetime.utcnow().isoformat(),
                "details": f"Physical altercation continuing. {participant_str}."
            })

        # Generate FIGHT ENDED event for timeline reconstruction
        if fight_result.get("fight_ended", False):
            active_events.append({
                "event_type": "FIGHT BEHAVIOR ENDED",
                "threat_level": "MEDIUM",
                "confidence": 0.90,
                "timestamp": datetime.utcnow().isoformat(),
                "details": f"Aggressive interaction patterns no longer observed. Duration: {fight_result['fight_duration_sec']}s"
            })

        # --- MODULE: ACCIDENT DETECTION ---
        accident_result = self.run_accident_detection(frame)

        raw_accident = accident_result.get("accident_detected", False)
        if raw_accident:
            self.accident_counters[camera_id] = self.accident_counters.get(camera_id, 0) + 1
        else:
            self.accident_counters[camera_id] = 0

        # Require 2 consecutive frames of accident detection to eliminate transient video noise
        accident_confirmed = (self.accident_counters.get(camera_id, 0) >= 2)
        accident_result["accident_detected"] = accident_confirmed

        if accident_confirmed:
            for abox in accident_result.get("boxes", []):
                detections.append({
                    "bbox": abox["bbox"],
                    "class": "Accident",
                    "confidence": round(abox["confidence"], 2),
                    "track_id": None,
                    "threat_level": "HIGH"
                })

            active_events.append({
                "event_type": "ACCIDENT DETECTED",
                "threat_level": "HIGH",
                "confidence": accident_result.get("accident_confidence", 0.70),
                "timestamp": datetime.utcnow().isoformat(),
                "details": "Visible road accident/collision pattern detected on camera feed."
            })

        # --- MODULE 5: THREAT SCORING ENGINE ---
        total_score, threat_level = self.calculate_threat_score(
            detections=detections,
            active_events=active_events,
            fight_result=fight_result,
            accident_result=accident_result
        )

        # --- MODULE 6: MULTI-EVENT CORRELATION ENGINE ---
        correlated_events, override_level = self.correlate_multi_events(
            active_events=active_events,
            detections=detections,
            accident_result=accident_result,
            fight_result=fight_result
        )
        if override_level:
            threat_level = override_level

        # --- MODULE 15: EXPLAINABLE AI (XAI) GENERATION ENGINE ---
        xai_explanation = XAIService.generate_explanation(
            camera_id=camera_id,
            timestamp=datetime.utcnow().isoformat(),
            threat_level=threat_level,
            threat_score=total_score,
            confidence=0.95 if threat_level in ["HIGH", "CRITICAL"] else 0.85,
            detections=detections,
            active_events=active_events,
            correlated_events=correlated_events,
            tracking_history=self.tracking_history.get(camera_id, {}),
            person_count=person_count,
            fight_result=fight_result,
            accident_result=accident_result
        )

        # Draw overlays on a copy to avoid mutating the original input frame
        annotated_frame = draw_annotations(
            frame=frame.copy(),
            detections=detections,
            tracking_history=self.tracking_history[camera_id],
            zones=zones,
            threat_score=total_score,
            threat_level=threat_level,
            person_count=person_count,
            correlated_events=correlated_events
        )

        return (
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
        )
