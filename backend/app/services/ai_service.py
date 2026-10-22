import os
import math
import time
import threading
from collections import deque
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

def _box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def _box_distance(box1, box2):
    """Euclidean distance between two bounding boxes (0.0 if overlapping)."""
    dx = max(0.0, box1[0] - box2[2], box2[0] - box1[2])
    dy = max(0.0, box1[1] - box2[3], box2[1] - box1[3])
    return (dx * dx + dy * dy) ** 0.5

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

        # Helper to resolve weights/ first, then fallback
        def _resolve_service_model(configured_path: str, weight_rel: str, fallback_rel: str) -> str:
            if configured_path and os.path.exists(configured_path):
                return configured_path
            p_weight = PROJECT_ROOT / "weights" / weight_rel
            if p_weight.exists():
                return str(p_weight)
            p_fallback = PROJECT_ROOT / fallback_rel
            if p_fallback.exists():
                return str(p_fallback)
            return configured_path or str(p_weight)

        # --- MODULE 0: DEDICATED PERSON DETECTION MODEL (COCO) ---
        person_model_path = _resolve_service_model(
            getattr(settings, "PERSON_MODEL_PATH", ""),
            "person_yolo11s.pt",
            "yolo11s.pt"
        )
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
        weapon_model_path = _resolve_service_model(
            settings.WEAPON_MODEL_PATH,
            "weapon_yolov5.pt",
            os.path.join(
                "OD-WeaponDetection",
                "Weapons and similar handled objects",
                "Sohas_weapon-Detection-YOLOv5",
                "runs",
                "detect",
                "weapon6cls",
                "weights",
                "best.pt"
            )
        )
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
        fight_model_path = _resolve_service_model(
            getattr(settings, "FIGHT_MODEL_PATH", ""),
            "fight_yolo11n_cls.pt",
            os.path.join("runs", "classify", "fight_classifier", "weights", "best.pt")
        )
        try:
            if os.path.exists(fight_model_path):
                self.fight_model = YOLO(fight_model_path)
                self.fight_model.to(self.device)
                logger.info(f"Fight Classification Model loaded ({self.device}): {fight_model_path}")
                print(f"✅ Fight model loaded successfully from {fight_model_path} [{self.device}]")
            else:
                logger.error(f"Fight model file not found at {fight_model_path}")
                print(f"❌ Fight model load failed: path not found at {fight_model_path}")
                self.fight_model = None
        except Exception as e:
            logger.error(f"Failed to load fight model: {e}")
            print(f"❌ Fight model load failed: {e}")
            self.fight_model = None

        # --- MODULE: ACCIDENT DETECTION MODEL ---
        accident_model_path = _resolve_service_model(
            getattr(settings, "ACCIDENT_MODEL_PATH", ""),
            "accident_yolov8.pt",
            os.path.join("runs", "detect", "accident_detector", "weights", "best.pt")
        )
        try:
            if os.path.exists(accident_model_path):
                self.accident_model = YOLO(accident_model_path)
                self.accident_model.to(self.device)
                logger.info(f"Accident Detection Model loaded ({self.device}): {accident_model_path}")
                print(f"✅ Accident model loaded successfully from {accident_model_path} [{self.device}]")
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
        # Rolling 5-frame binary deque per camera per class for weapon confirmation
        self.weapon_buffers = {}
        
        # 2-Frame Consecutive Accident Validation Counter per camera: { camera_id: int }
        self.accident_counters = {}
        # Rolling 4-frame deque for accident confirmation & 15-frame scene motion deque
        self.accident_buffers = {}
        self.accident_zone_motion = {}

        # Frame counters
        self.frame_counters = {}
        self.event_collection = None

        # Temporal Fight Buffer: { camera_id: deque of last 10 fight predictions (True/False) }
        self.fight_buffers = {}

        # Temporal Person Gate Buffer: { camera_id: deque of last 5 bools (2+ persons present) }
        self.person_gate_buffers = {}

        # Fight duration & cooldown tracking: { camera_id: {"start_time": float | None, "active": bool, "non_fight_consecutive": int, "participant_memory": dict, "smoothed_participants": list} }
        self.fight_duration = {}
        self._last_fight_process_time = {}
        # Dynamic centroid velocity tracking history for interacting fight pairs: { camera_id: deque(maxlen=4) }
        self.fight_pair_history = {}

        # Per-camera ByteTrack tracker isolation state and concurrency locks
        self._person_model_lock = threading.Lock()
        self._weapon_model_lock = threading.Lock()
        self._person_trackers = {}  # { camera_id: [tracker instances] }
        self._weapon_trackers = {}  # { camera_id: [tracker instances] }
        self._warned_missing_trackers = False

        # Per-camera unattended weapon tracking state
        self.unattended_weapon_state = {}  # { camera_id: { weapon_key: {"absent_frames": int, "unattended_since": float | None, "is_unattended": bool, "last_seen": float} } }

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
        self.weapon_buffers.pop(camera_id, None)
        self.accident_counters.pop(camera_id, None)
        self.accident_buffers.pop(camera_id, None)
        self.accident_zone_motion.pop(camera_id, None)
        self.fight_buffers.pop(camera_id, None)
        self.person_gate_buffers.pop(camera_id, None)
        self.fight_duration.pop(camera_id, None)
        self._last_fight_process_time.pop(camera_id, None)
        self.fight_pair_history.pop(camera_id, None)
        self.frame_counters.pop(camera_id, None)
        self.unattended_weapon_state.pop(camera_id, None)
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
        raw_score = 0
        fight_result = fight_result or {}
        fight_detected = fight_result.get("fight_detected", False)
        fight_conf = fight_result.get("fight_confidence", 0.0)

        accident_result = accident_result or {}
        accident_detected = accident_result.get("accident_detected", False)
        accident_conf = accident_result.get("accident_confidence", 0.0)

        has_weapon = False
        has_attended_weapon = False
        has_unattended_weapon = False
        max_unattended_dur = 0.0
        has_running = False
        has_intrusion = False

        person_score = 0
        for det in detections:
            cls_name = det.get("class", "").lower()
            conf = det.get("confidence", 0.0)
            is_unatt = det.get("is_unattended", False)
            u_dur = det.get("unattended_duration", 0.0)

            if "pistol" in cls_name or "gun" in cls_name:
                has_weapon = True
                if is_unatt:
                    has_unattended_weapon = True
                    max_unattended_dur = max(max_unattended_dur, u_dur)
                    raw_score += int(40 * conf)
                else:
                    has_attended_weapon = True
                    base = settings.BASE_THREAT_SCORES.get("pistol", 80)
                    raw_score += int(base * conf)
            elif "knife" in cls_name or "scissors" in cls_name:
                has_weapon = True
                if is_unatt:
                    has_unattended_weapon = True
                    max_unattended_dur = max(max_unattended_dur, u_dur)
                    raw_score += int(30 * conf)
                else:
                    has_attended_weapon = True
                    base = settings.BASE_THREAT_SCORES.get("knife", 60)
                    raw_score += int(base * conf)
            elif cls_name == "person":
                person_score = min(person_score + 5, 15)  # Cap person contribution at +15 max
        # Person score adds threat weight only when an active threat, incident, or event is present
        if raw_score > 0 or active_events or (fight_detected and fight_conf >= 0.85) or accident_detected:
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

        # Check for confirmed attended weapons in active_events (Item 1)
        confirmed_attended_weapons = [
            e for e in (active_events or [])
            if e.get("event_type", "").startswith("WEAPON DETECTED") and not e.get("is_unattended", False)
        ]
        has_confirmed_attended_weapon = (len(confirmed_attended_weapons) > 0)

        # Multi-signal escalation overrides guarantee minimum priority levels
        # (Uses confirmed attended weapons only; unattended weapons must not escalate to CRITICAL)
        if fight_detected and fight_conf >= 0.85:
            if has_confirmed_attended_weapon and has_intrusion:
                score = max(score, 90)
            elif has_confirmed_attended_weapon and has_running:
                score = max(score, 100)
            elif has_confirmed_attended_weapon:
                score = max(score, 70)

        # Accident escalation and calibrated scoring (per user specifications):
        # (Uses confirmed attended weapons only; unattended weapons must not escalate to CRITICAL)
        if accident_detected:
            if has_confirmed_attended_weapon and (fight_detected and fight_conf >= 0.85):
                score = 100
            elif has_confirmed_attended_weapon:
                score = max(score, 85)
            elif fight_detected and fight_conf >= 0.85:
                score = max(score, 75)
            elif has_intrusion:
                score = 55
            elif not has_confirmed_attended_weapon and not (fight_detected and fight_conf >= 0.85):
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

        # Attended lethal weapon guarantees appropriate priority ONLY after passing persistence
        # (weapon_counters >= 3 for that class, reflected by WEAPON DETECTED in active_events)
        if confirmed_attended_weapons:
            has_attended_firearm = any(
                any(k in e.get("event_type", "").lower() for k in ["pistol", "gun", "firearm", "rifle"])
                for e in confirmed_attended_weapons
            )
            if has_attended_firearm:
                score = max(score, 85)
                level = "CRITICAL"
            elif level in ["LOW", "MEDIUM"]:
                score = max(score, 65)
                level = "HIGH"

        # Unattended weapon rules (Task E & Task 3):
        # When only unattended weapons are present (no attended weapon, fight, accident, or intrusion):
        # Detection threat_level="MEDIUM", frame threat_score capped below 60, highest_threat="MEDIUM".
        # Only upgrade to HIGH if the weapon remains unattended for >= 30 frames / > 2.0s.
        confirmed_unattended_weapons = [
            e for e in (active_events or [])
            if e.get("event_type", "").startswith("UNATTENDED WEAPON DETECTED")
        ]
        if confirmed_unattended_weapons and not confirmed_attended_weapons and not fight_detected and not accident_detected and not has_intrusion:
            if any(e.get("threat_level") == "HIGH" for e in confirmed_unattended_weapons):
                score = max(score, 65)
                score = min(score, 75)
                level = "HIGH"
            else:
                score = max(score, 45)
                score = min(score, 59)
                level = "MEDIUM"

        # Cap unconfirmed weapons (Item 2):
        # Explicit checks:
        # - Attended: event_type starts with "WEAPON DETECTED"
        # - Unattended: event_type starts with "UNATTENDED WEAPON DETECTED"
        # Handle unattended-HIGH path explicitly.
        # An unconfirmed pistol next to a confirmed unattended knife is still capped at MEDIUM.
        has_weapon_det = any(
            d.get("class", "").lower() in ["pistol", "gun", "firearm", "rifle", "knife", "scissors"]
            for d in detections
        )
        has_confirmed_attended = any(
            e.get("event_type", "").startswith("WEAPON DETECTED") and not e.get("is_unattended", False)
            for e in (active_events or [])
        )
        has_confirmed_unattended_high = any(
            e.get("event_type", "").startswith("UNATTENDED WEAPON DETECTED") and e.get("threat_level") == "HIGH"
            for e in (active_events or [])
        )

        if has_weapon_det and not has_confirmed_attended:
            has_independent_high = (
                (fight_detected and fight_conf >= 0.70) or
                accident_detected or
                has_intrusion or
                has_confirmed_unattended_high
            )
            if not has_independent_high:
                score = min(score, 59)
                if level in ["HIGH", "CRITICAL"]:
                    level = "MEDIUM"
            elif not (fight_detected and fight_conf >= 0.90) and not (accident_detected and has_confirmed_attended):
                # If unattended HIGH or accident alone is present, cap at HIGH (unconfirmed weapon must not push to CRITICAL)
                if level == "CRITICAL":
                    level = "HIGH"
                    score = min(score, 75)

        return score, level

    def run_fight_classification(
        self,
        frame: np.ndarray,
        camera_id: str,
        person_detections: list,
        now: float,
        raw_fight_override: bool = None,
        fight_conf_override: float = None,
        all_detections: list = None,
        is_screen_replay: bool = False
    ) -> dict:
        """
        MODULE: Fight Classification with Temporal Confirmation, Cooldown Period, and ID Reconciliation.
        - Multi-region screen & mobile crop analysis: isolates handheld phone screens and contours.
        - Uses a rolling window with responsive confirmation for mobile screens and surveillance.
        - Cooldown period (15 consecutive non-fight frames) ensures brief classification drops do NOT split an active fight into multiple incidents.
        - Participant ID Reconciliation & Smoothing handles ByteTrack ID drops, swaps, and occlusions.
        Returns fight_result dict.
        """
        COOLDOWN_FRAMES_THRESHOLD = 8  # Cooldown threshold to declare fight ended (optimized from 15 for fast calm recovery)

        # Wall-clock idle timeout: if no frame processed for > 5.0 seconds, clear fight buffers and duration
        last_processed = self._last_fight_process_time.get(camera_id)
        if last_processed is not None and (now - last_processed) > 5.0:
            self.fight_buffers.pop(camera_id, None)
            self.person_gate_buffers.pop(camera_id, None)
            self.fight_duration.pop(camera_id, None)
            self.fight_pair_history.pop(camera_id, None)
            self.weapon_buffers.pop(camera_id, None)
            self.accident_buffers.pop(camera_id, None)
            self.accident_zone_motion.pop(camera_id, None)
        self._last_fight_process_time[camera_id] = now

        # Initialize per-camera fight buffer, person gate, and duration state (Config 4)
        min_persons_to_run = getattr(settings, "FIGHT_MIN_PERSONS_TO_RUN", 2)
        confirm_min_persons = getattr(settings, "FIGHT_CONFIRM_MIN_PERSONS_SEEN", 4)

        if camera_id not in self.fight_buffers:
            self.fight_buffers[camera_id] = deque(maxlen=10)
        if camera_id not in self.person_gate_buffers:
            self.person_gate_buffers[camera_id] = deque(maxlen=10)
        if camera_id not in self.fight_duration:
            self.fight_duration[camera_id] = {
                "start_time": None,
                "active": False,
                "non_fight_consecutive": 0,
                "participant_memory": {},  # {track_id: last_seen_time}
                "smoothed_participants": [],
            }

        # Config 4: Qualified persons (conf >= 0.35)
        qualified_persons = [d for d in person_detections if d.get("confidence", 0.0) >= 0.35]

        # Spatial Interaction / Physical Proximity Gate:
        # A physical fight strictly requires at least two interacting persons in physical contact or striking range on the same depth plane.
        has_interacting_pair = False
        if len(qualified_persons) >= 2:
            for i in range(len(qualified_persons)):
                for j in range(i + 1, len(qualified_persons)):
                    b1 = qualified_persons[i].get("bbox", [0, 0, 0, 0])
                    b2 = qualified_persons[j].get("bbox", [0, 0, 0, 0])
                    w1, h1 = max(1.0, b1[2] - b1[0]), max(1.0, b1[3] - b1[1])
                    w2, h2 = max(1.0, b2[2] - b2[0]), max(1.0, b2[3] - b2[1])

                    # Depth ratio: fighters must be on roughly comparable depth planes
                    depth_ratio = max(h1, h2) / min(h1, h2)
                    if depth_ratio > 2.8:
                        continue

                    # Overlap / physical entanglement check
                    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                    overlap_w = max(0.0, xi2 - xi1)
                    overlap_h = max(0.0, yi2 - yi1)
                    min_w, min_h = min(w1, w2), min(h1, h2)
                    min_area = min_w * min_h

                    # Physical fight strictly requires actual physical contact/entanglement
                    # Discard slight boundary jitter (requires at least 5% width, 10% height, 4% area overlap)
                    if overlap_w >= 0.05 * min_w and overlap_h >= 0.10 * min_h:
                        inter_area = overlap_w * overlap_h
                        iou = inter_area / (w1 * h1 + w2 * h2 - inter_area) if (w1 * h1 + w2 * h2 - inter_area) > 0 else 0
                        overlap_ratio = inter_area / min_area if min_area > 0 else 0
                        if iou > 0.60:
                            continue  # Duplicate detection of same person
                        if overlap_ratio >= 0.04:
                            # Dynamic centroid velocity cross-check over last 4 frames
                            c1 = ((b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0)
                            c2 = ((b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0)
                            rel_dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])

                            if camera_id not in self.fight_pair_history:
                                self.fight_pair_history[camera_id] = deque(maxlen=4)
                            self.fight_pair_history[camera_id].append((rel_dist, now))

                            # Calculate displacement speed delta_d / delta_t across rolling window
                            history = list(self.fight_pair_history[camera_id])
                            is_velocity_valid = True
                            if len(history) >= 3:
                                speeds = []
                                for k in range(1, len(history)):
                                    d_prev, t_prev = history[k - 1]
                                    d_curr, t_curr = history[k]
                                    dt = max(1e-3, t_curr - t_prev)
                                    speeds.append(abs(d_curr - d_prev) / dt)
                                vel_variance = float(np.var(speeds)) if len(speeds) > 1 else 0.0
                                mean_speed = float(np.mean(speeds)) if speeds else 0.0
                                # Slow stationary hugging/walking has negligible relative displacement variance
                                if vel_variance < 0.25 and mean_speed < 2.0:
                                    is_velocity_valid = False

                            if is_velocity_valid:
                                has_interacting_pair = True
                                break
                if has_interacting_pair:
                    break

        # Motion Gate: A physical fight is an active, violent event.
        # Requires active motion (MSE >= 1.40) observed in at least 2 of the last 6 frames.
        motion_hist = self.health_buffers.get(camera_id, {}).get("motion_history", [])
        active_motion_frames = sum(
            1 for m in motion_hist
            if (float(m) if isinstance(m, (int, float)) and not isinstance(m, bool) else (1.40 if m else 0.0)) >= 1.40
        )
        is_motion_active = (active_motion_frames >= 2)

        # Sliding window records presence of interacting candidates
        has_qualifying_interaction = (len(qualified_persons) >= min_persons_to_run) and has_interacting_pair
        self.person_gate_buffers[camera_id].append(has_qualifying_interaction)

        # Gate to run classifier: min persons, spatial interaction, and non-frozen scene
        # Presentation & Screen-Replay Mode (CAM-001): DO NOT block fight classifier behind len(persons) >= 2
        can_run_classifier = (has_qualifying_interaction and is_motion_active) or is_screen_replay or (camera_id == "CAM-001")

        dur_tracker = self.fight_duration[camera_id]
        fight_conf = 0.0
        raw_fight = False

        if raw_fight_override is not None:
            raw_fight = raw_fight_override
            fight_conf = fight_conf_override if fight_conf_override is not None else (0.90 if raw_fight else 0.0)
        elif self.fight_model is not None and frame is not None and frame.size > 0:
            try:
                if can_run_classifier:
                    results = self.fight_model.predict([frame], verbose=False)
                    if results and results[0].probs is not None:
                        probs = results[0].probs
                        names = results[0].names
                        top1_idx = int(probs.top1)
                        top1_conf = float(probs.top1conf)
                        top1_name = names.get(top1_idx, "").lower()

                        fight_thresh = 0.65 if (is_screen_replay or camera_id == "CAM-001") else 0.80
                        if top1_name == "fight" and top1_conf >= fight_thresh:
                            raw_fight = True
                            fight_conf = top1_conf
            except Exception as e:
                logger.warning(f"Fight model inference error on {camera_id}: {e}")

        # Push raw result into temporal buffer
        self.fight_buffers[camera_id].append((raw_fight, fight_conf))

        # Temporal confirmation:
        buf = self.fight_buffers[camera_id]
        fight_frames = sum(1 for f, c in buf if f and (c >= 0.65 if (is_screen_replay or camera_id == "CAM-001") else c >= 0.80))
        if is_screen_replay or camera_id == "CAM-001":
            # Screen replay / webcam: latch altercation immediately, require only 2 consecutive frames or 2 of 4
            recent_4 = list(buf)[-4:]
            fight_in_recent_4 = sum(1 for f, c in recent_4 if f and c >= 0.65)
            consec_2 = len(buf) >= 2 and buf[-1][0] and buf[-2][0]
            buffer_confirmed = raw_fight or (fight_in_recent_4 >= 2) or consec_2
        elif raw_fight_override is not None:
            buffer_confirmed = (fight_frames >= 6)
        else:
            persons_seen_in_last_10 = sum(1 for h in self.person_gate_buffers[camera_id] if h)
            buffer_confirmed = (
                len(buf) >= 6 and
                fight_frames >= 6 and
                persons_seen_in_last_10 >= confirm_min_persons
            )

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


    def run_accident_detection(self, frame: np.ndarray, screen_boxes: list = None, conf_threshold: float = None) -> dict:
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

            accident_conf_thresh = conf_threshold if conf_threshold is not None else getattr(settings, "ACCIDENT_CONFIDENCE_THRESHOLD", 0.52)

            # Full camera frame evaluation (calibrated threshold >= 0.52)
            results = self.accident_model(frame, verbose=False)
            if results and len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_idx = int(box.cls[0])
                    raw_cls = str(self.accident_model.names.get(cls_idx, "")).lower()
                    conf = float(box.conf[0])

                    if ("accident" in raw_cls or cls_idx == 0) and conf >= accident_conf_thresh:
                        bbox = box.xyxy[0].tolist()
                        boxes.append({
                            "bbox": bbox,
                            "confidence": conf,
                            "source": "full_frame"
                        })
                        if conf > best_conf:
                            best_conf = conf

            if best_conf >= accident_conf_thresh and len(boxes) > 0:
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
        
        correlated = []
        override_level = None

        has_fight = any("FIGHT" in e for e in event_types) or (fight_result and fight_result.get("fight_detected", False))
        has_crowd = any("CROWD" in e for e in event_types)
        has_intrusion = any("INTRUSION" in e for e in event_types)

        # Correlation rules use ONLY confirmed attended weapon events (starts with "WEAPON DETECTED", not "UNATTENDED")
        confirmed_attended_knife = any(
            e.get("event_type", "").startswith("WEAPON DETECTED") and "KNIFE" in e.get("event_type", "").upper() and not e.get("is_unattended", False)
            for e in active_events
        ) or any(
            "knife" in d.get("class", "").lower() and not d.get("is_unattended", False)
            for d in (detections or [])
        )
        confirmed_attended_pistol = any(
            e.get("event_type", "").startswith("WEAPON DETECTED") and any(k in e.get("event_type", "").upper() for k in ["PISTOL", "GUN", "FIREARM"]) and not e.get("is_unattended", False)
            for e in active_events
        ) or any(
            any(k in d.get("class", "").lower() for k in ["pistol", "gun", "firearm"]) and not d.get("is_unattended", False)
            for d in (detections or [])
        )
        # For accident, strictly require confirmed attended weapon from active_events (unattended weapons never escalate accident)
        confirmed_attended_weapon_event = any(
            e.get("event_type", "").startswith("WEAPON DETECTED") and not e.get("is_unattended", False)
            for e in active_events
        )

        accident_result = accident_result or {}
        has_accident = (
            any("ACCIDENT" in e for e in event_types) or
            accident_result.get("accident_detected", False)
        )

        if has_fight and confirmed_attended_knife:
            correlated.append("FIGHT + KNIFE COMBINATION (CRITICAL)")
            override_level = "CRITICAL"
        if has_fight and confirmed_attended_pistol:
            correlated.append("FIGHT + PISTOL THREAT (CRITICAL)")
            override_level = "CRITICAL"
        if confirmed_attended_pistol and has_intrusion:
            correlated.append("PISTOL IN RESTRICTED AREA (CRITICAL)")
            override_level = "CRITICAL"
        if confirmed_attended_knife and has_intrusion:
            correlated.append("KNIFE IN RESTRICTED AREA (HIGH)")
            if override_level != "CRITICAL":
                override_level = "HIGH"
        if has_crowd and confirmed_attended_pistol:
            correlated.append("CROWD + PISTOL THREAT (CRITICAL)")
            override_level = "CRITICAL"
        if has_crowd and confirmed_attended_knife:
            correlated.append("CROWD + KNIFE THREAT (HIGH)")
            if override_level != "CRITICAL":
                override_level = "HIGH"

        # --- Accident Correlations (Confirmed Attended Weapon Events Only) ---
        if has_accident and confirmed_attended_weapon_event:
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
        crowd_threshold: int = 10,
        is_demo_mode: bool = False
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
        current_frame_idx = self.frame_counters[camera_id]

        # Identify Presentation & Screen-Replay Mode (CAM-001 / Demo Webcam)
        is_screen_replay = (
            is_demo_mode
            or camera_id in ["CAM-001", "CAM-01", "webcam"]
            or camera_id.startswith("CAM-001")
            or (camera_name and "webcam" in camera_name.lower())
            or "demo" in camera_id.lower()
        )

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

        if "motion_history" not in self.health_buffers[camera_id]:
            self.health_buffers[camera_id]["motion_history"] = deque(maxlen=6)
        self.health_buffers[camera_id]["motion_history"].append(float(mse))

        # ===== CONTINUOUS DEBUG LOG (every 30 frames) =====
        freeze_count = self.health_buffers[camera_id]["freeze_counter"]
        if self.frame_counters[camera_id] % 30 == 0:
            logger.debug(f"[FRAME METRICS] cam={camera_id} | frame={self.frame_counters[camera_id]} | "
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
            logger.warning(f"[TAMPER ALERT] Cam={camera_id} | {tamper_reason} | Brightness={brightness:.2f} | StdDev={stddev:.2f} | MSE={mse:.4f}")

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
                    logger.warning(f"[TAMPER ALERT] Cam={camera_id} | {tamper_reason} | Brightness={brightness:.2f} | StdDev={stddev:.2f} | MSE={mse:.4f}")

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
            logger.warning(f"[TAMPER ALERT] Cam={camera_id} | {tamper_reason} | Brightness={brightness:.2f} | StdDev={stddev:.2f} | MSE={mse:.4f}")

            active_events.append({
                "event_type": "CAMERA DISCONNECTED",
                "threat_level": "HIGH",
                "confidence": 0.99,
                "timestamp": datetime.utcnow().isoformat()
            })

        self.health_buffers[camera_id]["last_frame"] = gray.copy()

        # Initialize weapon persistence counter and deque buffer for camera
        if camera_id not in self.weapon_counters:
            self.weapon_counters[camera_id] = {"pistol": 0, "knife": 0}
        if camera_id not in self.weapon_buffers:
            self.weapon_buffers[camera_id] = {
                "pistol": deque(maxlen=5),
                "knife": deque(maxlen=5)
            }

        frame_detected_attended_weapons = set()

        # --- MODULE 0: DEDICATED PERSON & VEHICLE DETECTION (ByteTrack) ---
        if self.person_model is not None:
            try:
                person_results = self._track_with_camera_state(
                    self.person_model,
                    frame,
                    camera_id,
                    self._person_trackers,
                    self._person_model_lock,
                    classes=[0, 2, 3, 5, 7],
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False
                )
                if person_results and person_results[0].boxes is not None:
                    for box in person_results[0].boxes:
                        cls_idx = int(box.cls[0])
                        conf = float(box.conf[0])
                        raw_name = self.person_model.names.get(cls_idx, "").lower() if hasattr(self.person_model, "names") else ""

                        if cls_idx == 0 or raw_name == "person":
                            if conf < 0.30:
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
                        elif cls_idx in [2, 3, 5, 7] or raw_name in ["car", "motorcycle", "bus", "truck"]:
                            if conf < 0.35:
                                continue
                            bbox = box.xyxy[0].tolist()
                            track_id = int(box.id[0]) if box.id is not None else None
                            display_name = raw_name.capitalize() if raw_name else "Vehicle"
                            detections.append({
                                "bbox": bbox,
                                "class": display_name,
                                "confidence": conf,
                                "track_id": track_id,
                                "threat_level": "LOW"
                            })
            except Exception as e:
                logger.error(f"Person/Vehicle track error: {e}")

        # --- MODULE 1: WEAPON & OBJECT DETECTION (ByteTrack) ---
        # Single person set (conf >= 0.30) for weapon ceiling (Item 1)
        ceiling_persons = [
            d for d in detections
            if d.get("class", "").lower() == "person" and d.get("confidence", 0.0) >= 0.30
        ]
        # Qualified persons (conf >= 0.30) for attended/unattended status (Patch 1: lowered from 0.50 to 0.30)
        qualified_persons = [
            d for d in ceiling_persons
            if d.get("confidence", 0.0) >= 0.30
        ]

        current_frame_weapon_keys = set()

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

                        # Calibrated thresholds (Patch 1 & Screen Replay Mode):
                        # On screen replay/webcam (CAM-001), pistol or knife conf >= 0.32
                        is_security_weapon = raw_cls in settings.SECURITY_CLASSES or raw_cls in ["gun", "firearm", "rifle", "knife", "scissors", "pistol"]
                        if is_screen_replay and (raw_cls in ["pistol", "gun", "knife", "scissors"] or is_security_weapon):
                            required_conf = 0.32
                        elif raw_cls in ["pistol", "gun"]:
                            required_conf = getattr(settings, "WEAPON_CONFIDENCE_THRESHOLD", 0.38)
                        elif raw_cls in ["knife", "scissors"]:
                            required_conf = 0.40
                        elif is_security_weapon:
                            required_conf = getattr(settings, "WEAPON_CONFIDENCE_THRESHOLD", 0.38)
                        else:
                            required_conf = 0.25

                        if conf < required_conf:
                            continue

                        bbox = box.xyxy[0].tolist()
                        wx1, wy1, wx2, wy2 = bbox
                        weapon_area = max(0.0, (wx2 - wx1) * (wy2 - wy1))
                        frame_area = float(frame.shape[0] * frame.shape[1])

                        # Area ceiling check (Item 1):
                        # On screen replay mode, bypass 25% ceiling check to allow close-ups on screen
                        rejected_by_ceiling = False
                        if not is_screen_replay:
                            overlapping_person_areas = []
                            for p in ceiling_persons:
                                px1, py1, px2, py2 = p["bbox"]
                                ix1 = max(wx1, px1)
                                iy1 = max(wy1, py1)
                                ix2 = min(wx2, px2)
                                iy2 = min(wy2, py2)
                                if ix2 > ix1 and iy2 > iy1:
                                    p_area = max(1.0, (px2 - px1) * (py2 - py1))
                                    overlapping_person_areas.append(p_area)

                            if overlapping_person_areas:
                                largest_p_area = max(overlapping_person_areas)
                                if weapon_area > 0.50 * largest_p_area:
                                    rejected_by_ceiling = True
                            else:
                                if frame_area > 0 and (weapon_area / frame_area) > 0.25:
                                    rejected_by_ceiling = True

                        if rejected_by_ceiling:
                            continue

                        # --- UNATTENDED WEAPON LOGIC (Items 3 & 4) ---
                        wcx = (wx1 + wx2) / 2.0
                        wcy = (wy1 + wy2) / 2.0

                        # On screen replay mode, bypass attended person proximity check (trigger even if no person body recognized)
                        if is_screen_replay:
                            person_present = True
                        else:
                            person_present = False
                            for p in qualified_persons:
                                px1, py1, px2, py2 = p["bbox"]
                                ph = max(1.0, py2 - py1)
                                # 1. Overlap condition
                                if px1 < wx2 and px2 > wx1 and py1 < wy2 and py2 > wy1:
                                    person_present = True
                                    break
                                # 2. Distance to person box edge <= 0.45 * person box height (Patch 1: relaxed from 0.25)
                                dist = _box_distance([wx1, wy1, wx2, wy2], [px1, py1, px2, py2])
                                if dist <= 0.45 * ph:
                                    person_present = True
                                    break

                        track_id = int(box.id[0]) if box.id is not None else None
                        weapon_key = str(track_id) if track_id is not None else f"untracked_{raw_cls}"
                        current_frame_weapon_keys.add(weapon_key)

                        if camera_id not in self.unattended_weapon_state:
                            self.unattended_weapon_state[camera_id] = {}

                        is_first_sight = (weapon_key not in self.unattended_weapon_state[camera_id])

                        if is_first_sight:
                            # Item 4: IoU inheritance for new weapon track of same class appearing within 10 frames
                            # with box IoU > 0.30 or center dist < 50 px inherits state
                            best_match_key = None
                            best_match_metric = -1.0
                            for cand_key, cand_state in self.unattended_weapon_state[camera_id].items():
                                if cand_key in current_frame_weapon_keys:
                                    continue
                                if cand_state.get("raw_cls") != raw_cls:
                                    continue
                                cand_last_frame = cand_state.get("last_seen_frame", 0)
                                if (current_frame_idx - cand_last_frame) <= 10:
                                    cand_bbox = cand_state.get("bbox")
                                    iou = _box_iou([wx1, wy1, wx2, wy2], cand_bbox) if cand_bbox else 0.0
                                    cand_center = cand_state.get("center", (0, 0))
                                    c_dist = math.hypot(wcx - cand_center[0], wcy - cand_center[1])
                                    if iou > 0.30 or c_dist < 50.0:
                                        metric = iou if iou > 0.30 else (1.0 / (1.0 + c_dist))
                                        if metric > best_match_metric:
                                            best_match_metric = metric
                                            best_match_key = cand_key

                            if best_match_key is not None:
                                cand = self.unattended_weapon_state[camera_id].pop(best_match_key)
                                w_state = {
                                    "raw_cls": raw_cls,
                                    "bbox": [wx1, wy1, wx2, wy2],
                                    "center": (wcx, wcy),
                                    "initial_center": cand.get("initial_center", (wcx, wcy)),
                                    "max_displacement": cand.get("max_displacement", 0.0),
                                    "absent_frames": cand.get("absent_frames", 0),
                                    "consecutive_unattended_frames": cand.get("consecutive_unattended_frames", 0),
                                    "is_unattended": cand.get("is_unattended", False),
                                    "first_seen_frame": cand.get("first_seen_frame", current_frame_idx),
                                    "last_seen_frame": current_frame_idx,
                                    "last_seen": now,
                                    "conf": conf
                                }
                            else:
                                if not person_present:
                                    w_state = {
                                        "raw_cls": raw_cls,
                                        "bbox": [wx1, wy1, wx2, wy2],
                                        "center": (wcx, wcy),
                                        "initial_center": (wcx, wcy),
                                        "max_displacement": 0.0,
                                        "absent_frames": 0,
                                        "consecutive_unattended_frames": 1,
                                        "is_unattended": True,
                                        "first_seen_frame": current_frame_idx,
                                        "last_seen_frame": current_frame_idx,
                                        "last_seen": now,
                                        "conf": conf
                                    }
                                else:
                                    w_state = {
                                        "raw_cls": raw_cls,
                                        "bbox": [wx1, wy1, wx2, wy2],
                                        "center": (wcx, wcy),
                                        "initial_center": (wcx, wcy),
                                        "max_displacement": 0.0,
                                        "absent_frames": 0,
                                        "consecutive_unattended_frames": 0,
                                        "is_unattended": False,
                                        "first_seen_frame": current_frame_idx,
                                        "last_seen_frame": current_frame_idx,
                                        "last_seen": now,
                                        "conf": conf
                                    }
                            self.unattended_weapon_state[camera_id][weapon_key] = w_state
                        else:
                            w_state = self.unattended_weapon_state[camera_id][weapon_key]
                            w_state["raw_cls"] = raw_cls
                            w_state["bbox"] = [wx1, wy1, wx2, wy2]
                            w_state["center"] = (wcx, wcy)
                            w_state["last_seen_frame"] = current_frame_idx
                            w_state["last_seen"] = now
                            w_state["conf"] = conf

                        # Update displacement from initial_center
                        init_cx, init_cy = w_state.get("initial_center", (wcx, wcy))
                        disp = math.hypot(wcx - init_cx, wcy - init_cy)
                        w_state["max_displacement"] = max(w_state.get("max_displacement", 0.0), disp)

                        # Update attended / unattended state transitions
                        if person_present:
                            # Immediate recovery when a qualified person is present
                            w_state["absent_frames"] = 0
                            w_state["is_unattended"] = False
                            w_state["consecutive_unattended_frames"] = 0
                            w_state["initial_center"] = (wcx, wcy)
                            w_state["max_displacement"] = 0.0
                        else:
                            if w_state["is_unattended"]:
                                if not is_first_sight:
                                    w_state["consecutive_unattended_frames"] += 1
                            else:
                                # 10-frame hysteresis applies ONLY to attended -> unattended transition after person leaves
                                w_state["absent_frames"] += 1
                                if w_state["absent_frames"] >= 10:
                                    w_state["is_unattended"] = True
                                    w_state["consecutive_unattended_frames"] = 1
                                    w_state["initial_center"] = (wcx, wcy)
                                    w_state["max_displacement"] = 0.0

                        is_unattended = w_state["is_unattended"]
                        u_frames = w_state.get("consecutive_unattended_frames", 0)
                        max_disp = w_state.get("max_displacement", 0.0)
                        is_stationary = (max_disp < 20.0)

                        display_name = raw_cls.capitalize()

                        unattended_high_frames = getattr(settings, "UNATTENDED_HIGH_AFTER_FRAMES", 30)
                        if is_unattended:
                            if u_frames >= unattended_high_frames and is_stationary:
                                threat_level = "HIGH"
                            elif u_frames >= 15 and is_stationary:
                                threat_level = "MEDIUM"
                            else:
                                threat_level = "LOW"
                        else:
                            if raw_cls in ["pistol", "gun"]:
                                threat_level = "CRITICAL"
                            elif raw_cls in ["knife", "scissors"]:
                                threat_level = "HIGH"
                            else:
                                threat_level = "LOW"

                        if not is_unattended:
                            if raw_cls in ["pistol", "gun"]:
                                frame_detected_attended_weapons.add("pistol")
                            elif raw_cls in ["knife", "scissors"]:
                                frame_detected_attended_weapons.add("knife")

                        detections.append({
                            "bbox": bbox,
                            "class": display_name,
                            "confidence": conf,
                            "track_id": track_id,
                            "threat_level": threat_level,
                            "is_unattended": is_unattended,
                            "unattended_duration": round(u_frames / 15.0, 2)
                        })
            except Exception as e:
                logger.error(f"YOLO track error: {e}")

        # Item 3: Prune entries not seen for 5 seconds or 30 frames and reset absent_frames for undetected attended weapons
        if camera_id in self.unattended_weapon_state:
            stale_keys = []
            for k, s in self.unattended_weapon_state[camera_id].items():
                if now - s.get("last_seen", now) > 5.0 or (current_frame_idx - s.get("last_seen_frame", 0)) > 30:
                    stale_keys.append(k)
                elif k not in current_frame_weapon_keys:
                    if not s.get("is_unattended", False):
                        s["absent_frames"] = 0
            for k in stale_keys:
                del self.unattended_weapon_state[camera_id][k]

        # --- ATTENDED WEAPON PERSISTENCE (Rolling 5-frame binary deque: >= 2 of last 4 or >= 3 of 5) ---
        for sec_cls in ["pistol", "knife"]:
            if sec_cls not in self.weapon_buffers[camera_id]:
                self.weapon_buffers[camera_id][sec_cls] = deque(maxlen=5)

            is_present = 1 if sec_cls in frame_detected_attended_weapons else 0
            self.weapon_buffers[camera_id][sec_cls].append(is_present)

            # Maintain legacy integer counter for test suite compatibility
            if is_present:
                self.weapon_counters[camera_id][sec_cls] = self.weapon_counters[camera_id].get(sec_cls, 0) + 1
            else:
                self.weapon_counters[camera_id][sec_cls] = 0

            w_buf = self.weapon_buffers[camera_id][sec_cls]
            recent_4 = list(w_buf)[-4:]
            if is_screen_replay:
                weapon_confirmed = (is_present == 1 and len(w_buf) == 1) or (sum(recent_4) >= 2) or (self.weapon_counters[camera_id][sec_cls] >= 2)
            else:
                weapon_confirmed = (sum(recent_4) >= 2) or (sum(w_buf) >= 3) or (self.weapon_counters[camera_id][sec_cls] >= 3)

            if weapon_confirmed:
                matching_dets = [
                    d for d in detections
                    if (sec_cls in d["class"].lower() or (sec_cls == "pistol" and "gun" in d["class"].lower()))
                    and not d.get("is_unattended", False)
                ]
                max_conf = max([d["confidence"] for d in matching_dets], default=0.85) if matching_dets else 0.85
                event_type = f"WEAPON DETECTED ({sec_cls.upper()})"
                threat_level = "CRITICAL" if sec_cls == "pistol" else "HIGH"

                active_events.append({
                    "event_type": event_type,
                    "class_name": sec_cls,
                    "threat_level": threat_level,
                    "confidence": round(max_conf, 2),
                    "timestamp": datetime.utcnow().isoformat(),
                    "is_unattended": False,
                    "unattended_duration": 0.0
                })

        # --- UNATTENDED WEAPON PERSISTENCE (15 consecutive frames + stationary < 20 px) ---
        unattended_event_classes = set()
        if camera_id in self.unattended_weapon_state:
            for k, s in self.unattended_weapon_state[camera_id].items():
                if k in current_frame_weapon_keys and s.get("is_unattended", False):
                    u_frames = s.get("consecutive_unattended_frames", 0)
                    disp = s.get("max_displacement", 0.0)
                    if u_frames >= 15 and disp < 20.0:
                        raw_c = s.get("raw_cls", "")
                        sec_cls = "pistol" if raw_c in ["pistol", "gun"] else ("knife" if raw_c in ["knife", "scissors"] else raw_c)
                        if sec_cls in ["pistol", "knife"] and sec_cls not in unattended_event_classes:
                            unattended_event_classes.add(sec_cls)
                            event_type = f"UNATTENDED WEAPON DETECTED ({sec_cls.upper()})"
                            threat_level = "HIGH" if u_frames >= unattended_high_frames else "MEDIUM"
                            conf = s.get("conf", 0.85)

                            active_events.append({
                                "event_type": event_type,
                                "class_name": sec_cls,
                                "threat_level": threat_level,
                                "confidence": round(conf, 2),
                                "timestamp": datetime.utcnow().isoformat(),
                                "is_unattended": True,
                                "unattended_duration": round(u_frames / 15.0, 2)
                            })

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
        fight_result = self.run_fight_classification(
            frame, camera_id, person_dets, now, all_detections=detections, is_screen_replay=is_screen_replay
        )

        # Generate state-aware fight active events
        if fight_result.get("fight_started", False) or (is_screen_replay and fight_result.get("fight_detected")):
            participants = fight_result.get("fight_participants", [])
            participant_str = ", ".join([f"Person #{p}" for p in participants]) if participants else "Screen Replay / Visual Altercation"
            event_name = "ALTERCATION / FIGHT DETECTED" if is_screen_replay else "FIGHT STARTED"
            active_events.append({
                "event_type": event_name,
                "threat_level": "HIGH",
                "confidence": max(fight_result.get("fight_confidence", 0.65), 0.65),
                "participants": participants,
                "duration_sec": fight_result.get("fight_duration_sec", 0.0),
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
        if camera_id not in self.accident_buffers:
            self.accident_buffers[camera_id] = deque(maxlen=4)
        if camera_id not in self.accident_zone_motion:
            self.accident_zone_motion[camera_id] = deque(maxlen=15)
        self.accident_zone_motion[camera_id].append(float(mse))

        # Run accident detection (confidence >= 0.40 on screen replay / CAM-001)
        acc_conf_req = 0.40 if is_screen_replay else getattr(settings, "ACCIDENT_CONFIDENCE_THRESHOLD", 0.52)
        accident_result = self.run_accident_detection(frame, conf_threshold=acc_conf_req)
        raw_accident = accident_result.get("accident_detected", False)

        if is_screen_replay:
            # Presentation & Screen-Replay Mode (CAM-001):
            # Bypass vehicle correlation check & bypass total scene motion gate
            valid_accident_boxes = accident_result.get("boxes", [])
            self.accident_buffers[camera_id].append(1 if raw_accident else 0)
            if raw_accident:
                self.accident_counters[camera_id] = self.accident_counters.get(camera_id, 0) + 1
            else:
                self.accident_counters[camera_id] = 0

            # Latch detection if present in 2 consecutive frames or single-frame simulation
            accident_confirmed = raw_accident or (self.accident_counters.get(camera_id, 0) >= 2) or (sum(self.accident_buffers[camera_id]) >= 2)
            accident_result["accident_detected"] = accident_confirmed
        else:
            # Standard CCTV accident pipeline with vehicle overlap and motion gate
            valid_accident_boxes = []
            if raw_accident and accident_result.get("boxes"):
                vehicle_classes = {"car", "motorcycle", "bus", "truck", "vehicle"}
                tracked_vehicles = [
                    d for d in detections
                    if d.get("class", "").lower() in vehicle_classes
                ]
                for abox in accident_result.get("boxes", []):
                    abox_coords = abox.get("bbox", [0, 0, 0, 0])
                    acx = (abox_coords[0] + abox_coords[2]) / 2.0
                    acy = (abox_coords[1] + abox_coords[3]) / 2.0
                    vehicle_matched = False
                    for v in tracked_vehicles:
                        v_bbox = v.get("bbox", [0, 0, 0, 0])
                        iou = _box_iou(abox_coords, v_bbox)
                        vcx = (v_bbox[0] + v_bbox[2]) / 2.0
                        vcy = (v_bbox[1] + v_bbox[3]) / 2.0
                        c_dist = math.hypot(acx - vcx, acy - vcy)
                        if iou >= 0.15 or c_dist < 80.0:
                            vehicle_matched = True
                            break
                    if vehicle_matched:
                        valid_accident_boxes.append(abox)

                # Standalone accident boxes with no vehicles nearby are suppressed
                if valid_accident_boxes:
                    accident_result["boxes"] = valid_accident_boxes
                else:
                    raw_accident = False

            # Total Scene Motion Gate (Patch 3)
            # Discard accident detections if total scene motion (MSE) across prior 15 frames is < 0.20
            motion_15 = list(self.accident_zone_motion[camera_id])
            if len(motion_15) >= 5 and float(np.mean(motion_15)) < 0.20:
                raw_accident = False

            # Sustained Detection: Accident must be detected across >= 2 out of 4 frames in a rolling window
            self.accident_buffers[camera_id].append(1 if raw_accident else 0)
            if raw_accident:
                self.accident_counters[camera_id] = self.accident_counters.get(camera_id, 0) + 1
            else:
                self.accident_counters[camera_id] = 0

            accident_confirmed = (sum(self.accident_buffers[camera_id]) >= 2) or (self.accident_counters.get(camera_id, 0) >= 2)
            accident_result["accident_detected"] = accident_confirmed

        if accident_confirmed:
            boxes_to_add = valid_accident_boxes if valid_accident_boxes else accident_result.get("boxes", [])
            for abox in boxes_to_add:
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

        # Presentation & Screen-Replay Mode Threat Escalation:
        if is_screen_replay:
            if accident_result.get("accident_detected", False):
                total_score = max(total_score, 80)
                threat_level = "CRITICAL"
            if fight_result.get("fight_detected", False):
                total_score = max(total_score, 75)
                if threat_level in ["LOW", "MEDIUM"]:
                    threat_level = "HIGH"

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
