"""
Eagle Eye SOC — Fight Detection Pipeline Test Suite
Tests the temporal confirmation buffer, confidence-weighted threat scoring,
and multi-signal escalation rules for the fight detection module.

Run from backend directory:
    cd D:\\AI SUrvellinace\\backend
    python test_fight_pipeline.py
"""

import sys
import os
import numpy as np
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))

# -----------------------------------------------------------------------
# Test helpers
# -----------------------------------------------------------------------

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    suffix = f" — {detail}" if detail else ""
    print(f"  {status} {label}{suffix}")
    return condition

# -----------------------------------------------------------------------
# Test 1: Fight model loads and classifies a fight image
# -----------------------------------------------------------------------

def test_fight_model_load_and_detect():
    print("\n=== TEST 1: Fight Model Load & Fight Classification ===")
    model_path = r"D:\AI SUrvellinace\runs\classify\fight_classifier\weights\best.pt"
    ok = True

    if not os.path.exists(model_path):
        print(f"  \033[93m[SKIP]\033[0m Model not found at {model_path} — skipping inference test.")
        return True

    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        ok &= check("Model loaded", model is not None)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = model.predict(frame, verbose=False)
        ok &= check("Prediction returned results", results is not None and len(results) > 0)

        if results and results[0].probs is not None:
            probs = results[0].probs
            names = results[0].names
            top1_name = names.get(int(probs.top1), "").lower()
            top1_conf = float(probs.top1conf)
            ok &= check("Top-1 class is valid ('fight' or 'nonfight')",
                        top1_name in ["fight", "nonfight"],
                        f"got: {top1_name} ({top1_conf:.3f})")
            print(f"  [INFO] Model classified frame as: {top1_name.upper()} ({top1_conf*100:.1f}%)")
    except Exception as e:
        ok = check("Model inference no exception", False, str(e))

    return ok

# -----------------------------------------------------------------------
# Test 2: Temporal Confirmation Buffer — 7/10 frames triggers alert
# -----------------------------------------------------------------------

def test_temporal_confirmation():
    print("\n=== TEST 2: Temporal Confirmation Buffer (7/10 Frames) ===")
    ok = True

    # --- Part A: 6/10 should NOT confirm ---
    buf_a = deque(maxlen=10)
    for i in range(10):
        is_fight = i < 6  # 6 Fight, 4 NonFight
        buf_a.append((is_fight, 0.90 if is_fight else 0.0))

    count_a = sum(1 for f, _ in buf_a if f)
    confirmed_a = len(buf_a) >= 10 and count_a >= 7
    ok &= check("6/10 frames does NOT confirm fight", not confirmed_a, f"fight_frames={count_a}")

    # --- Part B: 7/10 SHOULD confirm (fresh buffer) ---
    buf_b = deque(maxlen=10)
    for i in range(10):
        is_fight = i < 7  # exactly 7 Fight, 3 NonFight
        buf_b.append((is_fight, 0.91 if is_fight else 0.0))

    count_b = sum(1 for f, _ in buf_b if f)
    confirmed_b = len(buf_b) >= 10 and count_b >= 7
    ok &= check("7/10 frames DOES confirm fight", confirmed_b, f"fight_frames={count_b}")

    # --- Part C: Confidence averaging ---
    fight_confs = [c for f, c in buf_b if f]
    avg_conf = round(sum(fight_confs) / len(fight_confs), 4)
    ok &= check("Average fight confidence is reasonable", 0.85 <= avg_conf <= 1.0, f"avg_conf={avg_conf}")

    # --- Part D: Add 3 NonFight frames → pushes 3 oldest Fight frames out → 4 remain ---
    for _ in range(3):
        buf_b.append((False, 0.0))
    count_d = sum(1 for f, _ in buf_b if f)
    confirmed_d = len(buf_b) >= 10 and count_d >= 7
    ok &= check("After 3 NonFight frames, confirmation resets", not confirmed_d, f"fight_frames={count_d}")

    return ok

# -----------------------------------------------------------------------
# Test 3: Threat Score — Fight + Weapon → Target ≥ 70, Level = HIGH
# -----------------------------------------------------------------------

def test_fight_weapon_threat_score():
    print("\n=== TEST 3: Fight + Weapon → Score ≥ 70 (HIGH) ===")
    ok = True

    from app.services.ai_service import AIService
    svc = AIService()

    detections = [{"class": "pistol", "confidence": 0.88, "track_id": 5}]
    active_events = []
    fight_result = {
        "fight_detected": True,
        "fight_confidence": 0.92,
        "fight_participants": [5, 12],
        "fight_duration_sec": 4.0
    }

    score, level = svc.calculate_threat_score(detections, active_events, fight_result)
    weapon_base = int(80 * 0.88)  # = 70

    ok &= check(f"Score >= 70 (got {score})", score >= 70,
                f"weapon={weapon_base}, fight_escalation applied")
    ok &= check(f"Level is HIGH or CRITICAL (got {level})", level in ["HIGH", "CRITICAL"])

    return ok

# -----------------------------------------------------------------------
# Test 4: Fight + Weapon + Zone Intrusion → Score ≥ 90, Level = CRITICAL
# -----------------------------------------------------------------------

def test_fight_weapon_intrusion_critical():
    print("\n=== TEST 4: Fight + Weapon + Intrusion -> Score >= 90 (CRITICAL) ===")
    ok = True

    from app.services.ai_service import AIService
    svc = AIService()

    detections = [{"class": "knife", "confidence": 0.95, "track_id": 3}]
    active_events = [
        {"event_type": "ZONE INTRUSION", "threat_level": "HIGH", "person_id": 3}
    ]
    fight_result = {
        "fight_detected": True,
        "fight_confidence": 0.96,
        "fight_participants": [3, 8],
        "fight_duration_sec": 7.5
    }

    score, level = svc.calculate_threat_score(detections, active_events, fight_result)

    ok &= check(f"Score >= 90 (got {score})", score >= 90, f"score={score}")
    ok &= check(f"Level is CRITICAL (got {level})", level == "CRITICAL")

    return ok

# -----------------------------------------------------------------------
# Test 5: NonFight — score should NOT include fight bonus
# -----------------------------------------------------------------------

def test_nonfight_no_score_bonus():
    print("\n=== TEST 5: NonFight — No Fight Score Bonus ===")
    ok = True

    from app.services.ai_service import AIService
    svc = AIService()

    detections = [{"class": "person", "confidence": 0.90, "track_id": 1}]
    active_events = []
    fight_result = {
        "fight_detected": False,
        "fight_confidence": 0.0,
        "fight_participants": [],
        "fight_duration_sec": 0.0
    }

    score, level = svc.calculate_threat_score(detections, active_events, fight_result)
    ok &= check(f"Score is low without fight (got {score})", score <= 20,
                "Only person detection should yield <= 20")
    ok &= check(f"Level is LOW (got {level})", level == "LOW")

    return ok

# -----------------------------------------------------------------------
# Test 6: Single-person Shadow Boxing — No multi-person fight trigger
# -----------------------------------------------------------------------

def test_shadow_boxing_single_person():
    print("\n=== TEST 6: Single-Person Shadow Boxing ===")
    ok = True
    from app.services.ai_service import AIService
    svc = AIService()
    camera_id = "test_cam_shadow_boxing"
    
    person_dets = [{"class": "person", "confidence": 0.90, "track_id": 1}]
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for i in range(10):
        svc.fight_buffers[camera_id] = svc.fight_buffers.get(camera_id, deque(maxlen=10))
        svc.fight_buffers[camera_id].append((False, 0.10))
        res = svc.run_fight_classification(dummy_frame, camera_id, person_dets, now=float(i))

    ok &= check("Single-person nonfight does NOT trigger fight alert", not res["fight_detected"])
    ok &= check("No fight participants listed", len(res["fight_participants"]) == 0)
    return ok

# -----------------------------------------------------------------------
# Test 7: Dancing Crowd — Crowd Anomaly Only, No Fight
# -----------------------------------------------------------------------

def test_dancing_crowd():
    print("\n=== TEST 7: Dancing Crowd (Crowd Anomaly Only) ===")
    ok = True
    from app.services.ai_service import AIService
    svc = AIService()

    detections = [{"class": "person", "confidence": 0.88, "track_id": i} for i in range(1, 13)]
    active_events = [{"event_type": "CROWD DETECTION (12 People)", "threat_level": "MEDIUM", "confidence": 0.90}]
    fight_result = {
        "fight_detected": False,
        "fight_confidence": 0.0,
        "fight_participants": [],
        "fight_duration_sec": 0.0
    }

    score, level = svc.calculate_threat_score(detections, active_events, fight_result)
    ok &= check("Dancing crowd yields MEDIUM threat level", level == "MEDIUM", f"score={score}, level={level}")
    ok &= check("Score is within reasonable bounds (crowd+person)", 30 <= score <= 60, f"score={score}")
    return ok

# -----------------------------------------------------------------------
# Test 8: Fast Sports Activity — Transient Motion Filtered
# -----------------------------------------------------------------------

def test_sports_activity_transient_motion():
    print("\n=== TEST 8: Sports Activity (Transient Motion Filtered) ===")
    ok = True
    from app.services.ai_service import AIService
    svc = AIService()
    camera_id = "test_cam_sports"
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    svc.fight_buffers[camera_id] = deque(maxlen=10)
    for i in range(10):
        is_transient = i in [2, 3, 6, 7]
        svc.fight_buffers[camera_id].append((is_transient, 0.85 if is_transient else 0.10))

    res = svc.run_fight_classification(dummy_frame, camera_id, [], now=10.0)
    ok &= check("4/10 sports motion frames do NOT trigger fight alert", not res["fight_detected"], f"raw_fight_frames={res['raw_fight_frames']}")
    return ok

# -----------------------------------------------------------------------
# Test 9: Two People Hugging — Close Proximity NonFight
# -----------------------------------------------------------------------

def test_two_people_hugging():
    print("\n=== TEST 9: Two People Hugging (Close Proximity NonFight) ===")
    ok = True
    from app.services.ai_service import AIService
    svc = AIService()
    camera_id = "test_cam_hugging"
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    person_dets = [
        {"class": "person", "confidence": 0.92, "track_id": 10},
        {"class": "person", "confidence": 0.91, "track_id": 11}
    ]

    svc.fight_buffers[camera_id] = deque(maxlen=10)
    for i in range(10):
        svc.fight_buffers[camera_id].append((False, 0.05))
        res = svc.run_fight_classification(dummy_frame, camera_id, person_dets, now=float(i))

    score, level = svc.calculate_threat_score(person_dets, [], res)
    ok &= check("Hugging without fight classification does NOT trigger fight alert", not res["fight_detected"])
    ok &= check("Threat level is LOW for hugging", level == "LOW", f"score={score}, level={level}")
    return ok

# -----------------------------------------------------------------------
# Test 10: Brief 3-Frame Fight Spike — Buffer Filtering
# -----------------------------------------------------------------------

def test_brief_3frame_fight_spike():
    print("\n=== TEST 10: Brief 3-Frame Fight Spike ===")
    ok = True
    from app.services.ai_service import AIService
    svc = AIService()
    camera_id = "test_cam_3frame_spike"
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    svc.fight_buffers[camera_id] = deque(maxlen=10)
    for i in range(10):
        is_spike = i < 3
        svc.fight_buffers[camera_id].append((is_spike, 0.95 if is_spike else 0.05))

    res = svc.run_fight_classification(dummy_frame, camera_id, [], now=1.0)
    ok &= check("3/10 fight frames do NOT confirm fight", not res["fight_detected"], f"raw_fight_frames={res['raw_fight_frames']}")
    return ok

# -----------------------------------------------------------------------
# Test 11: Fight with Tracker ID Swap & Reconciliation
# -----------------------------------------------------------------------

def test_tracker_id_swap_reconciliation():
    print("\n=== TEST 11: Tracker ID Swap & Participant Reconciliation ===")
    ok = True
    from app.services.ai_service import AIService
    svc = AIService()
    camera_id = "test_cam_id_swap"
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    p_dets_initial = [
        {"class": "person", "confidence": 0.90, "track_id": 5},
        {"class": "person", "confidence": 0.90, "track_id": 12}
    ]

    svc.fight_buffers[camera_id] = deque(maxlen=10)
    for i in range(10):
        res_initial = svc.run_fight_classification(
            dummy_frame, camera_id, p_dets_initial, now=float(i), raw_fight_override=True, fight_conf_override=0.92
        )

    ok &= check("Initial fight confirmed with participants [5, 12]", res_initial["fight_detected"] and 5 in res_initial["fight_participants"] and 12 in res_initial["fight_participants"])

    p_dets_swapped = [
        {"class": "person", "confidence": 0.90, "track_id": 5},
        {"class": "person", "confidence": 0.88, "track_id": 18}
    ]
    res_swapped = svc.run_fight_classification(
        dummy_frame, camera_id, p_dets_swapped, now=11.0, raw_fight_override=True, fight_conf_override=0.92
    )

    parts = res_swapped["fight_participants"]
    ok &= check("Participant reconciliation retains memory during ID swap", 5 in parts and 12 in parts and 18 in parts, f"got: {parts}")
    return ok

# -----------------------------------------------------------------------
# Test 12: Fight Leaving and Re-entering Frame (Cooldown Period)
# -----------------------------------------------------------------------

def test_cooldown_period_leaves_and_reenters():
    print("\n=== TEST 12: Cooldown Period (Frame Leaves & Re-entry) ===")
    ok = True
    from app.services.ai_service import AIService
    svc = AIService()
    camera_id = "test_cam_cooldown"
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    svc.fight_buffers[camera_id] = deque(maxlen=10)
    for i in range(10):
        svc.run_fight_classification(
            dummy_frame, camera_id, [], now=float(i), raw_fight_override=True, fight_conf_override=0.94
        )

    for i in range(10, 15):
        res_drop = svc.run_fight_classification(
            dummy_frame, camera_id, [], now=float(i), raw_fight_override=False, fight_conf_override=0.05
        )

    ok &= check("5 nonfight frames during active fight stay ACTIVE via cooldown", res_drop["fight_detected"] and res_drop["cooldown_active"])
    ok &= check("Fight has NOT ended after 5 nonfight frames", not res_drop["fight_ended"])

    ended_triggered = False
    for i in range(15, 30):
        res_cooldown_end = svc.run_fight_classification(
            dummy_frame, camera_id, [], now=float(i), raw_fight_override=False, fight_conf_override=0.05
        )
        if res_cooldown_end["fight_ended"]:
            ended_triggered = True

    ok &= check("After 15 consecutive nonfight frames, fight officially ENDS", ended_triggered and not res_cooldown_end["fight_detected"])
    return ok

# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Eagle Eye SOC — Fight Detection Pipeline Test Suite")
    print("=" * 60)

    results = []
    results.append(("TEST 1: Model Load & Classify",          test_fight_model_load_and_detect()))
    results.append(("TEST 2: Temporal Confirmation",           test_temporal_confirmation()))
    results.append(("TEST 3: Fight+Weapon >= 70",              test_fight_weapon_threat_score()))
    results.append(("TEST 4: Fight+Weapon+Intrusion",          test_fight_weapon_intrusion_critical()))
    results.append(("TEST 5: NonFight No Bonus",               test_nonfight_no_score_bonus()))
    results.append(("TEST 6: Single-Person Shadow Boxing",     test_shadow_boxing_single_person()))
    results.append(("TEST 7: Dancing Crowd Anomaly",           test_dancing_crowd()))
    results.append(("TEST 8: Sports Activity Filtered",        test_sports_activity_transient_motion()))
    results.append(("TEST 9: Two People Hugging NonFight",     test_two_people_hugging()))
    results.append(("TEST 10: Brief 3-Frame Fight Spike",      test_brief_3frame_fight_spike()))
    results.append(("TEST 11: Tracker ID Swap Reconciliation", test_tracker_id_swap_reconciliation()))
    results.append(("TEST 12: Cooldown & Re-entry Handling",   test_cooldown_period_leaves_and_reenters()))

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    passed = 0
    for name, result in results:
        status = PASS if result else FAIL
        print(f"  {status} {name}")
        if result:
            passed += 1

    total = len(results)
    print(f"\n  {passed}/{total} tests passed")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)

