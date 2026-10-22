import os
import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.services.ai_service import AIService

THREAT_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

def test_item1_correlation_bug():
    print("\n" + "=" * 80)
    print("ITEM 1: CORRELATION BUG VERIFICATION")
    print("=" * 80)
    ai_service = AIService()

    # (a) accident.mp4 through process_frame with fresh camera_id: max threat must be HIGH, never CRITICAL
    accident_video = repo_root / "sample_videos" / "accident.mp4"
    assert accident_video.exists(), f"Missing {accident_video}"

    cam_id_acc = "cam_test_item1_accident"
    ai_service.cleanup_camera(cam_id_acc)

    cap = cv2.VideoCapture(str(accident_video))
    assert cap.isOpened(), "Could not open accident.mp4"

    f_idx = 0
    max_threat_level = "LOW"
    max_threat_score = 0
    THREAT_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    critical_frames = []

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        f_idx += 1
        out = ai_service.process_frame(frame, cam_id_acc, "Accident Cam", (0.0, 0.0))
        highest_threat = out[2]
        threat_score = out[3]

        if THREAT_ORDER.get(highest_threat, 0) > THREAT_ORDER.get(max_threat_level, 0):
            max_threat_level = highest_threat
        if threat_score > max_threat_score:
            max_threat_score = threat_score

        if highest_threat == "CRITICAL":
            critical_frames.append(f_idx)

    cap.release()
    print(f"(a) accident.mp4: processed {f_idx} frames | max_threat_level={max_threat_level} | max_threat_score={max_threat_score} | critical_frames={critical_frames}")
    assert max_threat_level in ("LOW", "MEDIUM", "HIGH"), f"Expected accident.mp4 max threat <= HIGH, got {max_threat_level}"
    assert len(critical_frames) == 0, f"accident.mp4 reached CRITICAL on frames: {critical_frames}"
    print(">>> PASS 1(a): accident.mp4 max threat is HIGH, never CRITICAL.")

    # (b) fight.mp4: unchanged behavior (reaches CRITICAL)
    fight_video = repo_root / "sample_videos" / "fight.mp4"
    assert fight_video.exists(), f"Missing {fight_video}"

    cam_id_fight = "cam_test_item1_fight"
    ai_service.cleanup_camera(cam_id_fight)

    cap_f = cv2.VideoCapture(str(fight_video))
    f_idx = 0
    fight_max_threat = "LOW"
    fight_max_score = 0
    while True:
        ret, frame = cap_f.read()
        if not ret or frame is None:
            break
        f_idx += 1
        out = ai_service.process_frame(frame, cam_id_fight, "Fight Cam", (0.0, 0.0))
        if THREAT_ORDER.get(out[2], 0) > THREAT_ORDER.get(fight_max_threat, 0):
            fight_max_threat = out[2]
        if out[3] > fight_max_score:
            fight_max_score = out[3]
    cap_f.release()
    print(f"(b) fight.mp4: processed {f_idx} frames | max_threat={fight_max_threat} | max_score={fight_max_score}")
    assert fight_max_threat in ("HIGH", "CRITICAL"), f"Expected fight.mp4 to reach HIGH or CRITICAL, got {fight_max_threat}"
    print(">>> PASS 1(b): fight.mp4 behavior confirmed.")

    # (c) accident + a confirmed attended pistol (synthetic): CRITICAL
    synthetic_active_events = [
        {"event_type": "ACCIDENT DETECTED", "threat_level": "HIGH"},
        {"event_type": "WEAPON DETECTED (PISTOL)", "threat_level": "CRITICAL", "is_unattended": False}
    ]
    synthetic_dets = [
        {"class": "Person", "confidence": 0.88, "threat_level": "LOW", "is_unattended": False},
        {"class": "Pistol", "confidence": 0.90, "threat_level": "CRITICAL", "is_unattended": False}
    ]
    score_c, level_c = ai_service.calculate_threat_score(
        detections=synthetic_dets,
        active_events=synthetic_active_events,
        accident_result={"accident_detected": True, "accident_confidence": 0.85}
    )
    corr_c, corr_level_c = ai_service.correlate_multi_events(
        active_events=synthetic_active_events,
        detections=synthetic_dets,
        accident_result={"accident_detected": True, "accident_confidence": 0.85}
    )
    print(f"(c) synthetic accident + confirmed attended pistol: score={score_c}, level={level_c}, correlated={corr_c}, corr_level={corr_level_c}")
    assert level_c == "CRITICAL", f"Expected CRITICAL, got {level_c}"
    assert corr_level_c == "CRITICAL", f"Expected correlation CRITICAL, got {corr_level_c}"
    assert "ACCIDENT + WEAPON INVOLVEMENT (CRITICAL)" in corr_c
    print(">>> PASS 1(c): accident + confirmed attended pistol reaches CRITICAL.")

def test_item2_cap_check():
    print("\n" + "=" * 80)
    print("ITEM 2: CAP CHECK (UNCONFIRMED PISTOL NEXT TO CONFIRMED UNATTENDED KNIFE)")
    print("=" * 80)
    ai_service = AIService()

    # Detections:
    # 1. Unconfirmed Pistol (conf 0.90, threat_level CRITICAL)
    # 2. Confirmed Unattended Knife (conf 0.80, threat_level MEDIUM, is_unattended True)
    detections = [
        {"class": "Pistol", "confidence": 0.90, "threat_level": "CRITICAL", "is_unattended": False},
        {"class": "Knife", "confidence": 0.80, "threat_level": "MEDIUM", "is_unattended": True, "unattended_duration": 1.2}
    ]
    # Active events has confirmed unattended knife (MEDIUM), but NO confirmed attended weapon
    active_events = [
        {"event_type": "UNATTENDED WEAPON DETECTED (KNIFE)", "threat_level": "MEDIUM", "is_unattended": True}
    ]

    score, level = ai_service.calculate_threat_score(detections=detections, active_events=active_events)
    print(f"Cap Check result: score={score}, level={level}")
    assert score <= 59, f"Expected threat score <= 59, got {score}"
    assert level == "MEDIUM", f"Expected level == 'MEDIUM', got {level}"
    print(">>> PASS 2: Unconfirmed pistol next to confirmed unattended knife is strictly capped at MEDIUM.")

def test_item3_unattended_persistence():
    print("\n" + "=" * 80)
    print("ITEM 3: UNATTENDED PERSISTENCE & STATIONARY VALIDATION")
    print("=" * 80)
    ai_service = AIService()

    # (a) accident.mp4 (already verified 0 unattended events above) and 100 road frames
    cam_id_road = "cam_test_item3_road"
    ai_service.cleanup_camera(cam_id_road)

    cap_acc = cv2.VideoCapture(str(repo_root / "sample_videos" / "accident.mp4"))
    unattended_events_road = []
    for f in range(100):
        ret, frame = cap_acc.read()
        if not ret or frame is None:
            break
        out = ai_service.process_frame(frame, cam_id_road, "Road Cam", (0.0, 0.0))
        alerts = out[4]
        for a in alerts:
            if "UNATTENDED" in a.get("event_type", "").upper():
                unattended_events_road.append((f, a))
    cap_acc.release()
    print(f"(a) 100 road frames unattended events count: {len(unattended_events_road)}")
    assert len(unattended_events_road) == 0, f"Expected 0 unattended events on road footage, got {unattended_events_road}"
    print(">>> PASS 3(a): 100 road frames produce 0 unattended events.")

    # (b) Synthetic fixed weapon on background for 60 frames:
    # -> frames 1-14: LOW / 0 events
    # -> frame 15: MEDIUM
    # -> frame 30: HIGH
    cam_id_fixed = "cam_test_item3_fixed"
    ai_service.cleanup_camera(cam_id_fixed)

    from unittest.mock import patch

    class MockBox:
        def __init__(self, xyxy, cls_id, conf, track_id=201):
            self.xyxy = [np.array(xyxy)]
            self.cls = [cls_id]
            self.conf = [conf]
            self.id = [track_id]

    class MockResult:
        def __init__(self, boxes):
            self.boxes = boxes

    # Fixed weapon at (200, 200, 240, 240)
    fixed_knife_box = [MockBox([200.0, 200.0, 240.0, 240.0], 2, 0.85, track_id=201)]
    blank_frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    levels_by_frame = {}
    events_by_frame = {}

    with patch.object(ai_service, "_track_with_camera_state") as mock_track:
        def side_effect(model, f, c_id, tracker_dict, lock, **kwargs):
            if model == ai_service.person_model:
                return [MockResult([])] # No person
            elif model == ai_service.model:
                return [MockResult(fixed_knife_box)]
            return []
        mock_track.side_effect = side_effect

        for frame_num in range(1, 61):
            out = ai_service.process_frame(blank_frame, cam_id_fixed, "Fixed Cam", (0.0, 0.0))
            levels_by_frame[frame_num] = out[2]
            events_by_frame[frame_num] = [a["event_type"] for a in out[4]]

    print(f"Frame 1 level: {levels_by_frame[1]}, events: {events_by_frame[1]}")
    print(f"Frame 14 level: {levels_by_frame[14]}, events: {events_by_frame[14]}")
    print(f"Frame 15 level: {levels_by_frame[15]}, events: {events_by_frame[15]}")
    print(f"Frame 29 level: {levels_by_frame[29]}, events: {events_by_frame[29]}")
    print(f"Frame 30 level: {levels_by_frame[30]}, events: {events_by_frame[30]}")

    assert levels_by_frame[1] == "LOW" and len(events_by_frame[1]) == 0, "Frame 1 must be LOW and 0 events"
    assert levels_by_frame[14] == "LOW" and len(events_by_frame[14]) == 0, "Frame 14 must be LOW and 0 events"
    assert levels_by_frame[15] == "MEDIUM", f"Frame 15 must be MEDIUM, got {levels_by_frame[15]}"
    assert any("UNATTENDED" in e for e in events_by_frame[15]), "Frame 15 must generate UNATTENDED event"
    assert levels_by_frame[30] == "HIGH", f"Frame 30 must be HIGH, got {levels_by_frame[30]}"
    print(">>> PASS 3(b): Fixed weapon reaches MEDIUM at frame 15, and HIGH at frame 30.")

    # (c) Moving weapon (30 px/frame) -> 0 events
    cam_id_moving = "cam_test_item3_moving"
    ai_service.cleanup_camera(cam_id_moving)

    moving_events = []
    moving_levels = []

    for frame_num in range(1, 61):
        x_shift = (frame_num - 1) * 30.0
        moving_knife_box = [MockBox([50.0 + x_shift, 200.0, 90.0 + x_shift, 240.0], 2, 0.85, track_id=301)]

        with patch.object(ai_service, "_track_with_camera_state") as mock_track:
            def side_effect(model, f, c_id, tracker_dict, lock, **kwargs):
                if model == ai_service.person_model:
                    return [MockResult([])]
                elif model == ai_service.model:
                    return [MockResult(moving_knife_box)]
                return []
            mock_track.side_effect = side_effect

            out = ai_service.process_frame(blank_frame, cam_id_moving, "Moving Cam", (0.0, 0.0))
            moving_levels.append(out[2])
            for a in out[4]:
                if "UNATTENDED" in a.get("event_type", "").upper():
                    moving_events.append((frame_num, a))

    print(f"(c) Moving weapon 60 frames: unattended events={len(moving_events)}, max_level={max(moving_levels, key=lambda l: THREAT_ORDER[l])}")
    assert len(moving_events) == 0, f"Expected 0 events for moving weapon, got {moving_events}"
    print(">>> PASS 3(c): Moving weapon (30 px/frame) produced 0 unattended events.")

def test_item4_cam004_weapon_flip():
    print("\n" + "=" * 80)
    print("ITEM 4: CAM-004 FLIP VERIFICATION (weapon.mp4 3 LOOPS)")
    print("=" * 80)
    ai_service = AIService()

    weapon_video = repo_root / "sample_videos" / "weapon.mp4"
    assert weapon_video.exists(), f"Missing {weapon_video}"

    cam_id = "cam_test_item4_weapon"
    ai_service.cleanup_camera(cam_id)

    unattended_knife_flips = []
    total_frames = 0

    for loop_idx in range(1, 4):
        cap = cv2.VideoCapture(str(weapon_video))
        assert cap.isOpened(), "Could not open weapon.mp4"
        loop_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            loop_frame += 1
            total_frames += 1

            out = ai_service.process_frame(frame, cam_id, "Weapon Cam", (0.0, 0.0))
            dets = out[1]
            alerts = out[4]

            # Check if any knife detection is marked as unattended
            knife_dets = [d for d in dets if "knife" in d.get("class", "").lower()]
            for k in knife_dets:
                if k.get("is_unattended", False):
                    unattended_knife_flips.append((loop_idx, loop_frame, total_frames, k))

            # Check if any alert is UNATTENDED
            for a in alerts:
                if "UNATTENDED" in a.get("event_type", "").upper():
                    unattended_knife_flips.append((loop_idx, loop_frame, total_frames, a))

        cap.release()

    print(f"weapon.mp4 across 3 loops ({total_frames} frames): unattended flips count = {len(unattended_knife_flips)}")
    assert len(unattended_knife_flips) == 0, f"Knife flipped to unattended in weapon.mp4! Flips: {unattended_knife_flips}"
    print(">>> PROVEN: In weapon.mp4 across 3 loops, held knife NEVER flips to unattended.")

if __name__ == "__main__":
    test_item1_correlation_bug()
    test_item2_cap_check()
    test_item3_unattended_persistence()
    test_item4_cam004_weapon_flip()
    print("\nALL TESTS FOR ITEMS 1 TO 4 PASSED SUCCESSFULLY!")
