import os
import sys
import time
import asyncio
from unittest.mock import patch, AsyncMock
from pathlib import Path
import numpy as np
import cv2

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.services.ai_service import AIService
from app.services import event_service
from app.services.email_service import EmailService

def test_ceiling_largest_person():
    print("=" * 80)
    print("ITEM 1: CEILING BUG FIX TEST (LARGEST OVERLAPPING PERSON)")
    print("=" * 80)

    ai_service = AIService()
    cam_id = "test_cam_ceiling_eval"
    ai_service.cleanup_camera(cam_id)

    # Frame dimensions: 640x480
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)

    # Mock track method to simulate:
    # 1. Near person: [100, 100, 300, 500] (area = 200 * 400 = 80,000 px, conf=0.92)
    # 2. Far small person: [180, 200, 220, 250] (area = 40 * 50 = 2,000 px, conf=0.65)
    # 3. Knife: [170, 190, 210, 230] (area = 40 * 40 = 1,600 px, conf=0.88)
    # The knife overlaps BOTH the near person and the far small person!
    # Knife area (1600) is > 50% of far small person area (2000 * 0.5 = 1000).
    # BUT knife area (1600) is << 50% of largest person area (80000 * 0.5 = 40000).
    # Expected result: KNIFE IS NOT REJECTED!

    class MockBox:
        def __init__(self, xyxy, cls_id, conf, track_id=1):
            self.xyxy = [np.array(xyxy)]
            self.cls = [cls_id]
            self.conf = [conf]
            self.id = [track_id]

    class MockResult:
        def __init__(self, boxes):
            self.boxes = boxes
            self.names = {0: "person", 1: "knife", 2: "pistol"}

    # In best.pt: {0: 'pistol', 1: 'smartphone', 2: 'knife', 3: 'monedero', 4: 'billete', 5: 'tarjeta'}
    person_boxes = [
        MockBox([100.0, 100.0, 300.0, 500.0], 0, 0.92, track_id=1), # Near large person (80k px)
        MockBox([180.0, 200.0, 220.0, 250.0], 0, 0.65, track_id=2), # Far small person (2k px)
    ]
    knife_boxes = [
        MockBox([170.0, 190.0, 210.0, 230.0], 2, 0.88, track_id=101) # Knife (1.6k px)
    ]

    with patch.object(ai_service, "_track_with_camera_state") as mock_track:
        def side_effect(model, f, c_id, tracker_dict, lock, **kwargs):
            if model == ai_service.person_model:
                return [MockResult(person_boxes)]
            elif model == ai_service.model:
                return [MockResult(knife_boxes)]
            return []
        mock_track.side_effect = side_effect

        out = ai_service.process_frame(frame, cam_id, "Ceiling Test Cam", (0.0, 0.0))
        detections = out[1]
        det_classes = [d["class"] for d in detections]
        print(f"Detections with near person (80k px) and far small person (2k px) overlapping knife (1.6k px): {det_classes}")
        assert "Knife" in det_classes, "FAIL: Knife was erroneously rejected by ceiling!"
        print(">>> PROVEN: Knife overlapping both near person and small far person is NOT rejected (compared to largest person).")

    # Now verify that an oversized weapon (> 50% of largest overlapping person) IS rejected
    oversized_knife_boxes = [
        MockBox([100.0, 100.0, 310.0, 350.0], 2, 0.88, track_id=102) # area = 210 * 250 = 52,500 > 50% of 80,000
    ]
    with patch.object(ai_service, "_track_with_camera_state") as mock_track:
        def side_effect(model, f, c_id, tracker_dict, lock, **kwargs):
            if model == ai_service.person_model:
                return [MockResult(person_boxes)]
            elif model == ai_service.model:
                return [MockResult(oversized_knife_boxes)]
            return []
        mock_track.side_effect = side_effect

        out = ai_service.process_frame(frame, cam_id, "Ceiling Test Cam", (0.0, 0.0))
        detections = out[1]
        det_classes = [d["class"] for d in detections]
        print(f"Detections with oversized weapon (52.5k px > 50% of 80k px): {det_classes}")
        assert "Knife" not in det_classes, "FAIL: Oversized weapon was not rejected by ceiling!"
        print(">>> PROVEN: Oversized weapon (> 50% of largest overlapping person) IS rejected by ceiling.")

def test_persistence_gating_on_attended_weapon():
    print("\n" + "=" * 80)
    print("ITEM 1 TESTS: ATTENDED WEAPON PERSISTENCE GATING")
    print("=" * 80)
    ai_service = AIService()
    cam_id = "test_persistence_cam"
    ai_service.cleanup_camera(cam_id)
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    class MockBox:
        def __init__(self, xyxy, cls_id, conf, track_id=1):
            self.xyxy = [np.array(xyxy)]
            self.cls = [cls_id]
            self.conf = [conf]
            self.id = [track_id]

    class MockResult:
        def __init__(self, boxes):
            self.boxes = boxes

    pistol_box = [MockBox([280.0, 200.0, 340.0, 260.0], 0, 0.88, track_id=50)]
    person_box = [MockBox([250.0, 150.0, 360.0, 400.0], 0, 0.90, track_id=1)]

    # (a) Single-frame pistol with a qualified person -> highest_threat must NOT be CRITICAL
    with patch.object(ai_service, "_track_with_camera_state") as mock_track:
        def side_effect(model, f, c_id, tracker_dict, lock, **kwargs):
            if model == ai_service.person_model:
                return [MockResult(person_box)]
            elif model == ai_service.model:
                return [MockResult(pistol_box)]
            return []
        mock_track.side_effect = side_effect

        out1 = ai_service.process_frame(frame, cam_id, "Cam Persist", (0.0, 0.0))
        highest_threat1 = out1[2]
        alerts1 = out1[4]
        print(f"Single frame result: highest_threat={highest_threat1}, alerts={[a['event_type'] for a in alerts1]}")
        assert highest_threat1 != "CRITICAL", f"Expected highest_threat != 'CRITICAL' on single frame, got {highest_threat1}"
        assert len(alerts1) == 0, f"Expected 0 alerts on single frame, got {len(alerts1)}"
        print(">>> PASS (a): Single-frame pistol with qualified person is NOT CRITICAL.")

    # (b) The same pistol on 3 consecutive frames -> CRITICAL
    with patch.object(ai_service, "_track_with_camera_state") as mock_track:
        def side_effect(model, f, c_id, tracker_dict, lock, **kwargs):
            if model == ai_service.person_model:
                return [MockResult(person_box)]
            elif model == ai_service.model:
                return [MockResult(pistol_box)]
            return []
        mock_track.side_effect = side_effect

        out2 = ai_service.process_frame(frame, cam_id, "Cam Persist", (0.0, 0.0))
        out3 = ai_service.process_frame(frame, cam_id, "Cam Persist", (0.0, 0.0))
        highest_threat3 = out3[2]
        alerts3 = out3[4]
        alert_types3 = [a["event_type"] for a in alerts3]
        print(f"3 consecutive frames result: highest_threat={highest_threat3}, alerts={alert_types3}")
        assert highest_threat3 == "CRITICAL", f"Expected highest_threat == 'CRITICAL' after 3 frames, got {highest_threat3}"
        assert "WEAPON DETECTED (PISTOL)" in alert_types3, "Expected 'WEAPON DETECTED (PISTOL)' in alerts after 3 frames"
        print(">>> PASS (b): 3 consecutive frames of pistol with qualified person reaches CRITICAL.")

async def run_end_to_end_pipeline():
    print("\n" + "=" * 80)
    print("ITEM 4: END-TO-END PIPELINE & EMAIL SUPPRESSION TEST (REAL CODE)")
    print("=" * 80)

    ai_service = AIService()
    cam_id = "test_cam_e2e_real"
    ai_service.cleanup_camera(cam_id)

    frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    class MockBox:
        def __init__(self, xyxy, cls_id, conf, track_id=1):
            self.xyxy = [np.array(xyxy)]
            self.cls = [cls_id]
            self.conf = [conf]
            self.id = [track_id]

    class MockResult:
        def __init__(self, boxes):
            self.boxes = boxes
            self.names = {0: "person", 1: "knife", 2: "pistol"}

    # In best.pt: class 0 is pistol
    pistol_box = [MockBox([280.0, 200.0, 340.0, 260.0], 0, 0.88, track_id=50)]
    person_qualified_box = [MockBox([250.0, 150.0, 360.0, 400.0], 0, 0.90, track_id=1)]

    # Mock MongoDB collections so tests run in memory without live DB connection
    mock_event_col = AsyncMock()
    mock_event_col.find_one.return_value = None
    mock_insert_res = AsyncMock()
    mock_insert_res.inserted_id = "test_event_id"
    mock_event_col.insert_one.return_value = mock_insert_res
    mock_event_col.update_one.return_value = AsyncMock()

    mock_cam_col = AsyncMock()
    mock_cam_col.find_one.return_value = {"camera_id": cam_id, "latitude": 0.0, "longitude": 0.0}
    mock_cam_col.update_one.return_value = AsyncMock()

    # Mock EmailService.send_alert_email
    email_calls = []
    async def mock_send_email(**kwargs):
        email_calls.append(kwargs)
        return True

    with patch.object(EmailService, "send_alert_email", side_effect=mock_send_email), \
         patch("app.services.event_service.get_event_collection", return_value=mock_event_col), \
         patch("app.services.event_service.get_camera_collection", return_value=mock_cam_col):
        # -------------------------------------------------------------
        # (a) Weapon, NO person, first sight -> MEDIUM, 0 emails
        # -------------------------------------------------------------
        print("\n--- (a) WEAPON, NO PERSON, FIRST SIGHT ---")
        email_calls.clear()
        event_service._email_cooldown_cache.clear()

        # Run 3 consecutive frames to pass 3-frame persistence
        for f_idx in range(3):
            with patch.object(ai_service, "_track_with_camera_state") as mock_track:
                def side_effect(model, f, c_id, tracker_dict, lock, **kwargs):
                    if model == ai_service.person_model:
                        return [MockResult([])]  # No person
                    elif model == ai_service.model:
                        return [MockResult(pistol_box)]
                    return []
                mock_track.side_effect = side_effect

                out = ai_service.process_frame(frame, cam_id, "Cam E2E", (0.0, 0.0))
                highest_threat = out[2]
                threat_score = out[3]
                alerts = out[4]

                for alert in alerts:
                    await event_service.log_threat_event(
                        camera_id=cam_id,
                        event_type=alert["event_type"],
                        threat_level=alert["threat_level"],
                        threat_score=threat_score,
                        confidence=alert["confidence"],
                        frame=frame,
                        frame_buffer=[]
                    )
        await asyncio.sleep(0.05)

        event_types = [a["event_type"] for a in alerts]
        print(f"Result (a): events={event_types}, highest_threat={highest_threat}, threat_score={threat_score}, email_calls={len(email_calls)}")
        assert highest_threat == "MEDIUM", f"Expected highest_threat='MEDIUM', got {highest_threat}"
        assert threat_score < 60, f"Expected threat_score < 60, got {threat_score}"
        assert len(email_calls) == 0, f"Expected 0 emails for unattended at MEDIUM, got {len(email_calls)}"
        print(">>> PASS (a): Weapon with no person on first sight produces MEDIUM, score < 60, and 0 emails.")

        # -------------------------------------------------------------
        # (b) Same weapon still present after > 2.0 s -> HIGH, 1 email
        # -------------------------------------------------------------
        print("\n--- (b) SAME WEAPON STILL PRESENT AFTER > 2.0s ---")
        email_calls.clear()
        event_service._email_cooldown_cache.clear()

        # Mock clock advancing by 2.5 seconds
        t_start = time.time()
        with patch("time.time", side_effect=lambda: t_start + 2.5):
            with patch.object(ai_service, "_track_with_camera_state") as mock_track:
                def side_effect(model, f, c_id, tracker_dict, lock, **kwargs):
                    if model == ai_service.person_model:
                        return [MockResult([])]  # Still no person
                    elif model == ai_service.model:
                        return [MockResult(pistol_box)]
                    return []
                mock_track.side_effect = side_effect

                out = ai_service.process_frame(frame, cam_id, "Cam E2E", (0.0, 0.0))
                highest_threat = out[2]
                threat_score = out[3]
                alerts = out[4]

                for alert in alerts:
                    await event_service.log_threat_event(
                        camera_id=cam_id,
                        event_type=alert["event_type"],
                        threat_level=alert["threat_level"],
                        threat_score=threat_score,
                        confidence=alert["confidence"],
                        frame=frame,
                        frame_buffer=[]
                    )
        await asyncio.sleep(0.05)

        event_types = [a["event_type"] for a in alerts]
        print(f"Result (b): events={event_types}, highest_threat={highest_threat}, threat_score={threat_score}, email_calls={len(email_calls)}")
        assert highest_threat == "HIGH", f"Expected highest_threat='HIGH', got {highest_threat}"
        assert len(email_calls) == 1, f"Expected exactly 1 email for unattended after >2.0s, got {len(email_calls)}"
        print(">>> PASS (b): Unattended weapon after > 2.0s upgrades to HIGH and sends exactly 1 email.")

        # -------------------------------------------------------------
        # (c) Weapon with a qualified person -> CRITICAL, 1 email (3 frames persistence)
        # -------------------------------------------------------------
        print("\n--- (c) WEAPON WITH A QUALIFIED PERSON (3 FRAMES PERSISTENCE) ---")
        email_calls.clear()
        event_service._email_cooldown_cache.clear()

        # Run 3 consecutive frames to satisfy 3-frame persistence validator
        for f_idx in range(3):
            with patch.object(ai_service, "_track_with_camera_state") as mock_track:
                def side_effect(model, f, c_id, tracker_dict, lock, **kwargs):
                    if model == ai_service.person_model:
                        return [MockResult(person_qualified_box)]  # Qualified person present
                    elif model == ai_service.model:
                        return [MockResult(pistol_box)]
                    return []
                mock_track.side_effect = side_effect

                out = ai_service.process_frame(frame, cam_id, "Cam E2E", (0.0, 0.0))
                highest_threat = out[2]
                threat_score = out[3]
                alerts = out[4]

                for alert in alerts:
                    await event_service.log_threat_event(
                        camera_id=cam_id,
                        event_type=alert["event_type"],
                        threat_level=alert["threat_level"],
                        threat_score=threat_score,
                        confidence=alert["confidence"],
                        frame=frame,
                        frame_buffer=[]
                    )
        await asyncio.sleep(0.05)

        event_types = [a["event_type"] for a in alerts]
        print(f"Result (c): events={event_types}, highest_threat={highest_threat}, threat_score={threat_score}, email_calls={len(email_calls)}")
        assert highest_threat == "CRITICAL", f"Expected highest_threat='CRITICAL', got {highest_threat}"
        assert len(email_calls) == 1, f"Expected 1 email for attended pistol at CRITICAL, got {len(email_calls)}"
        print(">>> PASS (c): Attended weapon with qualified person produces CRITICAL and sends 1 email.")

        # -------------------------------------------------------------
        # (d) Weapon with a person who leaves for 10+ frames -> becomes unattended, MEDIUM
        # -------------------------------------------------------------
        print("\n--- (d) PERSON LEAVES FOR 10+ FRAMES -> BECOMES UNATTENDED (MEDIUM) ---")
        # Run 9 frames with no person (still within 10-frame hysteresis -> still attended)
        for f_idx in range(1, 10):
            with patch.object(ai_service, "_track_with_camera_state") as mock_track:
                def side_effect(model, f, c_id, tracker_dict, lock, **kwargs):
                    if model == ai_service.person_model:
                        return [MockResult([])]  # Person left
                    elif model == ai_service.model:
                        return [MockResult(pistol_box)]
                    return []
                mock_track.side_effect = side_effect
                out = ai_service.process_frame(frame, cam_id, "Cam E2E", (0.0, 0.0))

        # At frame 10: 10th consecutive frame without person -> transitions to unattended (MEDIUM)
        email_calls.clear()
        event_service._email_cooldown_cache.clear()

        with patch.object(ai_service, "_track_with_camera_state") as mock_track:
            def side_effect(model, f, c_id, tracker_dict, lock, **kwargs):
                if model == ai_service.person_model:
                    return [MockResult([])]
                elif model == ai_service.model:
                    return [MockResult(pistol_box)]
                return []
            mock_track.side_effect = side_effect

            out = ai_service.process_frame(frame, cam_id, "Cam E2E", (0.0, 0.0))
            highest_threat = out[2]
            threat_score = out[3]
            alerts = out[4]

            for alert in alerts:
                await event_service.log_threat_event(
                    camera_id=cam_id,
                    event_type=alert["event_type"],
                    threat_level=alert["threat_level"],
                    threat_score=threat_score,
                    confidence=alert["confidence"],
                    frame=frame,
                    frame_buffer=[]
                )
        await asyncio.sleep(0.05)

        event_types = [a["event_type"] for a in alerts]
        print(f"Result (d at frame 10): events={event_types}, highest_threat={highest_threat}, threat_score={threat_score}, email_calls={len(email_calls)}")
        assert highest_threat == "MEDIUM", f"Expected highest_threat='MEDIUM' at frame 10, got {highest_threat}"
        assert threat_score < 60, f"Expected threat_score < 60, got {threat_score}"
        assert len(email_calls) == 0, f"Expected 0 emails for transition to unattended at MEDIUM, got {len(email_calls)}"
        print(">>> PASS (d): Person leaving for 10+ frames transitions weapon to unattended MEDIUM with 0 emails.")

    print("\n" + "=" * 80)
    print("ALL END-TO-END PIPELINE & CEILING TESTS PASSED!")
    print("=" * 80)

def test_unconfirmed_weapon_capped():
    print("\n" + "=" * 80)
    print("ITEM 4: UNCONFIRMED WEAPON CAPPING TEST (0_DzLlklZa0_3.avi)")
    print("=" * 80)

    clip_path = repo_root / "RWF-2000" / "NonFight" / "0_DzLlklZa0_3.avi"
    assert clip_path.exists(), f"Clip not found: {clip_path}"

    ai_service = AIService()
    cam_id = "test_cam_cap_unconfirmed"
    ai_service.cleanup_camera(cam_id)

    cap = cv2.VideoCapture(str(clip_path))
    assert cap.isOpened(), "Could not open clip"

    frame_idx = 0
    frame_36_tested = False
    max_threat_overall = "LOW"
    THREAT_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_idx += 1

        out = ai_service.process_frame(
            frame=frame,
            camera_id=cam_id,
            camera_name="0_DzLlklZa0_3.avi",
            gps_coords=(0.0, 0.0)
        )
        dets = out[1]
        highest_threat = out[2]
        threat_score = out[3]
        alerts = out[4]

        if THREAT_ORDER.get(highest_threat, 0) > THREAT_ORDER.get(max_threat_overall, 0):
            max_threat_overall = highest_threat

        # Frame 36 has the raw 0.76 pistol detection
        pistol_dets = [d for d in dets if d.get("class", "").lower() in ("pistol", "gun")]
        if pistol_dets:
            conf = pistol_dets[0]["confidence"]
            print(f"Detected unconfirmed pistol on frame {frame_idx}: conf={conf:.3f}, threat={highest_threat}, score={threat_score}")
            assert threat_score <= 59, f"Expected threat_score <= 59 for unconfirmed weapon, got {threat_score}"
            assert highest_threat in ("LOW", "MEDIUM"), f"Expected highest_threat <= MEDIUM, got {highest_threat}"
            assert len(alerts) == 0, f"Expected 0 alerts for unconfirmed weapon, got {alerts}"
            frame_36_tested = True

        # Assert no frame ever reaches HIGH or CRITICAL
        assert highest_threat not in ("HIGH", "CRITICAL"), f"Frame {frame_idx} reached {highest_threat}!"

    cap.release()
    assert frame_36_tested, "Expected to encounter the unconfirmed pistol frame during clip evaluation"
    assert max_threat_overall in ("LOW", "MEDIUM"), f"Expected overall max threat <= MEDIUM, got {max_threat_overall}"
    print(f">>> PASS: 0_DzLlklZa0_3.avi stayed below HIGH across all {frame_idx} frames (max threat: {max_threat_overall}).")

def main():
    test_ceiling_largest_person()
    test_persistence_gating_on_attended_weapon()
    test_unconfirmed_weapon_capped()
    asyncio.run(run_end_to_end_pipeline())

if __name__ == "__main__":
    main()
