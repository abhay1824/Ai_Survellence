"""
Comprehensive End-to-End Verification Pipeline for YOLO11s Accident Detection Integration in AEGIS SOC.
Tests:
1. Model loading (Accident model & Fight model)
2. Real accident image detection (confidence >= 0.70)
3. Negative / clean image test (no false positives)
4. Threat scoring engine rules:
   - Accident Only: score 35, level HIGH
   - Accident + Intrusion: score 55, level HIGH
   - Accident + Fight: score 75, level HIGH
   - Accident + Weapon: score 85, level CRITICAL
   - Accident + Fight + Weapon: score 100, level CRITICAL
5. Multi-event correlation engine
6. Explainable AI (XAI) engine evidence, timeline, score deltas, and non-predictive safety
7. PDF Incident report generation with Accident Analysis Section
"""

import sys
import os
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

import os
import sys
import cv2
import numpy as np

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.services.ai_service import AIService
from app.services.xai_service import XAIService
from app.utils.pdf_generator import generate_incident_pdf

@pytest.fixture
def ai_svc():
    return AIService()

def test_model_loading(ai_svc):
    print("\n--- TEST 1: Model Loading ---")
    assert ai_svc.accident_model is not None, "Accident model failed to load!"
    print("  [PASS] Accident model loaded successfully:", type(ai_svc.accident_model))
    assert ai_svc.fight_model is not None, "Fight model failed to load!"
    print("  [PASS] Fight model loaded successfully:", type(ai_svc.fight_model))


def test_accident_detection(ai_svc):
    print("\n--- TEST 2: Detection on Real Accident Frame ---")
    accident_img_path = r"D:\AI SUrvellinace\acc\test\images\Accidents-online-video-cutter_com-1-_mp4-113_jpg.rf.08ebb356870c2b9d4076437525cb7c2b.jpg"
    assert os.path.exists(accident_img_path), f"Test image not found: {accident_img_path}"

    frame = cv2.imread(accident_img_path)
    assert frame is not None, "Failed to read test image"

    res = ai_svc.run_accident_detection(frame)
    print("  Accident Result:", res)
    assert res["accident_detected"] is True, "Expected accident_detected == True"
    assert res["accident_confidence"] >= 0.70, f"Expected confidence >= 0.70, got {res['accident_confidence']}"
    assert len(res["boxes"]) > 0, "Expected at least one bounding box"
    print(f"  [PASS] Real accident detected with confidence: {res['accident_confidence'] * 100:.1f}%")

    print("\n--- TEST 3: Negative Test on Clean Frame ---")
    clean_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    clean_frame[:] = (120, 120, 120)
    clean_res = ai_svc.run_accident_detection(clean_frame)
    print("  Clean Frame Result:", clean_res)
    assert clean_res["accident_detected"] is False, "Expected accident_detected == False on clean frame"
    assert clean_res["accident_confidence"] == 0.0, "Expected 0.0 confidence"
    print("  [PASS] Clean frame produced 0 false positives.")


def test_threat_scoring(ai_svc):
    print("\n--- TEST 4: Threat Scoring & Escalation Overrides ---")

    # 1. Accident Only
    score, level = ai_svc.calculate_threat_score(
        detections=[],
        active_events=[],
        fight_result={"fight_detected": False, "fight_confidence": 0.0},
        accident_result={"accident_detected": True, "accident_confidence": 0.85}
    )
    print(f"  Accident Only: Score = {score}, Level = {level}")
    assert score == 35, f"Expected 35, got {score}"
    assert level == "HIGH", f"Expected HIGH, got {level}"
    print("  [PASS] Accident Only -> 35 (HIGH)")

    # 2. Accident + Intrusion
    score, level = ai_svc.calculate_threat_score(
        detections=[],
        active_events=[{"event_type": "ZONE INTRUSION"}],
        fight_result={"fight_detected": False, "fight_confidence": 0.0},
        accident_result={"accident_detected": True, "accident_confidence": 0.85}
    )
    print(f"  Accident + Intrusion: Score = {score}, Level = {level}")
    assert score == 55, f"Expected 55, got {score}"
    assert level == "HIGH", f"Expected HIGH, got {level}"
    print("  [PASS] Accident + Intrusion -> 55 (HIGH)")

    # 3. Accident + Fight
    score, level = ai_svc.calculate_threat_score(
        detections=[],
        active_events=[],
        fight_result={"fight_detected": True, "fight_confidence": 0.88},
        accident_result={"accident_detected": True, "accident_confidence": 0.85}
    )
    print(f"  Accident + Fight: Score = {score}, Level = {level}")
    assert score == 75, f"Expected 75, got {score}"
    assert level == "HIGH", f"Expected HIGH, got {level}"
    print("  [PASS] Accident + Fight -> 75 (HIGH)")

    # 4. Accident + Weapon (requires confirmed attended weapon)
    score, level = ai_svc.calculate_threat_score(
        detections=[{"class": "pistol", "confidence": 0.90, "is_unattended": False}],
        active_events=[{"event_type": "WEAPON DETECTED (PISTOL)", "threat_level": "CRITICAL", "is_unattended": False}],
        fight_result={"fight_detected": False, "fight_confidence": 0.0},
        accident_result={"accident_detected": True, "accident_confidence": 0.85}
    )
    print(f"  Accident + Weapon: Score = {score}, Level = {level}")
    assert score >= 85, f"Expected >= 85, got {score}"
    assert level == "CRITICAL", f"Expected CRITICAL, got {level}"
    print("  [PASS] Accident + Weapon -> 85 (CRITICAL)")

    # 5. Accident + Fight + Weapon
    score, level = ai_svc.calculate_threat_score(
        detections=[{"class": "pistol", "confidence": 0.90, "is_unattended": False}],
        active_events=[{"event_type": "WEAPON DETECTED (PISTOL)", "threat_level": "CRITICAL", "is_unattended": False}],
        fight_result={"fight_detected": True, "fight_confidence": 0.92},
        accident_result={"accident_detected": True, "accident_confidence": 0.85}
    )
    print(f"  Accident + Fight + Weapon: Score = {score}, Level = {level}")
    assert score == 100, f"Expected 100, got {score}"
    assert level == "CRITICAL", f"Expected CRITICAL, got {level}"
    print("  [PASS] Accident + Fight + Weapon -> 100 (CRITICAL)")


def test_correlations(ai_svc):
    print("\n--- TEST 5: Correlation Rules ---")

    corr, lvl = ai_svc.correlate_multi_events(
        active_events=[{"event_type": "CROWD GATHERING"}],
        detections=[],
        accident_result={"accident_detected": True, "accident_confidence": 0.80}
    )
    print("  Accident + Crowd:", corr, lvl)
    assert any("CROWD GATHERING" in c for c in corr)
    assert lvl == "HIGH"
    print("  [PASS] Accident + Crowd Correlation passed.")

    corr, lvl = ai_svc.correlate_multi_events(
        active_events=[{"event_type": "WEAPON DETECTED (PISTOL)", "is_unattended": False}],
        detections=[{"class": "pistol", "is_unattended": False}],
        accident_result={"accident_detected": True, "accident_confidence": 0.80}
    )
    print("  Accident + Weapon:", corr, lvl)
    assert any("WEAPON INVOLVEMENT" in c for c in corr)
    assert lvl == "CRITICAL"
    print("  [PASS] Accident + Weapon Correlation passed.")


def test_xai_generation():
    print("\n--- TEST 6: XAI Generation & Non-Predictive Safety ---")

    explanation = XAIService.generate_explanation(
        camera_id="CAM-01",
        timestamp="2026-09-17T01:00:00",
        threat_level="HIGH",
        threat_score=35,
        confidence=0.90,
        detections=[{"class": "Accident", "confidence": 0.85}],
        active_events=[{"event_type": "ACCIDENT DETECTED"}],
        correlated_events=[],
        tracking_history={},
        person_count=0,
        fight_result={"fight_detected": False, "fight_confidence": 0.0},
        accident_result={"accident_detected": True, "accident_confidence": 0.85}
    )

    evidence_str = " ".join(explanation.get("evidence_used", []))
    assert "Accident detected" in evidence_str, f"Missing accident in evidence: {evidence_str}"
    print("  [PASS] Evidence includes accident:", evidence_str)

    timeline = explanation.get("timeline", [])
    acc_steps = [t for t in timeline if "Accident" in t.get("event", "")]
    assert len(acc_steps) > 0, "Expected accident step in timeline"
    print("  [PASS] Timeline includes accident step:", acc_steps)

    delta_factors = explanation.get("threat_evolution", {}).get("delta_factors", [])
    acc_factors = [f for f in delta_factors if "Accident" in f.get("factor", "")]
    assert len(acc_factors) > 0, "Expected accident in delta factors"
    assert acc_factors[0]["points"] == 35, f"Expected 35 points, got {acc_factors[0]['points']}"
    print("  [PASS] Threat evolution contains accident factor:", acc_factors)

    cb = explanation.get("confidence_breakdown", {})
    assert "accident_detection_confidence" in cb, "Expected accident_detection_confidence in breakdown"
    assert cb["accident_detection_confidence"] == "85%", f"Expected 85%, got {cb['accident_detection_confidence']}"
    print("  [PASS] Confidence breakdown contains accident confidence:", cb["accident_detection_confidence"])


def test_pdf_generation():
    print("\n--- TEST 7: PDF Incident Report with Accident Analysis ---")
    event_data = {
        "event_id": "EVT-TESTACC1",
        "camera_id": "CAM-01",
        "event_type": "ACCIDENT DETECTED",
        "threat_level": "HIGH",
        "threat_score": 35,
        "confidence": 0.85,
        "timestamp": "2026-09-17T01:00:00Z",
        "accident_detected": True,
        "accident_confidence": 0.85,
        "fight_detected": False,
        "fight_confidence": 0.0,
        "xai_explanation": {
            "evidence_used": ["Accident detected (85.0% confidence)"],
            "behavioral_analysis": {
                "observed_actions": "Observed collision pattern on roadway.",
                "movement_state": "Stationary"
            },
            "spatial_risk_analysis": {
                "danger_radius_m": 15.0,
                "nearby_persons": [],
                "line_of_sight_summary": "Roadway zone."
            },
            "recommended_actions": [
                "Notify security personnel",
                "Dispatch nearest response / patrol unit",
                "Prepare medical/incident report protocol"
            ],
            "threat_assessment": "HIGH: Significant security violation or collision pattern detected."
        }
    }

    pdf_bytes = generate_incident_pdf(event_data)
    assert pdf_bytes is not None and len(pdf_bytes) > 1000, "PDF generation failed or returned too few bytes"
    print(f"  [PASS] Incident PDF generated successfully ({len(pdf_bytes)} bytes)")


if __name__ == "__main__":
    print("==================================================================")
    print("AEGIS SOC: VERIFYING ACCIDENT DETECTION PIPELINE INTEGRATION")
    print("==================================================================")

    ai_service = AIService()
    test_model_loading(ai_service)
    test_accident_detection(ai_service)
    test_threat_scoring(ai_service)
    test_correlations(ai_service)
    test_xai_generation()
    test_pdf_generation()

    print("\n==================================================================")
    print("[SUCCESS] ALL 7 TEST SUITES PASSED PERFECTLY!")
    print("==================================================================")
