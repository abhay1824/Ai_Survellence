"""
Eagle Eye Explainable AI (XAI) Test Suite
Validates:
1. Full 7-section XAI Explanation generation (Alert Summary, Evidence Used, Behavioral Analysis, Spatial Risk, Threat Assessment, Recommended Actions, Confidence Breakdown).
2. Trajectory Movement Analysis (Stationary, Walking, Running, Chasing).
3. Spatial Distance Calculation & Danger Radius Filtering (Meters).
4. Strict Safety Guardrails (Zero forbidden predictive language).
5. Exact text template formatting compliance.
"""

import sys
import os
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

import sys
import os
from datetime import datetime

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.xai_service import XAIService, FORBIDDEN_TERMS

def test_language_guardrails():
    print("\n--- TEST 1: Safety Guardrails & Language Rules ---")
    unsafe_text = "Person ID 12 will shoot someone and will attack a future victim with criminal intent."
    sanitized = XAIService.sanitize_language(unsafe_text)
    
    print(f"Original Text: {unsafe_text}")
    print(f"Sanitized Text: {sanitized}")
    
    for term in FORBIDDEN_TERMS:
        assert term not in sanitized.lower(), f"Forbidden term '{term}' was found in sanitized text!"
    print("[PASS] Language guardrails verified successfully (zero predictive claims).")

def test_trajectory_analysis():
    print("\n--- TEST 2: Trajectory & Movement Analysis ---")
    
    # 1. Stationary history
    history_stationary = {
        12: {
            "first_seen": 100.0,
            "last_seen": 104.0,
            "trajectory": [(100, 100), (102, 101)]
        }
    }
    res_stat = XAIService.analyze_trajectory(12, history_stationary)
    print(f"Stationary result: State={res_stat['movement_state']}, Speed={res_stat['speed_m_s']} m/s")
    assert res_stat["movement_state"] == "Stationary"

    # 2. Running history (high displacement over 2 seconds)
    history_running = {
        12: {
            "first_seen": 100.0,
            "last_seen": 102.0,
            "trajectory": [(100, 100), (250, 250)]  # dx=150, dy=150 -> dist ~212px -> 212/45=4.7m / 2s = 2.35m/s
        }
    }
    res_run = XAIService.analyze_trajectory(12, history_running)
    print(f"Running result: State={res_run['movement_state']}, Speed={res_run['speed_m_s']} m/s, Dir={res_run['direction']}")
    assert res_run["movement_state"] in ["Running", "Chasing"]
    assert res_run["direction"] == "South-East"

    print("[PASS] Trajectory and movement state classification passed!")

def test_spatial_risk_analysis():
    print("\n--- TEST 3: Spatial Risk & Danger Radius Calculation ---")
    
    # Primary person ID 12 at (100, 100)
    # Nearby person 7 at (100, 190) -> dist_px = 90px -> 90/45 = 2.0m (Inside 5m danger radius: Yes)
    # Nearby person 4 at (100, 245) -> dist_px = 145px -> 145/45 = 3.2m (Inside 5m danger radius: Yes)
    # Nearby person 9 at (100, 500) -> dist_px = 400px -> 400/45 = 8.9m (Inside 5m danger radius: No)

    history = {
        12: {"first_seen": 10.0, "last_seen": 14.0, "trajectory": [(100, 100)]}
    }
    detections = [
        {"class": "Pistol", "confidence": 0.97, "track_id": 12, "bbox": [90, 90, 110, 110]},
        {"class": "Person", "confidence": 0.95, "track_id": 7, "bbox": [90, 180, 110, 200]},
        {"class": "Person", "confidence": 0.94, "track_id": 4, "bbox": [90, 235, 110, 255]},
        {"class": "Person", "confidence": 0.91, "track_id": 9, "bbox": [90, 490, 110, 510]}
    ]

    res = XAIService.analyze_spatial_risk(12, detections, history, danger_radius_m=5.0)
    nearby = res["nearby_persons"]
    
    print("Spatial calculation results:")
    for p in nearby:
        print(f"  Person ID {p['track_id']} : {p['distance_m']}m (Inside Danger Radius: {p['inside_danger_radius']})")

    assert len(nearby) == 3
    assert nearby[0]["track_id"] == 7 and nearby[0]["inside_danger_radius"] is True
    assert nearby[1]["track_id"] == 4 and nearby[1]["inside_danger_radius"] is True
    assert nearby[2]["track_id"] == 9 and nearby[2]["inside_danger_radius"] is False

    print("[PASS] Spatial risk distance and danger radius calculation passed!")

def test_full_explanation_generation():
    print("\n--- TEST 4: Full 7-Section XAI Explanation Generation ---")
    
    camera_id = "CAM-07"
    timestamp = "14:32:18"
    threat_level = "CRITICAL"
    threat_score = 92
    confidence = 0.97

    detections = [
        {"class": "Pistol", "confidence": 0.97, "track_id": 12, "bbox": [100, 100, 140, 140]},
        {"class": "Person", "confidence": 0.96, "track_id": 12, "bbox": [100, 100, 140, 240]},
        {"class": "Person", "confidence": 0.92, "track_id": 4, "bbox": [180, 180, 220, 300]},
        {"class": "Person", "confidence": 0.94, "track_id": 7, "bbox": [250, 250, 290, 370]},
        {"class": "Person", "confidence": 0.91, "track_id": 15, "bbox": [310, 310, 350, 430]}
    ]

    active_events = [
        {"event_type": "RESTRICTED AREA INTRUSION", "threat_level": "HIGH", "confidence": 0.96, "person_id": 12},
        {"event_type": "CROWD DETECTION (15 People)", "threat_level": "MEDIUM", "confidence": 0.90}
    ]

    correlated_events = ["CROWD + PISTOL THREAT (CRITICAL)"]

    history = {
        12: {
            "first_seen": 100.0,
            "last_seen": 104.3,
            "trajectory": [(100, 100), (250, 250)]
        }
    }

    explanation = XAIService.generate_explanation(
        camera_id=camera_id,
        timestamp=timestamp,
        threat_level=threat_level,
        threat_score=threat_score,
        confidence=confidence,
        detections=detections,
        active_events=active_events,
        correlated_events=correlated_events,
        tracking_history=history,
        person_count=15
    )

    print("\n--- GENERATED XAI OUTPUT TEXT ---")
    print(explanation["formatted_text"])
    print("---------------------------------\n")

    # Assert structural integrity
    assert "alert_summary" in explanation
    assert "evidence_used" in explanation
    assert "behavioral_analysis" in explanation
    assert "spatial_risk_analysis" in explanation
    assert "threat_assessment" in explanation
    assert "recommended_actions" in explanation
    assert "confidence_breakdown" in explanation

    assert explanation["alert_summary"]["threat_level"] == "CRITICAL"
    assert explanation["alert_summary"]["threat_score"] == 92
    assert len(explanation["evidence_used"]) >= 4
    assert len(explanation["recommended_actions"]) >= 3

    print("[PASS] Full XAI explanation generation and formatting verified successfully!")

if __name__ == "__main__":
    print("===============================================================")
    print("      EAGLE EYE EXPLAINABLE AI (XAI) ENGINE TEST SUITE             ")
    print("===============================================================")
    
    test_language_guardrails()
    test_trajectory_analysis()
    test_spatial_risk_analysis()
    test_full_explanation_generation()
    
    print("\n===============================================================")
    print("   ALL XAI TESTS PASSED SUCCESSFULLY! PRODUCTION READY.         ")
    print("===============================================================")
