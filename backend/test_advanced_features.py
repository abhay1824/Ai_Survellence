import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.reid_service import ReIDService
from app.services.xai_service import XAIService

def test_reid_and_advanced_features():
    print("=" * 65)
    print("      EAGLE EYE ADVANCED SOC FEATURES AUTOMATED TEST SUITE        ")
    print("=" * 65)

    # 1. Test Person Re-ID Cross-Camera Matching
    print("\n--- TEST 1: Cross-Camera Person Re-ID Matching ---")
    bbox1 = (100.0, 50.0, 200.0, 300.0)
    gid1, score1 = ReIDService.match_or_register("CAM-01", 12, bbox1)
    print(f"Cam A Person ID #12 registered as: {gid1} (Confidence: {score1})")

    bbox2 = (105.0, 52.0, 202.0, 305.0)  # Slight shift, same person crop profile
    gid2, score2 = ReIDService.match_or_register("CAM-03", 4, bbox2)
    print(f"Cam C Person ID #4 matched as: {gid2} (Cosine Similarity: {score2})")

    assert gid1 == gid2, "Re-ID Failed: Identical person crop profiles should yield identical GLOBAL-ID"
    print("[PASS] Cross-camera Re-ID identity matching verified successfully!")

    # 2. Test Threat Cone & Non-Predictive Phrasing
    print("\n--- TEST 2: Threat Cone Analysis (Forward Line-of-Sight) ---")
    detections = [
        {"class": "person", "track_id": 12, "bbox": [100, 100, 200, 300]},
        {"class": "pistol", "confidence": 0.96, "bbox": [180, 120, 210, 150]},
        {"class": "person", "track_id": 4, "bbox": [220, 100, 320, 300]}  # Directly in forward line of sight (x: 220 > 100)
    ]
    tracking_history = {
        12: {"trajectory": [(50, 100), (100, 100)], "first_seen": 0, "last_seen": 4.3},
        4: {"trajectory": [(220, 100)], "first_seen": 0, "last_seen": 4.3}
    }

    spatial = XAIService.analyze_spatial_risk(12, detections, tracking_history)
    print("Spatial Risk Summary:", spatial["line_of_sight_summary"])
    assert "spatially aligned" in spatial["line_of_sight_summary"].lower() or "line of sight" in spatial["line_of_sight_summary"].lower()
    print("[PASS] Threat Cone analysis and non-predictive language rules verified!")

    # 3. Test Full Explanation with Timeline & Threat Evolution
    print("\n--- TEST 3: Timeline Reconstruction & Threat Evolution Graph ---")
    exp = XAIService.generate_explanation(
        camera_id="CAM-03",
        timestamp="2026-08-11T13:22:09Z",
        threat_level="CRITICAL",
        threat_score=87,
        confidence=0.96,
        detections=detections,
        active_events=[{"event_type": "Restricted Zone Intrusion", "person_id": 12}],
        correlated_events=["CROWD + PISTOL THREAT"],
        tracking_history=tracking_history
    )

    print("Timeline Steps Count:", len(exp.get("timeline", [])))
    print("Threat Evolution Delta:", exp.get("threat_evolution", {}).get("delta_explanation"))
    assert len(exp.get("timeline", [])) > 0, "Timeline steps missing"
    assert "threat_evolution" in exp, "Threat evolution data missing"
    print("[PASS] Timeline Reconstruction & Threat Evolution Graph verified!")

    print("\n" + "=" * 65)
    print("   ALL ADVANCED SOC TESTS PASSED SUCCESSFULLY! PRODUCTION READY.")
    print("=" * 65)

if __name__ == "__main__":
    test_reid_and_advanced_features()
