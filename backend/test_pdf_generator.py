import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.utils.pdf_generator import generate_incident_pdf

def test_pdf_generation():
    print("=" * 60)
    print("     EAGLE EYE INCIDENT PDF GENERATOR TEST SUITE")
    print("=" * 60)

    sample_event = {
        "event_id": "INC-20260811-001",
        "timestamp": "2026-08-11T13:22:09Z",
        "camera_id": "CAM-03",
        "threat_level": "CRITICAL",
        "threat_score": 87,
        "event_type": "Pistol Detected",
        "confidence": 0.96,
        "image_path": None,
        "xai_explanation": {
            "summary": "A pistol has been detected on Person ID 12 with score 87 (CRITICAL).",
            "evidence_used": [
                "Person ID 12 detected",
                "Pistol detected (96.0% confidence)",
                "Restricted Area Intrusion detected",
                "Crowd Density HIGH"
            ],
            "behavioral_analysis": {
                "movement_state": "Running",
                "speed_m_s": 2.4,
                "observed_actions": "Person ID 12 is moving south-east at 2.4 m/s while carrying a detected pistol."
            },
            "spatial_risk": {
                "nearby_persons": [
                    {"track_id": 4, "distance_meters": 2.1, "inside_danger_radius": True},
                    {"track_id": 7, "distance_meters": 3.8, "inside_danger_radius": True}
                ]
            },
            "recommended_actions": [
                "Dispatch nearest security unit",
                "Lock nearby access points",
                "Escalate to command center"
            ]
        }
    }

    print("\n--- Generating PDF Report Bytes ---")
    pdf_bytes = generate_incident_pdf(sample_event)
    print(f"Generated PDF Size: {len(pdf_bytes)} bytes")
    assert len(pdf_bytes) > 1000, "PDF bytes output too small"

    output_filename = "test_incident_report.pdf"
    output_filepath = os.path.join(os.path.dirname(__file__), output_filename)
    with open(output_filepath, "wb") as f:
        f.write(pdf_bytes)

    print(f"Saved test PDF report to: {output_filepath}")
    print("\n" + "=" * 60)
    print("   PDF GENERATOR TEST PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_pdf_generation()
