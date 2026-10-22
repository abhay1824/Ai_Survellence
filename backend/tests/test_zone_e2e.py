import os
import sys
import json
import time
from pathlib import Path
from fastapi.testclient import TestClient

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

from app.main import app
from app.core.security import create_access_token

def main():
    print("--- Testing Zones End-to-End ---")
    token = create_access_token(subject="admin", role="Super Admin")
    headers = {"Authorization": f"Bearer {token}"}
    
    with TestClient(app) as client:
        # 1. POST /api/zones with a zone covering CAM-002 (where persons are in view)
        zone_payload = {
            "camera_id": "CAM-002",
            "name": "Perimeter Gate Test",
            "points": [[0.0, 0.0], [1280.0, 0.0], [1280.0, 720.0], [0.0, 720.0]],
            "threat_level": "HIGH"
        }
        
        zone_id = None
        try:
            post_res = client.post("/api/zones", json=zone_payload, headers=headers)
            print(f"POST /api/zones Status Code: {post_res.status_code}")
            print("POST /api/zones Response:", json.dumps(post_res.json(), indent=2))
            assert post_res.status_code == 201, f"Expected 201, got {post_res.status_code}"
            zone_id = post_res.json()["zone_id"]
            
            # 2. Connect to WebSocket /api/ws/camera/CAM-002
            print(f"\nOpening WebSocket for CAM-002 to verify intrusion into {zone_id}...")
            intrusion_found = False
            
            with client.websocket_connect(f"/api/ws/camera/CAM-002?token={token}") as ws:
                for frame_idx in range(40):
                    msg_str = ws.receive_text()
                    msg = json.loads(msg_str)
                    alerts = msg.get("alerts", [])
                    person_count = msg.get("person_count", 0)
                    
                    for alert in alerts:
                        if alert.get("event_type") == "RESTRICTED AREA INTRUSION":
                            print(f"\n✅ SUCCESS! Received RESTRICTED AREA INTRUSION in WebSocket frame #{frame_idx}:")
                            print(json.dumps(alert, indent=2))
                            print(f"Payload context: person_count={person_count}, threat_score={msg.get('threat_score')}, highest_threat={msg.get('highest_threat')}")
                            intrusion_found = True
                            break
                            
                    if intrusion_found:
                        break
                        
            assert intrusion_found, "RESTRICTED AREA INTRUSION not found in websocket frames!"
            print("\nAll End-to-End Zone Verification checks passed!")
        finally:
            if zone_id:
                del_res = client.delete(f"/api/zones/{zone_id}", headers=headers)
                print(f"Cleanup: Deleted test zone {zone_id}, status: {del_res.status_code}")

def test_zone_e2e():
    main()

if __name__ == "__main__":
    main()
