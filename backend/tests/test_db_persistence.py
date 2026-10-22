import os
import sys
import json
from pathlib import Path
from pymongo import MongoClient
from fastapi.testclient import TestClient

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

from app.main import app
from app.core.security import create_access_token

def run_test():
    mongo_client = MongoClient("mongodb://localhost:27017")
    db = mongo_client["surveillance_db"]
    
    print("=================================================================")
    print("STEP 1: INITIAL STARTUP & CREATING TEST ZONE")
    print("=================================================================")
    token = create_access_token(subject="admin", role="Super Admin")
    headers = {"Authorization": f"Bearer {token}"}
    
    created_zone_id = None
    try:
        with TestClient(app) as client:
            # Create zone
            zone_payload = {
                "camera_id": "CAM-001",
                "name": "Persistent Server Room",
                "points": [[10.0, 10.0], [500.0, 10.0], [500.0, 500.0], [10.0, 500.0]],
                "threat_level": "HIGH"
            }
            res = client.post("/api/zones", json=zone_payload, headers=headers)
            print(f"Created zone status: {res.status_code}")
            created_zone_id = res.json()["zone_id"]
            print(f"Created Zone ID: {created_zone_id}")

        print("\n=================================================================")
        print("STEP 2: BACKEND RESTART #1 (Preserving Collections)")
        print("=================================================================")
        with TestClient(app) as client:
            # 1. API Verification
            cams_res = client.get("/api/cameras", headers=headers)
            zones_res = client.get("/api/zones/camera/CAM-001", headers=headers)
            zones = zones_res.json() if zones_res.status_code == 200 else []
            zone_ids = [z["zone_id"] for z in zones]
            
            # 2. Direct MongoDB collection verification
            user_count = db["users"].count_documents({})
            cam_count = db["cameras"].count_documents({})
            zone_count = db["restricted_zones"].count_documents({})
            
            print(f"After Restart 1 -> Users count: {user_count} (Expected: 4)")
            print(f"After Restart 1 -> Cameras count: {cam_count} (Expected: 5)")
            print(f"After Restart 1 -> Restricted Zones in DB: {zone_count}")
            print(f"After Restart 1 -> CAM-001 API Zones: {zone_ids}")
            
            assert user_count == 4, f"Expected 4 users, got {user_count}"
            assert cam_count == 5, f"Expected 5 cameras, got {cam_count}"
            assert created_zone_id in zone_ids, f"Zone {created_zone_id} did not persist after restart 1!"
            print("✅ Restart #1 Verification Passed: No duplicates, collections preserved, zone persisted!")

        print("\n=================================================================")
        print("STEP 3: BACKEND RESTART #2 (Second Preservation Check)")
        print("=================================================================")
        with TestClient(app) as client:
            cams_res = client.get("/api/cameras", headers=headers)
            zones_res = client.get("/api/zones/camera/CAM-001", headers=headers)
            zones = zones_res.json() if zones_res.status_code == 200 else []
            zone_ids = [z["zone_id"] for z in zones]
            
            user_count = db["users"].count_documents({})
            cam_count = db["cameras"].count_documents({})
            zone_count = db["restricted_zones"].count_documents({})
            
            print(f"After Restart 2 -> Users count: {user_count} (Expected: 4)")
            print(f"After Restart 2 -> Cameras count: {cam_count} (Expected: 5)")
            print(f"After Restart 2 -> Restricted Zones in DB: {zone_count}")
            print(f"After Restart 2 -> CAM-001 API Zones: {zone_ids}")
            
            assert user_count == 4, f"Expected 4 users, got {user_count}"
            assert cam_count == 5, f"Expected 5 cameras, got {cam_count}"
            assert created_zone_id in zone_ids, f"Zone {created_zone_id} did not persist after restart 2!"
            print("✅ Restart #2 Verification Passed: No duplicates, collections preserved, zone persisted!")

        print("\n=================================================================")
        print("ALL RESTART PERSISTENCE TESTS PASSED!")
        print("=================================================================")
    finally:
        if created_zone_id:
            del_res = db["restricted_zones"].delete_one({"zone_id": created_zone_id})
            print(f"Cleanup: Deleted test zone {created_zone_id} (count: {del_res.deleted_count})")

def test_db_persistence():
    run_test()

if __name__ == "__main__":
    run_test()
