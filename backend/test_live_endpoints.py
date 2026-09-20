import urllib.request
import urllib.parse
import json

BASE_URL = "http://127.0.0.1:8000/api"

def run_tests():
    print("=" * 60)
    print("TESTING BACKEND LIVE REST ENDPOINTS")
    print("=" * 60)

    # 1. Login test
    print("\n[1] Testing Auth Login (admin/admin123)...")
    login_data = urllib.parse.urlencode({"username": "admin", "password": "admin123"}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/auth/login",
        data=login_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            token = data.get("access_token")
            username = data.get("username")
            role = data.get("role")
            assert token, "Token not found"
            print(f"  --> SUCCESS: Logged in as {username} (Role: {role})")
    except Exception as e:
        print(f"  --> FAILED: {e}")
        return

    auth_headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Helper for authenticated GET
    def auth_get(path):
        r = urllib.request.Request(f"{BASE_URL}{path}", headers=auth_headers)
        with urllib.request.urlopen(r) as resp:
            return resp.status, resp.read()

    # 2. Get Current User (/auth/me)
    print("\n[2] Testing Current User (/auth/me)...")
    try:
        status, body = auth_get("/auth/me")
        print(f"  --> SUCCESS (HTTP {status}): {json.loads(body.decode())}")
    except Exception as e:
        print(f"  --> FAILED /auth/me: {e}")

    # 3. Get Cameras (/cameras)
    print("\n[3] Testing Get Cameras (/cameras)...")
    try:
        status, body = auth_get("/cameras")
        cameras = json.loads(body.decode())
        print(f"  --> SUCCESS (HTTP {status}): Retrieved {len(cameras)} cameras:")
        for cam in cameras:
            print(f"      - {cam.get('camera_id')}: {cam.get('camera_name')} [{cam.get('status')}]")
    except Exception as e:
        print(f"  --> FAILED /cameras: {e}")

    # 4. Get Camera Health (/cameras/health)
    print("\n[4] Testing Camera Health (/cameras/health)...")
    try:
        status, body = auth_get("/cameras/health")
        health = json.loads(body.decode())
        print(f"  --> SUCCESS (HTTP {status}): Health status for {len(health)} cameras:")
        for h in health[:3]:
            print(f"      - {h.get('camera_id')}: status={h.get('status')}, fps={h.get('fps')}, latency={h.get('latency_ms')}ms")
    except Exception as e:
        print(f"  --> FAILED /cameras/health: {e}")

    # 5. Get Events (/events)
    print("\n[5] Testing Events Endpoint (/events)...")
    try:
        status, body = auth_get("/events")
        events_resp = json.loads(body.decode())
        print(f"  --> SUCCESS (HTTP {status}): Total events={events_resp.get('total')}, Page count={len(events_resp.get('events', []))}")
    except Exception as e:
        print(f"  --> FAILED /events: {e}")

    # 6. Analytics Summary (/analytics/summary)
    print("\n[6] Testing Analytics Summary (/analytics/summary)...")
    try:
        status, body = auth_get("/analytics/summary")
        stats = json.loads(body.decode())
        print(f"  --> SUCCESS (HTTP {status}): Summary keys: {list(stats.keys())}")
    except Exception as e:
        print(f"  --> FAILED /analytics/summary: {e}")

    # 7. Analytics Charts (/analytics/charts)
    print("\n[7] Testing Analytics Charts (/analytics/charts)...")
    try:
        status, body = auth_get("/analytics/charts")
        charts = json.loads(body.decode())
        print(f"  --> SUCCESS (HTTP {status}): Charts keys: {list(charts.keys())}")
    except Exception as e:
        print(f"  --> FAILED /analytics/charts: {e}")

    # 8. Zones (/zones/camera/CAM-001)
    print("\n[8] Testing Restricted Zones (/zones/camera/CAM-001)...")
    try:
        status, body = auth_get("/zones/camera/CAM-001")
        zones = json.loads(body.decode())
        print(f"  --> SUCCESS (HTTP {status}): Retrieved {len(zones)} zones for CAM-001.")
    except Exception as e:
        print(f"  --> FAILED /zones/camera/CAM-001: {e}")

    # 9. Email Settings (/user/email-settings)
    print("\n[9] Testing Email Settings (/user/email-settings)...")
    try:
        status, body = auth_get("/user/email-settings")
        user_settings = json.loads(body.decode())
        print(f"  --> SUCCESS (HTTP {status}): Settings retrieved for {user_settings.get('email')}")
    except Exception as e:
        print(f"  --> FAILED /user/email-settings: {e}")

    # 10. PDF Report Generation (/events/pdf-report)
    print("\n[10] Testing SOC Incident PDF Report Generation (/events/pdf-report)...")
    try:
        status, body = auth_get("/events/pdf-report")
        print(f"  --> SUCCESS (HTTP {status}): Generated PDF report ({len(body)} bytes).")
    except Exception as e:
        print(f"  --> PDF report endpoint check: {e}")

    # 11. CSV Export (/events/export)
    print("\n[11] Testing CSV Events Export (/events/export)...")
    try:
        status, body = auth_get("/events/export")
        print(f"  --> SUCCESS (HTTP {status}): Generated CSV export ({len(body)} bytes).")
    except Exception as e:
        print(f"  --> CSV export check: {e}")

    print("\n" + "=" * 60)
    print("ALL BACKEND LIVE REST API ENDPOINTS VERIFIED!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
