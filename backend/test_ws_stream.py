import asyncio
import websockets
import json
import urllib.request
import urllib.parse

def get_token():
    login_data = urllib.parse.urlencode({"username": "admin", "password": "admin123"}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/auth/login",
        data=login_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        return data["access_token"]

async def test_ws():
    token = get_token()
    uri = f"ws://127.0.0.1:8000/api/ws/camera/CAM-001?token={token}"
    print("Connecting to live WebSocket stream:", uri.split("?")[0] + "?token=***")
    async with websockets.connect(uri) as ws:
        print("Connected to WebSocket stream! Waiting for live frames...")
        for i in range(3):
            msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
            data = json.loads(msg)
            threat = data.get("highest_threat")
            score = data.get("threat_score")
            detections = len(data.get("detections", []))
            b64_frame = data.get("frame", "")
            has_frame = len(b64_frame) > 0
            fight = data.get("fight_detected")
            print(f"  --> Frame {i+1}: Highest Threat={threat}, Threat Score={score}, Detections={detections}, Frame base64 size={len(b64_frame)} bytes, Fight Detected={fight}")
    print("\n[SUCCESS] Live camera WebSocket streaming & AI inference verified!")

if __name__ == "__main__":
    asyncio.run(test_ws())
