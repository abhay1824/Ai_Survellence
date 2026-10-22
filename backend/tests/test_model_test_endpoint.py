import os
import sys
import time
import subprocess
import requests
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
demo_images_dir = repo_root / "demo_images"

DEMO_FILES = [
    ("pistol", demo_images_dir / "pistol" / "pistol_1.jpg"),
    ("pistol", demo_images_dir / "pistol" / "pistol_2.jpg"),
    ("pistol", demo_images_dir / "pistol" / "pistol_3.jpg"),
    ("knife", demo_images_dir / "knife" / "knife_1.jpg"),
    ("knife", demo_images_dir / "knife" / "knife_2.jpg"),
    ("knife", demo_images_dir / "knife" / "knife_3.jpg"),
    ("fight", demo_images_dir / "fight" / "fight_1.jpg"),
    ("fight", demo_images_dir / "fight" / "fight_2.jpg"),
    ("fight", demo_images_dir / "fight" / "fight_3.jpg"),
    ("accident", demo_images_dir / "accident" / "accident_1.jpg"),
    ("accident", demo_images_dir / "accident" / "accident_2.jpg"),
    ("accident", demo_images_dir / "accident" / "accident_3.jpg"),
]

def main():
    port = 8899
    base_url = f"http://127.0.0.1:{port}"
    predict_url = f"{base_url}/api/model-test/predict"

    print("=" * 100)
    print(f"STARTING REAL HTTP SERVER ON {base_url} TO TEST /api/model-test/predict")
    print("=" * 100)

    # Launch uvicorn server
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    log_path = repo_root / "scratch" / "test_model_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")

    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(backend_dir),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Wait for server to be responsive
    print("Waiting for server startup...")
    ready = False
    for attempt in range(40):
        try:
            r = requests.get(f"{base_url}/", timeout=1.0)
            if r.status_code == 200:
                ready = True
                print("✅ Server is online and responding!")
                break
        except Exception:
            time.sleep(1.0)

    if not ready:
        print("❌ Server failed to start in time. Terminating...")
        server_proc.terminate()
        return

    results = []
    try:
        print("\n" + "=" * 100)
        print("UPLOADING 12 DEMO IMAGES OVER REAL HTTP TO /api/model-test/predict")
        print("=" * 100)

        for cat, img_path in DEMO_FILES:
            if not img_path.exists():
                print(f"File not found: {img_path}")
                continue

            with open(img_path, "rb") as f:
                files = {"file": (img_path.name, f, "image/jpeg")}
                data = {
                    "model_type": "all",
                    "confidence_threshold": "0.35"
                }
                t0 = time.time()
                resp = requests.post(predict_url, files=files, data=data, timeout=30.0)
                dur_ms = round((time.time() - t0) * 1000, 1)

            if resp.status_code != 200:
                print(f"❌ Error uploading {img_path.name}: HTTP {resp.status_code} - {resp.text}")
                continue

            res_json = resp.json()
            threat = res_json.get("threat_assessment", {})
            weapon_res = res_json.get("weapon_result", {})
            accident_res = res_json.get("accident_result", {})
            fight_res = res_json.get("fight_result") or {}

            results.append({
                "category": cat,
                "file": img_path.name,
                "status_code": resp.status_code,
                "latency_ms": dur_ms,
                "threat_score": threat.get("score"),
                "threat_level": threat.get("level"),
                "summary": threat.get("summary"),
                "weapons": [f"{d['class']}:{d['confidence']:.2f}" for d in weapon_res.get("detections", [])],
                "accidents": [f"Accident:{d['confidence']:.2f}" for d in accident_res.get("detections", [])],
                "fight_top": fight_res.get("top_class"),
                "fight_conf": fight_res.get("confidence", 0.0),
                "is_fight": fight_res.get("is_fight", False)
            })

            print(f"  [{cat.upper():<8}] {img_path.name:<20} | Status: {resp.status_code} | Level: {threat.get('level'):<8} | Score: {threat.get('score'):<3} | Latency: {dur_ms}ms")

        # Print formatted report table
        print("\n" + "=" * 120)
        print("REAL HTTP /api/model-test/predict RETURNS FOR 12 DEMO IMAGES")
        print("=" * 120)
        print(f"{'Category':<10} | {'File':<18} | {'Level':<8} | {'Score':<5} | {'Weapons':<22} | {'Fight':<20} | {'Accident':<18} | {'Latency'}")
        print("-" * 120)
        for r in results:
            wpn_str = ", ".join(r["weapons"]) if r["weapons"] else "None"
            fgt_str = f"{r['fight_top']}:{r['fight_conf']:.2f}" if r["fight_top"] else "None"
            acc_str = ", ".join(r["accidents"]) if r["accidents"] else "None"
            print(f"{r['category']:<10} | {r['file']:<18} | {r['threat_level']:<8} | {r['threat_score']:<5} | {wpn_str:<22} | {fgt_str:<20} | {acc_str:<18} | {r['latency_ms']}ms")
        print("=" * 120)

    finally:
        print("\nShutting down test HTTP server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except Exception:
            server_proc.kill()
        log_file.close()
        print("Server shutdown complete.")

if __name__ == "__main__":
    main()
