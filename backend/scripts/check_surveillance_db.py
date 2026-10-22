import os
import sys
import asyncio
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

from app.core.config import settings

async def check_real_surveillance_db():
    print("=" * 80)
    print("CHECKING REAL SURVEILLANCE_DB FOR TEST ARTIFACTS AND EVENTS")
    print("=" * 80)

    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client["surveillance_db"]

    # Test patterns for camera_id or event_type
    test_prefixes = ["test", "cam-test", "cam-thread", "cam_test", "eval_cam", "cam_accident", "test_d"]

    events_cursor = db["events"].find({})
    test_events = []
    prod_events = []

    async for evt in events_cursor:
        cam_id = str(evt.get("camera_id", "")).lower()
        evt_type = str(evt.get("event_type", "")).lower()
        
        is_test = any(cam_id.startswith(p) or p in cam_id for p in test_prefixes) or "test" in evt_type
        if is_test:
            test_events.append(evt)
        else:
            prod_events.append(evt)

    print(f"Total events in surveillance_db: {len(test_events) + len(prod_events)}")
    print(f"Test events found: {len(test_events)}")
    print(f"Real / camera events: {len(prod_events)}")

    # Check referenced image_path and video_path in test_events
    test_screenshots = []
    test_clips = []
    for evt in test_events:
        img_p = evt.get("image_path")
        vid_p = evt.get("video_path")
        if img_p and os.path.exists(img_p):
            test_screenshots.append(img_p)
        if vid_p and os.path.exists(vid_p):
            test_clips.append(vid_p)

    # Also check screenshots/ and event_clips/ directories for files matching test patterns
    screenshots_dir = Path(settings.SCREENSHOTS_DIR)
    clips_dir = Path(settings.EVENT_CLIPS_DIR)

    disk_test_screenshots = []
    if screenshots_dir.exists():
        for f in screenshots_dir.glob("*"):
            fname = f.name.lower()
            if any(fname.startswith(p) or p in fname for p in test_prefixes):
                disk_test_screenshots.append(str(f))

    disk_test_clips = []
    if clips_dir.exists():
        for f in clips_dir.glob("*"):
            fname = f.name.lower()
            if any(fname.startswith(p) or p in fname for p in test_prefixes):
                disk_test_clips.append(str(f))

    print(f"\n--- ALL 18 EVENTS IN SURVEILLANCE_DB ---")
    events_cursor2 = db["events"].find({})
    idx = 0
    async for evt in events_cursor2:
        idx += 1
        print(f"  {idx:2d}. Camera: {evt.get('camera_id')} | Event: {evt.get('event_type')} | Level: {evt.get('threat_level')} | Status: {evt.get('status')} | Time: {evt.get('timestamp')}")

    print(f"\n--- DISK SCREENSHOTS MATCHING TEST PREFIXES ({len(disk_test_screenshots)}) ---")
    for s in disk_test_screenshots:
        print(f"  {Path(s).name}")

    print(f"\n--- DISK CLIPS MATCHING TEST PREFIXES ({len(disk_test_clips)}) ---")
    for c in disk_test_clips:
        print(f"  {Path(c).name}")

    print(f"\nTest event screenshot files linked: {len(test_screenshots)}")
    print(f"Test event clip files linked: {len(test_clips)}")
    print(f"Disk screenshots matching test prefixes: {len(disk_test_screenshots)}")
    print(f"Disk clips matching test prefixes: {len(disk_test_clips)}")

    client.close()

if __name__ == "__main__":
    asyncio.run(check_real_surveillance_db())
