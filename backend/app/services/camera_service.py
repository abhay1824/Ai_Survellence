from datetime import datetime
import logging
from bson import ObjectId
from fastapi import HTTPException, status
from app.core.database import get_camera_collection
from app.models.camera import CameraDB
from app.schemas.camera import CameraCreate, CameraUpdate

logger = logging.getLogger(__name__)

def get_access_location() -> tuple[float, float]:
    """Fetch live access point geolocation from public IP, fallback to Bengaluru coordinates if offline."""
    try:
        import urllib.request, json
        req = urllib.request.Request("http://ip-api.com/json/", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            if data.get("status") == "success":
                lat = float(data.get("lat"))
                lon = float(data.get("lon"))
                logger.info(f"Detected live access location: Lat {lat}, Lon {lon} ({data.get('city')}, {data.get('country')})")
                return lat, lon
    except Exception as e:
        logger.warning(f"Could not resolve IP geolocation: {e}")
    return 12.9716, 77.5946

async def seed_cameras():
    """Seed initial smart city / SOC demonstration cameras centered on Bengaluru with live OpenCV / Stream URLs."""
    camera_col = get_camera_collection()
    count = await camera_col.count_documents({})
    
    # Bengaluru key surveillance nodes with live webcam and sample dataset video feeds
    bengaluru_cameras = [
        ("CAM-001", "MG Road Command Center Node", 12.9756, 77.6066, "0", "Online"),
        ("CAM-002", "Indiranagar 100ft Road Junction", 12.9784, 77.6408, "sample_videos/fight.mp4", "Online"),
        ("CAM-003", "Koramangala Signal Node", 12.9352, 77.6245, "sample_videos/accident.mp4", "Online"),
        ("CAM-004", "Electronic City Phase 1 Gate", 12.8452, 77.6602, "sample_videos/weapon.mp4", "Online"),
        ("CAM-005", "Manyata Tech Park Perimeter", 13.0454, 77.6200, "sample_videos/normal.mp4", "Online")
    ]
    
    if count == 0:
        logger.info("Seeding command center surveillance nodes in Bengaluru...")
        for cid, name, lat, lon, stream, status_val in bengaluru_cameras:
            camera_in = CameraDB(
                camera_id=cid,
                camera_name=name,
                latitude=lat,
                longitude=lon,
                stream_url=stream,
                status=status_val,
                fps=15.0,
                latency_ms=18,
                created_at=datetime.utcnow()
            )
            await camera_col.insert_one(camera_in.model_dump())
        logger.info("Seeded 5 surveillance nodes in Bengaluru.")
    else:
        # Migrate existing DB entries away from synthetic to sample video files
        for cid, name, lat, lon, stream, status_val in bengaluru_cameras:
            existing = await camera_col.find_one({"camera_id": cid})
            if existing:
                updates = {}
                if existing.get("stream_url") == "synthetic":
                    updates["stream_url"] = stream
                    updates["status"] = status_val
                # Ensure coordinates are in Bengaluru area
                if abs(existing.get("latitude", 0) - 12.97) > 2.0:
                    updates["latitude"] = lat
                    updates["longitude"] = lon
                    updates["camera_name"] = name
                if updates:
                    await camera_col.update_one({"camera_id": cid}, {"$set": updates})

async def get_all_cameras() -> list[dict]:
    camera_col = get_camera_collection()
    cursor = camera_col.find({})
    cameras = [doc async for doc in cursor]
    for c in cameras:
        c["_id"] = str(c["_id"])
    return cameras

async def get_camera_by_id(id_str: str) -> dict:
    if not ObjectId.is_valid(id_str):
        raise HTTPException(status_code=400, detail="Invalid MongoDB ID")
    camera_col = get_camera_collection()
    camera = await camera_col.find_one({"_id": ObjectId(id_str)})
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    camera["_id"] = str(camera["_id"])
    return camera

async def get_camera_by_code(camera_id: str) -> dict:
    camera_col = get_camera_collection()
    camera = await camera_col.find_one({"camera_id": camera_id})
    if camera:
        camera["_id"] = str(camera["_id"])
    return camera

async def update_camera_by_code(camera_id: str, update_dict: dict) -> dict:
    """Update camera fields by camera_id code (e.g. CAM-001)."""
    camera_col = get_camera_collection()
    res = await camera_col.update_one({"camera_id": camera_id}, {"$set": update_dict})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    return await get_camera_by_code(camera_id)

async def get_camera_health() -> list[dict]:
    """
    MODULE 10: Returns health status, FPS, ping latency, and tamper flags for all cameras.
    """
    camera_col = get_camera_collection()
    cursor = camera_col.find({})
    health_list = []
    async for cam in cursor:
        status_val = cam.get("status", "online")
        health_list.append({
            "id": str(cam["_id"]),
            "camera_id": cam.get("camera_id"),
            "camera_name": cam.get("camera_name"),
            "status": status_val,
            "fps": cam.get("fps", 15.0),
            "latency_ms": cam.get("latency_ms", 22),
            "freeze_detected": cam.get("freeze_detected", False) or str(status_val).lower() == "frozen",
            "tamper_detected": cam.get("tamper_detected", False) or str(status_val).lower() == "tampered",
            "last_ping": cam.get("last_ping", datetime.utcnow()).isoformat() if isinstance(cam.get("last_ping"), datetime) else cam.get("last_ping")
        })
    return health_list

async def create_camera(camera_data: CameraCreate) -> dict:
    camera_col = get_camera_collection()
    existing = await camera_col.find_one({"camera_id": camera_data.camera_id})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Camera with ID {camera_data.camera_id} already exists"
        )
        
    db_obj = CameraDB(
        camera_id=camera_data.camera_id,
        camera_name=camera_data.camera_name,
        latitude=camera_data.latitude,
        longitude=camera_data.longitude,
        stream_url=camera_data.stream_url,
        status=camera_data.status,
        created_at=datetime.utcnow()
    )
    res = await camera_col.insert_one(db_obj.model_dump())
    created = await camera_col.find_one({"_id": res.inserted_id})
    created["_id"] = str(created["_id"])
    return created

async def update_camera(id_str: str, camera_data: CameraUpdate) -> dict:
    if not ObjectId.is_valid(id_str):
        raise HTTPException(status_code=400, detail="Invalid MongoDB ID")
    camera_col = get_camera_collection()
    update_dict = {k: v for k, v in camera_data.model_dump().items() if v is not None}
    if not update_dict:
        return await get_camera_by_id(id_str)
        
    res = await camera_col.update_one({"_id": ObjectId(id_str)}, {"$set": update_dict})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Camera not found")
    return await get_camera_by_id(id_str)

async def delete_camera(id_str: str) -> bool:
    if not ObjectId.is_valid(id_str):
        raise HTTPException(status_code=400, detail="Invalid MongoDB ID")
    camera_col = get_camera_collection()
    res = await camera_col.delete_one({"_id": ObjectId(id_str)})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Camera not found")
    return True
