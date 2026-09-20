from fastapi import APIRouter, Depends, status, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.routers.auth import RoleChecker
from app.schemas.camera import CameraCreate, CameraResponse, CameraUpdate
from app.services import camera_service
from app.services.camera_manager import camera_manager

router = APIRouter(prefix="/cameras", tags=["cameras"])

allow_all = RoleChecker(["Super Admin", "Police Admin", "Control Room Operator", "Viewer"])
allow_admin = RoleChecker(["Super Admin", "Police Admin"])


# === Request Models ===

class CameraConnectRequest(BaseModel):
    source_url: Optional[str] = None  # Override source URL; if None, uses DB value


class CameraSourceUpdate(BaseModel):
    stream_url: str  # New source URL


# === CRUD Endpoints ===

@router.get("", response_model=list[CameraResponse])
async def get_cameras(current_user: dict = Depends(allow_all)):
    return await camera_service.get_all_cameras()


@router.get("/health")
async def get_camera_health(current_user: dict = Depends(allow_all)):
    """Real-time camera health from CameraManager workers."""
    health_data = camera_manager.get_all_health()
    if health_data:
        return health_data
    # Fallback to DB-based health if no workers running
    return await camera_service.get_camera_health()


@router.get("/health/{camera_id}")
async def get_single_camera_health(camera_id: str, current_user: dict = Depends(allow_all)):
    """Real-time health for a single camera."""
    health = camera_manager.get_health(camera_id)
    if health:
        return health
    raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found or not running")


@router.post("/detect-sources")
async def detect_available_sources(current_user: dict = Depends(allow_admin)):
    """Auto-detect available local webcam devices."""
    sources = camera_manager.detect_available_webcams(max_index=5)
    return {"sources": sources, "count": len(sources)}


@router.post("/connect/{camera_id}")
async def connect_camera(camera_id: str, req: CameraConnectRequest = None,
                         current_user: dict = Depends(allow_admin)):
    """Start streaming a camera. Optionally override source URL."""
    camera = await camera_service.get_camera_by_code(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    source_url = camera.get("stream_url", "0")
    if req and req.source_url:
        source_url = req.source_url
        # Persist updated source URL to DB
        await camera_service.update_camera_by_code(camera_id, {"stream_url": source_url})

    success = camera_manager.start_camera(
        camera_id=camera_id,
        camera_name=camera.get("camera_name", camera_id),
        source_url=source_url,
    )
    return {"camera_id": camera_id, "connected": success, "source_url": source_url}


@router.post("/disconnect/{camera_id}")
async def disconnect_camera(camera_id: str, current_user: dict = Depends(allow_admin)):
    """Stop streaming a camera."""
    stopped = camera_manager.stop_camera(camera_id)
    return {"camera_id": camera_id, "disconnected": stopped}


@router.post("/{camera_id}/source")
async def update_camera_source(camera_id: str, req: CameraSourceUpdate,
                                current_user: dict = Depends(allow_admin)):
    """Update source URL for a camera at runtime and reconnect."""
    camera = await camera_service.get_camera_by_code(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    # Update in DB
    await camera_service.update_camera_by_code(camera_id, {"stream_url": req.stream_url})

    # Update running worker
    if camera_manager.is_camera_running(camera_id):
        camera_manager.update_camera_source(camera_id, req.stream_url)
    else:
        camera_manager.start_camera(
            camera_id=camera_id,
            camera_name=camera.get("camera_name", camera_id),
            source_url=req.stream_url,
        )

    return {"camera_id": camera_id, "stream_url": req.stream_url, "status": "updated"}


@router.get("/{id}", response_model=CameraResponse)
async def get_camera(id: str, current_user: dict = Depends(allow_all)):
    return await camera_service.get_camera_by_id(id)


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(camera_in: CameraCreate, current_user: dict = Depends(allow_admin)):
    return await camera_service.create_camera(camera_in)


@router.put("/{id}", response_model=CameraResponse)
async def update_camera(id: str, camera_in: CameraUpdate, current_user: dict = Depends(allow_admin)):
    updated_cam = await camera_service.update_camera(id, camera_in)
    if updated_cam:
        cam_code = updated_cam.get("camera_id")
        new_url = updated_cam.get("stream_url")
        if cam_code and new_url:
            if camera_manager.is_camera_running(cam_code):
                camera_manager.update_camera_source(cam_code, new_url)
            else:
                camera_manager.start_camera(cam_code, updated_cam.get("camera_name", cam_code), new_url)
    return updated_cam


@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(id: str, current_user: dict = Depends(allow_admin)):
    await camera_service.delete_camera(id)
    return None
