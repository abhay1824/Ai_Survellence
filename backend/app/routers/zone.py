import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from app.core.database import get_database
from app.schemas.zone import ZoneCreate, ZoneResponse, ZoneUpdate
from app.routers.auth import RoleChecker

router = APIRouter(prefix="/zones", tags=["zones"])

allow_all = RoleChecker(["Super Admin", "Police Admin", "Control Room Operator", "Viewer"])
allow_admin = RoleChecker(["Super Admin", "Police Admin"])

def get_zone_collection():
    db = get_database()
    return db["restricted_zones"]

@router.get("/camera/{camera_id}", response_model=list[ZoneResponse])
async def get_zones_by_camera(camera_id: str, current_user: dict = Depends(allow_all)):
    col = get_zone_collection()
    cursor = col.find({"camera_id": camera_id})
    zones = [doc async for doc in cursor]
    for z in zones:
        z["_id"] = str(z["_id"])
    return zones

@router.post("", response_model=ZoneResponse, status_code=status.HTTP_201_CREATED)
async def create_zone(zone_in: ZoneCreate, current_user: dict = Depends(allow_admin)):
    col = get_zone_collection()
    zone_id = f"ZONE-{uuid.uuid4().hex[:6].upper()}"
    doc = {
        "zone_id": zone_id,
        "camera_id": zone_in.camera_id,
        "name": zone_in.name,
        "points": zone_in.points,
        "threat_level": zone_in.threat_level,
        "created_at": datetime.utcnow()
    }
    result = await col.insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    return doc

@router.put("/{zone_id}", response_model=ZoneResponse)
async def update_zone(zone_id: str, zone_in: ZoneUpdate, current_user: dict = Depends(allow_admin)):
    col = get_zone_collection()
    update_data = {k: v for k, v in zone_in.model_dump().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    result = await col.find_one_and_update(
        {"zone_id": zone_id},
        {"$set": update_data},
        return_document=True
    )
    if not result:
        raise HTTPException(status_code=404, detail="Zone not found")
    result["_id"] = str(result["_id"])
    return result

@router.delete("/{zone_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_zone(zone_id: str, current_user: dict = Depends(allow_admin)):
    col = get_zone_collection()
    res = await col.delete_one({"zone_id": zone_id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Zone not found")
    return None
