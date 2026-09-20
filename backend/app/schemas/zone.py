# backend/app/schemas/zone.py
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional

class ZoneBase(BaseModel):
    name: str
    points: List[List[float]]
    threat_level: str = "HIGH"

class ZoneCreate(ZoneBase):
    camera_id: str

class ZoneUpdate(BaseModel):
    name: Optional[str] = None
    points: Optional[List[List[float]]] = None
    threat_level: Optional[str] = None

class ZoneResponse(ZoneBase):
    zone_id: str
    camera_id: str
    created_at: datetime

    class Config:
        from_attributes = True
