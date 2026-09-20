# backend/app/models/zone.py
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List

class ZoneDB(BaseModel):
    zone_id: str
    camera_id: str
    name: str
    points: List[List[float]]  # Polygon points: [[x1, y1], [x2, y2], ...]
    threat_level: str = "HIGH"
    created_at: datetime = Field(default_factory=datetime.utcnow)
