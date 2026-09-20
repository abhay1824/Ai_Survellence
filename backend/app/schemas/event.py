from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Dict, Any
from datetime import datetime
from bson import ObjectId
from pydantic import BeforeValidator
from typing_extensions import Annotated

PyObjectId = Annotated[str, BeforeValidator(str)]

class EventBase(BaseModel):
    event_id: str
    camera_id: str
    event_type: str
    threat_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    threat_score: int = 0
    confidence: float
    latitude: float
    longitude: float
    person_id: Optional[int] = None
    zone_id: Optional[str] = None
    duration: Optional[float] = None
    correlated_events: List[str] = Field(default_factory=list)
    xai_explanation: Optional[Dict[str, Any]] = None
    status: str = "ACTIVE"
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    # --- Fight Detection Fields ---
    fight_detected: bool = False
    fight_confidence: float = 0.0
    fight_participants: List[int] = Field(default_factory=list)
    fight_duration_sec: float = 0.0
    # --- Accident Detection Fields ---
    accident_detected: bool = False
    accident_confidence: float = 0.0


class EventCreate(EventBase):
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class EventResponse(EventBase):
    id: PyObjectId = Field(alias="_id")
    timestamp: datetime

    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str}
        from_attributes = True

class AlarmAction(BaseModel):
    user: Optional[str] = "Operator"
