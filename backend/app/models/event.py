# backend/app/models/event.py
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class EventDB(BaseModel):
    event_id: str
    camera_id: str
    event_type: str
    threat_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    threat_score: int = 0
    confidence: float
    latitude: float
    longitude: float
    person_id: Optional[int] = None
    zone_id: Optional[str] = None
    duration: Optional[float] = None  # for loitering / active duration
    correlated_events: List[str] = Field(default_factory=list)
    xai_explanation: Optional[Dict[str, Any]] = None
    status: str = "ACTIVE"  # ACTIVE, ACKNOWLEDGED, RESOLVED, ESCALATED
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    # --- Fight Detection Fields ---
    fight_detected: bool = False
    fight_confidence: float = 0.0
    fight_participants: List[int] = Field(default_factory=list)
    fight_duration_sec: float = 0.0
    # --- Accident Detection Fields ---
    accident_detected: bool = False
    accident_confidence: float = 0.0

