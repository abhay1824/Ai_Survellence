# backend/app/models/camera.py
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class CameraStatus(str, Enum):
    ONLINE = "online"
    NO_VIDEO_INPUT = "no_video_input"
    OFFLINE = "offline"
    FROZEN = "frozen"
    TAMPERED = "tampered"
    WARNING = "warning"
    CRITICAL_ALERT = "critical_alert"

class CameraDB(BaseModel):
    camera_id: str
    camera_name: str
    latitude: float
    longitude: float
    stream_url: str
    status: str = CameraStatus.ONLINE.value  # Online, No Video Input, Warning, Critical Alert, Offline, Tampered, Frozen
    fps: float = 15.0
    latency_ms: int = 24
    freeze_detected: bool = False
    tamper_detected: bool = False
    last_ping: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)

