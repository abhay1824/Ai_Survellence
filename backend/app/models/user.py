# backend/app/models/user.py
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field

class UserDB(BaseModel):
    username: str
    password_hash: str
    role: str
    email: str = ""
    notification_emails: List[str] = Field(default_factory=list)
    email_notifications_enabled: bool = True
    critical_alerts_enabled: bool = True
    daily_report_enabled: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
