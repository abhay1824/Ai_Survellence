from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional


class EmailSettingsResponse(BaseModel):
    """Response schema for email settings GET endpoint."""
    email: str
    notification_emails: List[str] = []
    email_notifications_enabled: bool = True
    critical_alerts_enabled: bool = True
    daily_report_enabled: bool = False


class EmailSettingsUpdate(BaseModel):
    """Schema for updating all email settings at once."""
    email: Optional[str] = None
    notification_emails: Optional[List[str]] = None
    email_notifications_enabled: Optional[bool] = None
    critical_alerts_enabled: Optional[bool] = None
    daily_report_enabled: Optional[bool] = None


class AddRecipientRequest(BaseModel):
    """Schema for adding a single notification recipient."""
    email: str


class RemoveRecipientRequest(BaseModel):
    """Schema for removing a single notification recipient."""
    email: str


class TestEmailRequest(BaseModel):
    """Schema for sending a test email notification."""
    recipient: Optional[str] = None  # If None, send to all configured recipients


class EmailNotificationResult(BaseModel):
    """Result of an email notification attempt."""
    success: bool
    recipients_count: int = 0
    message: str = ""
