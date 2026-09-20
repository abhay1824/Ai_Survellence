"""
AEGIS SOC — User Email Settings Router
Manages email notification preferences and recipient configuration.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from app.routers.auth import get_current_user, RoleChecker
from app.core.database import get_user_collection
from app.schemas.user import (
    EmailSettingsResponse,
    EmailSettingsUpdate,
    AddRecipientRequest,
    RemoveRecipientRequest,
    TestEmailRequest,
    EmailNotificationResult,
)
from app.services.email_service import EmailService

router = APIRouter(prefix="/user", tags=["user"])

allow_all = RoleChecker(["Super Admin", "Police Admin", "Control Room Operator", "Viewer"])
allow_admin = RoleChecker(["Super Admin", "Police Admin"])


@router.get("/email-settings", response_model=EmailSettingsResponse)
async def get_email_settings(current_user: dict = Depends(get_current_user)):
    """Retrieve the current user's email notification settings."""
    return EmailSettingsResponse(
        email=current_user.get("email", ""),
        notification_emails=current_user.get("notification_emails", []),
        email_notifications_enabled=current_user.get("email_notifications_enabled", True),
        critical_alerts_enabled=current_user.get("critical_alerts_enabled", True),
        daily_report_enabled=current_user.get("daily_report_enabled", False),
    )


@router.put("/email-settings", response_model=EmailSettingsResponse)
async def update_email_settings(
    payload: EmailSettingsUpdate,
    current_user: dict = Depends(get_current_user),
):
    """Update the current user's email notification settings."""
    user_col = get_user_collection()
    username = current_user.get("username")

    update_fields = {}
    if payload.email is not None:
        update_fields["email"] = payload.email
    if payload.notification_emails is not None:
        # Deduplicate and validate
        seen = set()
        clean_emails = []
        for e in payload.notification_emails:
            e_lower = e.strip().lower()
            if e_lower and e_lower not in seen:
                seen.add(e_lower)
                clean_emails.append(e.strip())
        update_fields["notification_emails"] = clean_emails
    if payload.email_notifications_enabled is not None:
        update_fields["email_notifications_enabled"] = payload.email_notifications_enabled
    if payload.critical_alerts_enabled is not None:
        update_fields["critical_alerts_enabled"] = payload.critical_alerts_enabled
    if payload.daily_report_enabled is not None:
        update_fields["daily_report_enabled"] = payload.daily_report_enabled

    if not update_fields:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields provided for update",
        )

    await user_col.update_one(
        {"username": username},
        {"$set": update_fields},
    )

    updated_user = await user_col.find_one({"username": username})
    return EmailSettingsResponse(
        email=updated_user.get("email", ""),
        notification_emails=updated_user.get("notification_emails", []),
        email_notifications_enabled=updated_user.get("email_notifications_enabled", True),
        critical_alerts_enabled=updated_user.get("critical_alerts_enabled", True),
        daily_report_enabled=updated_user.get("daily_report_enabled", False),
    )


@router.post("/email-settings/add-recipient", response_model=EmailSettingsResponse)
async def add_recipient(
    payload: AddRecipientRequest,
    current_user: dict = Depends(get_current_user),
):
    """Add a new email recipient to the current user's notification list."""
    user_col = get_user_collection()
    username = current_user.get("username")

    new_email = payload.email.strip()
    if not new_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email address cannot be empty",
        )

    existing_emails = current_user.get("notification_emails", [])
    if new_email.lower() in [e.lower() for e in existing_emails]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Recipient {new_email} already exists",
        )

    await user_col.update_one(
        {"username": username},
        {"$push": {"notification_emails": new_email}},
    )

    updated_user = await user_col.find_one({"username": username})
    return EmailSettingsResponse(
        email=updated_user.get("email", ""),
        notification_emails=updated_user.get("notification_emails", []),
        email_notifications_enabled=updated_user.get("email_notifications_enabled", True),
        critical_alerts_enabled=updated_user.get("critical_alerts_enabled", True),
        daily_report_enabled=updated_user.get("daily_report_enabled", False),
    )


@router.delete("/email-settings/remove-recipient", response_model=EmailSettingsResponse)
async def remove_recipient(
    payload: RemoveRecipientRequest,
    current_user: dict = Depends(get_current_user),
):
    """Remove an email recipient from the current user's notification list."""
    user_col = get_user_collection()
    username = current_user.get("username")

    target_email = payload.email.strip()
    existing_emails = current_user.get("notification_emails", [])

    if target_email.lower() not in [e.lower() for e in existing_emails]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recipient {target_email} not found in notification list",
        )

    # Remove case-insensitive match
    updated_list = [e for e in existing_emails if e.lower() != target_email.lower()]

    await user_col.update_one(
        {"username": username},
        {"$set": {"notification_emails": updated_list}},
    )

    updated_user = await user_col.find_one({"username": username})
    return EmailSettingsResponse(
        email=updated_user.get("email", ""),
        notification_emails=updated_user.get("notification_emails", []),
        email_notifications_enabled=updated_user.get("email_notifications_enabled", True),
        critical_alerts_enabled=updated_user.get("critical_alerts_enabled", True),
        daily_report_enabled=updated_user.get("daily_report_enabled", False),
    )


@router.post("/email-settings/test", response_model=EmailNotificationResult)
async def send_test_email(
    payload: TestEmailRequest = TestEmailRequest(),
    current_user: dict = Depends(get_current_user),
):
    """Send a test email to verify notification delivery works."""
    if payload.recipient:
        recipients = [payload.recipient.strip()]
    else:
        # Send to all configured recipients for this user
        recipients = []
        primary = current_user.get("email", "")
        if primary:
            recipients.append(primary)
        for email in current_user.get("notification_emails", []):
            if email and email not in recipients:
                recipients.append(email)

    if not recipients:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No email recipients configured. Add a primary email or notification recipients first.",
        )

    result = await EmailService.send_test_email(recipients)
    return EmailNotificationResult(**result)
