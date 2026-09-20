from fastapi import APIRouter, Depends
from app.routers.auth import RoleChecker
from app.services import event_service
from app.services.email_service import EmailService

router = APIRouter(prefix="/analytics", tags=["analytics"])

allow_all = RoleChecker(["Super Admin", "Police Admin", "Control Room Operator", "Viewer"])
allow_admin = RoleChecker(["Super Admin", "Police Admin", "Control Room Operator"])

@router.get("/summary")
async def get_summary(current_user: dict = Depends(allow_all)):
    stats = await event_service.get_dashboard_summary()
    # Serialize ObjectId of recent events
    for event in stats.get("recent_events", []):
        event["_id"] = str(event["_id"])
    return stats

@router.get("/charts")
async def get_charts(current_user: dict = Depends(allow_all)):
    chart_data = await event_service.get_analytics_charts()
    return chart_data

@router.post("/daily-digest/send")
async def trigger_daily_digest(current_user: dict = Depends(allow_admin)):
    """Triggers the Daily Security Report delivery to all subscribed users."""
    stats = await event_service.get_daily_digest_stats()
    result = await EmailService.send_daily_digest(stats)
    return result
