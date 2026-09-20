from datetime import datetime
import logging
from fastapi import HTTPException, status
from app.core.database import get_user_collection
from app.core.security import get_password_hash, verify_password, create_access_token
from app.models.user import UserDB
from app.schemas.auth import UserCreate

logger = logging.getLogger(__name__)

async def seed_users():
    """Seed initial roles/users for testing and operation."""
    user_col = get_user_collection()
    
    # Check if any user exists
    count = await user_col.count_documents({})
    if count == 0:
        logger.info("No users found in DB. Seeding initial command center roles...")
        initial_users = [
            ("admin", "admin123", "Super Admin", "admin@company.com"),
            ("police", "police123", "Police Admin", "police@company.com"),
            ("operator", "operator123", "Control Room Operator", "operator@company.com"),
            ("viewer", "viewer123", "Viewer", "viewer@company.com"),
        ]
        for username, password, role, email in initial_users:
            user_in = UserDB(
                username=username,
                password_hash=get_password_hash(password),
                role=role,
                email=email,
                notification_emails=[],
                email_notifications_enabled=True,
                critical_alerts_enabled=True,
                daily_report_enabled=False,
                created_at=datetime.utcnow()
            )
            await user_col.insert_one(user_in.model_dump())
        logger.info("Successfully seeded 4 roles/users with email notification fields.")
    else:
        logger.info("Users exist in database. Skipping seed.")

async def authenticate_user(username: str, password: str) -> dict:
    user_col = get_user_collection()
    user = await user_col.find_one({"username": username})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    if not verify_password(password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    return user
