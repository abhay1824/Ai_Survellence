import logging
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

logger = logging.getLogger(__name__)

class Database:
    client: AsyncIOMotorClient = None
    db = None

db_helper = Database()

async def connect_to_mongo():
    logger.info("Connecting to MongoDB...")
    db_helper.client = AsyncIOMotorClient(settings.MONGODB_URL)
    db_helper.db = db_helper.client[settings.DATABASE_NAME]
    logger.info("Connected to MongoDB successfully!")

async def close_mongo_connection():
    logger.info("Closing MongoDB connection...")
    if db_helper.client:
        db_helper.client.close()
    logger.info("MongoDB connection closed.")

def get_database():
    return db_helper.db

# Collection Helper references
def get_user_collection():
    return db_helper.db["users"]

def get_camera_collection():
    return db_helper.db["cameras"]

def get_event_collection():
    return db_helper.db["events"]

def get_zone_collection():
    return db_helper.db["restricted_zones"]
