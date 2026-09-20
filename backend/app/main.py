import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import connect_to_mongo, close_mongo_connection, get_camera_collection, get_event_collection, get_user_collection
from app.routers import auth, camera, event, stream, analytics, zone, user, model_test
from app.services.auth_service import seed_users
from app.services.camera_service import seed_cameras
from app.services.camera_manager import camera_manager

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    await connect_to_mongo()
    
    from app.core.database import get_zone_collection
    if getattr(settings, "RESET_DB_ON_START", False):
        try:
            await get_user_collection().drop()
            await get_camera_collection().drop()
            await get_event_collection().drop()
            await get_zone_collection().drop()
            logger.info("RESET_DB_ON_START is True: Cleared previous database collections.")
        except Exception as e:
            logger.error(f"Error dropping collections: {e}")
    else:
        logger.info("RESET_DB_ON_START is False: Preserving database collections across restart.")
        
    # Seed initial workspace database objects
    try:
        await seed_users()
        await seed_cameras()
        logger.info("Database initialized with user and camera assets. Events log cleared.")
    except Exception as e:
        logger.error(f"Startup DB seeding failed: {e}")

    # Start CameraManager — auto-connect all seeded cameras
    try:
        cam_col = get_camera_collection()
        cursor = cam_col.find({})
        async for cam in cursor:
            camera_manager.start_camera(
                camera_id=cam["camera_id"],
                camera_name=cam["camera_name"],
                source_url=cam.get("stream_url", "0"),
            )
        logger.info(f"CameraManager started with {len(camera_manager._workers)} camera workers.")
    except Exception as e:
        logger.error(f"CameraManager startup failed: {e}")
        
    yield

    # Shutdown actions
    camera_manager.stop_all()
    logger.info("CameraManager shut down all camera workers.")
    await close_mongo_connection()

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan,
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount screenshots and video clip directories for UI display
app.mount("/screenshots", StaticFiles(directory=settings.SCREENSHOTS_DIR), name="screenshots")
app.mount("/event_clips", StaticFiles(directory=settings.EVENT_CLIPS_DIR), name="event_clips")

# Include Routers
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(camera.router, prefix=settings.API_V1_STR)
app.include_router(event.router, prefix=settings.API_V1_STR)
app.include_router(stream.router, prefix=settings.API_V1_STR)
app.include_router(analytics.router, prefix=settings.API_V1_STR)
app.include_router(zone.router, prefix=settings.API_V1_STR)
app.include_router(user.router, prefix=settings.API_V1_STR)
app.include_router(model_test.router, prefix=settings.API_V1_STR)

@app.get("/")
def read_root():
    return {"status": "Online", "service": "Eagle Eye Smart Surveillance SOC API", "version": "2.0.0"}
