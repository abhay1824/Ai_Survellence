import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env file automatically from backend dir, project root, or current directory
_current_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(os.path.dirname(_current_dir))
_project_root = os.path.dirname(_backend_dir)

load_dotenv(os.path.join(_backend_dir, ".env"))
load_dotenv(os.path.join(_project_root, ".env"))
load_dotenv()

def _resolve_model_path(primary_rel: str, fallback_rel: str) -> str:
    primary = os.path.join(_project_root, primary_rel)
    if os.path.exists(primary):
        return primary
    return os.path.join(_project_root, fallback_rel)

class Settings(BaseSettings):
    PROJECT_NAME: str = "Smart Surveillance & Crime Monitoring"
    API_V1_STR: str = "/api"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "SUPER_SECRET_SURVEILLANCE_KEY_CHANGE_IN_PRODUCTION")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # MongoDB Config
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    DATABASE_NAME: str = "surveillance_db"
    RESET_DB_ON_START: bool = os.getenv("RESET_DB_ON_START", "false").lower() in ("true", "1", "yes")
    
    # AI Config — Unified weights/ registry with legacy fallbacks
    WEAPON_MODEL_PATH: str = os.getenv(
        "WEAPON_MODEL_PATH",
        _resolve_model_path(
            os.path.join("weights", "weapon_yolov5.pt"),
            os.path.join(
                "OD-WeaponDetection",
                "Weapons and similar handled objects",
                "Sohas_weapon-Detection-YOLOv5",
                "runs",
                "detect",
                "weapon6cls",
                "weights",
                "best.pt"
            )
        )
    )
    FIGHT_MODEL_PATH: str = os.getenv(
        "FIGHT_MODEL_PATH",
        _resolve_model_path(
            os.path.join("weights", "fight_yolo11n_cls.pt"),
            os.path.join("runs", "classify", "fight_classifier", "weights", "best.pt")
        )
    )
    FIGHT_MIN_PERSONS_TO_RUN: int = int(os.getenv("FIGHT_MIN_PERSONS_TO_RUN", "2"))
    FIGHT_CONFIRM_MIN_PERSONS_SEEN: int = int(os.getenv("FIGHT_CONFIRM_MIN_PERSONS_SEEN", "4"))
    ACCIDENT_MODEL_PATH: str = os.getenv(
        "ACCIDENT_MODEL_PATH",
        _resolve_model_path(
            os.path.join("weights", "accident_yolov8.pt"),
            os.path.join("runs", "detect", "accident_detector", "weights", "best.pt")
        )
    )
    ACCIDENT_CONFIDENCE_THRESHOLD: float = float(os.getenv("ACCIDENT_CONFIDENCE_THRESHOLD", "0.52"))
    PERSON_MODEL_PATH: str = os.getenv(
        "PERSON_MODEL_PATH",
        _resolve_model_path(
            os.path.join("weights", "person_yolo11s.pt"),
            "yolo11s.pt"
        )
    )
    YOLO_MODEL: str = WEAPON_MODEL_PATH
    CONFIDENCE_THRESHOLD: float = 0.35
    WEAPON_CONFIDENCE_THRESHOLD: float = 0.38
    SECURITY_CLASSES: set = {"pistol", "knife"}
    BASE_THREAT_SCORES: dict = {"pistol": 80, "knife": 60}
    UNATTENDED_HIGH_AFTER_FRAMES: int = int(os.getenv("UNATTENDED_HIGH_AFTER_FRAMES", "30"))

    # Paths
    SCREENSHOTS_DIR: str = os.getenv("SCREENSHOTS_DIR", os.path.join(_project_root, "screenshots"))
    EVENT_CLIPS_DIR: str = os.getenv("EVENT_CLIPS_DIR", os.path.join(_project_root, "event_clips"))

    # Email / SMTP Configuration
    SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    SMTP_FROM_EMAIL: str = os.getenv("SMTP_FROM_EMAIL", "aegis-alerts@surveillance.local")
    SMTP_FROM_NAME: str = os.getenv("SMTP_FROM_NAME", "Eagle SOC")
    SMTP_USE_TLS: bool = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
    EMAIL_ENABLED: bool = os.getenv("EMAIL_ENABLED", "true").lower() == "true"
    EMAIL_RETRY_ATTEMPTS: int = int(os.getenv("EMAIL_RETRY_ATTEMPTS", "3"))
    EMAIL_RETRY_DELAY: int = int(os.getenv("EMAIL_RETRY_DELAY", "5"))
    ALERT_RECIPIENT_EMAIL: str = os.getenv("ALERT_RECIPIENT_EMAIL", "shabella271@gmail.com")

    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "allow"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(settings.EVENT_CLIPS_DIR, exist_ok=True)
