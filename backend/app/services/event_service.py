from datetime import datetime, timedelta
import asyncio
import logging
import uuid
from bson import ObjectId
from fastapi import HTTPException
import pandas as pd
from app.core.config import settings
from app.core.database import get_event_collection, get_camera_collection
from app.models.event import EventDB
from app.models.camera import CameraStatus
from app.utils.cv_helpers import save_evidence_screenshot, save_evidence_clip

logger = logging.getLogger(__name__)

# Email trigger event types
EMAIL_TRIGGER_EVENTS = {
    "WEAPON", "PISTOL", "GUN", "KNIFE", "RIFLE", "FIREARM",
    "FIGHT", "FIGHT BEHAVIOR DETECTED",
    "ACCIDENT", "COLLISION",
    "INTRUSION", "INTRUDER", "ZONE INTRUSION",
}

# In-memory cache for email cooldown (Event Flood Protection)
_email_cooldown_cache = {}
EMAIL_COOLDOWN_MINUTES = 5

def _is_email_cooldown_active(camera_id: str, event_type: str) -> bool:
    """Verifies and updates the cooldown to prevent event flooding."""
    key = f"{camera_id}_{event_type.upper()}"
    last_sent = _email_cooldown_cache.get(key)
    now = datetime.utcnow()
    
    if last_sent:
        if now - last_sent < timedelta(minutes=EMAIL_COOLDOWN_MINUTES):
            return True
            
    _email_cooldown_cache[key] = now
    return False

def _should_trigger_email(camera_id: str, event_type: str, threat_level: str, accident_detected: bool = False) -> bool:
    """Determine if an event should trigger email notification with flood protection."""
    if not getattr(settings, "EMAIL_ENABLED", True):
        return False

    event_upper = event_type.upper()
    is_unattended = "UNATTENDED" in event_upper

    # An unattended weapon event at MEDIUM must NEVER email
    if is_unattended and threat_level == "MEDIUM":
        return False

    will_trigger = False
    
    # Trigger for HIGH and CRITICAL threat levels or accident detections
    if accident_detected or threat_level in ("HIGH", "CRITICAL"):
        will_trigger = True
    elif not is_unattended:
        # Trigger for specific dangerous attended event types regardless of level
        for trigger in EMAIL_TRIGGER_EVENTS:
            if trigger in event_upper:
                will_trigger = True
                break
                
    if will_trigger:
        if _is_email_cooldown_active(camera_id, event_type):
            logger.info(f"Skipping email alert for {camera_id} - {event_type} (Cooldown active)")
            return False
        return True
        
    return False


async def _fire_email_notification(
    event_type: str,
    threat_level: str,
    threat_score: int,
    camera_id: str,
    xai_explanation: dict = None,
    accident_detected: bool = False,
    accident_confidence: float = 0.0,
    event_doc: dict = None,
):
    """Fire-and-forget email notification for dangerous events."""
    try:
        from app.services.email_service import EmailService
        from app.services.xai_service import XAIService
        from app.utils.pdf_generator import generate_incident_pdf

        timestamp_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        events_list = [event_type]

        # Build XAI summary for email
        xai_summary = ""
        if xai_explanation:
            xai_summary = XAIService.generate_email_summary(xai_explanation)
        else:
            xai_summary = f"{event_type} detected on {camera_id}. Threat score: {threat_score}."

        is_accident = accident_detected or "ACCIDENT" in event_type.upper() or "COLLISION" in event_type.upper()

        if is_accident:
            pdf_bytes = None
            if event_doc:
                try:
                    pdf_bytes = generate_incident_pdf(event_doc)
                except Exception as pdf_err:
                    logger.warning(f"Failed to generate accident incident PDF: {pdf_err}")

            await EmailService.send_accident_alert_email(
                threat_level=threat_level,
                threat_score=threat_score,
                camera_id=camera_id,
                timestamp=timestamp_str,
                accident_confidence=accident_confidence if accident_confidence > 0 else 0.75,
                xai_summary=xai_summary,
                pdf_bytes=pdf_bytes,
                pdf_filename=f"AEGIS_ACCIDENT_{camera_id}_{int(datetime.utcnow().timestamp())}.pdf"
            )
        else:
            await EmailService.send_alert_email(
                threat_level=threat_level,
                threat_score=threat_score,
                camera_id=camera_id,
                timestamp=timestamp_str,
                events=events_list,
                xai_summary=xai_summary,
            )
    except Exception as e:
        logger.error(f"Email notification dispatch error: {e}")

async def log_threat_event(
    camera_id: str,
    event_type: str,
    threat_level: str,
    threat_score: int,
    confidence: float,
    frame,
    frame_buffer: list,
    person_id: int = None,
    zone_id: str = None,
    duration: float = None,
    correlated_events: list = None,
    xai_explanation: dict = None,
    fight_detected: bool = False,
    fight_confidence: float = 0.0,
    fight_participants: list = None,
    fight_duration_sec: float = 0.0,
    accident_detected: bool = False,
    accident_confidence: float = 0.0
) -> dict:
    """
    MODULE 7 & 8: Log threat event, trigger evidence video/screenshot capture, update MongoDB and camera status.
    """
    event_col = get_event_collection()
    camera_col = get_camera_collection()
    
    # --- FIGHT EVENT STATE MACHINE (Prevent MongoDB duplication) ---
    is_fight_event = "FIGHT" in event_type.upper() or fight_detected

    if is_fight_event:
        if "ENDED" in event_type.upper():
            # Update existing active fight event to RESOLVED
            active_fight = await event_col.find_one({
                "camera_id": camera_id,
                "fight_detected": True,
                "status": "ACTIVE"
            })
            if active_fight:
                await event_col.update_one(
                    {"_id": active_fight["_id"]},
                    {"$set": {
                        "status": "RESOLVED",
                        "fight_duration_sec": fight_duration_sec,
                        "ended_at": datetime.utcnow()
                    }}
                )
                active_fight["status"] = "RESOLVED"
                active_fight["fight_duration_sec"] = fight_duration_sec
                active_fight["_id"] = str(active_fight["_id"])
                return active_fight
        elif "ONGOING" in event_type.upper() or fight_detected:
            # Check for existing ACTIVE fight event
            active_fight = await event_col.find_one({
                "camera_id": camera_id,
                "fight_detected": True,
                "status": "ACTIVE"
            })
            if active_fight:
                new_score = max(active_fight.get("threat_score", 0), threat_score)
                await event_col.update_one(
                    {"_id": active_fight["_id"]},
                    {"$set": {
                        "fight_duration_sec": fight_duration_sec,
                        "fight_participants": fight_participants or [],
                        "fight_confidence": max(active_fight.get("fight_confidence", 0.0), fight_confidence),
                        "threat_score": new_score,
                        "last_updated": datetime.utcnow()
                    }}
                )
                active_fight["fight_duration_sec"] = fight_duration_sec
                active_fight["fight_participants"] = fight_participants or []
                active_fight["threat_score"] = new_score
                active_fight["_id"] = str(active_fight["_id"])
                return active_fight

    # Cooldown check: prevent duplicate non-fight event logs for the same camera/type in the last 10 seconds
    cooldown_time = datetime.utcnow() - timedelta(seconds=10)
    existing = await event_col.find_one({
        "camera_id": camera_id,
        "event_type": event_type,
        "timestamp": {"$gte": cooldown_time}
    })
    
    if existing:
        return existing

    # Fetch camera location
    camera = await camera_col.find_one({"camera_id": camera_id})
    lat = camera.get("latitude", 0.0) if camera else 0.0
    lon = camera.get("longitude", 0.0) if camera else 0.0

    # Capture Evidence files (Screenshot & 10s pre/post Video Clip for HIGH or CRITICAL)
    image_path = None
    video_path = None
    if threat_level in ["HIGH", "CRITICAL"]:
        try:
            image_path = save_evidence_screenshot(frame, camera_id, event_type.replace(" ", "_"))
            video_path = save_evidence_clip(frame_buffer, camera_id, event_type.replace(" ", "_"))
        except Exception as e:
            logger.error(f"Evidence capture error: {e}")

    # Build DB Record
    event_id = f"EVT-{uuid.uuid4().hex[:8].upper()}"
    db_obj = EventDB(
        event_id=event_id,
        camera_id=camera_id,
        event_type=event_type,
        threat_level=threat_level,
        threat_score=threat_score,
        confidence=confidence,
        latitude=lat,
        longitude=lon,
        person_id=person_id,
        zone_id=zone_id,
        duration=duration,
        correlated_events=correlated_events or [],
        xai_explanation=xai_explanation,
        status="ACTIVE",
        image_path=image_path,
        video_path=video_path,
        timestamp=datetime.utcnow(),
        fight_detected=fight_detected,
        fight_confidence=fight_confidence,
        fight_participants=fight_participants or [],
        fight_duration_sec=fight_duration_sec,
        accident_detected=accident_detected,
        accident_confidence=accident_confidence
    )

    doc = db_obj.model_dump()
    await event_col.insert_one(doc)

    # Update camera status based on threat
    if camera:
        new_status = camera.get("status", CameraStatus.ONLINE.value)
        if threat_level == "CRITICAL":
            new_status = CameraStatus.CRITICAL_ALERT.value
        elif "NO VIDEO INPUT" in event_type.upper() or "NO_VIDEO_INPUT" in event_type.upper():
            new_status = CameraStatus.NO_VIDEO_INPUT.value
        elif "TAMPERING" in event_type.upper() or "OBSTRUCTED" in event_type.upper():
            new_status = CameraStatus.TAMPERED.value
        elif "FREEZE" in event_type.upper():
            new_status = CameraStatus.FROZEN.value
        elif threat_level == "HIGH":
            new_status = CameraStatus.WARNING.value
        else:
            new_status = CameraStatus.ONLINE.value

        await camera_col.update_one({"camera_id": camera_id}, {"$set": {"status": new_status, "last_ping": datetime.utcnow()}})

    logger.info(f"Logged Surveillance Event [{event_id}]: {event_type} - {threat_level} (Score: {threat_score})")

    # Trigger email notification for qualifying events (fire-and-forget)
    if _should_trigger_email(camera_id, event_type, threat_level, accident_detected=accident_detected):
        asyncio.ensure_future(
            _fire_email_notification(
                event_type=event_type,
                threat_level=threat_level,
                threat_score=threat_score,
                camera_id=camera_id,
                xai_explanation=xai_explanation,
                accident_detected=accident_detected,
                accident_confidence=accident_confidence,
                event_doc=doc,
            )
        )

    return doc

async def acknowledge_alarm(event_id: str, username: str = "Operator") -> dict:
    """MODULE 7: Acknowledge an active alarm event."""
    event_col = get_event_collection()
    res = await event_col.find_one_and_update(
        {"event_id": event_id},
        {"$set": {
            "status": "ACKNOWLEDGED",
            "acknowledged_by": username,
            "acknowledged_at": datetime.utcnow()
        }},
        return_document=True
    )
    if not res:
        raise HTTPException(status_code=404, detail="Alarm event not found")
    res["_id"] = str(res["_id"])
    return res

async def resolve_alarm(event_id: str) -> dict:
    """MODULE 7: Resolve an active alarm event."""
    event_col = get_event_collection()
    res = await event_col.find_one_and_update(
        {"event_id": event_id},
        {"$set": {"status": "RESOLVED"}},
        return_document=True
    )
    if not res:
        raise HTTPException(status_code=404, detail="Alarm event not found")
    res["_id"] = str(res["_id"])
    return res

async def escalate_alarm(event_id: str) -> dict:
    """MODULE 7: Escalate an alarm event to CRITICAL priority."""
    event_col = get_event_collection()
    res = await event_col.find_one_and_update(
        {"event_id": event_id},
        {"$set": {
            "status": "ESCALATED",
            "threat_level": "CRITICAL",
            "threat_score": 120
        }},
        return_document=True
    )
    if not res:
        raise HTTPException(status_code=404, detail="Alarm event not found")
    res["_id"] = str(res["_id"])
    return res

async def get_events(
    page: int = 1,
    limit: int = 20,
    start_date: str = None,
    end_date: str = None,
    camera_id: str = None,
    threat_level: str = None,
    event_type: str = None,
    status_filter: str = None,
    search_query: str = None
) -> tuple[list[dict], int]:
    """
    MODULE 12 & 13: Fetch, filter, and search events.
    """
    event_col = get_event_collection()
    query = {}

    if start_date or end_date:
        query["timestamp"] = {}
        if start_date:
            try:
                query["timestamp"]["$gte"] = datetime.fromisoformat(start_date)
            except Exception:
                pass
        if end_date:
            try:
                query["timestamp"]["$lte"] = datetime.fromisoformat(end_date)
            except Exception:
                pass

    if camera_id:
        query["camera_id"] = camera_id
    if threat_level:
        query["threat_level"] = threat_level
    if event_type:
        query["event_type"] = {"$regex": event_type, "$options": "i"}
    if status_filter:
        query["status"] = status_filter
    if search_query:
        query["$or"] = [
            {"event_type": {"$regex": search_query, "$options": "i"}},
            {"camera_id": {"$regex": search_query, "$options": "i"}},
            {"event_id": {"$regex": search_query, "$options": "i"}}
        ]

    total = await event_col.count_documents(query)
    skip = (page - 1) * limit

    cursor = event_col.find(query).sort("timestamp", -1).skip(skip).limit(limit)
    events = [doc async for doc in cursor]
    for e in events:
        e["_id"] = str(e["_id"])
    return events, total

async def export_events_csv(start_date: str = None, end_date: str = None, camera_id: str = None) -> str:
    """Export events to CSV format."""
    event_col = get_event_collection()
    query = {}
    if camera_id:
        query["camera_id"] = camera_id
        
    cursor = event_col.find(query).sort("timestamp", -1).limit(1000)
    events = [doc async for doc in cursor]
    
    if not events:
        return "event_id,camera_id,event_type,threat_level,threat_score,status,timestamp\n"
        
    df = pd.DataFrame(events)
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].apply(lambda t: t.isoformat() if isinstance(t, datetime) else t)
        
    return df.to_csv(index=False)

async def get_dashboard_summary() -> dict:
    """MODULE 9 & 11: Real-time statistics for command center KPI cards."""
    camera_col = get_camera_collection()
    event_col = get_event_collection()
    
    total_cameras = await camera_col.count_documents({})
    active_cameras = await camera_col.count_documents({"status": {"$ne": "Offline"}})
    
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_events = await event_col.count_documents({"timestamp": {"$gte": today_start}})
    
    critical_alerts = await event_col.count_documents({
        "threat_level": "CRITICAL",
        "status": "ACTIVE"
    })
    
    active_alarms_count = await event_col.count_documents({"status": "ACTIVE"})

    total_accidents_today = await event_col.count_documents({
        "timestamp": {"$gte": today_start},
        "$or": [
            {"accident_detected": True},
            {"event_type": {"$regex": "ACCIDENT|COLLISION", "$options": "i"}}
        ]
    })

    cursor = event_col.find({}).sort("timestamp", -1).limit(10)
    recent_events = [doc async for doc in cursor]
    for e in recent_events:
        e["_id"] = str(e["_id"])

    # Threat Distribution
    pipeline = [{"$group": {"_id": "$threat_level", "count": {"$sum": 1}}}]
    cursor = event_col.aggregate(pipeline)
    threat_distribution = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    async for doc in cursor:
        if doc["_id"] in threat_distribution:
            threat_distribution[doc["_id"]] = doc["count"]

    return {
        "total_cameras": total_cameras,
        "active_cameras": active_cameras,
        "today_events": today_events,
        "critical_alerts": critical_alerts,
        "active_alarms_count": active_alarms_count,
        "total_accidents_today": total_accidents_today,
        "recent_events": recent_events,
        "threat_distribution": threat_distribution
    }

async def get_analytics_charts() -> dict:
    """MODULE 11 & 14: Heatmap analytics and chart aggregations."""
    event_col = get_event_collection()
    camera_col = get_camera_collection()
    
    events_per_day = []
    accident_trend = []
    today = datetime.utcnow().date()
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        day_start = datetime.combine(day, datetime.min.time())
        day_end = datetime.combine(day, datetime.max.time())
        count = await event_col.count_documents({"timestamp": {"$gte": day_start, "$lte": day_end}})
        events_per_day.append({"date": day.strftime("%b %d"), "Events": count})

        acc_count = await event_col.count_documents({
            "timestamp": {"$gte": day_start, "$lte": day_end},
            "$or": [
                {"accident_detected": True},
                {"event_type": {"$regex": "ACCIDENT|COLLISION", "$options": "i"}}
            ]
        })
        accident_trend.append({"date": day.strftime("%b %d"), "accidents": acc_count})

    pipeline = [{"$group": {"_id": "$threat_level", "count": {"$sum": 1}}}]
    cursor = event_col.aggregate(pipeline)
    threat_dist = []
    async for doc in cursor:
        threat_dist.append({"name": doc["_id"], "value": doc["count"]})
    if not threat_dist:
        threat_dist = [
            {"name": "LOW", "value": 0},
            {"name": "MEDIUM", "value": 0},
            {"name": "HIGH", "value": 0},
            {"name": "CRITICAL", "value": 0}
        ]

    pipeline = [
        {"$group": {"_id": "$camera_id", "incidents": {"$sum": 1}}},
        {"$sort": {"incidents": -1}},
        {"$limit": 5}
    ]
    cursor = event_col.aggregate(pipeline)
    camera_ranking = []
    async for doc in cursor:
        cam = await camera_col.find_one({"camera_id": doc["_id"]})
        name = cam["camera_name"] if cam else doc["_id"]
        camera_ranking.append({"name": name, "incidents": doc["incidents"]})

    # Location Heatmap Data
    cursor = event_col.find({"threat_level": {"$in": ["MEDIUM", "HIGH", "CRITICAL"]}}).limit(200)
    heatmap_data = []
    async for doc in cursor:
        intensity = 0.5 if doc["threat_level"] == "MEDIUM" else (0.8 if doc["threat_level"] == "HIGH" else 1.0)
        heatmap_data.append({
            "lat": doc["latitude"],
            "lng": doc["longitude"],
            "intensity": intensity,
            "event_type": doc["event_type"],
            "camera_id": doc["camera_id"],
            "threat_level": doc["threat_level"]
        })

    return {
        "events_per_day": events_per_day,
        "accident_trend": accident_trend,
        "threat_distribution": threat_dist,
        "camera_ranking": camera_ranking,
        "heatmap_data": heatmap_data
    }

async def get_daily_digest_stats() -> dict:
    """MODULE 9 & 11: Daily Incident Digest stats aggregation spanning the last 24 hours."""
    event_col = get_event_collection()
    camera_col = get_camera_collection()
    
    now = datetime.utcnow()
    last_24h = now - timedelta(hours=24)
    
    # Query for last 24h events
    cursor = event_col.find({"timestamp": {"$gte": last_24h}})
    events = [doc async for doc in cursor]
    
    total_alerts = len(events)
    critical_events = sum(1 for e in events if e.get("threat_level") == "CRITICAL")
    high_events = sum(1 for e in events if e.get("threat_level") == "HIGH")
    
    weapon_detections = 0
    fight_detections = 0
    accident_detections = 0
    
    cam_incidents = {}
    
    for e in events:
        event_type = e.get("event_type", "").upper()
        if any(w in event_type for w in ["WEAPON", "PISTOL", "GUN", "KNIFE", "RIFLE"]):
            weapon_detections += 1
        if "FIGHT" in event_type or e.get("fight_detected"):
            fight_detections += 1
        if any(a in event_type for a in ["ACCIDENT", "COLLISION"]) or e.get("accident_detected"):
            accident_detections += 1
            
        cam_id = e.get("camera_id")
        cam_incidents[cam_id] = cam_incidents.get(cam_id, 0) + 1

    # Get top 3 cameras
    sorted_cams = sorted(cam_incidents.items(), key=lambda item: item[1], reverse=True)[:3]
    top_cameras = []
    for cid, count in sorted_cams:
        cam_info = await camera_col.find_one({"camera_id": cid})
        cam_name = cam_info["camera_name"] if cam_info else cid
        top_cameras.append({"camera_id": cid, "camera_name": cam_name, "count": count})

    return {
        "date": now.strftime("%B %d, %Y"),
        "total_alerts": total_alerts,
        "critical_events": critical_events,
        "high_events": high_events,
        "weapon_detections": weapon_detections,
        "fight_detections": fight_detections,
        "accident_detections": accident_detections,
        "top_cameras": top_cameras
    }

async def delete_event(event_id: str) -> dict:
    """Delete a single event and its associated evidence files."""
    event_col = get_event_collection()
    query = {"$or": [{"event_id": event_id}]}
    if ObjectId.is_valid(event_id):
        query["$or"].append({"_id": ObjectId(event_id)})

    event_doc = await event_col.find_one(query)
    if not event_doc:
        raise HTTPException(status_code=404, detail="Event not found")

    img_path = event_doc.get("image_path")
    if img_path:
        try:
            from pathlib import Path
            clean_path = img_path.lstrip("/\\")
            p = Path(__file__).resolve().parent.parent.parent / clean_path
            if p.exists() and p.is_file():
                p.unlink()
        except Exception as e:
            logger.warning(f"Could not delete image file {img_path}: {e}")

    res = await event_col.delete_one({"_id": event_doc["_id"]})
    return {"success": res.deleted_count > 0, "event_id": event_id}

