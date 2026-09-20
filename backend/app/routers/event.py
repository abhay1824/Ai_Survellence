from datetime import datetime
import asyncio
from fastapi import APIRouter, Depends, Query, status, HTTPException
from fastapi.responses import StreamingResponse, Response
from app.routers.auth import RoleChecker
from app.schemas.event import EventResponse, AlarmAction
from app.services import event_service
from app.services.xai_service import XAIService
from app.services.email_service import EmailService
from app.core.database import get_event_collection
from app.utils.pdf_generator import generate_pdf_report, generate_incident_pdf

router = APIRouter(prefix="/events", tags=["events"])

allow_all = RoleChecker(["Super Admin", "Police Admin", "Control Room Operator", "Viewer"])
allow_operator = RoleChecker(["Super Admin", "Police Admin", "Control Room Operator"])

@router.get("", response_model=dict)
async def get_events(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
    start_date: str = Query(None),
    end_date: str = Query(None),
    camera_id: str = Query(None),
    threat_level: str = Query(None),
    event_type: str = Query(None),
    status_filter: str = Query(None, alias="status"),
    search: str = Query(None),
    current_user: dict = Depends(allow_all)
):
    events, total = await event_service.get_events(
        page=page,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        camera_id=camera_id,
        threat_level=threat_level,
        event_type=event_type,
        status_filter=status_filter,
        search_query=search
    )
    return {
        "events": events,
        "total": total,
        "page": page,
        "limit": limit
    }

@router.get("/{event_id}/explain")
async def get_event_explanation(event_id: str, current_user: dict = Depends(allow_all)):
    """MODULE 15: Retrieve or generate XAI explanation for an event."""
    event_col = get_event_collection()
    event = await event_col.find_one({"event_id": event_id})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    if event.get("xai_explanation"):
        return event["xai_explanation"]

    # Generate on the fly if missing
    timestamp_str = event.get("timestamp").isoformat() if isinstance(event.get("timestamp"), datetime) else str(event.get("timestamp"))
    explanation = XAIService.generate_explanation(
        camera_id=event.get("camera_id", "UNKNOWN"),
        timestamp=timestamp_str,
        threat_level=event.get("threat_level", "LOW"),
        threat_score=event.get("threat_score", 0),
        confidence=event.get("confidence", 0.95),
        detections=[{"class": event.get("event_type", "Event"), "confidence": event.get("confidence", 0.95), "track_id": event.get("person_id")}],
        active_events=[{"event_type": event.get("event_type"), "threat_level": event.get("threat_level"), "person_id": event.get("person_id")}],
        correlated_events=event.get("correlated_events", []),
        tracking_history={}
    )

    # Cache back into database
    await event_col.update_one({"event_id": event_id}, {"$set": {"xai_explanation": explanation}})
    return explanation

@router.post("/explain")
async def generate_custom_explanation(payload: dict, current_user: dict = Depends(allow_operator)):
    """MODULE 15: Generate custom XAI explanation on the fly."""
    return XAIService.generate_explanation(
        camera_id=payload.get("camera_id", "CAM-01"),
        timestamp=payload.get("timestamp", datetime.utcnow().isoformat()),
        threat_level=payload.get("threat_level", "LOW"),
        threat_score=payload.get("threat_score", 0),
        confidence=payload.get("confidence", 0.95),
        detections=payload.get("detections", []),
        active_events=payload.get("active_events", []),
        correlated_events=payload.get("correlated_events", []),
        tracking_history=payload.get("tracking_history", {}),
        person_count=payload.get("person_count", 0),
        danger_radius_m=payload.get("danger_radius_m", 5.0)
    )

@router.post("/{event_id}/acknowledge")
async def acknowledge_event(event_id: str, action: AlarmAction = None, current_user: dict = Depends(allow_operator)):
    """MODULE 7: Acknowledge Alarm Event."""
    username = current_user.get("username", "Operator")
    return await event_service.acknowledge_alarm(event_id, username=username)

@router.post("/{event_id}/resolve")
async def resolve_event(event_id: str, current_user: dict = Depends(allow_operator)):
    """MODULE 7: Resolve Alarm Event."""
    return await event_service.resolve_alarm(event_id)

@router.post("/{event_id}/escalate")
async def escalate_event(event_id: str, current_user: dict = Depends(allow_operator)):
    """MODULE 7: Escalate Alarm Event."""
    return await event_service.escalate_alarm(event_id)

@router.get("/export")
async def export_events(
    start_date: str = Query(None),
    end_date: str = Query(None),
    camera_id: str = Query(None),
    current_user: dict = Depends(allow_all)
):
    csv_content = await event_service.export_events_csv(
        start_date=start_date,
        end_date=end_date,
        camera_id=camera_id
    )
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=surveillance_export_{int(datetime.utcnow().timestamp())}.csv"
        }
    )

@router.get("/pdf-report")
async def get_pdf_report(
    start_date: str = Query(None),
    end_date: str = Query(None),
    current_user: dict = Depends(allow_all)
):
    """MODULE 14: PDF Security Intelligence Report Generation."""
    events, total = await event_service.get_events(
        page=1,
        limit=1000,
        start_date=start_date,
        end_date=end_date
    )
    
    stats = {
        "total_events": total,
        "critical_count": sum(1 for e in events if e.get("threat_level") == "CRITICAL"),
        "high_count": sum(1 for e in events if e.get("threat_level") == "HIGH"),
        "medium_count": sum(1 for e in events if e.get("threat_level") == "MEDIUM"),
        "low_count": sum(1 for e in events if e.get("threat_level") == "LOW"),
    }
    
    pdf_buffer = generate_pdf_report(
        start_date=start_date,
        end_date=end_date,
        events=events,
        stats=stats
    )
    
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=security_intelligence_report_{int(datetime.utcnow().timestamp())}.pdf"
        }
    )

@router.get("/{event_id}/pdf")
async def get_single_event_pdf(event_id: str, current_user: dict = Depends(allow_all)):
    """Generate official single-event PDF Incident Report."""
    collection = get_event_collection()
    event_doc = await collection.find_one({"event_id": event_id})
    if not event_doc:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
    
    # Format MongoDB document
    event_doc["_id"] = str(event_doc["_id"])

    # If XAI explanation is missing, generate on the fly
    if not event_doc.get("xai_explanation"):
        exp = XAIService.generate_explanation(
            camera_id=event_doc.get("camera_id", "CAM-01"),
            timestamp=str(event_doc.get("timestamp", datetime.utcnow().isoformat())),
            threat_level=event_doc.get("threat_level", "LOW"),
            threat_score=event_doc.get("threat_score", 0),
            confidence=event_doc.get("confidence", 0.90),
            detections=[{"class": event_doc.get("event_type", "Object"), "confidence": event_doc.get("confidence", 0.90), "threat_level": event_doc.get("threat_level", "LOW")}],
            active_events=[event_doc.get("event_type", "Event")]
        )
        event_doc["xai_explanation"] = exp

    pdf_bytes = generate_incident_pdf(event_doc)

    # Fire-and-forget: email the PDF to all configured recipients
    if event_doc.get("threat_level") in ("HIGH", "CRITICAL"):
        asyncio.ensure_future(
            EmailService.send_incident_pdf(event_data=event_doc, pdf_bytes=pdf_bytes)
        )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=incident_report_{event_id}.pdf"
        }
    )

@router.get("/analytics/heatmap")
async def get_threat_heatmap(current_user: dict = Depends(allow_all)):
    """MODULE 12: Spatial Threat Heatmap Endpoint."""
    collection = get_event_collection()
    cursor = collection.find().limit(500)
    heatmap_points = []
    
    # Campus reference GPS center (e.g. 12.9716, 77.5946)
    async for event_doc in cursor:
        lat = event_doc.get("latitude", 12.9716)
        lng = event_doc.get("longitude", 77.5946)
        score = event_doc.get("threat_score", 50)
        level = event_doc.get("threat_level", "MEDIUM")
        
        # Calculate intensity scaling 0.1 to 1.0
        intensity = round(min(max(score / 100.0, 0.2), 1.0), 2)
        heatmap_points.append({
            "lat": lat,
            "lng": lng,
            "intensity": intensity,
            "threat_score": score,
            "threat_level": level,
            "camera_id": event_doc.get("camera_id", "CAM-01"),
            "event_type": event_doc.get("event_type", "Threat Detected")
        })

    # If empty, return realistic sample campus hotspot clusters
    if not heatmap_points:
        heatmap_points = [
            {"lat": 12.9716, "lng": 77.5946, "intensity": 0.95, "threat_score": 92, "threat_level": "CRITICAL", "camera_id": "CAM-01", "event_type": "Pistol Detected"},
            {"lat": 12.9720, "lng": 77.5950, "intensity": 0.85, "threat_score": 85, "threat_level": "HIGH", "camera_id": "CAM-03", "event_type": "Restricted Area Intrusion"},
            {"lat": 12.9710, "lng": 77.5940, "intensity": 0.45, "threat_score": 45, "threat_level": "MEDIUM", "camera_id": "CAM-02", "event_type": "Loitering Detected"},
            {"lat": 12.9725, "lng": 77.5955, "intensity": 0.20, "threat_score": 20, "threat_level": "LOW", "camera_id": "CAM-04", "event_type": "Unattended Object"}
        ]
        
    return {"points": heatmap_points, "total_hotspots": len(heatmap_points)}
