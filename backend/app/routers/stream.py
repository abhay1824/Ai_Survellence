"""
Eagle Eye WebSocket Streaming Router — Thin consumer layer.
Reads frames from CameraManager, runs AI inference, broadcasts to clients.
No cv2.VideoCapture. No browser frame uploads. Pure server-push architecture.
"""

import asyncio
import cv2
import numpy as np
import time
import logging
import jwt
import base64
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, status
from app.core.config import settings
from app.core.database import get_user_collection, get_database
from app.services import camera_service, event_service
from app.services.ai_service import AIService
from app.services.camera_manager import camera_manager, CameraStatus
from app.utils.cv_helpers import encode_frame_to_base64

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws", tags=["stream"])

ai_service = AIService()


# ============================================================
# Offline / No Video Input Frame Generator
# ============================================================

def generate_offline_frame(camera_name: str, frame_num: int) -> np.ndarray:
    """
    Generates a clean, modern 'NO VIDEO INPUT / CAMERA OFFLINE' frame.
    Displayed when no live stream URL or OpenCV local camera video is active.
    """
    w, h = 640, 480
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (18, 18, 22)  # Dark cyber background

    # Subtle Grid
    for x in range(0, w, 40):
        cv2.line(frame, (x, 0), (x, h), (28, 28, 34), 1)
    for y in range(0, h, 40):
        cv2.line(frame, (0, y), (w, y), (28, 28, 34), 1)

    # Offline Warning Card
    cv2.rectangle(frame, (100, 170), (w - 100, 310), (28, 32, 42), -1)
    cv2.rectangle(frame, (100, 170), (w - 100, 310), (60, 70, 90), 1)

    # Pulsing status text
    cv2.putText(frame, "NO LIVE VIDEO INPUT", (185, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 180, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, f"NODE: {camera_name.upper()}", (140, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 205), 1, cv2.LINE_AA)
    cv2.putText(frame, "Configure RTSP Stream URL or OpenCV Local Camera (0)", (120, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 130, 150), 1, cv2.LINE_AA)

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp_str, (w - 190, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 140), 1, cv2.LINE_AA)

    return frame


# ============================================================
# WebSocket Authentication
# ============================================================

async def authenticate_ws(token: str) -> dict:
    """Verify WebSocket token."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        username = payload.get("sub")
        if not username:
            return None
        user = await get_user_collection().find_one({"username": username})
        return user
    except Exception:
        return None


# ============================================================
# WebSocket Stream Endpoint
# ============================================================

@router.websocket("/camera/{camera_id}")
async def ws_stream(websocket: WebSocket, camera_id: str, token: str = Query(...)):
    """
    WebSocket endpoint for real-time AI-annotated video stream.
    Architecture: CameraManager → get_frame → AI → encode → send
    """
    await websocket.accept()

    # Authenticate
    user = await authenticate_ws(token)
    if not user:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Resolve camera from DB or generate fallback
    camera = await camera_service.get_camera_by_code(camera_id)
    if not camera:
        if camera_id.startswith("CAM-"):
            camera = {
                "camera_id": camera_id,
                "camera_name": f"Surveillance Node {camera_id.split('-')[-1]}",
                "stream_url": "synthetic",
                "latitude": 12.9716,
                "longitude": 77.5946,
                "status": "Online"
            }
        else:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            return

    # Ensure camera worker is running
    if not camera_manager.is_camera_running(camera_id):
        camera_manager.start_camera(
            camera_id=camera_id,
            camera_name=camera.get("camera_name", camera_id),
            source_url=camera.get("stream_url", "synthetic"),
        )

    # Fetch zones
    db = get_database()
    cursor = db["zones"].find({"camera_id": camera_id})
    zones = [doc async for doc in cursor]

    frame_num = 0
    camera_name = camera.get("camera_name", camera_id)
    gps_coords = (camera.get("latitude", 0.0), camera.get("longitude", 0.0))
    is_synthetic = camera.get("stream_url", "").lower() == "synthetic"

    logger.info(f"[WebSocket] Client connected to {camera_id}")

    client_frame = None

    async def receive_client_frames():
        nonlocal client_frame
        try:
            while True:
                msg_text = await websocket.receive_text()
                if msg_text:
                    import json
                    data = json.loads(msg_text)
                    if "frame" in data and data["frame"]:
                        b64_str = data["frame"].split(",")[-1]
                        img_bytes = base64.b64decode(b64_str)
                        np_arr = np.frombuffer(img_bytes, np.uint8)
                        decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if decoded is not None:
                            client_frame = decoded
        except Exception:
            pass

    reader_task = asyncio.create_task(receive_client_frames())

    last_fight_telemetry = {
        "fight_detected": False,
        "fight_confidence": 0.0,
        "fight_participants": [],
        "fight_duration_sec": 0.0
    }

    try:
        while True:
            try:
                # 1. Get latest frame (prefer client browser webcam frame if present, else CameraManager worker frame)
                frame = client_frame if client_frame is not None else camera_manager.get_frame(camera_id)

                # 2. Fallback to offline frame if no frame available
                if frame is None:
                    frame_num += 1
                    frame = generate_offline_frame(camera_name, frame_num)
                else:
                    frame_num += 1

                # 3. AI Processing (Offloaded to worker thread to prevent event-loop freezing)
                (
                    annotated_frame,
                    detections,
                    highest_threat,
                    total_score,
                    alerts,
                    correlated,
                    person_count,
                    xai_explanation,
                    fight_result,
                    accident_result
                ) = await asyncio.to_thread(
                    ai_service.process_frame,
                    camera_id=camera_id,
                    frame=frame.copy(),
                    camera_name=camera_name,
                    gps_coords=gps_coords,
                    zones=zones
                )

                # 4. Log threat events
                for alert in alerts:
                    asyncio.create_task(
                        event_service.log_threat_event(
                            camera_id=camera_id,
                            event_type=alert["event_type"],
                            threat_level=alert["threat_level"],
                            threat_score=total_score,
                            confidence=alert["confidence"],
                            frame=annotated_frame,
                            frame_buffer=camera_manager.get_frame_buffer(camera_id),
                            person_id=alert.get("person_id"),
                            zone_id=alert.get("zone_id"),
                            duration=alert.get("duration"),
                            correlated_events=correlated,
                            xai_explanation=xai_explanation,
                            fight_detected=fight_result.get("fight_detected", False),
                            fight_confidence=fight_result.get("fight_confidence", 0.0),
                            fight_participants=fight_result.get("fight_participants", []),
                            fight_duration_sec=fight_result.get("fight_duration_sec", 0.0),
                            accident_detected=accident_result.get("accident_detected", False),
                            accident_confidence=accident_result.get("accident_confidence", 0.0)
                        )
                    )

                # 5. Encode and send (Offloaded to worker thread)
                if annotated_frame is None:
                    annotated_frame = frame

                encoded_frame = await asyncio.to_thread(encode_frame_to_base64, annotated_frame)

                # Include health stats in payload
                health = camera_manager.get_health(camera_id) or {}

                # Throttled Fight Telemetry evaluation
                curr_fight_detected = fight_result.get("fight_detected", False)
                curr_conf = fight_result.get("fight_confidence", 0.0)
                curr_participants = fight_result.get("fight_participants", [])
                curr_dur = fight_result.get("fight_duration_sec", 0.0)

                fight_updated = (
                    curr_fight_detected != last_fight_telemetry["fight_detected"] or
                    abs(curr_conf - last_fight_telemetry["fight_confidence"]) >= 0.05 or
                    curr_participants != last_fight_telemetry["fight_participants"] or
                    int(curr_dur) != int(last_fight_telemetry["fight_duration_sec"]) or
                    fight_result.get("fight_started", False) or
                    fight_result.get("fight_ended", False)
                )

                if fight_updated:
                    last_fight_telemetry["fight_detected"] = curr_fight_detected
                    last_fight_telemetry["fight_confidence"] = curr_conf
                    last_fight_telemetry["fight_participants"] = curr_participants
                    last_fight_telemetry["fight_duration_sec"] = curr_dur

                payload = {
                    "frame": encoded_frame,
                    "detections": detections,
                    "highest_threat": highest_threat,
                    "threat_score": total_score,
                    "alerts": alerts,
                    "correlated": correlated,
                    "person_count": person_count,
                    "xai_explanation": xai_explanation,
                    "fight_detected": curr_fight_detected,
                    "fight_confidence": curr_conf,
                    "fight_participants": curr_participants,
                    "fight_duration_sec": curr_dur,
                    "fight_telemetry_updated": fight_updated,
                    "accident_detected": accident_result.get("accident_detected", False),
                    "accident_confidence": accident_result.get("accident_confidence", 0.0),
                    "health": {
                        "fps": health.get("fps", 0),
                        "status": health.get("status", "offline"),
                        "latency_ms": health.get("latency_ms", 0),
                        "frames_captured": health.get("frames_captured", 0),
                    }
                }
                await websocket.send_json(payload)

            except (WebSocketDisconnect, RuntimeError):
                break  # Exit stream loop cleanly on client disconnect
            except Exception as frame_err:
                # Per-frame error isolation — log but don't crash the stream
                logger.warning(f"[WebSocket] Frame error on {camera_id}: {frame_err}")

            # 6. Frame pacing — minimal yield to keep event loop responsive
            await asyncio.sleep(0.02)

    except WebSocketDisconnect:
        logger.info(f"[WebSocket] Client disconnected from {camera_id}")
    except Exception as e:
        logger.error(f"[WebSocket] Stream error on {camera_id}: {e}")
    finally:
        reader_task.cancel()
