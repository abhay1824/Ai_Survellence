import os
import cv2
import numpy as np
import time
import base64
from datetime import datetime
from app.core.config import settings

def point_in_polygon(point, polygon):
    """
    Ray-casting algorithm to test if (x, y) point is inside polygon [[x1, y1], [x2, y2], ...].
    """
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def draw_annotations(
    frame: np.ndarray,
    detections: list,
    tracking_history: dict = None,
    zones: list = None,
    threat_score: int = 0,
    threat_level: str = "LOW",
    person_count: int = 0,
    correlated_events: list = None
) -> np.ndarray:
    """
    Renders high-tech SOC overlays:
    1. Restricted Area Polygons
    2. Person Trajectories (breadcrumb polylines)
    3. Bounding Boxes with "Person #1" labels
    4. Top Live HUD bar (Threat Score, Person Count, Active Correlations)
    """
    h, w = frame.shape[:2]
    
    # 1. Draw Restricted Polygon Zones
    if zones:
        for zone in zones:
            pts = np.array(zone.get("points", []), np.int32)
            if len(pts) > 2:
                pts = pts.reshape((-1, 1, 2))
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 0, 200))  # Translucent red
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
                zone_name = zone.get("name", "RESTRICTED ZONE")
                cv2.putText(
                    frame,
                    f"ZONE: {zone_name.upper()}",
                    (int(pts[0][0][0]), max(20, int(pts[0][0][1]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )

    # 2. Draw Trajectories for tracked persons
    if tracking_history:
        for track_id, info in tracking_history.items():
            traj = info.get("trajectory", [])
            if len(traj) > 1:
                # Draw breadcrumb line
                pts = np.array(traj, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (255, 200, 0), 2, cv2.LINE_AA)
                # Draw breadcrumb dots
                for p in traj[-5:]:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 255, 255), -1)

    # 3. Draw Bounding Boxes and Labels
    for det in detections:
        bbox = det.get("bbox")  # [x1, y1, x2, y2]
        cls_name = det.get("class", "")
        conf = det.get("confidence", 0.0)
        track_id = det.get("track_id")
        threat = det.get("threat_level", "LOW")
        
        if threat == "CRITICAL":
            color = (0, 0, 255)  # Bright Red
        elif threat == "HIGH":
            color = (0, 140, 255)  # Orange
        elif threat == "MEDIUM":
            color = (0, 230, 255)  # Yellow
        else:
            color = (0, 255, 120)  # Cyber Green
            
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Corner bracket bounding box design for SOC look
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label formatting
        if cls_name.lower() == "person" and track_id is not None:
            label = f"Person #{track_id} ({int(conf * 100)}%)"
        else:
            label = f"{cls_name} ({int(conf * 100)}%)"
            
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # 4. Top HUD Overlay
    hud_bg = frame.copy()
    cv2.rectangle(hud_bg, (0, 0), (w, 40), (15, 15, 15), -1)
    cv2.addWeighted(hud_bg, 0.7, frame, 0.3, 0, frame)
    
    # Threat Score Badge
    if threat_score > 100 or threat_level == "CRITICAL":
        badge_color = (0, 0, 255)  # Red
    elif threat_score > 50 or threat_level == "HIGH":
        badge_color = (0, 140, 255)  # Orange
    elif threat_score > 20 or threat_level == "MEDIUM":
        badge_color = (0, 220, 255)  # Yellow
    else:
        badge_color = (0, 200, 0)  # Green
        
    cv2.rectangle(frame, (10, 8), (170, 32), badge_color, -1)
    cv2.putText(frame, f"SCORE: {threat_score} [{threat_level}]", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Live Person Count
    cv2.putText(frame, f"PEOPLE: {person_count}", (185, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    
    # Correlated events or active alert notification
    if correlated_events:
        corr_text = f"CORRELATION: {correlated_events[0]}"
        cv2.putText(frame, corr_text, (300, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1, cv2.LINE_AA)
    elif threat_level in ["HIGH", "CRITICAL"]:
        has_fight = any("fight" in d.get("class", "").lower() for d in detections)
        banner = "⚠️ PHYSICAL ALTERCATION DETECTED" if has_fight else f"⚠️ ALERT: {threat_level} THREAT DETECTED"
        cv2.putText(frame, banner, (280, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 140, 255) if threat_level == "HIGH" else (0, 0, 255), 1, cv2.LINE_AA)

    return frame

def save_evidence_screenshot(frame: np.ndarray, camera_id: str, event_type: str) -> str:
    """Save screenshot of a threat event."""
    os.makedirs(settings.SCREENSHOTS_DIR, exist_ok=True)
    filename = f"{camera_id}_{event_type}_{int(time.time())}.jpg"
    filepath = os.path.join(settings.SCREENSHOTS_DIR, filename)
    cv2.imwrite(filepath, frame)
    return f"/screenshots/{filename}"

def save_evidence_clip(frames: list, camera_id: str, event_type: str, fps: int = 15) -> str:
    """
    MODULE 8: Save 10s pre/post video clip of a threat event into /event_clips.
    """
    if not frames:
        return None
    os.makedirs(settings.EVENT_CLIPS_DIR, exist_ok=True)
    filename = f"{camera_id}_{event_type}_{int(time.time())}.mp4"
    filepath = os.path.join(settings.EVENT_CLIPS_DIR, filename)
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
    
    for f in frames:
        out.write(f)
    out.release()
    
    return f"/event_clips/{filename}"

def encode_frame_to_base64(frame: np.ndarray) -> str:
    """Encode OpenCV frame to base64 jpeg string."""
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{jpg_as_text}"
