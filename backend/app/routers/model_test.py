import os
import time
import math
import base64
import logging
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from app.services.ai_service import AIService
from app.core.config import settings

logger = logging.getLogger(__name__)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

router = APIRouter(
    prefix="/model-test",
    tags=["Model Testing & Inference Sandbox"]
)

# Preset sample test images available on the system
PRESET_SAMPLES = {
    "fight_sample_1": {
        "id": "fight_sample_1",
        "title": "Violent Fight Scene #1",
        "category": "fight",
        "description": "High-intensity physical altercation between individuals",
        "path": str(PROJECT_ROOT / "fight_frames" / "val" / "Fight" / "-1l5631l3fg_0_100.jpg")
    },
    "fight_sample_2": {
        "id": "fight_sample_2",
        "title": "Violent Fight Scene #2",
        "category": "fight",
        "description": "Crowd disturbance and combat frame",
        "path": str(PROJECT_ROOT / "fight_frames" / "val" / "Fight" / "0H2s9UJcNJ0_0_10.jpg")
    },
    "normal_scene_1": {
        "id": "normal_scene_1",
        "title": "Normal Pedestrian Scene",
        "category": "normal",
        "description": "Clean non-fight baseline civilian activity",
        "path": str(PROJECT_ROOT / "fight_frames" / "val" / "NonFight" / "-1l5631l3fg_0_50.jpg")
    },
    "accident_sample_1": {
        "id": "accident_sample_1",
        "title": "Traffic Accident Scene",
        "category": "accident",
        "description": "Vehicular collision and traffic accident incident",
        "path": str(PROJECT_ROOT / "acc" / "test" / "images" / "Accidents-online-video-cutter_com-1-_mp4-113_jpg.rf.08ebb356870c2b9d4076437525cb7c2b.jpg")
    },
    "weapon_sample_1": {
        "id": "weapon_sample_1",
        "title": "Armed Weapon (Knife)",
        "category": "weapon",
        "description": "Concealed or brandished edged weapon incident",
        "path": str(PROJECT_ROOT / "OD-WeaponDetection" / "Knife classification" / "AAAKnife" / "-0.jpg")
    },
    "surveillance_cam": {
        "id": "surveillance_cam",
        "title": "Surveillance Camera Stream Frame",
        "category": "surveillance",
        "description": "Actual facility live camera snapshot",
        "path": str(PROJECT_ROOT / "test_cam1.jpg")
    }
}


@router.get("")
@router.get("/")
@router.get("/status")
def get_models_status():
    """Returns operational status, loaded weights, and class mappings for all models."""
    ai_service = AIService()

    models_info = {
        "person_detector": {
            "loaded": getattr(ai_service, "person_model", None) is not None,
            "type": "Object Detection (YOLO COCO)",
            "classes": ["person"],
            "model_file": getattr(settings, "PERSON_MODEL_PATH", "yolo11s.pt")
        },
        "fight_classifier": {
            "loaded": ai_service.fight_model is not None,
            "type": "Classification",
            "classes": list(ai_service.fight_model.names.values()) if ai_service.fight_model else ["Fight", "NonFight"],
            "model_file": getattr(settings, "FIGHT_MODEL_PATH", "runs/classify/fight_classifier/weights/best.pt")
        },
        "weapon_detector": {
            "loaded": ai_service.model is not None,
            "type": "Object Detection (YOLO)",
            "classes": list(ai_service.model.names.values()) if ai_service.model else ["pistol", "knife", "smartphone"],
            "model_file": settings.WEAPON_MODEL_PATH
        },
        "accident_detector": {
            "loaded": ai_service.accident_model is not None,
            "type": "Object Detection (YOLO)",
            "classes": list(ai_service.accident_model.names.values()) if ai_service.accident_model else ["Accident"],
            "model_file": getattr(settings, "ACCIDENT_MODEL_PATH", "runs/detect/accident_detector/weights/best.pt")
        }
    }

    available_samples = []
    for k, v in PRESET_SAMPLES.items():
        exists = os.path.exists(v["path"])
        available_samples.append({
            "id": v["id"],
            "title": v["title"],
            "category": v["category"],
            "description": v["description"],
            "available": exists
        })

    return {
        "service": "Eagle SOC Model Testing Laboratory",
        "models": models_info,
        "available_samples": available_samples
    }


@router.get("/samples/{sample_id}")
def get_sample_image(sample_id: str):
    """Serves the raw image file for a preset sample."""
    sample = PRESET_SAMPLES.get(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample preset not found")

    if not os.path.exists(sample["path"]):
        raise HTTPException(status_code=404, detail=f"Sample image file not found on disk: {sample['path']}")

    return FileResponse(sample["path"], media_type="image/jpeg")


@router.post("/predict")
async def run_model_inference(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    model_type: str = Form("all"),  # "all", "fight", "weapon", "accident"
    confidence_threshold: float = Form(0.35)
):
    """
    Executes trained AI model inference on an uploaded image or sample preset.
    Returns detected categories, bounding boxes, fight probability, annotated image,
    and a centralized threat assessment.
    """
    start_time = time.time()
    ai_service = AIService()

    # 1. Retrieve Image Buffer
    img_array = None
    original_b64 = None

    if file and file.filename:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_array is None:
            raise HTTPException(status_code=400, detail="Uploaded file could not be decoded as an image.")
        _, orig_buf = cv2.imencode(".jpg", img_array)
        original_b64 = f"data:image/jpeg;base64,{base64.b64encode(orig_buf).decode('utf-8')}"
    elif sample_id and sample_id in PRESET_SAMPLES:
        sample_path = PRESET_SAMPLES[sample_id]["path"]
        if not os.path.exists(sample_path):
            raise HTTPException(status_code=404, detail=f"Preset file {sample_path} not found.")
        img_array = cv2.imread(sample_path)
        if img_array is None:
            raise HTTPException(status_code=500, detail="Failed to load preset sample image.")
        _, orig_buf = cv2.imencode(".jpg", img_array)
        original_b64 = f"data:image/jpeg;base64,{base64.b64encode(orig_buf).decode('utf-8')}"
    else:
        raise HTTPException(status_code=400, detail="Please provide either an image file or a valid sample_id.")

    h, w = img_array.shape[:2]
    annotated = img_array.copy()

    fight_result = None
    weapon_detections = []
    accident_detections = []

    # 2. Execute Fight Classifier if requested
    if model_type in ["all", "fight"] and ai_service.fight_model is not None:
        try:
            res_fight = ai_service.fight_model.predict(img_array, verbose=False)
            if res_fight and len(res_fight) > 0 and res_fight[0].probs is not None:
                probs = res_fight[0].probs
                top1_idx = int(probs.top1)
                top_class = str(ai_service.fight_model.names.get(top1_idx, "Unknown"))
                top_conf = float(probs.top1conf)

                probs_dict = {}
                for idx, name in ai_service.fight_model.names.items():
                    probs_dict[str(name)] = round(float(probs.data[idx]), 4)

                is_fight = (top_class.lower() == "fight") and (top_conf >= 0.50)
                fight_result = {
                    "top_class": top_class,
                    "confidence": round(top_conf, 4),
                    "is_fight": is_fight,
                    "probabilities": probs_dict,
                    "model": getattr(settings, "FIGHT_MODEL_PATH", "weights/fight_yolo11n_cls.pt")
                }
        except Exception as e:
            logger.error(f"Fight prediction error: {e}")
            fight_result = {"error": str(e), "is_fight": False}

    # 3. Execute Weapon Object Detector if requested
    if model_type in ["all", "weapon"] and ai_service.model is not None:
        try:
            res_weapon = ai_service.model.predict(img_array, conf=confidence_threshold, verbose=False)
            if res_weapon and len(res_weapon) > 0 and res_weapon[0].boxes is not None:
                for box in res_weapon[0].boxes:
                    coords = [int(v) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = str(ai_service.model.names.get(cls_id, f"weapon_{cls_id}"))

                    weapon_detections.append({
                        "class": cls_name,
                        "confidence": round(conf, 4),
                        "box": coords  # [x1, y1, x2, y2]
                    })
        except Exception as e:
            logger.error(f"Weapon detection error: {e}")

    # 4. Execute Accident Detector if requested
    if model_type in ["all", "accident"] and ai_service.accident_model is not None:
        try:
            res_acc = ai_service.accident_model.predict(img_array, conf=confidence_threshold, verbose=False)
            if res_acc and len(res_acc) > 0 and res_acc[0].boxes is not None:
                for box in res_acc[0].boxes:
                    coords = [int(v) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = str(ai_service.accident_model.names.get(cls_id, "Accident"))

                    accident_detections.append({
                        "class": cls_name,
                        "confidence": round(conf, 4),
                        "box": coords
                    })
        except Exception as e:
            logger.error(f"Accident detection error: {e}")

    # 5. Render Visual Annotations on Image
    # A. Draw Weapon Detections
    for det in weapon_detections:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['class'].upper()} {det['confidence'] * 100:.1f}%"
        # Tactical red/orange box
        color = (0, 70, 255) if det["class"] in ["pistol", "knife"] else (0, 165, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # Label background badge
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        badge_h = label_size[1] + 10
        badge_w = label_size[0] + 12
        badge_y1 = max(0, y1 - badge_h)
        cv2.rectangle(annotated, (x1, badge_y1), (x1 + badge_w, badge_y1 + badge_h), color, -1)
        cv2.putText(annotated, label, (x1 + 6, badge_y1 + badge_h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # B. Draw Accident Detections
    for det in accident_detections:
        x1, y1, x2, y2 = det["box"]
        label = f"ACCIDENT {det['confidence'] * 100:.1f}%"
        color = (0, 215, 255)  # Amber / Yellow
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        badge_h = label_size[1] + 10
        badge_w = label_size[0] + 12
        badge_y1 = max(0, y1 - badge_h)
        cv2.rectangle(annotated, (x1, badge_y1), (x1 + badge_w, badge_y1 + badge_h), color, -1)
        cv2.putText(annotated, label, (x1 + 6, badge_y1 + badge_h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    # C. Draw Fight Overlay Header Banner
    if fight_result and "is_fight" in fight_result:
        is_fight = fight_result["is_fight"]
        conf = fight_result["confidence"]
        banner_h = 46

        # Semi-transparent overlay banner at top
        overlay = annotated.copy()
        if is_fight:
            bg_color = (20, 20, 210)  # Intense Red
            banner_text = f"CRITICAL ALERT: FIGHT DETECTED ({conf * 100:.1f}% CONFIDENCE)"
            icon_tag = "[!]"
        else:
            bg_color = (35, 130, 45)  # Clean Emerald
            banner_text = f"NORMAL / NON-FIGHT SCENE ({conf * 100:.1f}% CONFIDENCE)"
            icon_tag = "[OK]"

        cv2.rectangle(overlay, (0, 0), (w, banner_h), bg_color, -1)
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

        # White border line below banner
        line_color = (60, 60, 255) if is_fight else (80, 220, 100)
        cv2.line(annotated, (0, banner_h), (w, banner_h), line_color, 2)

        # Render Header text
        display_str = f"{icon_tag}  {banner_text}"
        cv2.putText(annotated, display_str, (16, 30), cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

    # Encode annotated image to Base64
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

    # 6. Overall Threat Score & Categorization
    threat_score = 0
    threat_level = "SAFE"
    summary_text = "No security threats or violent anomalies identified."

    is_fight = fight_result.get("is_fight", False) if fight_result else False
    fight_conf = fight_result.get("confidence", 0.0) if fight_result else 0.0
    weapon_count = len(weapon_detections)
    accident_count = len(accident_detections)

    if is_fight and weapon_count > 0:
        threat_score = 98
        threat_level = "CRITICAL"
        summary_text = f"ARMED COMBAT: Physical fight detected with active weapon present ({weapon_detections[0]['class']})."
    elif is_fight:
        threat_score = int(75 + (fight_conf * 20))
        threat_score = min(threat_score, 95)
        threat_level = "CRITICAL" if threat_score >= 80 else "HIGH"
        summary_text = f"FIGHT DETECTED: Physical violence detected with {fight_conf * 100:.1f}% model confidence."
    elif weapon_count > 0:
        threat_score = 80
        threat_level = "HIGH"
        summary_text = f"WEAPON DETECTED: Found {weapon_count} weapon object(s) ({weapon_detections[0]['class']})."
    elif accident_count > 0:
        threat_score = 70
        threat_level = "HIGH"
        summary_text = f"TRAFFIC ACCIDENT: Detected {accident_count} vehicular collision/hazard zone(s)."
    else:
        threat_score = 10
        threat_level = "SAFE"
        summary_text = "Clean environment. Zero threat signals detected."

    latency_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "success": True,
        "inference_time_ms": latency_ms,
        "image_dimensions": {"width": w, "height": h},
        "original_image": original_b64,
        "annotated_image": annotated_b64,
        "fight_result": fight_result,
        "weapon_result": {
            "count": weapon_count,
            "detections": weapon_detections
        },
        "accident_result": {
            "count": accident_count,
            "detections": accident_detections
        },
        "threat_assessment": {
            "score": threat_score,
            "level": threat_level,
            "summary": summary_text
        }
    }
