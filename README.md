# AEGIS: Enterprise Autonomous Edge-to-Cloud Surveillance SOC

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.2.8-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5.3-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-8.2.0-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0+CUDA-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11%20%7C%20v8%20%7C%20v5-000000?logo=github&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![MongoDB](https://img.shields.io/badge/MongoDB-Async%20Motor-47A248?logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4.19-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![License](https://img.shields.io/badge/License-Proprietary-blue.svg)]()

> **AEGIS** is a production-grade, AI-powered Security Operations Center (SOC) platform engineered for real-time computer vision threat detection, multi-model spatial correlation, explainable AI (XAI) incident diagnostics, and autonomous multi-channel alerting across distributed camera networks.

---

## Quick Navigation

* 📖 **Full File-by-File Codebase Manifest**: See [**`FILE_DIRECTORY.md`**](file:///d:/AI%20SUrvellinace/FILE_DIRECTORY.md) for comprehensive descriptions of every file in the repository.
* 🎯 **Live Demo Runbook**: See [**`DEMO.md`**](file:///d:/AI%20SUrvellinace/DEMO.md) for live presentation sequences, presentation accounts, and certified test patterns.

---

## Table of Contents
1. [System Architecture](#1-system-architecture)
2. [Dual-Mode AI Inference Pipeline](#2-dual-mode-ai-inference-pipeline)
3. [Unified Model Weights Registry](#3-unified-model-weights-registry)
4. [Monorepo Directory Layout](#4-monorepo-directory-layout)
5. [Quick-Start Guide](#5-quick-start-guide)
6. [Demonstration Credentials & Roles](#6-demonstration-credentials--roles)
7. [Preflight Verification & Health Diagnostics](#7-preflight-verification--health-diagnostics)
8. [Camera Channel Matrix](#8-camera-channel-matrix)
9. [Explainable AI (XAI) & Incident Reporting](#9-explainable-ai-xai--incident-reporting)
10. [REST & WebSocket API Reference](#10-rest--websocket-api-reference)
11. [Environment Configuration Reference](#11-environment-configuration-reference)
12. [Automated Test Suites](#12-automated-test-suites)
13. [Operational Diagnostics & Evaluation Scripts](#13-operational-diagnostics--evaluation-scripts)
14. [Security & Deployment Notes](#14-security--deployment-notes)

---

## 1. System Architecture

```text
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                              VIDEO INGESTION LAYER                                     │
 │  RTSP IP Streams │ USB / Laptop Webcams │ Recorded Forensic MP4s │ Synthetic Pipelines │
 └─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                           │ OpenCV VideoCapture (Threaded Workers)
                                           ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                    DUAL-MODE MULTI-MODEL AI PIPELINE (CUDA)                            │
 │                                                                                        │
 │  ┌─────────────────────────────────┐      ┌─────────────────────────────────────────┐  │
 │  │ 0. Person Detector (YOLO11s)    │ ───► │ 1. Weapon Detector (YOLOv5 Custom)      │  │
 │  │    Person bounding boxes, Re-ID │      │    Pistol, knife, phone/filter, boxes   │  │
 │  └────────────────┬────────────────┘      └─────────────────────────────────────────┘  │
 │                   │                                                                    │
 │                   ├──────────────────────► 2. Fight Classifier (YOLO11n-cls)           │
 │                   │                           Multi-person aggression & brawl action   │
 │                   │                                                                    │
 │                   └──────────────────────► 3. Accident Detector (YOLOv8 Custom)        │
 │                                               Vehicular collision & roadway trauma     │
 └─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                           │ Detections & Bounding Boxes
                                           ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                         THREAT SCORING & CORRELATION ENGINE                            │
 │  • Spatial Proximity Cone Analysis (Weapon in hand vs. bystander distance)             │
 │  • Temporal Stability Buffer (Multi-frame verification & cooldown suppression)         │
 │  • Restricted Area Polygonal Intersection (Zone breach scoring)                        │
 │  • Loitering & Unattended Weapon Tracking (Escalated threat timers)                    │
 │  • Multi-Event Dynamic Matrix (Accident + Intrusion + Weapon = CRITICAL 100)           │
 └─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                           │
                                           ├──────────────────────────┐
                                           ▼                          ▼
 ┌────────────────────────────────────────────────────┐ ┌─────────────────────────────────┐
 │               EXPLAINABLE AI (XAI)                 │ │     NOTIFICATION DISPATCHER     │
 │  • Natural language evidence summarization         │ │  • SMTP TLS Alerts (Gmail)      │
 │  • Key-factor contribution breakdown               │ │  • Automated PDF Incident Brief │
 │  • Spatial threat maps & risk recommendations     │ │  • Annotated Evidence Crop      │
 └─────────────────────────┬──────────────────────────┘ └─────────────────────────────────┘
                           │
                           ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                              PERSISTENCE & API LAYER                                   │
 │  • MongoDB (Motor Async): Events, Cameras, Restricted Zones, User Roles                │
 │  • FastAPI Core (Port 8000): WebSocket feeds, REST endpoints, Static clips / frames    │
 └─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                           │ WebSockets & REST (JSON)
                                           ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                       AEGIS COMMAND CENTER (VITE + REACT 19)                           │
 │  • Real-Time Multi-Cam Grid & Live HUD Overlays    • Interactive Zone Polygon Editor   │
 │  • Instant Forensic Filter & Clip Playback         • Comprehensive Telemetry & Health  │
 │  • Deep XAI Forensic Modal & PDF Download          • Analytical Charts & CSV Exports   │
 │  • Cross-Camera Person Re-ID Trajectory Search     • Interactive GIS Map & Heatmap     │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Dual-Mode AI Inference Pipeline

AEGIS dynamically operates across two specialized computer vision operating modes:

| Feature / Mode | CCTV Surveillance Mode | Webcam Presentation Mode |
| :--- | :--- | :--- |
| **Primary Ingestion** | RTSP IP Cameras, synthetic video streams (`sample_videos/`) | Direct USB or integrated laptop webcam (`source_url: "0"`) |
| **Spatial Resolution** | Wide-angle, distant person tracking (320px–1080px) | Close-up to mid-range dynamic user presentation |
| **Brightness Adaptation** | Static exposure with ambient lighting adjustments | Dynamic multi-frame brightness normalizer & contrast balancer |
| **Model Concurrency** | Full 4-model pipeline concurrently across camera workers | Person detector + Weapon + Threat cone HUD overlay |
| **False-Positive Gate** | Multi-frame persistence gates & unattended item tracking | Instantaneous HUD bounding boxes with safety class filtering |
| **Threat Gating** | 2+ person threshold for physical altercations | Single / multi-person weapon proximity cone analysis |

---

## 3. Unified Model Weights Registry

All active production model weights are stored in the standardized, top-level `weights/` directory. The backend configuration ([`backend/app/core/config.py`](file:///d:/AI%20SUrvellinace/backend/app/core/config.py) and [`backend/app/services/ai_service.py`](file:///d:/AI%20SUrvellinace/backend/app/services/ai_service.py)) dynamically resolves these models with graceful fallback to legacy paths:

| Model Role | Registry Path | Architecture | Primary Classes / Tasks | Size |
| :--- | :--- | :--- | :--- | :--- |
| **Person Detector** | [`weights/person_yolo11s.pt`](file:///d:/AI%20SUrvellinace/weights/person_yolo11s.pt) | YOLO11s COCO | `person` tracking, Re-ID bounding boxes | 19.3 MB |
| **Weapon Detector** | [`weights/weapon_yolov5.pt`](file:///d:/AI%20SUrvellinace/weights/weapon_yolov5.pt) | YOLOv5 6-Class | `pistol`, `knife`, `smartphone` (filter), objects | 5.5 MB |
| **Fight Classifier** | [`weights/fight_yolo11n_cls.pt`](file:///d:/AI%20SUrvellinace/weights/fight_yolo11n_cls.pt) | YOLO11n-cls | Binary classification: `Fight` vs. `NonFight` | 3.2 MB |
| **Accident Detector** | [`weights/accident_yolov8.pt`](file:///d:/AI%20SUrvellinace/weights/accident_yolov8.pt) | YOLOv8 Detect | Vehicular collisions, car damage, roadway impact | 19.2 MB |

---

## 4. Monorepo Directory Layout

> For an in-depth, file-by-file breakdown of every single file, check [**`FILE_DIRECTORY.md`**](file:///d:/AI%20SUrvellinace/FILE_DIRECTORY.md).

```text
d:/AI SUrvellinace/
├── weights/                                # Unified Model Weights Registry
│   ├── person_yolo11s.pt                   # COCO Person Detector
│   ├── weapon_yolov5.pt                    # 6-Class Weapon Detector
│   ├── fight_yolo11n_cls.pt                # Violence / Brawl Classifier
│   └── accident_yolov8.pt                  # Vehicular Accident Detector
│
├── backend/                                # FastAPI High-Performance Backend
│   ├── app/                                # Core Application Package
│   │   ├── core/                           # Database (MongoDB), Config & Security
│   │   │   ├── config.py                   # Global BaseSettings & model path resolution
│   │   │   ├── database.py                 # Async Motor MongoDB connection pool
│   │   │   └── security.py                 # JWT access token generation & hashing
│   │   ├── models/                         # PyMongo Database Entities (Camera, Event, User, Zone)
│   │   ├── schemas/                        # Pydantic Validation Schemas
│   │   ├── routers/                        # REST & WebSocket Endpoints (8 routers)
│   │   ├── services/                       # Business Logic (AIService, CameraManager, XAI, Email)
│   │   ├── utils/                          # Utilities (cv_helpers, pdf_generator)
│   │   └── main.py                         # FastAPI lifespan, static mounts & routers
│   ├── tests/                              # Comprehensive Pytest Test Suite (21 test suites)
│   ├── scripts/                            # Operational Tools, Benchmarks & Prep (35 scripts)
│   ├── .env.example                        # Documented environment template
│   └── requirements.txt                    # Python dependencies
│
├── frontend/                               # React 19 + Vite 8 + TypeScript Frontend
│   ├── public/                             # Public icons & static web assets
│   ├── src/
│   │   ├── assets/                         # SVG graphics & brand logos
│   │   ├── components/                     # UI components (common, map, modals, monitoring)
│   │   ├── constants/                      # Alert levels & system thresholds
│   │   ├── layouts/                        # CommandCenterLayout
│   │   ├── pages/                          # Routed views (11 views)
│   │   ├── services/                       # Axios & WebSocket API integrations
│   │   ├── types/                          # Surveillance TypeScript contracts
│   │   ├── App.tsx                         # Client router & navigation
│   │   ├── index.css                       # Global styles & Tailwind
│   │   └── main.tsx                        # React DOM mounting
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
│
├── docs/                                   # Documentation & Reports
│   └── reports/                            # Regression summaries, PDFs, and evaluation logs
│
├── sample_videos/                          # Test video clips (fight.mp4, accident.mp4, weapon.mp4, normal.mp4)
├── event_clips/                            # Recorded event video clips (.mp4)
├── screenshots/                            # Bounding box annotated event screenshots (.jpg)
├── DEMO.md                                 # Live presentation runbook
├── FILE_DIRECTORY.md                       # Comprehensive file directory manifest
└── README.md                               # Complete system documentation
```

---

## 5. Quick-Start Guide

### System Prerequisites
* **Operating System**: Windows 10/11 or Ubuntu 22.04 LTS
* **Python**: 3.10+ (with PyTorch and CUDA support enabled)
* **Node.js**: v18.0+ & npm
* **MongoDB Community Server**: Running locally on port `27017`

---

### Step 1: Start MongoDB
Ensure MongoDB is running locally:
```powershell
# Windows PowerShell
net start MongoDB
# Or run standalone:
mongod --dbpath "C:\data\db"
```

---

### Step 2: Configure & Start Backend (Port 8000)
```powershell
cd "d:\AI SUrvellinace\backend"

# Copy environment template if not present
cp .env.example .env

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server (production mode without auto-reload)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
* **API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **System Health Check**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

> [!NOTE]
> Run Uvicorn **WITHOUT `--reload`** in production to prevent file watcher overhead from throttling high-frequency CUDA inference.

---

### Step 3: Start Frontend Command Center (Port 5173)
```powershell
cd "d:\AI SUrvellinace\frontend"

# Install frontend dependencies
npm install

# Start Vite preview or development server
npm run preview -- --host 127.0.0.1 --port 5173
# Or development mode:
npm run dev -- --host 127.0.0.1 --port 5173
```
* **Command Center UI**: [http://localhost:5173/](http://localhost:5173/)

---

## 6. Demonstration Credentials & Roles

The system is pre-seeded on startup with 4 distinct Role-Based Access Control (RBAC) user profiles:

| Role | Username | Password | Email | System Privileges |
| :--- | :--- | :--- | :--- | :--- |
| **Chief SOC Director (Super Admin)** | `admin` | `admin123` | `admin@company.com` | Full administrative control: Live Monitoring, Model Sandbox, Forensic Search, Event History, Zone Polygon Editor, Email Settings. |
| **Control Room Operator** | `operator` | `operator123` | `operator@company.com` | Daily operations: Multi-cam feeds, incident acknowledgment, alarm silencing, and event timeline tracking. |
| **Police Admin** | `police` | `police123` | `police@company.com` | Law enforcement escalation view: Forensic event review, suspect search, and official PDF incident brief export. |
| **Guest / Viewer** | `viewer` | `viewer123` | `viewer@company.com` | Read-only telemetry, camera health, and analytical charts. |

---

## 7. Preflight Verification & Health Diagnostics

Run the automated preflight verification suite before demonstrations or production deployment:

```powershell
cd "d:\AI SUrvellinace\backend"
python scripts/demo_preflight.py
```

### Preflight Verification Matrix
1. **MongoDB Connection**: Connects and checks collections (`users`, `cameras`, `events`, `zones`).
2. **CUDA & PyTorch**: Confirms GPU availability, CUDA version, and VRAM.
3. **Model Weights**: Resolves all 4 production checkpoints in `weights/`.
4. **Camera Database**: Validates presence of the 5 seeded channels.
5. **Media Directories**: Verifies write permissions on `screenshots/` and `event_clips/`.
6. **Network Ports**: Verifies ports `8000` (Backend) and `5173` (Frontend) are accessible.

### Clean Slate Reset (Optional):
To clear prior test event logs and archive existing incident media while keeping users, cameras, and zone configurations intact:
```powershell
python scripts/demo_preflight.py --clear-events
```

---

## 8. Camera Channel Matrix

AEGIS comes pre-seeded with 5 dedicated camera channels covering critical surveillance scenarios:

| Channel ID | Camera Name | Default Source | Primary Detection Focus |
| :--- | :--- | :--- | :--- |
| **CAM-001** | North Gate Entrance | `sample_videos/fight.mp4` / Webcam `0` | Person Re-ID, Face & Physical Altercations |
| **CAM-002** | Perimeter Fence Alpha | Synthetic / RTSP Stream | Restricted Polygonal Zone Intrusion |
| **CAM-003** | Logistics Highway & Parking | `sample_videos/accident.mp4` | Vehicular Collisions & Roadway Hazards |
| **CAM-004** | Central Bank Lobby | `sample_videos/weapon.mp4` | Concealed & Unattended Firearms / Knives |
| **CAM-005** | Auxiliary Rooftop Feed | `sample_videos/normal.mp4` | Cross-Camera Appearance Trajectory |

---

## 9. Explainable AI (XAI) & Incident Reporting

AEGIS enforces zero black-box decisions through a 7-section structured Explainable AI diagnostic engine:

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      AEGIS XAI INCIDENT MODAL                          │
 │                                                                        │
 │  SECTION 1: Executive Incident Summary                                 │
 │  • Natural language summary synthesized from multi-model detections    │
 │                                                                        │
 │  SECTION 2: Key Evidence & Confidence Factors                          │
 │  • Primary object confidence, frame count persistence, model details   │
 │                                                                        │
 │  SECTION 3: Spatial Threat Cone & Proximity Analysis                   │
 │  • Distance to nearest bystander, threat angle, danger radius breach   │
 │                                                                        │
 │  SECTION 4: Temporal Progression Timeline                              │
 │  • Millisecond-accurate event development from inception to peak       │
 │                                                                        │
 │  SECTION 5: Dynamic Threat Score Breakdown                             │
 │  • Base threat + Proximity penalty + Zone multiplier + Multi-event     │
 │                                                                        │
 │  SECTION 6: Actionable SOC Recommendations                             │
 │  • Recommended SOP protocols, dispatch priority, evacuation guidance   │
 │                                                                        │
 │  SECTION 7: Automated Vector PDF Export                                │
 │  • Full multi-page report with annotated crops and legal audit trail   │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 10. REST & WebSocket API Reference

### Authentication (`/api/auth`)
* `POST /api/auth/token` — Exchange credentials for JWT bearer token
* `POST /api/auth/register` — Create new operator profile (Admin only)
* `GET /api/user/me` — Retrieve current authenticated user profile

### Cameras & Streaming (`/api/camera` & `/api/stream`)
* `GET /api/camera/` — List all configured camera channels
* `POST /api/camera/` — Add a new camera (RTSP URL, USB index, file)
* `PUT /api/camera/{id}` — Update camera configuration & status
* `DELETE /api/camera/{id}` — Remove camera channel
* `GET /api/camera/{id}/health` — Live FPS, dropped frames, worker uptime
* `WS /api/stream/ws/live/{camera_id}` — Real-time MJPEG / JPEG WebSocket video feed
* `WS /api/stream/ws/alerts` — Real-time event notifications broadcast
* `POST /api/stream/search/reid` — Search person appearance trajectory across feeds

### Incidents & Events (`/api/event`)
* `GET /api/event/` — Query incident events (with filters for threat level, camera, date)
* `GET /api/event/{id}` — Fetch detailed incident payload including XAI explanation
* `POST /api/event/{id}/acknowledge` — Operator incident acknowledgment
* `GET /api/event/{id}/pdf` — Download official vector incident PDF brief
* `GET /api/event/dashboard/summary` — Aggregated incident metrics & threat counts
* `GET /api/event/pdf-report` — Compile date-filtered executive intelligence PDF

### Restricted Zones (`/api/zone`)
* `GET /api/zone/` — List polygonal restricted zones
* `POST /api/zone/` — Create or update polygon vertices and threat multipliers
* `DELETE /api/zone/{id}` — Delete restricted area zone

### Model Testing Playground (`/api/model-test`)
* `POST /api/model-test/run` — Upload custom image or select preset for multi-model inference
* `GET /api/model-test/status` — Live status of all 4 AI models loaded on CUDA

---

## 11. Environment Configuration Reference

All options can be configured in `backend/.env` (templated from [`backend/.env.example`](file:///d:/AI%20SUrvellinace/backend/.env.example)):

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `PROJECT_NAME` | `"Smart Surveillance & Crime Monitoring"` | SOC system branding title |
| `SECRET_KEY` | `SUPER_SECRET_SURVEILLANCE_KEY...` | Cryptographic secret for JWT signing |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `10080` (7 days) | Session token expiration duration |
| `MONGODB_URL` | `"mongodb://localhost:27017"` | MongoDB database connection URI |
| `DATABASE_NAME` | `"surveillance_db"` | Primary MongoDB database name |
| `RESET_DB_ON_START` | `false` | If `true`, drops collections and re-seeds on boot |
| `PERSON_MODEL_PATH` | `"weights/person_yolo11s.pt"` | Person detection model weights path |
| `WEAPON_MODEL_PATH` | `"weights/weapon_yolov5.pt"` | Weapon detection model weights path |
| `FIGHT_MODEL_PATH` | `"weights/fight_yolo11n_cls.pt"` | Violence classification model weights path |
| `ACCIDENT_MODEL_PATH` | `"weights/accident_yolov8.pt"` | Accident detection model weights path |
| `CONFIDENCE_THRESHOLD` | `0.35` | Baseline object detection confidence gate |
| `WEAPON_CONFIDENCE_THRESHOLD` | `0.38` | Specific weapon validation threshold |
| `ACCIDENT_CONFIDENCE_THRESHOLD` | `0.52` | Specific accident confirmation threshold |
| `FIGHT_MIN_PERSONS_TO_RUN` | `2` | Minimum persons required to activate fight model |
| `FIGHT_CONFIRM_MIN_PERSONS_SEEN` | `4` | Frame persistence threshold for fight alerts |
| `UNATTENDED_HIGH_AFTER_FRAMES` | `30` | Frame count before unattended item escalates |
| `EMAIL_ENABLED` | `true` | Toggles autonomous SMTP alert dispatching |
| `SMTP_HOST` | `"smtp.gmail.com"` | Outgoing mail server address |
| `SMTP_PORT` | `587` | Outgoing mail server TLS port |
| `SMTP_USERNAME` | `"..."` | SMTP authentication username / Gmail address |
| `SMTP_PASSWORD` | `"..."` | 16-character Google App Password |
| `ALERT_RECIPIENT_EMAIL` | `"..."` | Default security officer alert recipient |

---

## 12. Automated Test Suites

All tests are centralized in [`backend/tests/`](file:///d:/AI%20SUrvellinace/backend/tests/) and run using `pytest`:

```powershell
# Run the entire test suite
pytest backend/tests/ -v

# Run targeted core verification suites
pytest backend/tests/test_weapon_pipeline.py -v
pytest backend/tests/test_accident_pipeline.py -v
pytest backend/tests/test_fight_pipeline.py -v
pytest backend/tests/test_db_persistence.py -v
pytest backend/tests/test_zone_e2e.py -v
pytest backend/tests/test_pdf_generator.py -v
pytest backend/tests/test_xai_pipeline.py -v
```

### Complete Catalog of 21 Test Suites
1. [`test_accident_pipeline.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_accident_pipeline.py) — End-to-end accident inference, threat scoring, and XAI
2. [`test_weapon_pipeline.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_weapon_pipeline.py) — Consecutive-frame weapon verification & bystander proximity
3. [`test_fight_pipeline.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_fight_pipeline.py) — Multi-person physical altercation temporal validation
4. [`test_db_persistence.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_db_persistence.py) — MongoDB CRUD consistency & camera persistence
5. [`test_zone_e2e.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_zone_e2e.py) — Interactive polygon intersection & intrusion alerting
6. [`test_pdf_generator.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_pdf_generator.py) — Automated incident brief generation with ReportLab
7. [`test_xai_pipeline.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_xai_pipeline.py) — Explainable AI evidence compilation & safety rules
8. [`test_email_accident.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_email_accident.py) — HTML email template generation & dispatch
9. [`test_smtp_connection.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_smtp_connection.py) — Outgoing TLS connection verification
10. [`test_model_test_endpoint.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_model_test_endpoint.py) — Model sandbox execution & class verification
11. [`test_live_endpoints.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_live_endpoints.py) — REST routing, CORS, and authentication verification
12. [`test_cam001_live_brightness.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_cam001_live_brightness.py) — Dynamic lighting & contrast adaptation
13. [`test_disconnect_reconnect.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_disconnect_reconnect.py) — Resilience against RTSP stream drops
14. [`test_held_still_200.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_held_still_200.py) — Stationary object gating & persistence
15. [`test_items_1_to_4.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_items_1_to_4.py) — Core baseline pipeline integration
16. [`test_multi_frame_live.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_multi_frame_live.py) — Multi-camera concurrent frame processing
17. [`test_real_webcam.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_real_webcam.py) — Physical USB webcam feed ingestion
18. [`test_unattended_weapon.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_unattended_weapon.py) — Abandoned firearm / knife tracking logic
19. [`test_ws_multi.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_ws_multi.py) — Concurrent WebSocket client streaming
20. [`test_ws_stream.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_ws_stream.py) — Low-latency frame serialization & delivery
21. [`test_advanced_features.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_advanced_features.py) — Multi-hazard correlation regression

---

## 13. Operational Diagnostics & Evaluation Scripts

Located in [`backend/scripts/`](file:///d:/AI%20SUrvellinace/backend/scripts/):

* `demo_preflight.py` — Automated system health check before presentations.
* `run_40clip_regression.py` — Batch verification against 40 benchmark videos.
* `run_regression_csv.py` — Generates `docs/reports/regression_results.csv`.
* `check_surveillance_db.py` — Verifies database collections and indexes.
* `diag_weapon_eval.py` & `diag_weapon_variants.py` — Precision testing across weapon geometries.
* `diag_fight.py` & `diag_fight_missed.py` — Fight temporal progression diagnostics.
* `diag_cctv_scale_eval.py` — Evaluates detections from 320px to 1080px.
* `eval_cooldown_tradeoff.py` — Optimizes notification cooldown parameters.
* `manage_zones.py` — CLI tool for updating polygonal tripwire boundaries.

---

## 14. Security & Deployment Notes

1. **Production Uvicorn Process**:
   Always execute the backend without `--reload` to prevent filesystem watcher interruptions during CUDA inference:
   ```powershell
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
   ```
2. **GPU Memory Allocation**:
   The multi-model pipeline uses PyTorch CUDA stream caching. A minimum of **4 GB dedicated VRAM** (GTX 1650 or higher) is recommended for concurrent multi-camera feeds.
3. **Environment Security**:
   Ensure `SECRET_KEY` and `SMTP_PASSWORD` are stored exclusively in `backend/.env` and never committed to version control. The repository `.gitignore` explicitly prevents credential leaks.
