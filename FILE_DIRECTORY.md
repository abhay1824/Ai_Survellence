# AEGIS AI Surveillance SOC — Comprehensive Codebase File Directory & Manifest

> **Document Version**: 2.0.0  
> **Platform**: AEGIS Autonomous Edge-to-Cloud Surveillance SOC  
> **Repository Root**: `d:\AI SUrvellinace`  
> **Target Audience**: SOC Engineers, Computer Vision Researchers, Full-Stack Developers, Security Auditors

This document provides a line-item architectural catalog and comprehensive description of **every file, script, component, model, and directory** in the AEGIS surveillance platform.

---

## Table of Contents
1. [Repository Root Files](#1-repository-root-files)
2. [Production AI Model Weights (`weights/`)](#2-production-ai-model-weights-weights)
3. [Sample Surveillance Video Streams (`sample_videos/`)](#3-sample-surveillance-video-streams-sample_videos)
4. [Backend Core Application (`backend/app/`)](#4-backend-core-application-backendapp)
   - [Core Infrastructure (`backend/app/core/`)](#core-infrastructure-backendappcore)
   - [Database Models (`backend/app/models/`)](#database-models-backendappmodels)
   - [Pydantic Schemas (`backend/app/schemas/`)](#pydantic-schemas-backendappschemas)
   - [API Routers (`backend/app/routers/`)](#api-routers-backendapprouters)
   - [Domain Services (`backend/app/services/`)](#domain-services-backendappservices)
   - [Utilities (`backend/app/utils/`)](#utilities-backendapputils)
5. [Backend Operational Scripts (`backend/scripts/`)](#5-backend-operational-scripts-backendscripts)
   - [Data Preparation (`backend/scripts/data_prep/`)](#data-preparation-backendscriptsdata_prep)
   - [Diagnostics & Evaluations (`backend/scripts/`)](#diagnostics--evaluations-backendscripts)
6. [Backend Automated Test Suites (`backend/tests/`)](#6-backend-automated-test-suites-backendtests)
7. [Frontend Command Center Application (`frontend/`)](#7-frontend-command-center-application-frontend)
   - [Root Configs & Build Assets](#root-configs--build-assets)
   - [Source Entrypoints & Styles (`frontend/src/`)](#source-entrypoints--styles-frontendsrc)
   - [UI Components (`frontend/src/components/`)](#ui-components-frontendcomponents)
   - [Command Center Pages (`frontend/src/pages/`)](#command-center-pages-frontendpages)
   - [Frontend Services & API Clients (`frontend/src/services/`)](#frontend-services--api-clients-frontendservices)
   - [TypeScript Contracts & Constants (`frontend/src/types/` & `constants/`)](#typescript-contracts--constants)
8. [Documentation & Evaluation Reports (`docs/`)](#8-documentation--evaluation-reports-docs)
9. [Dataset & Training Directories](#9-dataset--training-directories)
10. [Media & Evidence Directories](#10-media--evidence-directories)

---

## 1. Repository Root Files

| File Name | Category | Description | Key Responsibilities & Notes |
| :--- | :--- | :--- | :--- |
| [`README.md`](file:///d:/AI%20SUrvellinace/README.md) | Documentation | Primary project documentation and executive overview. | Architecture overview, dual-mode CV pipeline, Quickstart steps, API reference, test commands, and system benchmarks. |
| [`DEMO.md`](file:///d:/AI%20SUrvellinace/DEMO.md) | Runbook | Live presentation runbook and demonstration guide. | Step-by-step procedures for live webcam demonstrations, demo credentials, prohibited test patterns, and troubleshooting. |
| [`FILE_DIRECTORY.md`](file:///d:/AI%20SUrvellinace/FILE_DIRECTORY.md) | Manifest | Codebase manifest and line-item file reference. | Exhaustive index detailing every file, module, router, and test suite in the entire monorepo. |
| [`.gitignore`](file:///d:/AI%20SUrvellinace/.gitignore) | Git Config | Git version control exclusion rules. | Excludes Python cache (`__pycache__`), virtual environments (`.venv`), temporary crops (`debug_crops/`), node dependencies (`node_modules`), secrets (`.env`), and large raw datasets. |
| [`yolo11s.pt`](file:///d:/AI%20SUrvellinace/yolo11s.pt) | Model Checkpoint | Baseline COCO YOLO11s model checkpoint (19.3 MB). | Standard COCO object detection weights used for person tracking and bounding box initialization. |
| [`yolo11n-cls.pt`](file:///d:/AI%20SUrvellinace/yolo11n-cls.pt) | Model Checkpoint | Baseline YOLO11 nano classifier checkpoint (5.8 MB). | Base classification weights used as backbone for fine-tuning the fight classification network. |
| [`yolo26n.pt`](file:///d:/AI%20SUrvellinace/yolo26n.pt) | Model Checkpoint | Lightweight experimental nano model checkpoint (5.5 MB). | Used for fast inference experiments and baseline feature extraction. |
| [`Videos - Shortcut.lnk`](file:///d:/AI%20SUrvellinace/Videos%20-%20Shortcut.lnk) | Shortcut | Windows Shell Shortcut to sample video assets. | Direct OS link to forensic surveillance footage folders. |

---

## 2. Production AI Model Weights (`weights/`)

This directory houses the standardized production checkpoints utilized across all inference channels. The backend service dynamically resolves models from this directory with automatic fallback handling.

| File Path | Architecture | Primary Classes / Target | Size | Role in Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| [`weights/person_yolo11s.pt`](file:///d:/AI%20SUrvellinace/weights/person_yolo11s.pt) | YOLO11s (COCO) | `person` (Class 0) | 19.3 MB | **Module 0**: Ingests video frames, localizes human subjects, tracks persistence, and computes Euclidean bystander distances. |
| [`weights/weapon_yolov5.pt`](file:///d:/AI%20SUrvellinace/weights/weapon_yolov5.pt) | YOLOv5 6-Class Custom | `pistol`, `knife`, `smartphone`, `wallet` | 5.5 MB | **Module 1**: Scans whole frames and person crops for firearms and bladed weapons; filters out mobile phones. |
| [`weights/fight_yolo11n_cls.pt`](file:///d:/AI%20SUrvellinace/weights/fight_yolo11n_cls.pt) | YOLO11n-cls Custom | `Fight` vs. `NonFight` | 3.2 MB | **Module 2**: Temporal action classifier evaluating multi-person physical altercations with 6/10 frame gating. |
| [`weights/accident_yolov8.pt`](file:///d:/AI%20SUrvellinace/weights/accident_yolov8.pt) | YOLOv8 Detect Custom | `Accident`, `Severe Collision` | 19.2 MB | **Module 3**: Detects roadway vehicular collisions, damaged chassis, and rollover incidents on traffic channels. |
| [`weights/.gitkeep`](file:///d:/AI%20SUrvellinace/weights/.gitkeep) | Meta | Git retention placeholder. | Preserves the `weights/` directory structure in version control. |

---

## 3. Sample Surveillance Video Streams (`sample_videos/`)

Pre-packaged test video clips used by default camera channels (`CAM-001` through `CAM-005`):

| File Path | Resolution / Format | Primary Scene / Event | Assigned Channel |
| :--- | :--- | :--- | :--- |
| [`sample_videos/fight.mp4`](file:///d:/AI%20SUrvellinace/sample_videos/fight.mp4) | 1080p MP4 H.264 | Physical brawl between multiple individuals in urban setting. | `CAM-001` (North Gate Entrance) |
| [`sample_videos/accident.mp4`](file:///d:/AI%20SUrvellinace/sample_videos/accident.mp4) | 1080p MP4 H.264 | Multiple vehicle collision on public roadway. | `CAM-003` (Logistics Highway & Parking) |
| [`sample_videos/weapon.mp4`](file:///d:/AI%20SUrvellinace/sample_videos/weapon.mp4) | 1080p MP4 H.264 | Subject drawing firearm and brandishing weapon in public area. | `CAM-004` (Central Bank Lobby) |
| [`sample_videos/normal.mp4`](file:///d:/AI%20SUrvellinace/sample_videos/normal.mp4) | 1080p MP4 H.264 | Normal pedestrian transit, baseline non-threatening surveillance. | `CAM-005` (Auxiliary Rooftop Feed) |

---

## 4. Backend Core Application (`backend/app/`)

The backend is built with **FastAPI**, **Motor (Async MongoDB driver)**, **PyTorch/Ultralytics**, and **OpenCV**.

### Backend Root Configuration Files

| File Path | Description | Key Responsibilities |
| :--- | :--- | :--- |
| [`backend/requirements.txt`](file:///d:/AI%20SUrvellinace/backend/requirements.txt) | Python dependencies specification. | Declares exact versions for FastAPI, Uvicorn, Motor, Pymongo, Ultralytics, OpenCV, PyJWT, ReportLab, aiosmtplib, etc. |
| [`backend/.env.example`](file:///d:/AI%20SUrvellinace/backend/.env.example) | Environment variable template. | Comprehensive, annotated template with defaults for database, model paths, confidence gates, and SMTP credentials. |
| [`backend/.env`](file:///d:/AI%20SUrvellinace/backend/.env) | Local environment configuration. | Loaded by `backend/app/core/config.py` at runtime for operational secrets and tuning constants. |
| [`backend/yolov8n.pt`](file:///d:/AI%20SUrvellinace/backend/yolov8n.pt) | Local auxiliary YOLOv8 nano checkpoint. | Standalone test weight file for fast CPU verification and test mocking. |
| [`backend/app/main.py`](file:///d:/AI%20SUrvellinace/backend/app/main.py) | Application entrypoint and lifespan manager. | Initializes FastAPI, establishes async MongoDB connection pool, mounts static screenshot/clip routes, mounts CORS middleware, seeds database on startup, registers all routers, and manages graceful shutdown. |

---

### Core Infrastructure (`backend/app/core/`)

| File Path | Module | Description & Key Exports |
| :--- | :--- | :--- |
| [`backend/app/core/config.py`](file:///d:/AI%20SUrvellinace/backend/app/core/config.py) | `Settings` | Pydantic BaseSettings class managing environment variables, model file paths with automatic fallback logic, database parameters, threat score weights, and SMTP settings. |
| [`backend/app/core/database.py`](file:///d:/AI%20SUrvellinace/backend/app/core/database.py) | Database Pool | Async Motor client connection management. Exports `connect_to_mongo()`, `close_mongo_connection()`, and accessor helpers: `get_database()`, `get_camera_collection()`, `get_event_collection()`, `get_user_collection()`, and `get_zone_collection()`. |
| [`backend/app/core/security.py`](file:///d:/AI%20SUrvellinace/backend/app/core/security.py) | Cryptography & JWT | Security utilities for hashing passwords using Passlib (bcrypt), validating hashes (`verify_password`), and creating signed JWT access tokens (`create_access_token`) with HS256. |

---

### Database Models (`backend/app/models/`)

PyMongo and Pydantic entities defining the database document schemas:

| File Path | Entity Types | Description |
| :--- | :--- | :--- |
| [`backend/app/models/camera.py`](file:///d:/AI%20SUrvellinace/backend/app/models/camera.py) | `CameraDB` | Database schema for camera feeds: `camera_id`, `camera_name`, `latitude`, `longitude`, `stream_url`, `status`, `fps`, and telemetry metrics. |
| [`backend/app/models/event.py`](file:///d:/AI%20SUrvellinace/backend/app/models/event.py) | `EventDB`, `Detection`, `SpatialRiskAnalysis`, `XAIExplanation` | Complete incident entity schema containing detection boxes, threat levels (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), alarm status, temporal markers, annotated screenshot paths, and deep XAI diagnostic breakdowns. |
| [`backend/app/models/user.py`](file:///d:/AI%20SUrvellinace/backend/app/models/user.py) | `UserDB`, `UserRole`, `EmailSettings` | User authentication profile, hashed password storage, role hierarchy (`ADMIN`, `OPERATOR`, `POLICE`, `VIEWER`), and personalized email notification settings. |
| [`backend/app/models/zone.py`](file:///d:/AI%20SUrvellinace/backend/app/models/zone.py) | `RestrictedZoneDB` | Geospatial polygon definition: `zone_id`, `camera_id`, `name`, coordinate vertex array `[[x, y], ...]`, and associated threat level multiplier. |

---

### Pydantic Schemas (`backend/app/schemas/`)

Data Transfer Objects (DTOs) used for request parsing and response serialization:

| File Path | Schemas | Description |
| :--- | :--- | :--- |
| [`backend/app/schemas/auth.py`](file:///d:/AI%20SUrvellinace/backend/app/schemas/auth.py) | `Token`, `TokenData`, `UserLogin` | Schemas for authentication tokens, bearer payload validation, and login credentials. |
| [`backend/app/schemas/camera.py`](file:///d:/AI%20SUrvellinace/backend/app/schemas/camera.py) | `CameraCreate`, `CameraUpdate`, `CameraResponse`, `CameraHealthResponse` | Validation schemas for creating, updating, and fetching camera channels and live worker telemetry. |
| [`backend/app/schemas/event.py`](file:///d:/AI%20SUrvellinace/backend/app/schemas/event.py) | `EventCreate`, `EventResponse`, `EventSummary`, `EventFilterParams` | Schemas for querying historical incidents, pagination, date-range filters, and analytics summaries. |
| [`backend/app/schemas/user.py`](file:///d:/AI%20SUrvellinace/backend/app/schemas/user.py) | `UserCreate`, `UserResponse`, `EmailSettingsUpdate` | Schemas for user onboarding, profile queries, and updating SMTP alert preferences. |
| [`backend/app/schemas/zone.py`](file:///d:/AI%20SUrvellinace/backend/app/schemas/zone.py) | `ZoneCreate`, `ZoneResponse` | Schemas for validating polygonal vertex arrays and zone threat properties. |

---

### API Routers (`backend/app/routers/`)

REST and WebSocket route handlers mounted under `/api`:

| File Path | Route Prefix | Key Endpoints | Description |
| :--- | :--- | :--- | :--- |
| [`backend/app/routers/auth.py`](file:///d:/AI%20SUrvellinace/backend/app/routers/auth.py) | `/api/auth` | `POST /token`, `POST /register` | Handles OAuth2 form login, credentials validation, and new operator registration. |
| [`backend/app/routers/camera.py`](file:///d:/AI%20SUrvellinace/backend/app/routers/camera.py) | `/api/camera` | `GET /`, `POST /`, `PUT /{id}`, `DELETE /{id}`, `GET /{id}/health` | CRUD operations for camera channels; exposes live worker FPS, uptime, and frame-drop telemetry. |
| [`backend/app/routers/event.py`](file:///d:/AI%20SUrvellinace/backend/app/routers/event.py) | `/api/event` | `GET /`, `GET /{id}`, `POST /{id}/acknowledge`, `GET /{id}/pdf`, `GET /dashboard/summary`, `GET /pdf-report` | Queries filtered security events, marks alarms acknowledged, triggers ReportLab PDF export, and returns aggregated SOC metric summaries. |
| [`backend/app/routers/stream.py`](file:///d:/AI%20SUrvellinace/backend/app/routers/stream.py) | `/api/stream` | `WS /ws/live/{camera_id}`, `WS /ws/alerts`, `POST /search/reid` | Delivers real-time video feeds with annotated HUD bounding boxes via WebSocket; broadcasts instant threat alerts to operators; executes cross-camera Re-ID search. |
| [`backend/app/routers/analytics.py`](file:///d:/AI%20SUrvellinace/backend/app/routers/analytics.py) | `/api/analytics` | `GET /charts`, `GET /trends`, `GET /threat-distribution` | Aggregates incident frequencies, severity trends, and hourly distribution data formatted for Recharts. |
| [`backend/app/routers/zone.py`](file:///d:/AI%20SUrvellinace/backend/app/routers/zone.py) | `/api/zone` | `GET /`, `POST /`, `DELETE /{id}` | Manages polygonal restricted zones per camera feed for virtual perimeter tripwire monitoring. |
| [`backend/app/routers/user.py`](file:///d:/AI%20SUrvellinace/backend/app/routers/user.py) | `/api/user` | `GET /me`, `GET /email-settings`, `PUT /email-settings`, `POST /test-email` | Handles authenticated user profile queries, email alert configurations, and live test email dispatches. |
| [`backend/app/routers/model_test.py`](file:///d:/AI%20SUrvellinace/backend/app/routers/model_test.py) | `/api/model-test` | `GET /status`, `POST /run` | Live AI sandbox allowing operators to upload test images or select sample benchmarks to evaluate raw multi-model inference. |

---

### Domain Services (`backend/app/services/`)

The core business logic, computer vision pipeline, and external integration engines:

| File Path | Service Class | Primary Capabilities & Logic |
| :--- | :--- | :--- |
| [`backend/app/services/ai_service.py`](file:///d:/AI%20SUrvellinace/backend/app/services/ai_service.py) | `AIService` | **Central AI Engine (1700+ lines)**. Singleton orchestrating 4 PyTorch/Ultralytics models on CUDA: Person Tracking (YOLO11s), Weapon Detection (YOLOv5 6-class), Fight Classification (YOLO11n-cls), and Accident Detection (YOLOv8). Implements spatial proximity cone analysis, temporal multi-frame confirmation buffers, unattended object tracking, and multi-hazard correlation scoring. |
| [`backend/app/services/camera_manager.py`](file:///d:/AI%20SUrvellinace/backend/app/services/camera_manager.py) | `CameraManager`, `CameraWorker` | Multi-threaded video ingestion engine. Spawns dedicated background worker threads for each RTSP or USB webcam channel, maintains latest frame buffers with lock protection, computes real-time FPS/latency telemetry, and pushes frames through `AIService`. |
| [`backend/app/services/camera_service.py`](file:///d:/AI%20SUrvellinace/backend/app/services/camera_service.py) | `CameraService` | Database abstraction for camera management. Seeds default camera channels (`CAM-001` through `CAM-005`), handles MongoDB CRUD, and aggregates health telemetry. |
| [`backend/app/services/auth_service.py`](file:///d:/AI%20SUrvellinace/backend/app/services/auth_service.py) | `AuthService` | User authentication service. Seeds initial RBAC accounts (`admin`, `operator`, `police`, `viewer`), verifies credentials, and issues JWT tokens. |
| [`backend/app/services/event_service.py`](file:///d:/AI%20SUrvellinace/backend/app/services/event_service.py) | `EventService` | Incident lifecycle service. Packages detections into events, executes deduplication cooldown gates (prevents alarm flooding), writes annotated screenshots to disk, saves video clips, and triggers email notifications. |
| [`backend/app/services/xai_service.py`](file:///d:/AI%20SUrvellinace/backend/app/services/xai_service.py) | `XAIService` | **Explainable AI Engine**. Translates numeric model outputs into a 7-section structured explanation: executive summary, primary evidence factors, bystander proximity cone metrics, temporal timeline, dynamic threat score math, SOP recommendations, and audit blocks. |
| [`backend/app/services/email_service.py`](file:///d:/AI%20SUrvellinace/backend/app/services/email_service.py) | `EmailService` | Asynchronous SMTP alert engine using `aiosmtplib`. Generates branded HTML email incident notifications with threat badges, key metrics, annotated screenshot attachments, and PDF incident reports. Includes retry logic. |
| [`backend/app/services/reid_service.py`](file:///d:/AI%20SUrvellinace/backend/app/services/reid_service.py) | `ReIDService` | Person Re-Identification engine. Extracts feature vectors from person bounding boxes and computes cosine similarity to trace individual transit trajectories across multiple camera channels. |

---

### Utilities (`backend/app/utils/`)

| File Path | Description | Key Functions & Capabilities |
| :--- | :--- | :--- |
| [`backend/app/utils/cv_helpers.py`](file:///d:/AI%20SUrvellinace/backend/app/utils/cv_helpers.py) | Computer Vision Rendering Utilities | `point_in_polygon()`: Ray-casting algorithm for zone breach detection.<br>`draw_annotations()`: Renders high-tech SOC overlays, translucent red restricted zone polygons, person breadcrumb trajectories, color-coded bounding boxes, and top live HUD status bar. |
| [`backend/app/utils/pdf_generator.py`](file:///d:/AI%20SUrvellinace/backend/app/utils/pdf_generator.py) | Vector Incident PDF Brief Generator | Vector document compiler using ReportLab. Generates multi-page official security incident briefs featuring corporate header, metadata tables, evidence images, dynamic threat breakdown, and sign-off audit stamps. |

---

## 5. Backend Operational Scripts (`backend/scripts/`)

Tools for dataset preparation, diagnostics, automated regression, and pre-demonstration verification.

### Data Preparation (`backend/scripts/data_prep/`)

| File Path | Purpose | Execution Example |
| :--- | :--- | :--- |
| [`backend/scripts/data_prep/extract.py`](file:///d:/AI%20SUrvellinace/backend/scripts/data_prep/extract.py) | Frame Extractor | Extracts sequential JPEG frames from raw surveillance MP4 videos at configurable sample rates for dataset creation. |
| [`backend/scripts/data_prep/split.py`](file:///d:/AI%20SUrvellinace/backend/scripts/data_prep/split.py) | Dataset Splitter | Partitions image frames and YOLO label files into 80/10/10 train, validation, and test directories. |
| [`backend/scripts/data_prep/test_fight.py`](file:///d:/AI%20SUrvellinace/backend/scripts/data_prep/test_fight.py) | Violence Classifier Check | Standalone CLI script testing the YOLO11n-cls fight model across test sequences and logging confidence scores. |
| [`backend/scripts/data_prep/test_plain_fight.py`](file:///d:/AI%20SUrvellinace/backend/scripts/data_prep/test_plain_fight.py) | Baseline Classifier Check | Minimal inference script testing raw single-frame classification without temporal buffering. |

---

### Diagnostics & Evaluations (`backend/scripts/`)

| File Path | Script Name | Description & Usage |
| :--- | :--- | :--- |
| [`backend/scripts/demo_preflight.py`](file:///d:/AI%20SUrvellinace/backend/scripts/demo_preflight.py) | Preflight Diagnostic Suite | **Must-run script before live demonstrations**. Checks MongoDB health, CUDA availability, 4 model weights, camera records, write permissions, and network ports. Run: `python scripts/demo_preflight.py [--clear-events]`. |
| [`backend/scripts/demo_prep.py`](file:///d:/AI%20SUrvellinace/backend/scripts/demo_prep.py) | Demo Scenarios Preparer | Prepares deterministic demonstration scenarios, seeds baseline events, and checks media directory integrity. |
| [`backend/scripts/check_surveillance_db.py`](file:///d:/AI%20SUrvellinace/backend/scripts/check_surveillance_db.py) | DB Health Inspector | Inspects MongoDB collections, outputs document counts, checks indexes, and verifies document structure consistency. |
| [`backend/scripts/collect_hard_negatives.py`](file:///d:/AI%20SUrvellinace/backend/scripts/collect_hard_negatives.py) | Hard Negative Collector | Extracts false-positive candidates (e.g. mobile phones misidentified as pistols) to feed back into model retraining. |
| [`backend/scripts/diag_cctv_scale_eval.py`](file:///d:/AI%20SUrvellinace/backend/scripts/diag_cctv_scale_eval.py) | Multi-Scale CCTV Evaluator | Benchmarks model detection accuracy across simulated CCTV distances and resolutions (320px to 1080px). |
| [`backend/scripts/diag_fight.py`](file:///d:/AI%20SUrvellinace/backend/scripts/diag_fight.py) | Fight Inference Diagnoser | Runs fight classification on sample videos and outputs temporal frame confidence progressions. |
| [`backend/scripts/diag_fight_missed.py`](file:///d:/AI%20SUrvellinace/backend/scripts/diag_fight_missed.py) | False Negative Diagnoser | Pinpoints frames where violence occurred but confidence fell below the persistence threshold. |
| [`backend/scripts/diag_gate_eval.py`](file:///d:/AI%20SUrvellinace/backend/scripts/diag_gate_eval.py) | Gating Logic Evaluator | Evaluates the effect of confidence gates and multi-frame persistence windows on precision and recall. |
| [`backend/scripts/diag_weapon_eval.py`](file:///d:/AI%20SUrvellinace/backend/scripts/diag_weapon_eval.py) | Weapon Precision Evaluator | Tests precision, recall, and mAP across weapon classes (`pistol`, `knife`) on evaluation sets. |
| [`backend/scripts/diag_weapon_test.py`](file:///d:/AI%20SUrvellinace/backend/scripts/diag_weapon_test.py) | Weapon Detection Sandbox | Fast CLI tester running weapon detection on user-specified image files with printed bounding boxes. |
| [`backend/scripts/diag_weapon_variants.py`](file:///d:/AI%20SUrvellinace/backend/scripts/diag_weapon_variants.py) | Weapon Variant Tester | Evaluates detection behavior across aspect ratios, bounding box ceilings, and partial weapon occlusions. |
| [`backend/scripts/diag_webcam_eval.py`](file:///d:/AI%20SUrvellinace/backend/scripts/diag_webcam_eval.py) | Webcam Presentation Diagnoser | Tests dynamic contrast balancing and brightness normalizers for laptop and USB webcam streams. |
| [`backend/scripts/eval_cctv_anomaly.py`](file:///d:/AI%20SUrvellinace/backend/scripts/eval_cctv_anomaly.py) | Anomaly Benchmark Runner | Evaluates threat scoring and anomaly detection against extended continuous CCTV recordings. |
| [`backend/scripts/eval_cooldown_tradeoff.py`](file:///d:/AI%20SUrvellinace/backend/scripts/eval_cooldown_tradeoff.py) | Cooldown Tradeoff Benchmarker | Models alert frequency vs. missed incidents across cooldown windows (5s to 60s) to optimize operator ergonomics. |
| [`backend/scripts/eval_demo_scenarios.py`](file:///d:/AI%20SUrvellinace/backend/scripts/eval_demo_scenarios.py) | Scenario Simulator | Runs automated simulated passes for knife, pistol, accident, and fight demonstration workflows. |
| [`backend/scripts/eval_regression_pipeline.py`](file:///d:/AI%20SUrvellinace/backend/scripts/eval_regression_pipeline.py) | Pipeline Regression Runner | Runs automated end-to-end regression across all 4 CV models and validates key performance indicators. |
| [`backend/scripts/eval_task_a.py`](file:///d:/AI%20SUrvellinace/backend/scripts/eval_task_a.py) | Task A Evaluator | Dedicated evaluation benchmark for Person Detection and Spatial Tracking. |
| [`backend/scripts/eval_task_b.py`](file:///d:/AI%20SUrvellinace/backend/scripts/eval_task_b.py) | Task B Evaluator | Dedicated evaluation benchmark for Weapon Detection, Threat Cones, and Proximity Analysis. |
| [`backend/scripts/eval_task_c.py`](file:///d:/AI%20SUrvellinace/backend/scripts/eval_task_c.py) | Task C Evaluator | Dedicated evaluation benchmark for Fight Classification and Vehicular Collision Detection. |
| [`backend/scripts/eval_weapon_before_after.py`](file:///d:/AI%20SUrvellinace/backend/scripts/eval_weapon_before_after.py) | Model Comparison Tool | Quantifies performance deltas before and after model fine-tuning or threshold adjustments. |
| [`backend/scripts/finetune_weapon.py`](file:///d:/AI%20SUrvellinace/backend/scripts/finetune_weapon.py) | YOLO Fine-Tuning Script | Orchestrates YOLO fine-tuning runs on custom weapon datasets with transfer learning and augmentation. |
| [`backend/scripts/generate_contact_sheets.py`](file:///d:/AI%20SUrvellinace/backend/scripts/generate_contact_sheets.py) | Contact Sheet Generator | Assembles grid image contact sheets of detected events for rapid visual audits by supervisors. |
| [`backend/scripts/inspect_frame16.py`](file:///d:/AI%20SUrvellinace/backend/scripts/inspect_frame16.py) | Single-Frame Inspector | Diagnostic script inspecting frame 16 of specific sample clips to debug edge-case detection anomalies. |
| [`backend/scripts/investigate_cam003.py`](file:///d:/AI%20SUrvellinace/backend/scripts/investigate_cam003.py) | CAM-003 Channel Diagnoser | Analyzes vehicular accident detection and video decoding performance on camera channel `CAM-003`. |
| [`backend/scripts/investigate_cam004.py`](file:///d:/AI%20SUrvellinace/backend/scripts/investigate_cam004.py) | CAM-004 Channel Diagnoser | Analyzes weapon detection and unattended item tracking on camera channel `CAM-004`. |
| [`backend/scripts/investigate_fight_bug.py`](file:///d:/AI%20SUrvellinace/backend/scripts/investigate_fight_bug.py) | Fight Bug Root-Cause Script | Investigates participant gating conditions and temporal buffer resets during fight classification. |
| [`backend/scripts/manage_zones.py`](file:///d:/AI%20SUrvellinace/backend/scripts/manage_zones.py) | CLI Zone Manager | Command-line utility to inspect, create, or clear restricted polygon zones in the database. |
| [`backend/scripts/record_webcam_clip.py`](file:///d:/AI%20SUrvellinace/backend/scripts/record_webcam_clip.py) | Webcam Clip Recorder | Records 10-second calibration MP4 clips from attached webcam devices for offline model testing. |
| [`backend/scripts/run_40clip_regression.py`](file:///d:/AI%20SUrvellinace/backend/scripts/run_40clip_regression.py) | 40-Clip Batch Benchmarker | Executes batch evaluation across 40 diverse surveillance clips and reports aggregate accuracy. |
| [`backend/scripts/run_regression_csv.py`](file:///d:/AI%20SUrvellinace/backend/scripts/run_regression_csv.py) | CSV Regression Reporter | Runs automated regression and outputs metrics to `docs/reports/regression_results.csv`. |
| [`backend/scripts/simulate_stream_cam004.py`](file:///d:/AI%20SUrvellinace/backend/scripts/simulate_stream_cam004.py) | Stream Simulator | Simulates continuous RTSP video streaming on `CAM-004` to stress-test WebSocket delivery. |

---

## 6. Backend Automated Test Suites (`backend/tests/`)

Centralized in `backend/tests/` and executed using `pytest`.

| Test File | Target Subsystem | Key Invariants Verified |
| :--- | :--- | :--- |
| [`backend/tests/conftest.py`](file:///d:/AI%20SUrvellinace/backend/tests/conftest.py) | Global Fixtures | Sets up Python import paths, mock MongoDB client fixtures, test video frames, and session teardown. |
| [`backend/tests/test_accident_pipeline.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_accident_pipeline.py) | Vehicular Accident Pipeline | Validates YOLOv8 accident detection, confidence thresholding, 2-frame persistence confirmation, and incident creation. |
| [`backend/tests/test_weapon_pipeline.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_weapon_pipeline.py) | Weapon Detection Pipeline | Validates firearm/knife detection, consecutive-frame verification, and bystander proximity cone calculations. |
| [`backend/tests/test_fight_pipeline.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_fight_pipeline.py) | Fight Classifier Pipeline | Validates 6/10 temporal sliding buffer, 2+ person threshold gating, and brawl confidence progression. |
| [`backend/tests/test_db_persistence.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_db_persistence.py) | Database CRUD & Persistence | Tests MongoDB connection, collection drops, user/camera seeding, event logging, and zone queries. |
| [`backend/tests/test_zone_e2e.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_zone_e2e.py) | Restricted Zone Logic | Tests ray-casting point-in-polygon calculations, perimeter breach detection, and threat score multipliers. |
| [`backend/tests/test_pdf_generator.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_pdf_generator.py) | PDF Incident Generator | Validates ReportLab vector layout rendering, table construction, and screenshot image embedding. |
| [`backend/tests/test_xai_pipeline.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_xai_pipeline.py) | Explainable AI Pipeline | Tests generation of all 7 XAI sections, natural language synthesis, and risk factor calculations. |
| [`backend/tests/test_email_accident.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_email_accident.py) | Email Alerting | Verifies HTML incident template compilation, recipient routing, and attachment handling. |
| [`backend/tests/test_smtp_connection.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_smtp_connection.py) | SMTP Connectivity | Validates TLS handshake, authentication with credentials, and error handling for unreachable mail hosts. |
| [`backend/tests/test_model_test_endpoint.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_model_test_endpoint.py) | Sandbox Inference Endpoint | Tests `/api/model-test/run` and `/status` with test image payloads and validates detection boxes. |
| [`backend/tests/test_live_endpoints.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_live_endpoints.py) | REST API & Security | Tests all REST routes, HTTP response codes, CORS headers, and JWT bearer authentication. |
| [`backend/tests/test_cam001_live_brightness.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_cam001_live_brightness.py) | Lighting Normalization | Tests dynamic brightness balancing and histogram contrast normalization on camera `CAM-001`. |
| [`backend/tests/test_disconnect_reconnect.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_disconnect_reconnect.py) | Stream Resilience | Tests worker reconnection logic when an RTSP camera stream disconnects unexpectedly. |
| [`backend/tests/test_held_still_200.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_held_still_200.py) | Loitering & Stationary Gating | Tests tracking stability and loitering triggers when an object or person remains stationary for 200+ frames. |
| [`backend/tests/test_items_1_to_4.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_items_1_to_4.py) | Core Verification Suite | End-to-end integration test validating the 4 core detection milestones. |
| [`backend/tests/test_multi_frame_live.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_multi_frame_live.py) | Multi-Camera Concurrency | Validates concurrent frame ingestion across 5 camera workers without race conditions or memory leaks. |
| [`backend/tests/test_real_webcam.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_real_webcam.py) | Physical Webcam Ingestion | Tests hardware webcam capture via OpenCV VideoCapture index `0` and frame encoding. |
| [`backend/tests/test_unattended_weapon.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_unattended_weapon.py) | Unattended Object Engine | Tests unattended weapon logic: weapon detected without nearby person triggers escalated threat alarm. |
| [`backend/tests/test_ws_multi.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_ws_multi.py) | Concurrent WebSockets | Tests simultaneous WebSocket streaming to multiple browser clients from a single worker. |
| [`backend/tests/test_ws_stream.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_ws_stream.py) | WebSocket Serialization | Tests low-latency JPEG encoding and binary WebSocket streaming throughput. |
| [`backend/tests/test_advanced_features.py`](file:///d:/AI%20SUrvellinace/backend/tests/test_advanced_features.py) | Advanced Features Suite | Tests multi-event correlation matrix (accident + intrusion + weapon = CRITICAL 100). |
| [`backend/tests/fixtures/`](file:///d:/AI%20SUrvellinace/backend/tests/fixtures/) | Test Fixtures Directory | Sample test frames (`cam0_test_*.jpg`, `plain_test.jpg`, `scratch_sp.jpg`) and sample incident PDF briefs. |

---

## 7. Frontend Command Center Application (`frontend/`)

Built with **React 19**, **Vite 8**, **TypeScript**, **Tailwind CSS**, **Leaflet**, and **Lucide React**.

### Root Configs & Build Assets

| File Path | Description | Key Responsibilities |
| :--- | :--- | :--- |
| [`frontend/package.json`](file:///d:/AI%20SUrvellinace/frontend/package.json) | NPM package definition. | Declares scripts (`dev`, `build`, `preview`, `lint`) and dependencies (React 19, React Router 7, Axios, Lucide, Recharts, Leaflet, Tailwind). |
| [`frontend/vite.config.ts`](file:///d:/AI%20SUrvellinace/frontend/vite.config.ts) | Vite build configuration. | Configures `@vitejs/plugin-react`, local dev server ports, and proxying. |
| [`frontend/tsconfig.json`](file:///d:/AI%20SUrvellinace/frontend/tsconfig.json) | TypeScript root configuration. | Base compiler options and project references. |
| [`frontend/tsconfig.app.json`](file:///d:/AI%20SUrvellinace/frontend/tsconfig.app.json) | TypeScript app compiler config. | Strict typing rules, DOM types, and module resolution for `src/`. |
| [`frontend/tsconfig.node.json`](file:///d:/AI%20SUrvellinace/frontend/tsconfig.node.json) | TypeScript node configuration. | Type rules for Vite build scripts and config files. |
| [`frontend/tailwind.config.js`](file:///d:/AI%20SUrvellinace/frontend/tailwind.config.js) | Tailwind CSS styling configuration. | Custom theme extensions: SOC dark-palette colors (`zinc-950`, `zinc-900`), threat color tokens, and custom keyframes. |
| [`frontend/postcss.config.js`](file:///d:/AI%20SUrvellinace/frontend/postcss.config.js) | PostCSS plugins config. | Configures Tailwind CSS and Autoprefixer. |
| [`frontend/index.html`](file:///d:/AI%20SUrvellinace/frontend/index.html) | Single Page Application HTML root. | Declares viewport meta, font links (Inter), title, and mounting div `#root`. |
| [`frontend/.oxlintrc.json`](file:///d:/AI%20SUrvellinace/frontend/.oxlintrc.json) | Oxlint linter rules. | High-speed Rust linter rules for React hooks and clean code formatting. |
| [`frontend/.gitignore`](file:///d:/AI%20SUrvellinace/frontend/.gitignore) | Frontend git exclusion rules. | Excludes `dist/`, `node_modules/`, and local build artifacts. |
| [`frontend/README.md`](file:///d:/AI%20SUrvellinace/frontend/README.md) | Frontend starter documentation. | Documentation on Vite React template structure. |

---

### Source Entrypoints & Styles (`frontend/src/`)

| File Path | Description | Key Responsibilities |
| :--- | :--- | :--- |
| [`frontend/src/main.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/main.tsx) | React DOM mounting point. | Initializes React 19 root and renders `<App />` inside `#root`. |
| [`frontend/src/App.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/App.tsx) | Main Router & Application Shell. | Configures client routing (`/`, `/monitoring`, `/forensics`, `/health`, `/cameras`, `/events`, `/reports`, `/email`, `/model-test`, `/login`) wrapped in `ProtectedRoute`. |
| [`frontend/src/App.css`](file:///d:/AI%20SUrvellinace/frontend/src/App.css) | Global component animations. | Custom CSS animations for pulsing threat badges, scanlines, and radar sweeps. |
| [`frontend/src/index.css`](file:///d:/AI%20SUrvellinace/frontend/src/index.css) | Global Tailwind CSS stylesheet. | Tailwind directives (`@tailwind base`, `components`, `utilities`), scrollbar styling, and high-tech SOC theme variables. |

---

### UI Components (`frontend/src/components/`)

#### Common Components (`frontend/src/components/common/`)

| File Path | Component Name | Description & Capabilities |
| :--- | :--- | :--- |
| [`frontend/src/components/common/Navbar.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/components/common/Navbar.tsx) | `Navbar` | Persistent top navigation bar displaying system title, active operational view tabs, live threat status pill, user role badge, and logout trigger. |
| [`frontend/src/components/common/ProtectedRoute.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/components/common/ProtectedRoute.tsx) | `ProtectedRoute` | Route guard component that verifies JWT session tokens in localStorage and redirects unauthenticated users to `/login`. |
| [`frontend/src/components/common/EventTimeline.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/components/common/EventTimeline.tsx) | `EventTimeline` | Real-time vertical incident activity feed showing recent events with timestamps, threat badges, and quick-inspection links. |

#### Map & Geospatial Components (`frontend/src/components/map/`)

| File Path | Component Name | Description & Capabilities |
| :--- | :--- | :--- |
| [`frontend/src/components/map/CameraMap.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/components/map/CameraMap.tsx) | `CameraMap` | Interactive Leaflet map rendering camera markers across GPS coordinates, displaying camera status indicators and field-of-view cones. |
| [`frontend/src/components/map/HeatmapOverlay.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/components/map/HeatmapOverlay.tsx) | `HeatmapOverlay` | Geospatial heatmap overlay rendering incident density and high-risk zones across geographic sectors. |

#### Modal Dialogs (`frontend/src/components/modals/`)

| File Path | Component Name | Description & Capabilities |
| :--- | :--- | :--- |
| [`frontend/src/components/modals/XAIExplanationModal.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/components/modals/XAIExplanationModal.tsx) | `XAIExplanationModal` | **Flagship 7-Section Explainable AI Dialog**. Displays natural language summaries, key confidence factors, bystander proximity analysis, temporal progression timeline, threat score breakdown, SOC recommendations, and one-click PDF incident report download. |
| [`frontend/src/components/modals/index.ts`](file:///d:/AI%20SUrvellinace/frontend/src/components/modals/index.ts) | Modal Barrel Export | Clean re-export barrel for frontend modal components. |

#### Monitoring & Video HUD Components (`frontend/src/components/monitoring/`)

| File Path | Component Name | Description & Capabilities |
| :--- | :--- | :--- |
| [`frontend/src/components/monitoring/CameraGridCell.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/components/monitoring/CameraGridCell.tsx) | `CameraGridCell` | Individual camera cell for multi-camera grid layouts (1x1, 2x2, 3x3). Embeds WebSocket video feed, telemetry badge, and quick action controls. |
| [`frontend/src/components/monitoring/LiveHUDOverlay.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/components/monitoring/LiveHUDOverlay.tsx) | `LiveHUDOverlay` | Real-time heads-up display overlay rendered on top of live camera feeds: threat level badge, active person count, and correlation alerts. |
| [`frontend/src/components/monitoring/PolygonEditorModal.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/components/monitoring/PolygonEditorModal.tsx) | `PolygonEditorModal` | Interactive canvas editor allowing operators to click and draw custom polygonal restricted zones directly onto camera video frames. |
| [`frontend/src/components/monitoring/SirenAudioPlayer.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/components/monitoring/SirenAudioPlayer.tsx) | `SirenAudioPlayer` | Web Audio synthesizer producing audible security sirens during CRITICAL security events with operator mute/silence control. |

---

### Command Center Pages (`frontend/src/pages/`)

| File Path | Route | Page Title | Primary Capabilities & Workflows |
| :--- | :--- | :--- | :--- |
| [`frontend/src/pages/CommandCenter.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/pages/CommandCenter.tsx) | `/` | Operational Command Center | Primary SOC overview: interactive Leaflet GIS map with GPS geolocation, quick landmark jumps, incident heatmap overlay, live camera preview, and polygon zone editor trigger. |
| [`frontend/src/pages/LiveMonitoring.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/pages/LiveMonitoring.tsx) | `/monitoring` | Multi-Camera Live Feeds | Real-time multi-camera grid (single, 1x2, 2x2), low-latency WebSocket video streaming with HUD overlays, live fight/accident indicators, audible siren alerts, and camera health telemetry. |
| [`frontend/src/pages/EventHistory.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/pages/EventHistory.tsx) | `/events` | Forensic Event History | Searchable incident database with multi-parameter filtering (date range, camera channel, threat level, event type), annotated screenshot preview, XAI explanation modal launcher, and CSV export. |
| [`frontend/src/pages/ForensicSearch.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/pages/ForensicSearch.tsx) | `/forensics` | Cross-Camera Re-ID Search | Visual suspect search across camera networks. Upload a person image crop or select an event detection to trace their trajectory across all camera feeds using Re-ID feature vectors. |
| [`frontend/src/pages/CameraManagement.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/pages/CameraManagement.tsx) | `/cameras` | Camera Channel Administration | Add, configure, test, and remove camera feeds (RTSP stream URLs, USB webcam indexes, and video files). |
| [`frontend/src/pages/CameraHealth.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/pages/CameraHealth.tsx) | `/health` | Telemetry & Infrastructure | Real-time telemetry dashboard monitoring FPS rates, stream latency, dropped frame counts, worker uptime, and camera connection stability. |
| [`frontend/src/pages/AnalyticsReport.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/pages/AnalyticsReport.tsx) | `/reports` | Security Intelligence Analytics | Interactive analytics with Recharts (incident trends, hourly threat distribution, camera breach comparisons) and automated PDF incident brief compiler. |
| [`frontend/src/pages/EmailSettings.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/pages/EmailSettings.tsx) | `/email` | Notification Configuration | Configure SMTP recipient lists, toggle alert levels, enable automated PDF incident attachments, and dispatch live test emails. |
| [`frontend/src/pages/ModelTesting.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/pages/ModelTesting.tsx) | `/model-test` | AI Model Sandbox Playground | Live evaluation sandbox allowing operators to upload custom images or run benchmark samples to test the 4 AI models on CUDA, inspect raw bounding boxes, and verify inference times. |
| [`frontend/src/pages/Login.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/pages/Login.tsx) | `/login` | Secure Operator Authentication | High-tech SOC login interface with role selection presets (`admin`, `operator`, `police`, `viewer`) and JWT session establishment. |
| [`frontend/src/pages/Dashboard.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/pages/Dashboard.tsx) | `/dashboard` | Legacy Overview Dashboard | Retained legacy dashboard view showing system status and quick incident counters. |

---

### Frontend Services & API Clients (`frontend/src/services/`)

Axios HTTP and WebSocket client integrations:

| File Path | Target Domain | Key Functions & Responsibilities |
| :--- | :--- | :--- |
| [`frontend/src/services/api.ts`](file:///d:/AI%20SUrvellinace/frontend/src/services/api.ts) | Base Axios Client | Configures base Axios instance, injects JWT Bearer tokens via request interceptors, and exports `API_BASE_URL` and `WS_BASE_URL`. |
| [`frontend/src/services/auth.ts`](file:///d:/AI%20SUrvellinace/frontend/src/services/auth.ts) | Authentication Client | `login()`: Posts credentials to `/api/auth/token`, stores JWT token in localStorage.<br>`logout()`: Clears session and redirects to login.<br>`getCurrentUser()`: Fetches authenticated user profile. |
| [`frontend/src/services/camera.ts`](file:///d:/AI%20SUrvellinace/frontend/src/services/camera.ts) | Camera API Client | `getCameras()`, `addCamera()`, `updateCamera()`, `deleteCamera()`, `getCameraHealth()`. |
| [`frontend/src/services/event.ts`](file:///d:/AI%20SUrvellinace/frontend/src/services/event.ts) | Incident API Client | `getEvents()`: Multi-filtered event search.<br>`getEvent()`: Detailed single-event query.<br>`acknowledgeEvent()`: Operator incident acknowledgment.<br>`getAnalyticsCharts()`: Aggregated analytics for Recharts.<br>`downloadEventPDF()`: Downloads vector incident report PDF. |
| [`frontend/src/services/user.ts`](file:///d:/AI%20SUrvellinace/frontend/src/services/user.ts) | User & Settings Client | `getEmailSettings()`, `updateEmailSettings()`, `sendTestEmail()`. |
| [`frontend/src/services/zone.ts`](file:///d:/AI%20SUrvellinace/frontend/src/services/zone.ts) | Restricted Zone Client | `getZones()`, `saveZone()`, `deleteZone()`. |

---

### TypeScript Contracts & Constants

| File Path | Description | Key Definitions |
| :--- | :--- | :--- |
| [`frontend/src/types/surveillance.ts`](file:///d:/AI%20SUrvellinace/frontend/src/types/surveillance.ts) | Core Domain Contracts | `ThreatLevel`, `AlarmStatus`, `RestrictedZone`, `Camera`, `NearbyPerson`, `SpatialRiskAnalysis`, `TimelineStep`, `XAIExplanation`, `SurveillanceEvent`, `HeatmapPoint`, `CameraHealth`. |
| [`frontend/src/types/index.ts`](file:///d:/AI%20SUrvellinace/frontend/src/types/index.ts) | Type Barrel Export | Re-exports all surveillance contracts and UI state types for clean imports. |
| [`frontend/src/constants/alertLevels.ts`](file:///d:/AI%20SUrvellinace/frontend/src/constants/alertLevels.ts) | Threat Level Constants | Color schemes, badge styles, threat thresholds, and siren trigger triggers (`CRITICAL`, `HIGH`). |
| [`frontend/src/layouts/CommandCenterLayout.tsx`](file:///d:/AI%20SUrvellinace/frontend/src/layouts/CommandCenterLayout.tsx) | Page Layout Container | Standardized container wrapping views with persistent navigation and responsive margins. |

---

## 8. Documentation & Evaluation Reports (`docs/`)

Contains evaluation benchmarks, regression CSVs, dataset citations, and sample incident briefs:

| File Path | Type | Description |
| :--- | :--- | :--- |
| [`docs/reports/regression_results.csv`](file:///d:/AI%20SUrvellinace/docs/reports/regression_results.csv) | CSV Data | Tabular benchmark metrics from automated multi-model regression runs. |
| [`docs/reports/fight_test_output.txt`](file:///d:/AI%20SUrvellinace/docs/reports/fight_test_output.txt) | Text Log | Execution log from fight classifier evaluation across test video sequences. |
| [`docs/reports/summary_report.pdf`](file:///d:/AI%20SUrvellinace/docs/reports/summary_report.pdf) | PDF Document | Sample compiled executive security intelligence report. |
| [`docs/reports/test_report.pdf`](file:///d:/AI%20SUrvellinace/docs/reports/test_report.pdf) | PDF Document | Sample automated vector incident report generated by ReportLab. |
| [`docs/reports/README.dataset.txt`](file:///d:/AI%20SUrvellinace/docs/reports/README.dataset.txt) | Text Reference | Citations and licensing details for the accident and violence surveillance datasets. |
| [`docs/reports/README.roboflow.txt`](file:///d:/AI%20SUrvellinace/docs/reports/README.roboflow.txt) | Text Reference | Roboflow dataset export metadata and class mapping reference. |

---

## 9. Dataset & Training Directories

These directories store training and validation data utilized during model training and offline evaluation:

| Directory Path | Content & Purpose |
| :--- | :--- |
| [`acc/`](file:///d:/AI%20SUrvellinace/acc/) | Roboflow vehicular accident dataset containing partitioned `train/`, `valid/`, and `test/` splits with YOLO labels. |
| [`ACCIDENT/`](file:///d:/AI%20SUrvellinace/ACCIDENT/) | Raw accident video clips and extracted frame sets used during accident model training. |
| [`accident_frames/`](file:///d:/AI%20SUrvellinace/accident_frames/) | Frame-by-frame extractions from road collision footage used for threshold calibration. |
| [`Dataaset2/` & `Dataaset2_YOLO/`](file:///d:/AI%20SUrvellinace/Dataaset2/) | Annotated weapon datasets containing bladed objects, knives, and firearms in YOLO annotation format. |
| [`OD-WeaponDetection/`](file:///d:/AI%20SUrvellinace/OD-WeaponDetection/) | YOLOv5 weapon detection repository containing architecture definitions and training runs. |
| [`RWF-2000/`](file:///d:/AI%20SUrvellinace/RWF-2000/) | Real-World Fight dataset (2,000 surveillance video clips) used to train the violence action classifier. |
| [`WD/`](file:///d:/AI%20SUrvellinace/WD/) | Weapon detection raw images and label annotations. |
| [`hard_negatives/`](file:///d:/AI%20SUrvellinace/hard_negatives/) | Collected candidate images of mobile phones and innocuous objects used to eliminate false positive weapon alerts. |
| [`runs/`](file:///d:/AI%20SUrvellinace/runs/) | Ultralytics YOLO run output directory containing checkpoints, precision-recall curves, and training metrics. |

---

## 10. Media & Evidence Directories

Persistent storage locations managed at runtime by FastAPI:

| Directory Path | Storage Role | Managed By |
| :--- | :--- | :--- |
| [`screenshots/`](file:///d:/AI%20SUrvellinace/screenshots/) | Stores bounding-box annotated JPEG evidence captures generated whenever a security event is triggered. Served at `/screenshots`. | `backend/app/services/event_service.py` |
| [`event_clips/`](file:///d:/AI%20SUrvellinace/event_clips/) | Stores 5–10 second MP4 video recordings centered around triggered security incidents. Served at `/event_clips`. | `backend/app/services/event_service.py` |
| [`demo_images/`](file:///d:/AI%20SUrvellinace/demo_images/) | Clean, certified test image frames used for live presentation demonstrations (e.g. `knife_1.jpg`, `pistol_1.jpg`, `accident_1.jpg`, `fight_1.jpg`). | Demo Presenter / `DEMO.md` |
| [`debug_crops/`](file:///d:/AI%20SUrvellinace/debug_crops/) | Diagnostic image crops of detected persons and candidate weapons for debugging spatial filters. | `backend/app/services/ai_service.py` |
| [`webcam_tests/`](file:///d:/AI%20SUrvellinace/webcam_tests/) | Temporary test frames saved during local webcam calibration tests. | `backend/scripts/record_webcam_clip.py` |
| [`scratch/`](file:///d:/AI%20SUrvellinace/scratch/) | Development scratch space for temporary scripts and experimental snippets. | Developer Workspace |

---

## Summary Statistics

```text
Total Top-Level Subsystems:   4 (weights, backend, frontend, docs)
Backend REST & WS Routers:    8 (auth, camera, event, stream, analytics, zone, user, model_test)
Core Backend Services:        8 (ai_service, camera_manager, camera_service, auth_service, event_service, xai_service, email_service, reid_service)
Operational & Diag Scripts:   35 scripts in backend/scripts/
Pytest Automated Test Suites: 21 test suites in backend/tests/
Frontend Routed Pages:        11 pages in frontend/src/pages/
Frontend UI Components:       11 components in frontend/src/components/
AI Production Models:         4 models on CUDA (YOLO11s Person, YOLOv5 Weapon, YOLO11n Fight, YOLOv8 Accident)
```
