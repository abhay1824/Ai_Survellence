"""
Eagle Eye Camera Manager — Production-grade camera ingestion subsystem.
Manages per-camera worker threads, frame buffers, validation, health monitoring,
auto-reconnect, and an AI processing queue decoupled from capture FPS.
"""

import asyncio
import os
from pathlib import Path
import cv2
import numpy as np
import threading
import time
import logging
import queue
import urllib.request
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[3]

logger = logging.getLogger(__name__)


# ============================================================
# Enums & Data Classes
# ============================================================

class CameraStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    RECONNECTING = "reconnecting"
    NO_VIDEO_INPUT = "no_video_input"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class CameraHealthStats:
    """Real-time health statistics for a camera worker."""
    camera_id: str = ""
    camera_name: str = ""
    status: CameraStatus = CameraStatus.OFFLINE
    source_type: str = "unknown"
    source_url: str = ""
    fps: float = 0.0
    latency_ms: float = 0.0
    frames_captured: int = 0
    frames_dropped: int = 0
    last_frame_time: float = 0.0
    uptime_seconds: float = 0.0
    start_time: float = 0.0
    consecutive_failures: int = 0
    reconnect_count: int = 0

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "status": self.status.value,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "fps": round(self.fps, 1),
            "latency_ms": round(self.latency_ms, 1),
            "frames_captured": self.frames_captured,
            "frames_dropped": self.frames_dropped,
            "last_frame_time": self.last_frame_time,
            "uptime_seconds": round(time.time() - self.start_time, 1) if self.start_time else 0.0,
            "consecutive_failures": self.consecutive_failures,
            "reconnect_count": self.reconnect_count,
        }


# ============================================================
# Frame Validator
# ============================================================

class FrameValidator:
    """Static frame quality validation methods."""

    @staticmethod
    def is_valid(frame) -> bool:
        """Check if a frame is usable (not None, not empty, not corrupted)."""
        if frame is None:
            return False
        if not isinstance(frame, np.ndarray):
            return False
        if frame.size == 0:
            return False
        if len(frame.shape) < 2:
            return False
        return True

    @staticmethod
    def get_metrics(frame: np.ndarray) -> dict:
        """Compute quality metrics for a valid frame."""
        brightness = float(np.mean(frame))
        stddev = float(np.std(frame))
        h, w = frame.shape[:2]
        return {
            "brightness": round(brightness, 2),
            "stddev": round(stddev, 2),
            "resolution": f"{w}x{h}",
            "is_black": brightness < 8.0 and stddev < 2.0,
        }

    @staticmethod
    def is_quality_ok(frame: np.ndarray) -> bool:
        """Returns True if frame passes minimum quality thresholds."""
        brightness = float(np.mean(frame))
        stddev = float(np.std(frame))
        # Reject pure black / obstructed frames
        if brightness < 5.0 and stddev < 2.0:
            return False
        return True


# ============================================================
# Frame Buffer (Thread-Safe Ring Buffer)
# ============================================================

class FrameBuffer:
    """Thread-safe ring buffer holding the latest N frames for a camera."""

    def __init__(self, capacity: int = 150):
        self._capacity = capacity
        self._buffer: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_count = 0

    def push(self, frame: np.ndarray):
        with self._lock:
            self._latest_frame = frame
            self._buffer.append(frame)
            if len(self._buffer) > self._capacity:
                self._buffer.pop(0)
            self._frame_count += 1

    def get_latest(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def get_all(self) -> list[np.ndarray]:
        """Get all buffered frames (for evidence clip generation)."""
        with self._lock:
            return [f.copy() for f in self._buffer]

    @property
    def count(self) -> int:
        with self._lock:
            return self._frame_count

    def clear(self):
        with self._lock:
            self._buffer.clear()
            self._latest_frame = None


# ============================================================
# Source Resolver — Auto-detect camera stream endpoints
# ============================================================

class SourceResolver:
    """Resolves camera source URLs to working VideoCapture endpoints non-blockingly."""

    @staticmethod
    def resolve(source_url: str) -> tuple[str, str]:
        """
        Returns (resolved_url, source_type).
        source_type: "webcam", "rtsp", "http_stream", "http_snapshot", "file"
        """
        url = source_url.strip()

        if not url or url.lower() == "synthetic":
            return "synthetic", "synthetic"

        if url.isdigit():
            return url, "webcam"

        if url.lower() == "webcam":
            return "0", "webcam"

        if url.startswith("rtsp://"):
            return url, "rtsp"

        if url.startswith("http://") or url.startswith("https://"):
            if url.rstrip("/").endswith("/shot.jpg") or url.rstrip("/").endswith("/photo.jpg"):
                return url, "http_snapshot"
            return url, "http_stream"

        # Check if relative path exists under project root
        if not os.path.isabs(url):
            abs_cand = os.path.normpath(os.path.join(PROJECT_ROOT, url))
            if os.path.exists(abs_cand):
                return abs_cand, "file"

        # Assume file path or device index
        return url, "file"



# ============================================================
# Camera Worker — Dedicated capture thread per camera
# ============================================================

class CameraWorker:
    """
    Runs in a dedicated daemon thread. Captures frames from a single camera source.
    Writes validated frames into a FrameBuffer. Auto-reconnects on failure.
    """

    RECONNECT_DELAY = 3.0       # Seconds between reconnect attempts
    MAX_CONSECUTIVE_FAIL = 15   # Failures before entering RECONNECTING state
    FPS_WINDOW = 30             # Frames over which to compute rolling FPS

    def __init__(self, camera_id: str, camera_name: str, source_url: str,
                 target_fps: int = 15, resolution: tuple = (640, 480)):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.source_url = source_url
        self.target_fps = target_fps
        self.resolution = resolution

        # Resolved source
        self.resolved_url, self.source_type = SourceResolver.resolve(source_url)

        # Frame buffer
        self.frame_buffer = FrameBuffer(capacity=150)

        # Health stats
        self.stats = CameraHealthStats(
            camera_id=camera_id,
            camera_name=camera_name,
            source_type=self.source_type,
            source_url=self.resolved_url,
        )

        # Internal state
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # FPS tracking
        self._frame_times: list[float] = []
        self._synthetic_warned = False

    def start(self):
        """Start the capture thread."""
        if self._running:
            return
        self._running = True
        self.stats.start_time = time.time()
        self.stats.status = CameraStatus.RECONNECTING
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name=f"cam-{self.camera_id}")
        self._thread.start()
        logger.info(f"[CameraWorker] Started worker for {self.camera_id} ({self.source_type}: {self.resolved_url})")

    def stop(self):
        """Stop the capture thread and release resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._release_cap()
        self.stats.status = CameraStatus.OFFLINE
        logger.info(f"[CameraWorker] Stopped worker for {self.camera_id}")

    def update_source(self, new_url: str):
        """Update the camera source URL and reconnect."""
        self._release_cap()
        self.source_url = new_url
        self.resolved_url, self.source_type = SourceResolver.resolve(new_url)
        self.stats.source_type = self.source_type
        self.stats.source_url = self.resolved_url
        self.stats.consecutive_failures = 0
        self.stats.status = CameraStatus.RECONNECTING
        self._synthetic_warned = False
        logger.info(f"[CameraWorker] Source updated for {self.camera_id}: {self.resolved_url} ({self.source_type})")

    def _run_loop(self):
        """Main capture loop — runs in a dedicated thread."""
        frame_interval = 1.0 / self.target_fps

        while self._running:
            try:
                t_iter_start = time.time()
                # Connect if needed
                if self._cap is None or not self._cap.isOpened():
                    if not self._connect():
                        time.sleep(self.RECONNECT_DELAY)
                        continue

                # Handle snapshot-based sources differently
                if self.source_type == "http_snapshot":
                    self._capture_snapshot()
                else:
                    self._capture_video_frame()

                # Dynamic throttle to target FPS to prevent driver queue lag & buffer freezing
                elapsed = time.time() - t_iter_start
                sleep_time = max(0.005, frame_interval - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"[CameraWorker] {self.camera_id} loop error: {e}")
                self.stats.consecutive_failures += 1
                if self.stats.consecutive_failures > self.MAX_CONSECUTIVE_FAIL:
                    self.stats.status = CameraStatus.RECONNECTING
                    self._release_cap()
                    time.sleep(self.RECONNECT_DELAY)

    def _connect(self) -> bool:
        """Attempt to open the camera source."""
        self.stats.status = CameraStatus.RECONNECTING

        if self.source_type == "synthetic":
            if not self._synthetic_warned:
                logger.warning(
                    f"[CameraWorker] Camera {self.camera_id} uses 'synthetic' source which has no live video feed. "
                    "Camera worker will not capture frames. Configure a valid stream URL or video file in settings."
                )
                self._synthetic_warned = True
            self.stats.status = CameraStatus.NO_VIDEO_INPUT
            time.sleep(10.0)  # Back off to avoid spamming connection loop
            return False

        if self.source_type == "http_snapshot":
            # Test snapshot endpoint
            if SourceResolver._test_snapshot(self.resolved_url):
                self.stats.status = CameraStatus.ONLINE
                self.stats.reconnect_count += 1
                logger.info(f"[CameraWorker] {self.camera_id} connected via snapshot: {self.resolved_url}")
                return True
            else:
                self.stats.consecutive_failures += 1
                return False

        # VideoCapture-based sources (webcam, rtsp, http_stream, file)
        try:
            if self.source_type == "webcam":
                idx = int(self.resolved_url) if self.resolved_url.isdigit() else 0
                self._cap = cv2.VideoCapture(idx)
                if not self._cap or not self._cap.isOpened():
                    self._cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if self._cap and self._cap.isOpened():
                    try:
                        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
            elif self.source_type == "rtsp":
                # Fast 1.5s timeout prevents 30-second FFmpeg blocking on unreachable camera streams
                try:
                    self._cap = cv2.VideoCapture(
                        self.resolved_url,
                        cv2.CAP_FFMPEG,
                        [
                            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1500,
                            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1500,
                        ]
                    )
                except Exception:
                    self._cap = cv2.VideoCapture(self.resolved_url)
            else:
                self._cap = cv2.VideoCapture(self.resolved_url)

            if self._cap.isOpened():
                # Read a test frame
                ret, test_frame = self._cap.read()
                if ret and test_frame is not None:
                    self.stats.status = CameraStatus.ONLINE
                    self.stats.consecutive_failures = 0
                    self.stats.reconnect_count += 1
                    logger.info(f"[CameraWorker] {self.camera_id} connected: {self.resolved_url}")
                    return True
                else:
                    self._release_cap()
                    self.stats.consecutive_failures += 1
                    return False
            else:
                self._release_cap()
                self.stats.consecutive_failures += 1
                return False
        except Exception as e:
            logger.error(f"[CameraWorker] {self.camera_id} connect error: {e}")
            self._release_cap()
            self.stats.consecutive_failures += 1
            return False

    def _capture_video_frame(self):
        """Capture a single frame from VideoCapture source."""
        if self._cap is None or not self._cap.isOpened():
            self.stats.consecutive_failures += 1
            return

        t_start = time.time()

        # Flush stale network stream buffer for live HTTP/RTSP streams to prevent freezing & latency buildup
        if self.source_type in ["http_stream", "rtsp"]:
            try:
                for _ in range(2):
                    self._cap.grab()
            except Exception:
                pass

        ret, frame = self._cap.read()

        if not ret or frame is None:
            # For file sources, seamlessly loop back to frame 0
            if self.source_type == "file":
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()

            if not ret or frame is None:
                self.stats.frames_dropped += 1
                self.stats.consecutive_failures += 1
                if self.stats.consecutive_failures > self.MAX_CONSECUTIVE_FAIL:
                    self.stats.status = CameraStatus.RECONNECTING
                    self._release_cap()
                return

        # Reset failure counter on success
        self.stats.consecutive_failures = 0

        # Validate frame
        if not FrameValidator.is_valid(frame):
            self.stats.frames_dropped += 1
            return

        # Resize to target resolution
        frame = cv2.resize(frame, self.resolution)

        # Quality check (flag but still buffer)
        if not FrameValidator.is_quality_ok(frame):
            self.stats.status = CameraStatus.NO_VIDEO_INPUT
        else:
            self.stats.status = CameraStatus.ONLINE

        # Push to buffer
        self.frame_buffer.push(frame)
        self.stats.frames_captured += 1
        self.stats.last_frame_time = time.time()
        self.stats.latency_ms = (time.time() - t_start) * 1000

        # Update FPS
        self._update_fps()

    def _capture_snapshot(self):
        """Capture a frame via HTTP snapshot polling."""
        t_start = time.time()
        try:
            req = urllib.request.urlopen(self.resolved_url, timeout=2.0)
            data = req.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                self.stats.frames_dropped += 1
                self.stats.consecutive_failures += 1
                return

            self.stats.consecutive_failures = 0
            frame = cv2.resize(frame, self.resolution)

            if not FrameValidator.is_quality_ok(frame):
                self.stats.status = CameraStatus.NO_VIDEO_INPUT
            else:
                self.stats.status = CameraStatus.ONLINE

            self.frame_buffer.push(frame)
            self.stats.frames_captured += 1
            self.stats.last_frame_time = time.time()
            self.stats.latency_ms = (time.time() - t_start) * 1000
            self._update_fps()

        except Exception as e:
            self.stats.frames_dropped += 1
            self.stats.consecutive_failures += 1
            if self.stats.consecutive_failures > self.MAX_CONSECUTIVE_FAIL:
                self.stats.status = CameraStatus.RECONNECTING
                logger.warning(f"[CameraWorker] {self.camera_id} snapshot failures exceeded threshold")

    def _update_fps(self):
        """Track rolling FPS over the last N frames."""
        now = time.time()
        self._frame_times.append(now)
        if len(self._frame_times) > self.FPS_WINDOW:
            self._frame_times.pop(0)
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self.stats.fps = (len(self._frame_times) - 1) / elapsed

    def _release_cap(self):
        """Safely release VideoCapture."""
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None


# ============================================================
# AI Processing Queue — Decouples capture from inference
# ============================================================

@dataclass
class AIQueueItem:
    """A frame queued for AI processing."""
    camera_id: str
    camera_name: str
    frame: np.ndarray
    gps_coords: tuple
    zones: list
    timestamp: float = field(default_factory=time.time)


class AIProcessingQueue:
    """
    Thread-safe queue that decouples camera capture from AI inference.
    Camera workers push frames at capture FPS. AI consumer processes at inference speed.
    Latest-frame-wins: if the queue is full, old frames are dropped.
    """

    def __init__(self, maxsize: int = 10):
        self._queue: queue.Queue[AIQueueItem] = queue.Queue(maxsize=maxsize)

    def submit(self, item: AIQueueItem) -> bool:
        """Submit a frame for AI processing. Drops oldest if full."""
        try:
            if self._queue.full():
                try:
                    self._queue.get_nowait()  # Drop oldest
                except queue.Empty:
                    pass
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            return False

    def get(self, timeout: float = 0.1) -> Optional[AIQueueItem]:
        """Get next frame for processing. Returns None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def pending(self) -> int:
        return self._queue.qsize()


# ============================================================
# Camera Manager — Singleton orchestrator
# ============================================================

class CameraManager:
    """
    Singleton service managing all camera workers, frame buffers,
    and the AI processing queue.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._workers: dict[str, CameraWorker] = {}
        self._lock = threading.Lock()
        self.ai_queue = AIProcessingQueue(maxsize=20)
        self._initialized = True
        logger.info("[CameraManager] Initialized")

    def start_camera(self, camera_id: str, camera_name: str, source_url: str,
                     target_fps: int = 15) -> bool:
        """Start a camera worker. Returns True if started successfully."""
        with self._lock:
            if camera_id in self._workers:
                # Already running — update source if different
                worker = self._workers[camera_id]
                if worker.source_url != source_url:
                    worker.update_source(source_url)
                return True

            worker = CameraWorker(
                camera_id=camera_id,
                camera_name=camera_name,
                source_url=source_url,
                target_fps=target_fps,
            )
            worker.start()
            self._workers[camera_id] = worker
            return True

    def stop_camera(self, camera_id: str) -> bool:
        """Stop a camera worker. Returns True if stopped."""
        with self._lock:
            worker = self._workers.pop(camera_id, None)
            if worker:
                worker.stop()
                try:
                    from app.services.ai_service import AIService
                    AIService().cleanup_camera(camera_id)
                except Exception:
                    pass
                return True
            return False

    def stop_all(self):
        """Stop all camera workers (called on app shutdown)."""
        with self._lock:
            for cid, worker in self._workers.items():
                logger.info(f"[CameraManager] Stopping {cid}...")
                worker.stop()
            self._workers.clear()
        logger.info("[CameraManager] All cameras stopped")

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get the latest captured frame for a camera."""
        with self._lock:
            worker = self._workers.get(camera_id)
        if worker:
            return worker.frame_buffer.get_latest()
        return None

    def get_frame_buffer(self, camera_id: str) -> list[np.ndarray]:
        """Get all buffered frames for evidence clip generation."""
        with self._lock:
            worker = self._workers.get(camera_id)
        if worker:
            return worker.frame_buffer.get_all()
        return []

    def get_health(self, camera_id: str) -> Optional[dict]:
        """Get health stats for a single camera."""
        with self._lock:
            worker = self._workers.get(camera_id)
        if worker:
            return worker.stats.to_dict()
        return None

    def get_all_health(self) -> list[dict]:
        """Get health stats for all cameras."""
        with self._lock:
            workers = list(self._workers.values())
        return [w.stats.to_dict() for w in workers]

    def is_camera_running(self, camera_id: str) -> bool:
        with self._lock:
            return camera_id in self._workers

    def update_camera_source(self, camera_id: str, new_url: str) -> bool:
        """Update the source URL for a running camera."""
        with self._lock:
            worker = self._workers.get(camera_id)
        if worker:
            worker.update_source(new_url)
            return True
        return False

    def update_camera_name(self, camera_id: str, new_name: str) -> bool:
        """Update the display name for a running camera worker."""
        with self._lock:
            worker = self._workers.get(camera_id)
            if worker:
                worker.camera_name = new_name
                worker.stats.camera_name = new_name
                return True
        return False

    @staticmethod
    def detect_available_webcams(max_index: int = 5) -> list[dict]:
        """Probe local webcam indices 0-N and return available devices."""
        available = []
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if ret and frame is not None:
                    available.append({
                        "index": idx,
                        "name": f"Webcam {idx}",
                        "resolution": f"{w}x{h}",
                        "available": True,
                    })
            else:
                cap.release()
        return available


# Module-level singleton
camera_manager = CameraManager()
