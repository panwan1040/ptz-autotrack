from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import CameraSection, VideoSection
from app.logging_config import get_logger
from app.models.runtime import FramePacket

logger = get_logger(__name__)


@dataclass(slots=True)
class ReaderStats:
    frames_received: int = 0
    reconnect_count: int = 0
    last_source_fps: float = 0.0


class RtspReader:
    """Low-latency RTSP reader with reconnects and frame dropping."""

    def __init__(self, camera: CameraSection, video: VideoSection) -> None:
        self._camera = camera
        self._video = video
        self._queue: queue.Queue[FramePacket] = queue.Queue(maxsize=video.queue_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._stats = ReaderStats()
        self._frame_index = 0
        self._last_frame_ts = 0.0

    @property
    def stats(self) -> ReaderStats:
        return self._stats

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="rtsp-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def read(self, timeout: float = 1.0) -> FramePacket | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if not self._open_capture():
                time.sleep(self._video.reconnect_backoff_seconds)
                continue
            while not self._stop_event.is_set():
                assert self._capture is not None
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    logger.warning("rtsp_read_failed")
                    self._stats.reconnect_count += 1
                    self._capture.release()
                    self._capture = None
                    time.sleep(self._video.read_retry_delay_seconds)
                    break
                self._frame_index += 1
                timestamp = time.monotonic()
                self._stats.frames_received += 1
                fps = self._compute_source_fps(timestamp)
                packet = FramePacket(
                    frame=self._resize(frame),
                    timestamp=timestamp,
                    source_fps=fps,
                    frame_index=self._frame_index,
                )
                self._push_latest(packet)

    def _open_capture(self) -> bool:
        try:
            self._capture = cv2.VideoCapture(self._camera.rtsp_url, cv2.CAP_FFMPEG)
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self._capture.isOpened():
                logger.warning("rtsp_open_failed")
                self._stats.reconnect_count += 1
                return False
            logger.info("rtsp_connected")
            return True
        except Exception as exc:
            logger.error("rtsp_open_exception", error=str(exc))
            self._stats.reconnect_count += 1
            return False

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, (self._video.resize_width, self._video.resize_height))

    def _push_latest(self, packet: FramePacket) -> None:
        while True:
            try:
                self._queue.put_nowait(packet)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    return

    def _compute_source_fps(self, timestamp: float) -> float:
        if self._last_frame_ts <= 0:
            self._last_frame_ts = timestamp
            return 0.0
        delta = max(1e-6, timestamp - self._last_frame_ts)
        self._last_frame_ts = timestamp
        self._stats.last_source_fps = 1.0 / delta
        return self._stats.last_source_fps
