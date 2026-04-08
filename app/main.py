from __future__ import annotations

from app.camera.rtsp_reader import RtspReader
from app.config import load_config
from app.detection.yolo_detector import YoloDetector
from app.logging_config import configure_logging, get_logger
from app.services.tracking_service import TrackingService
from app.tracking.tracker import Tracker
from app.control.ptz_client import DahuaPtzClient


def build_service(detect_only_override: bool | None = None) -> TrackingService:
    config = load_config()
    if detect_only_override is True:
        config.app.detect_only = True
    configure_logging(config.app.log_level)
    logger = get_logger(__name__)
    logger.info("config_loaded", config=config.sanitized_dump())
    reader = RtspReader(config.camera, config.video)
    detector = YoloDetector(config.detection)
    tracker = Tracker(config.tracking)
    ptz_client = DahuaPtzClient(
        config.camera,
        config.ptz,
        detect_only=config.app.detect_only,
    )
    return TrackingService(config, reader, detector, tracker, ptz_client)


def run() -> None:
    service = build_service()
    service.start()


if __name__ == "__main__":
    run()
